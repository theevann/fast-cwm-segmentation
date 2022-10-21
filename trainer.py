import os, sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.sync_batchnorm.replicate import patch_replication_callback
from models import SwiftNet, DeepLab

from dataloaders import CityscapesBase
from utils import Saver, Evaluator, TensorboardSummary


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.device = torch.device(args.gpu_ids[0] if args.cuda else 'cpu')

        # Define Dataloader
        self.train_loader, self.val_loader, num_classes = self.get_data_loaders()

        # Define Network
        model, train_params = self.get_model_and_params(num_classes)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        self.evaluator = Evaluator(num_classes)
 
        # Using DataParallel
        if args.cuda:
            self.model = nn.DataParallel(model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)

        if args.eval_only:
            self.init_eval_mode()
        else:
            self.init_train_mode(train_params)


    def init_train_mode(self, train_params):
        args = self.args

        # Define Saver and Tensorboard Summary
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.writer = TensorboardSummary(self.saver.experiment_dir, self.val_loader.dataset)
        self.logger.info('\nExperiment dir : %s', self.saver.experiment_dir)

        # Define Optimizer and Scheduler
        if args.optim == 'sgd':
            self.optimizer = SGD(train_params, weight_decay=args.weight_decay,
                                             momentum=args.momentum, nesterov=args.nesterov)
        elif args.optim == 'adam':
            self.optimizer = Adam(train_params, weight_decay=args.weight_decay)

        if args.scheduler == 'poly':
            update_func = lambda epoch: (1 - 1.0 * epoch / args.epochs) ** args.poly_power
            self.scheduler = LambdaLR(self.optimizer, update_func)
        elif args.scheduler == 'cos':
            self.scheduler = CosineAnnealingLR(self.optimizer, args.epochs, eta_min=args.cos_eta_min)

        # Resuming checkpoint
        self.best_pred = 0.0
        self.args.start_epoch = 0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("No checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                if 'scheduler' in checkpoint.keys():
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                else:
                    print("ALERT: No scheduler found in checkpoint")
                    for _ in range(args.start_epoch): self.scheduler.step()
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.logger.info("Loaded checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

        self.logger.info('Using %s architecture' % args.archi)
        self.logger.info('Using %s backbone' % args.backbone)
        self.logger.info('Using %s optimizer' % args.optim)
        self.logger.info('Using %s scheduler' % args.scheduler)
        self.logger.info('Using %s images / batch [train]' % args.batch_size)
        self.logger.info('Using %s images / batch [val]' % args.val_batch_size)


    def init_eval_mode(self):
        args = self.args

        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("No checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info("Loaded checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])

        self.logger.info('Using %s architecture' % args.archi)
        self.logger.info('Using %s backbone' % args.backbone)
        self.logger.info('Using %s images / batch [val]' % args.val_batch_size)


    def get_data_loaders(self):
        kwargs = {'num_workers': self.args.workers, 'pin_memory': True}
        train_set = CityscapesBase(**self.args.db, split='train')
        val_set = CityscapesBase(**self.args.db, split='val')
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=self.args.val_batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader, train_set.num_classes


    def get_model_and_params(self, num_classes):
        args = self.args
        norm_layer = SynchronizedBatchNorm2d if args.sync_bn else nn.BatchNorm2d

        if args.archi == 'deeplab':
            model = DeepLab(num_classes=num_classes,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            norm_layer=norm_layer,
                            freeze_bn=args.freeze_bn,
                            image_lvl_feat=not args.no_image_lvl_feat)

            train_params = [{'params': model.fine_tune_params(), 'lr': args.lr / 10, 'weight_decay': args.weight_decay / 10},
                            {'params': model.random_init_params(), 'lr': args.lr, 'weight_decay': args.weight_decay}]

        elif args.archi == 'swiftnet':
            model = SwiftNet(backbone=args.backbone, num_classes=num_classes)
            train_params = [{'params': model.random_init_params(), 'lr': args.lr, 'weight_decay': args.weight_decay},
                            {'params': model.fine_tune_params(), 'lr': args.lr / 4, 'weight_decay': args.weight_decay / 4}]
        
        else:
            raise NotImplementedError("Architecture {} is not implemented".format(args.archi))

        return model, train_params


    def run(self):
        if self.args.eval_only:
            self.validate()
        else:
            self.train()


    def train(self):
        self.logger.info('\n' + '-' * 20)
        self.logger.info('Starting Training')
        self.logger.info('Starting Epoch: %d' % self.args.start_epoch)
        self.logger.info('Total Epoches: %d\n' % self.args.epochs)

        for epoch in range(self.args.start_epoch, self.args.epochs):
            is_best = False
            is_last = (epoch + 1) == self.args.epochs
            
            self.train_epoch(epoch)

            if (epoch + 1) % self.args.eval_interval == 0 or is_last:
                metrics = self.validate(epoch)
                self.writer.write_metrics(metrics, epoch)
                is_best = metrics['mIoU'] > self.best_pred
                self.best_pred = max(metrics['mIoU'], self.best_pred)

            if is_best or self.args.save_every_epoch or is_last:
                self.save(epoch, is_best=is_best)

        self.writer.close()


    def train_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        train_loss = 0
        tbar = tqdm(self.train_loader, file=sys.stdout)
        num_img_tr = len(self.train_loader)
        self.train_loader.dataset.seed = epoch
        for i, sample in enumerate(tbar):
            loss, image, target, output = self.process_sample(sample, index=i)
            train_loss += loss

            if i % 50 == 0:
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss, i + num_img_tr * epoch)

            # Show inference results each half of epoch
            if i % (num_img_tr // 2) == 0:
                global_step = i + num_img_tr * epoch
                self.writer.visualize_image(image, target, output, global_step)

        self.scheduler.step()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        self.logger.info('[End of Epoch %d]' % epoch)
        self.logger.info('Loss: %.3f' % train_loss)


    def validate(self, epoch=0):
        torch.set_grad_enabled(False)
        self.evaluator.reset()
        self.model.eval()
        test_loss = 0.0
        tbar = tqdm(self.val_loader, desc='\r', file=sys.stdout)
        for i, sample in enumerate(tbar):
            loss, image, target, output = self.process_sample(sample, index=i, train=False)
            test_loss += loss

            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.argmax(dim=1).cpu().numpy()
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)

        metrics = self.evaluator.get_all_metrics()
        metrics['total_loss_epoch'] = test_loss

        self.logger.info('Validation: [Epoch: %d]' % epoch)
        self.logger.info("Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {fwIoU}".format(**metrics))
        self.logger.info('Loss: %.3f' % test_loss)

        return metrics


    def process_sample(self, sample, index, *, train=True):
        image = sample['image_0'].to(self.device)
        target = sample['label'].to(self.device)
        output = self.model(image)
        loss = self.criterion(output, target)
        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), image, target, output


    def save(self, epoch, is_best=False):
        self.saver.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_pred': self.best_pred,
        }, is_best=is_best)