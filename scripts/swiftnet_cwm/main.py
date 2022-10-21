import torch
from torch.utils.data import DataLoader

from trainer import Trainer
from models import SwiftNetCWM, Stepper, MaskGenerator
from dataloaders import CityscapesBase


class TrainerCWM(Trainer):
    def get_model_and_params(self, num_classes):
        
        self.stepper = Stepper()
        self.mask_generator = MaskGenerator(self.args.masks, generator=self.args.generator)

        model = SwiftNetCWM(self.args.backbone, num_classes)
        model.set_width_mult(self.args.width_mult)
        model.set_stepper(self.stepper)
        model.set_mask_generator(self.mask_generator)

        if self.args.pretrained is not None:
            model.load_state_dict(torch.load(self.args.pretrained)["state_dict"])
            train_params = [
                {'params': model.parameters(), 'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
            ]
        else:
            train_params = [
                {'params': model.random_init_params(), 'lr': self.args.lr, 'weight_decay': self.args.weight_decay},
                {'params': model.fine_tune_params(), 'lr': self.args.lr / 4, 'weight_decay': self.args.weight_decay / 4}
            ]
        
        return model, train_params

    def get_data_loaders(self):
        kwargs = {'num_workers': self.args.workers, 'pin_memory': True}
        kwargs_val = {'num_workers': self.args.workers_val, 'pin_memory': True}
        train_set = CityscapesBase(**self.args.db, split='train')
        val_set = CityscapesBase(**self.args.db, split='val')
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=self.args.val_batch_size, shuffle=False, **kwargs_val)
        return train_loader, val_loader, train_set.num_classes

    def process_sample(self, sample, index, train=True):
        out_loss = 0
        n = len(self.args.frame_numbers) - 1
        assert 0 in self.args.mix

        for s in self.args.mix:
            self.model.module.reset()
            self.stepper.reset()

            for i in range(s, n):
                image = sample['image_%d' % i].to(self.device)
                self.model(image)
                self.stepper.step()

            image = sample['image_%d' % n].to(self.device)
            target = sample['label'].to(self.device)
            output = self.model(image)

            loss = self.criterion(output, target)
            out_loss += loss.detach()

            if train:
                loss.backward()

        if train:
            if ((index+1) % self.args.optim_step_each == 0):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

        return out_loss.item(), image, target, output


def main(args):
    args.logger.info(vars(args))
    torch.manual_seed(args.seed)
    trainer = TrainerCWM(args)
    trainer.run()
