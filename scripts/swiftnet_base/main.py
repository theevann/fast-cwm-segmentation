import torch

from trainer import Trainer


def main(args):
    args.logger.info(vars(args))
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.run()
