import torch

from trainer import Trainer
import builtins
from inspect import getframeinfo, stack
original_print = print

def print_wrap(*args, **kwargs):
    caller = getframeinfo(stack()[1][0])
    original_print("FN:",caller.filename,"Line:", caller.lineno,"Func:", caller.function,":::", *args, **kwargs)

builtins.print = print_wrap

def main(args):
    args.logger.info(vars(args))
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.run()
