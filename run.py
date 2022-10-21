import argparse 
import importlib
from pathlib import Path

from utils import load_args

parser = argparse.ArgumentParser(description="Semantic Segmentation Training")
parser.add_argument('--script-dir', type=str, required=True, help='Indicate directory for main file')
parser.add_argument('--config-file', type=str, required=True, help='Path of JSON config file')
parser.add_argument('--eval', type=str, help='Run evaluation with model at given path')
args = parser.parse_args()

config_file = Path(__file__).parent / 'scripts' / args.script_dir / args.config_file
train_args = load_args(config_file)

if args.eval is not None:
    train_args.resume = args.eval
    train_args.eval_only = True

module = importlib.import_module("scripts.{}.main".format(args.script_dir))
module.main(train_args)