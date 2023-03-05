import argparse
from pathlib import Path

from local import checkpoint

parser = argparse.ArgumentParser(prog="Prune checkpoints")
parser.add_argument("path", type=Path)
parser.add_argument("--keep", default=4, type=int)
args = parser.parse_args()


if __name__ == "__main__":
    checkpoint.prune_epochs(args.path, args.keep)
