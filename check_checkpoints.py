import argparse
from pathlib import Path

from local import checkpoint

parser = argparse.ArgumentParser(prog="Check checkpoints")
parser.add_argument("path", type=Path)
args = parser.parse_args()


if __name__ == "__main__":
    missing = checkpoint.check_epochs(args.path, prefix=None)
    if len(missing) == 0:
        exit(1)
