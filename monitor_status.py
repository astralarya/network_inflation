import argparse
from os import environ
from pathlib import Path
from sys import stderr

from local import checkpoint
from local import storage
from local.inflate import SequenceInflate

parser = argparse.ArgumentParser(prog="ResNet validation script")
parser.add_argument("--all", dest="all", action="store_true")
parser.add_argument("--no_model_ema", dest="model_ema", action="store_false")
args = parser.parse_args()

if __name__ == "__main__":
    print("Fetching status...", flush=True, end="", file=stderr)
    todo = []
    for name in checkpoint.iter_models():
        outname = f"{name}--ema" if args.model_ema else name
        outfile = f"{outname}.__val__.log"
        epoch = checkpoint.get_epoch(name)
        log_epoch = checkpoint.log_epoch(outfile)
        if args.all or epoch != log_epoch:
            todo.append((name, f"{log_epoch}/{epoch}"))
    print("DONE", file=stderr)
    for name, status in todo:
        print(name, status)
