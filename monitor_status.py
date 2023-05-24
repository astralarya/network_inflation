import argparse
from os import environ
from pathlib import Path

from local import checkpoint
from local import storage
from local.inflate import SequenceInflate

parser = argparse.ArgumentParser(prog="ResNet validation script")
parser.add_argument("--no_model_ema", dest="model_ema", action="store_false")
parser.add_argument(
    "--model_path",
    default=environ.get("MODEL_PATH", "/mnt/models/data"),
    type=Path,
)
args = parser.parse_args()

if args.model_path:
    storage.set_file_path(args.model_path)

args = {key: vars(args)[key] for key in vars(args) if key not in ["model_path"]}

if __name__ == "__main__":
    todo = []
    for name in checkpoint.iter_models():
        outname = f"{name}--ema" if args.model_ema else name
        outfile = f"{outname}.__val__.log"
        epoch = checkpoint.get_epoch(name)
        log_epoch = checkpoint.log_epoch(outfile)
        if epoch != log_epoch:
            todo.append(name)
    print(name)
