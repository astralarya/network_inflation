import argparse
from os import environ
from pathlib import Path

from local import storage
from local import validate
from local.inflate import SequenceInflate

parser = argparse.ArgumentParser(prog="ResNet validation script")
parser.add_argument(
    "name", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument("--suffix", help="Name suffix for save/load")
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument(
    "--inflate_strategy",
    default="space-evenly",
    type=SequenceInflate,
    choices=list(SequenceInflate),
)
parser.add_argument("--inflate_unmasked", action="store_false", dest="mask_inflate")
parser.add_argument("--model_ema", action="store_true")
parser.add_argument("--nprocs", default=8, type=int)
parser.add_argument(
    "--epoch",
    dest="epochs",
    type=lambda x: x if x in ["pre", "all"] else int(x),
    action="append",
)
parser.add_argument(
    "--model_path",
    default=environ.get("MODEL_PATH", "/mnt/models/data"),
    type=Path,
)
parser.add_argument(
    "--imagenet_path",
    default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"),
    type=Path,
)
args = parser.parse_args()

if args.model_path:
    storage.set_file_path(args.model_path)

args = {key: args[key] for key in vars(args) if key not in ["model_path"]}

if __name__ == "__main__":
    validate.validate(**args)
