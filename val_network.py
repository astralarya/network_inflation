import argparse
from os import environ
from pathlib import Path

from local import validate
from local.inflate import SequenceInflate

parser = argparse.ArgumentParser(prog="ResNet validation script")
parser.add_argument(
    "name", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--inflate_strategy", default="align-start", type=SequenceInflate)
parser.add_argument("--inflate_unmasked", action="store_false", dest="mask_inflate")
parser.add_argument("--nprocs", default=8, type=int)
parser.add_argument(
    "--epoch",
    dest="epochs",
    type=lambda x: x if x in ["pre", "all"] else int(x),
    action="append",
)
parser.add_argument("--from_epoch", default=0, type=int)
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


if __name__ == "__main__":
    validate.validate(**vars(args))
