import argparse
from os import environ
from pathlib import Path

from local import validate

parser = argparse.ArgumentParser(prog="ResNet validation script")
parser.add_argument(
    "name", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument(
    "--epoch",
    type=lambda x: x if x in ["pre", "init", "all"] else int(x),
    action="append",
)
parser.add_argument("--model_path", default="models", type=Path)
parser.add_argument(
    "--imagenet_path",
    default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"),
    type=Path,
)
args = parser.parse_args()


if __name__ == "__main__":
    validate.validate(**vars(args))
