import argparse
from os import environ
from pathlib import Path

from local import train

parser = argparse.ArgumentParser(prog="ResNet training script")
parser.add_argument(
    "name",
    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
)
parser.add_argument("--num_epochs", default=200, type=int)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=0.0001, type=float)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--nprocs", default=8, type=int)
parser.add_argument("--model_path", default="models", type=Path)
parser.add_argument(
    "--imagenet_path",
    default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"),
    type=Path,
)
args = parser.parse_args()


if __name__ == "__main__":
    train.train(**vars(args))
