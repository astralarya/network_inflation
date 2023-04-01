import argparse
from os import environ
from pathlib import Path

from local import train
from local.inflate import SequenceInflate

parser = argparse.ArgumentParser(prog="ResNet training script")
parser.add_argument(
    "name",
    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
)
parser.add_argument("--suffix", help="Name suffix for save/load")
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument(
    "--inflate_strategy",
    default="space-evenly",
    type=SequenceInflate,
    choices=list(SequenceInflate),
)
parser.add_argument("--inflate_unmasked", action="store_false", dest="mask_inflate")
parser.add_argument("--nprocs", default=8, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_epochs", default=600, type=int)
parser.add_argument("--opt", default="sgd", type=str)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--lr", default=0.5, type=float)
parser.add_argument("--lr_scheduler", default="cosineannealinglr", type=str)
parser.add_argument("--lr_step_size", default=30, type=int)
parser.add_argument("--lr_gamma", default=0.1, type=float)
parser.add_argument("--lr_min", default=0.0, type=float)
parser.add_argument("--lr_warmup_epochs", default=5, type=int)
parser.add_argument("--weight_decay", default=2e-5, type=float)
parser.add_argument("--norm_weight_decay", default=0.0, type=float)
parser.add_argument("--label_smoothing", default=0.1, type=float)
parser.add_argument("--mixup_alpha", default=0.2, type=float)
parser.add_argument("--cutmix_alpha", default=1.0, type=float)
parser.add_argument("--random_erase", default=0.1, type=float)
parser.add_argument("--model_ema_steps", default=32, type=int)
parser.add_argument("--model_ema_decay", default=0.9998, type=float)
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
    train.train(**vars(args))
