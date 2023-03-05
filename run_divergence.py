import argparse
from os import environ
from pathlib import Path

parser = argparse.ArgumentParser(prog="ResNet divergence script")
parser.add_argument(
    "network0", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument(
    "network1", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument("--reset0", action="store_true")
parser.add_argument("--reset1", action="store_true")
parser.add_argument("--inflate0", choices=["resnet50", "resnet101"])
parser.add_argument("--inflate1", choices=["resnet50", "resnet101"])
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument(
    "--imagenet_path",
    default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"),
    type=Path,
)
args = parser.parse_args()

from local import resnet
from local import stats
from local import train


network0 = resnet.network_load(args.network0, args.inflate0, args.reset0)
name0 = resnet.network_name(args.network0, args.inflate0, args.reset0)
network1 = resnet.network_load(args.network1, args.inflate1, args.reset1)
name1 = resnet.network_name(args.network1, args.inflate1, args.reset1)

train_data = train.data(args.imagenet_path / "train")

print(f"Divergence: {name0} <-> {name1}")
stats.divergence(network1, network1, train_data)
