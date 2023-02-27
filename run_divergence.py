import argparse
from os import environ
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="ResNet divergence script"
)
parser.add_argument('network0', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('network1', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--reset0', action='store_true')
parser.add_argument('--reset1', action='store_true')
parser.add_argument('--inflate0', choices=['resnet50', 'resnet101'])
parser.add_argument('--inflate1', choices=['resnet50', 'resnet101'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--imagenet_path', default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"), type=Path)
args = parser.parse_args()


from local import inflate
from local import imagenet
from local import model
from local import resnet


network0 = getattr(resnet, args.network0, lambda: None)()
network1 = getattr(resnet, args.network1, lambda: None)()
name0 = args.network0
name1 = args.network1

if network0 is None:
    print(f"Unknown network: {args.network0}")
    exit(1)
if network1 is None:
    print(f"Unknown network: {args.network1}")
    exit(1)


if args.reset0 is None:
    print(f"Reset network: {args.network0}")
    model.reset(network0)
    name0 = f"{name0}-reset"
if args.reset1 is None:
    print(f"Reset network: {args.network1}")
    model.reset(network1)
    name1 = f"{name1}-reset"

if args.inflate0 is not None:
    inflate_source0 = getattr(resnet, args.inflate0, lambda: None)()
    if args.inflate0 is not None and inflate_source0 is None:
        print(f"Unknown network: {args.inflate0}")
        exit(1)
    print(f"Inflating network0 ({args.network0}) from {args.inflate0}")
    inflate.resnet(inflate_source0, network0)
    name0 = f"{name0}--inflate-{args.inflate0}"
if args.inflate1 is not None:
    inflate_source1 = getattr(resnet, args.inflate1, lambda: None)()
    if args.inflate1 is not None and inflate_source1 is None:
        print(f"Unknown network: {args.inflate1}")
        exit(1)
    print(f"Inflating network1 ({args.network1}) from {args.inflate1}")
    inflate.resnet(inflate_source1, network1)
    name1 = f"{name1}--inflate-{args.inflate1}"


train_data = imagenet.train_data(args.imagenet_path / "train")

print(f"Divergence: {name0} <-> {name1}")
imagenet.divergence(network1, network1, train_data)