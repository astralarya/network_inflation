import argparse

from local import imagenet
from local import model
from local import resnet


parser = argparse.ArgumentParser(
    prog="ResNet training script"
)
parser.add_argument('name')
parser.add_argument('--network', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--batch_size', default=64, type=int)
args = parser.parse_args()

name = args.name
network = getattr(resnet, args.network, lambda: None)()

if network is None:
    print(f"Unknown network: {args.network}")
    exit(1)


eval_data = imagenet.eval_data("/mnt/imagenet/imagenet-1k/val/")

imagenet.eval(network, name, eval_data, batch_size=args.batch_size)

