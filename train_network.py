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
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()

name = args.name
network = getattr(resnet, args.network, lambda: None)()

if network is None:
    print(f"Unknown network: {args.network}")
    exit(1)


train_data = imagenet.train_data("/mnt/imagenet/imagenet-1k/train/")

imagenet.train(
    network,
    name,
    train_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    init_fn=model.reset
)

