import argparse
from os import environ
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="ResNet training script"
)
parser.add_argument('network', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--finetune', action="store_true")
parser.add_argument('--inflate', choices=['resnet50', 'resnet101'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--model_dir', default="models", type=Path)
parser.add_argument('--imagenet_path', default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"), type=Path)
args = parser.parse_args()


from local import inflate
from local import imagenet
from local import model
from local import resnet


name = args.network
if args.finetune is True:
    name = f"{name}--finetune"
if args.inflate is not None:
    name = f"{name}--inflate-{args.inflate}"

network = getattr(resnet, args.network, lambda: None)()
if network is None:
    print(f"Unknown network: {args.network}")
    exit(1)

def inflate_fn(x):
    inflate_source = getattr(resnet, args.inflate, lambda: None)()
    if inflate_source is None:
        print(f"Unknown network: {args.inflate}")
        exit(1)
    print(f"Inflating network ({args.network}) from {args.inflate}")
    inflate.resnet(inflate_source, x)

train_data = imagenet.train_data(args.imagenet_path / "train")

init_fn=model.reset if args.inflate is None else inflate_fn
imagenet.train(
    network,
    args.model_dir / name,
    train_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    init_fn=init_fn if args.finetune is False else None,

)

