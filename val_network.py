import argparse
from os import environ
from pathlib import Path

parser = argparse.ArgumentParser(
    prog="ResNet validation script"
)
parser.add_argument('network', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--inflate', choices=['resnet50', 'resnet101'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default="pre", type=lambda x: x if x in ["pre", "init", "all"] else int(x))
parser.add_argument('--model_dir', default="models", type=Path)
parser.add_argument('--imagenet_path', default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"), type=Path)
args = parser.parse_args()


from local import inflate
from local import imagenet
from local import model
from local import resnet


name = args.network
if args.inflate is not None:
    name = f"{name}--inflate-{args.inflate}"

network = getattr(resnet, args.network, lambda: None)()
if network is None:
    print(f"Unknown network: {args.network}")
    exit(1)

val_data = imagenet.val_data(args.imagenet_path / "val")

print(f"Validating model {name}")
if args.epoch == "pre":
    if args.inflate is not None:
        inflate_source = getattr(resnet, args.inflate, lambda: None)()
        if inflate_source is None:
            print(f"Unknown network: {args.inflate}")
            exit(1)
        print(f"Inflating network ({args.network}) from {args.inflate}")
        inflate.resnet(inflate_source, network)
    else:
        print("Using pretrained weights")
elif args.epoch == "init":
    print(f"Reset network ({args.network})")
    model.reset(network)

if args.epoch == "all":
    print("Validating all epochs")
    imagenet.run_val(network, args.model_dir / name, val_data, batch_size=args.batch_size)
else:
    print(f"Validating epoch {args.epoch}")
    imagenet.val_epoch(network, args.model_dir / name, val_data, args.epoch, batch_size=args.batch_size)

