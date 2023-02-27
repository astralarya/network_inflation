import argparse

parser = argparse.ArgumentParser(
    prog="ResNet validation script"
)
parser.add_argument('name')
parser.add_argument('--network', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--inflate', choices=['resnet50', 'resnet101'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default="all", type=lambda x: x if x in ["pre", "init", "all"] else int(x))
args = parser.parse_args()


from local import inflate
from local import imagenet
from local import model
from local import resnet


name = args.name
network = getattr(resnet, args.network, lambda: None)()
inflate_source = getattr(resnet, args.network, lambda: None)()

if network is None:
    print(f"Unknown network: {args.network}")
    exit(1)

if args.inflate is not None and inflate_source is None:
    print(f"Unknown network: {args.inflate}")
    exit(1)


val_data = imagenet.val_data("/mnt/imagenet/imagenet-1k/val/")

if args.epoch == "pre":
    if args.inflate is not None:
        inflate.resnet(inflate_source, network)
    imagenet.run_val(network, name, val_data, batch_size=args.batch_size)
elif args.epoch == "init":
    model.reset(network)
    imagenet.run_val(network, name, val_data, batch_size=args.batch_size)
elif args.epoch == "all":
    imagenet.run_val(network, name, val_data, batch_size=args.batch_size)
elif type(args.epoch) == "int":
    imagenet.val_epoch(network, name, val_data, args.epoch, batch_size=args.batch_size)

