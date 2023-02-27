import argparse

parser = argparse.ArgumentParser(
    prog="ResNet training script"
)
parser.add_argument('name')
parser.add_argument('--network', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--inflate', choices=['resnet50', 'resnet101'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
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


train_data = imagenet.train_data("/mnt/imagenet/imagenet-1k/train/")

imagenet.train(
    network,
    name,
    train_data,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    init_fn=model.reset if inflate_source is None else lambda x: inflate.resnet(inflate_source, x)

)

