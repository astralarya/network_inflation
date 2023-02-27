import argparse

parser = argparse.ArgumentParser(
    prog="ResNet divergence script"
)
parser.add_argument('network0', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('network1', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()


from local import inflate
from local import imagenet
from local import resnet


network0 = getattr(resnet, args.network0, lambda: None)()
network1 = getattr(resnet, args.network1, lambda: None)()

if network0 is None:
    print(f"Unknown network: {args.network0}")
    exit(1)
if network1 is None:
    print(f"Unknown network: {args.network1}")
    exit(1)


train_data = imagenet.train_data("/mnt/imagenet/imagenet-1k/train/")

print(f"Self-divergence: {args.network1}")
imagenet.divergence(network1, network1, train_data)

print(f"Self-divergence: {args.network0}")
imagenet.divergence(network0, network0, train_data)

print(f"Divergence: {args.network0} - {args.network1}")
imagenet.divergence(network1, network1, train_data)

print(f"Divergence: {args.network0} - inflated({args.network0}, {args.network1})")
inflate.resnet(network0, network1)
imagenet.divergence(network0, network1, train_data)