import argparse

parser = argparse.ArgumentParser(
    prog="ResNet divergence script"
)
parser.add_argument('network0', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('network1', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--reset0', action='store_true')
parser.add_argument('--reset1', action='store_true')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=8, type=int)
args = parser.parse_args()


from local import inflate
from local import imagenet
from local import model
from local import resnet


network0 = getattr(resnet, args.network0, lambda: None)()
network1 = getattr(resnet, args.network1, lambda: None)()

if network0 is None:
    print(f"Unknown network: {args.network0}")
    exit(1)
if network1 is None:
    print(f"Unknown network: {args.network1}")
    exit(1)

if args.reset0 is None:
    print(f"Reset network: {args.network0}")
    model.reset(network0)
if args.reset1 is None:
    print(f"Reset network: {args.network1}")
    model.reset(network1)

name0 = "init-" if args.reset0 else "" + args.network0
name1 = "init-" if args.reset1 else "" + args.network1


train_data = imagenet.train_data()

print(f"Self-divergence: {name1}")
imagenet.divergence(network1, network1, train_data)

print(f"Self-divergence: {name0}")
imagenet.divergence(network0, network0, train_data)

print(f"Divergence: {name0} - {name1}")
imagenet.divergence(network1, network1, train_data)

print(f"Divergence: {name0} - inflated({name0}, {name1})")
inflate.resnet(network0, network1)
imagenet.divergence(network0, network1, train_data)