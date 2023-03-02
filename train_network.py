import argparse
from os import environ
from pathlib import Path

from local import imagenet
import local.device as device

parser = argparse.ArgumentParser(prog="ResNet training script")
parser.add_argument(
    "network",
    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--nprocs", default=8, type=int)
parser.add_argument("--model_path", default="models", type=Path)
parser.add_argument(
    "--imagenet_path",
    default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"),
    type=Path,
)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()


def main(idx: int, _args: dict):
    device.sync_seed()

    def reset_fn(x):
        if device.is_main():
            print(f"Reset network ({args.network})")
        model.reset(x)

    def inflate_fn(x):
        inflate_source = getattr(resnet, args.inflate, lambda: None)()
        if inflate_source is None:
            print(f"Unknown network: {args.inflate}")
            exit(1)
        print(f"Inflating network ({args.network}) from {args.inflate}")
        inflate.resnet(inflate_source, x)

    imagenet.train(
        _args["network"],
        _args["name"],
        _args["data"],
        batch_size=_args["batch_size"],
        num_workers=_args["num_workers"],
        init_fn=reset_fn if args.inflate is None else inflate_fn,
        force=_args["force"],
    )


if __name__ == "__main__":
    from local import inflate
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

    train_data = imagenet.train_data(args.imagenet_path / "train")

    network = device.model(network)

    device.spawn(
        main,
        (
            {
                "network": network,
                "name": args.model_path / name,
                "data": train_data,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "force": args.force,
            },
        ),
        nprocs=args.nprocs,
        start_method="fork",
    )
