import argparse
from os import environ
from pathlib import Path

import torch.optim as optim

from local import imagenet
import local.device as device

parser = argparse.ArgumentParser(prog="ResNet training script")
parser.add_argument(
    "network",
    choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
)
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--inflate", choices=["resnet50", "resnet101"])
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--lr_step", default=10, type=int)
parser.add_argument("--lr_gamma", default=0.5, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--dampening", default=0.0, type=float)
parser.add_argument("--weight_decay", default=0.0001, type=float)
parser.add_argument("--nesterov", default=True, type=bool)
parser.add_argument("--batch_size", default=128, type=int)
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

    imagenet.train(
        name=_args["name"],
        network=_args["network"],
        optimizer=_args["optimizer"],
        scheduler=_args["scheduler"],
        data=_args["data"],
        init_epoch=_args["init_epoch"],
        batch_size=_args["batch_size"],
        num_workers=_args["num_workers"],
    )


if __name__ == "__main__":
    from local import inflate
    from local import model
    from local import resnet

    name = args.model_path / args.network
    if args.finetune is True:
        name = f"{name}--finetune"
    if args.inflate is not None:
        name = f"{name}--inflate-{args.inflate}"

    network = getattr(resnet, args.network, lambda: None)()
    if network is None:
        print(f"Unknown network: {args.network}")
        exit(1)

    train_data = imagenet.train_data(args.imagenet_path / "train")

    save_epoch, save_state = model.load(name)
    if save_epoch is not None:
        print(f"Resuming from epoch {save_epoch}")
        optimizer = optim.SGD(
            network.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )
        network.load_state_dict(save_state["model"])
        optimizer.load_state_dict(save_state["optimizer"])
        scheduler.load_state_dict(save_state["scheduler"])

    else:
        if args.inflate:
            inflate_source = getattr(resnet, args.inflate, lambda: None)()
            if inflate_source is None:
                print(f"Unknown network: {args.inflate}")
                exit(1)
            print(f"Inflating network ({args.network}) from {args.inflate}")
            inflate.resnet(inflate_source, network)
        else:
            print(f"Reset network ({args.network})")
            model.reset(network)

        optimizer = optim.SGD(
            network.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )
        model.save(
            name,
            0,
            {
                "loss": None,
                "model": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": args,
            },
        )
    network = device.model(network)

    print(f"Spawning {args.nprocs} processes")
    device.spawn(
        main,
        (
            {
                "network": network,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "name": name,
                "data": train_data,
                "init_epoch": save_epoch + 1 if save_epoch else 1,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
        ),
        nprocs=args.nprocs,
        start_method="fork",
    )
