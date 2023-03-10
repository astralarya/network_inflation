import argparse
from os import environ
from pathlib import Path

from local.inflate import SequenceInflate

parser = argparse.ArgumentParser(prog="ResNet divergence script")
parser.add_argument(
    "network0", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument(
    "network1", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
)
parser.add_argument("--reset0", action="store_true")
parser.add_argument("--reset1", action="store_true")
parser.add_argument("--inflate0", choices=["resnet50", "resnet101"])
parser.add_argument("--inflate1", choices=["resnet50", "resnet101"])
parser.add_argument(
    "--inflate_strategy0",
    default="align-start",
    type=SequenceInflate,
    choices=list(SequenceInflate),
)
parser.add_argument(
    "--inflate_strategy1",
    default="align-start",
    type=SequenceInflate,
    choices=list(SequenceInflate),
)
parser.add_argument("--inflate_unmasked0", action="store_false", dest="mask_inflate0")
parser.add_argument("--inflate_unmasked1", action="store_false", dest="mask_inflate1")
parser.add_argument("--num_epochs", default=8, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument(
    "--imagenet_path",
    default=environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"),
    type=Path,
)
args = parser.parse_args()

if __name__ == "__main__":

    from local import data
    from local import divergence
    from local import resnet

    name0, network0, _, _ = resnet.network_load(
        args.network0,
        inflate=args.inflate0,
        reset=args.reset0,
        inflate_strategy=args.inflate_strategy0,
        mask_inflate=args.mask_inflate0,
    )
    name1, network1, _, _ = resnet.network_load(
        args.network1,
        inflate=args.inflate1,
        reset=args.reset1,
        inflate_strategy=args.inflate_strategy1,
        mask_inflate=args.mask_inflate1,
    )

    train_data = data.load_dataset(
        args.imagenet_path / "train", transform=data.train_transform()
    )

    print(f"Divergence: {name0} <-> {name1}")
    divergence.divergence(
        network0,
        network1,
        train_data,
        num_epochs=args.num_epochs,
    )
