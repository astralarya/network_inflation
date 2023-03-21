from os import environ
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader2
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from local import data
from local import checkpoint
from local import device
from local import resnet
from local.inflate import SequenceInflate
from local.extern.model_ema import ExponentialMovingAverage


def validate(
    name: str,
    epochs: Optional[Sequence[Union[str, int]]],
    modifier: Optional[str] = None,
    finetune: bool = False,
    inflate: Optional[str] = None,
    inflate_strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    mask_inflate: bool = True,
    model_ema: bool = False,
    batch_size: int = 32,
    num_workers: int = 8,
    nprocs: int = 8,
    model_path: Path = None,
    imagenet_path: Optional[Union[Path, str]] = None,
):
    if model_path is None:
        model_path = Path("models")
    elif type(imagenet_path) == str:
        model_path = Path(model_path)

    if imagenet_path is None:
        imagenet_path = Path(environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"))
    elif type(imagenet_path) == str:
        imagenet_path = Path(imagenet_path)

    val_data = data.load_dataset(imagenet_path / "val", transform=data.val_transform())

    print(f"Spawning {nprocs} processes")
    device.spawn(
        _worker,
        (
            {
                "network_spec": {
                    "name": model_path / name,
                    "modifier": modifier,
                    "inflate": inflate,
                    "reset": not finetune,
                    "inflate_strategy": inflate_strategy,
                    "mask_inflate": mask_inflate,
                },
                "data": val_data,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "model_ema": model_ema,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _validate(**_args)


@torch.no_grad()
def _validate(
    network_spec,
    data: datasets.DatasetFolder,
    epochs: Sequence[Union[int, str]],
    batch_size=64,
    num_workers=4,
    model_ema: bool = False,
):
    name = resnet.network_name(**network_spec)
    outname = f"{name}--ema" if model_ema else name
    outname = f"{outname}.__val__"
    if device.is_main():
        print(f"Validating {outname}")
    if epochs is None:
        epochs = ["all"] if checkpoint.get_epoch(name) else ["pre"]
    epochs = (
        checkpoint.iter_epochs(name, checkpoint.log_epoch(outname, 1))
        if "all" in epochs
        else epochs
    )

    for epoch in epochs:
        if device.is_main():
            print(f"Validating epoch {epoch}")

        _, network, save_epoch, save_state = resnet.network_load(
            **network_spec,
            epoch=epoch,
            device=device.device(),
            print_output=device.is_main(),
        )
        if model_ema and save_state and "model_ema" in save_state:
            network = ExponentialMovingAverage(network, decay=0, device=device.device())
            network.load_state_dict(save_state["model_ema"])
        network.eval()

        softmax = nn.Softmax(dim=2).to(device.device())

        data_sampler = (
            DistributedSampler(
                data,
                num_replicas=device.world_size(),
                rank=device.ordinal(),
            )
            if device.world_size() > 1
            else None
        )
        data_loader = device.loader(
            DataLoader2(
                data,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=data_sampler,
            )
        )

        total = len(data)
        if device.is_main():
            print(f"Iterating {total} samples")

        top1_accuracy = 0.0
        top5_accuracy = 0.0
        for inputs, labels in tqdm(data_loader, disable=not device.is_main()):
            inputs = inputs.to(device.device())
            labels = labels.to(device.device())
            bs, ncrops, c, h, w = inputs.shape
            k = 5

            outputs = network(inputs.view(-1, c, h, w))
            outputs = softmax(outputs.view(bs, ncrops, -1))

            top1_outputs = outputs.mean(dim=1).max(dim=1).indices.flatten()
            top1_part = (top1_outputs == labels).sum() / total
            top1_accuracy += device.mesh_reduce(
                "top1_accuracy", top1_part.item(), lambda x: sum(x)
            )

            top5_outputs = outputs.mean(dim=1).topk(k, dim=1).indices.view(bs, k)
            top5_part = (
                top5_outputs == labels.repeat(k).view(k, -1).transpose(0, 1)
            ).int().max(dim=1).values.sum() / total
            top5_accuracy += device.mesh_reduce(
                "top5_accuracy", top5_part.item(), lambda x: sum(x)
            )

            device.step()
        if device.is_main():
            print(f"Top1 accuracy: {top1_accuracy}")
            print(f"Top5 accuracy: {top5_accuracy}")
            checkpoint.write_log(
                outname,
                f"{epoch}\t{top1_accuracy}\t{top5_accuracy}\n",
            )
    device.rendezvous("end")
