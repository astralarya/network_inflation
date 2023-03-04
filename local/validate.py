from os import environ
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from local import model
from local import device
from local import inflate as _inflate
from local import resnet


def data(data_root: str):
    print(f"Loading val data `{data_root}`... ", flush=True, end="")
    r = datasets.ImageFolder(
        data_root,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([crop for crop in crops])),
            ]
        ),
    )
    print("DONE")
    return r


def validate(
    name: str,
    epochs: Sequence[Union[str, int]],
    finetune: bool = False,
    inflate: Optional[str] = None,
    batch_size: int = 64,
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

    val_data = data(imagenet_path / "val")

    model_name = model_path / name
    if finetune is True:
        model_name = f"{model_name}--finetune"
    if inflate is not None:
        model_name = f"{model_name}--inflate-{inflate}"

    network_pre = getattr(resnet, name, lambda: None)
    network_type = getattr(resnet, f"{name}_type", lambda: None)
    if network_pre is None:
        print(f"Unknown network: {name}")
        exit(1)

    inflate_network = None
    if inflate is not None:
        inflate_network = getattr(resnet, inflate, lambda: None)
        if inflate_network is None:
            print(f"Unknown network: {inflate}")
            exit(1)
        network_pre = lambda: _inflate.resnet(inflate_network(), network_type())

    print(f"Spawning {nprocs} processes")
    device.spawn(
        _worker,
        (
            {
                "name": model_name,
                "network_pre": network_pre,
                "network_type": network_type,
                "data": val_data,
                "epochs": epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _validate(
        name=_args["name"],
        network_pre=_args["network_pre"],
        network_type=_args["network_type"],
        data=_args["data"],
        epochs=_args["epochs"],
        batch_size=_args["batch_size"],
        num_workers=_args["num_workers"],
    )


@torch.no_grad()
def _validate(
    name: str,
    network_pre: Callable[[], nn.Module],
    network_type: Callable[[], nn.Module],
    data: datasets.DatasetFolder,
    epochs: Sequence[Union[int, str]],
    batch_size=64,
    num_workers=4,
):
    epochs = model.iter_epochs(name) if "all" in epochs else epochs
    for epoch in epochs:
        if device.is_main():
            print(f"Validating epoch {epoch}")

        if type(epoch) == int:
            save_epoch, save_state = model.load(
                name, epoch, print_output=device.is_main()
            )
            if save_epoch is None:
                raise Exception(f"Epoch not found for {name}: {epoch}")
            network = network_type()
            network.load_state_dict(save_state["model"])
        else:
            network = network_pre()
        network.eval()

        network = network.to(device.device())
        softmax = nn.Softmax(dim=2).to(device.device())

        data_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                data,
                num_replicas=device.world_size(),
                rank=device.ordinal(),
            )
            if device.world_size() > 1
            else None
        )
        data_loader = device.loader(
            torch.utils.data.DataLoader2(
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
            ).max(dim=1).values.sum() / total
            top5_accuracy += device.mesh_reduce(
                "top5_accuracy", top5_part.item(), lambda x: sum(x)
            )

            device.step()
        if device.is_main():
            print(f"Top1 accuracy: {top1_accuracy}")
            print(f"Top5 accuracy: {top5_accuracy}")
            model.write_log(
                f"{name}.__val__",
                f"{epoch}\t{top1_accuracy}\t{top5_accuracy}\n",
            )
    device.rendezvous("end")
