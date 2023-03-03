from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm

from . import model as model
import local.device as device


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


def val(network: nn.Module, data: datasets.DatasetFolder, batch_size=64):
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader2(
            data,
            batch_size=batch_size,
        )

        total = len(data_loader.dataset)
        print(f"Iterating {total} samples")

        softmax = nn.Softmax(dim=2).to(device.device())
        network.eval()
        network.to(device.device())

        top1_accuracy = 0.0
        top5_accuracy = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device.device())
            labels = labels.to(device.device())
            bs, ncrops, c, h, w = inputs.shape
            k = 5

            outputs = network(inputs.view(-1, c, h, w))
            outputs = softmax(outputs.view(bs, ncrops, -1))

            top1_outputs = outputs.mean(dim=1).max(dim=1).indices.flatten()
            top1_accuracy += (top1_outputs == labels).sum() / total
            top5_outputs = outputs.mean(dim=1).topk(k, dim=1).indices.view(bs, k)
            top5_accuracy += (
                top5_outputs == labels.repeat(k).view(k, -1).transpose(0, 1)
            ).max(dim=1).values.sum() / total
            device.step()
        print(f"Top1 accuracy: {top1_accuracy}")
        print(f"Top5 accuracy: {top5_accuracy}")
        return {
            1: top1_accuracy,
            5: top5_accuracy,
        }


def val_epoch(
    network: nn.Module,
    name: str,
    data: datasets.DatasetFolder,
    epoch: Optional[Union[int, str]] = None,
    batch_size=256,
):
    if type(epoch) == int:
        save_epoch, save_state = model.load(name, epoch)
        if save_epoch is None:
            raise Exception(f"Epoch not found for {name}: {epoch}")
        network.load_state_dict(save_state["model"])

    accuracy = val(network, data, batch_size=batch_size)
    model.write_log(
        f"{name}.__val__",
        f"{epoch}\t{accuracy[1]}\t{accuracy[5]}",
    )


def run_val(
    network: nn.Module,
    name: str,
    data: datasets.DatasetFolder,
    batch_size=256,
):
    for epoch in model.list_epochs(name):
        val_epoch(network, name, data, epoch, batch_size=batch_size)
