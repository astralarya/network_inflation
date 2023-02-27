from typing import Any, Callable, Optional
from os import environ

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from . import model as model
from .device import device, device_step


def train_data(data_root: Optional[str] = None):
    data_root = environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k") + "/train/" if data_root is None else data_root
    return datasets.ImageFolder(
        data_root,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )


def val_data(data_root: Optional[str] = None):
    data_root = environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k") + "/val/" if data_root is None else data_root
    return datasets.ImageFolder(
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


def train(
    network: nn.Module,
    name: str,
    data: datasets.DatasetFolder,
    batch_size=256,
    num_epochs=2048,
    num_workers=8,
    init_fn: Optional[Callable[[nn.Module], Any]] = None,
):
    data_loader = torch.utils.data.DataLoader2(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    optimizer = optim.AdamW(network.parameters())
    softmax = nn.Softmax(dim=1).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    state = {
        None: network,
        "optim": optimizer,
    }
    save_epoch = model.load_state(state, name, init_fn=init_fn)

    total = len(data_loader.dataset)
    print(f"Iterating {total} samples")

    network.train()
    network.to(device)

    for epoch in range(save_epoch + 1, num_epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = softmax(network(inputs))
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / total
            device_step()
        print(f"[epoch {epoch}]: loss: {epoch_loss}")
        model.save_state(state, name, epoch, log=f"{epoch}\t{epoch_loss}\n")


def val(network: nn.Module, data: datasets.DatasetFolder, batch_size=64):
    with torch.no_grad():
        data_loader = torch.utils.data.DataLoader2(
            data,
            batch_size=batch_size,
            shuffle=True,
        )

        total = len(data_loader.dataset)
        print(f"Iterating {total} samples")

        softmax = nn.Softmax(dim=2).to(device)
        network.eval()
        network.to(device)

        top1_accuracy = 0.0
        top5_accuracy = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            bs, ncrops, c, h, w = inputs.shape
            k = 5
            outputs = network(inputs.view(-1, c, h, w))
            outputs = softmax(outputs.view(bs, ncrops, -1))
            top1_outputs = outputs.mean(1).max(dim=1).indices.flatten()
            top1_accuracy += (top1_outputs == labels).sum() / total
            top5_outputs = outputs.mean(1).topk(k, dim=1).indices.view(bs, k)
            top5_accuracy += (top5_outputs == labels.repeat(1, k).view(bs, k).squeeze(0)).max(dim=1).values.sum() / total
            device_step()
            print(f"Top1 accuracy: {top1_accuracy}")
            print(f"Top5 accuracy: {top5_accuracy}")
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
    epoch: Optional[int] = None,
    batch_size=256,
):
    if epoch is None or model.load(network, name, epoch) is not None:
        model.write_record(
            name,
            "eval",
            f"{epoch}\t{val(network, data, batch_size=batch_size)}",
        )


def run_val(
    network: nn.Module,
    name: str,
    data: datasets.DatasetFolder,
    batch_size=256,
):
    for epoch in model.list_epochs(name):
        val_epoch(network, name, data, epoch, batch_size=batch_size)


def divergence(network0: nn.Module, network1: nn.Module, data: datasets.DatasetFolder, batch_size=256, num_workers=8):
    data_loader = torch.utils.data.DataLoader2(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    with torch.no_grad():
        softmax = nn.Softmax(dim=1).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        network0.eval()
        network0.to(device)
        network1.eval()
        network1.to(device)

        total_loss = 0.0
        total = len(data_loader.dataset)
        print(f"Iterating {total} samples")
        for inputs, _ in tqdm(data_loader):
            inputs = inputs.to(device)
            outputs0 = softmax(network0(inputs))
            outputs1 = softmax(network1(inputs))
            loss = criterion(outputs0, outputs1)
            total_loss += loss.item() / total
            device_step()
        print(f"Divergence: {total_loss}")
        return total_loss

