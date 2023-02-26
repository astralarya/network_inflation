from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from . import model as model
from .device import device, device_step


def train_data(data_root: str):
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


def val_data(data_root: str):
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
    criterion = nn.CrossEntropyLoss().to(device=device)

    state = {
        None: network,
        "optim": optimizer,
    }
    save_epoch = model.load_state(state, name, init_fn=init_fn)

    total = len(data_loader.dataset)
    print(f"Iterating {total} samples")

    network.train()
    network.to(device=device)

    for epoch in range(save_epoch + 1, num_epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            optimizer.zero_grad()
            outputs = network(inputs.to(device=device))
            loss = criterion(outputs, labels.to(device=device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / total
            device_step()
        print(f"[epoch {epoch}]: loss: {epoch_loss}")
        model.save_state(state, name, epoch, log=f"{epoch}\t{epoch_loss}\n")


def val(network: nn.Module, data: datasets.DatasetFolder, batch_size=64):
    data_loader = torch.utils.data.DataLoader2(
        data,
        batch_size=batch_size,
    )

    with torch.no_grad():
        network.eval()
        network.to(device=device)
        accuracy = 0.0
        total = len(data_loader.dataset)
        print(f"Iterating {total} samples")
        for inputs, labels in tqdm(data_loader):
            bs, ncrops, c, h, w = inputs.shape
            outputs = network(inputs.to(device=device).view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1).max(dim=1).indices.flatten()
            labels = labels.to(device=device)
            accuracy += (outputs == labels).sum() / total
            device_step()
        print(f"Top1 accuracy: {accuracy}")
        return accuracy


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
    criterion = nn.CrossEntropyLoss().to(device=device)


    with torch.no_grad():
        network0.eval()
        network0.to(device=device)
        network1.eval()
        network1.to(device=device)

        total_loss = 0.0
        total = len(data_loader.dataset)
        print(f"Iterating {total} samples")
        for inputs, _ in tqdm(data_loader):
            inputs = inputs.to(device=device)
            outputs0 = network0(inputs)
            outputs1 = network1(inputs)
            loss = criterion(outputs0, outputs1)
            total_loss += loss.item() / total
            device_step()
        print(f"Divergence: {total_loss}")
        return total_loss

