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


def eval_data(data_root: str):
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
    save_epoch = model.load_state(state, name)
    start_epoch = 1 if save_epoch is None else save_epoch + 1
    if save_epoch is None and init_fn is not None:
        init_fn(network)

    total = len(data_loader.dataset)
    print(f"Iterating {total} samples")

    network.train()
    network.to(device=device)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            optimizer.zero_grad()
            outputs = network(inputs.to(device=device))
            loss = criterion(outputs, labels.to(device=device))
            loss.backward()
            optimizer.step()
            device_step()
            epoch_loss += loss.item() / total
        print(f"[epoch {epoch}]: loss: {epoch_loss}")
        model.save_state(state, name, epoch, log=f"{epoch}\t{epoch_loss}\n")


def eval(model: nn.Module, data: datasets.DatasetFolder, batch_size=64):
    data_loader = torch.utils.data.DataLoader2(
        data,
        batch_size=batch_size,
    )

    with torch.no_grad():
        model.eval()
        model.to(device=device)
        accuracy = 0.0
        total = len(data_loader.dataset)
        print(f"Iterating {total} samples")
        for inputs, labels in tqdm(data_loader):
            bs, ncrops, c, h, w = inputs.shape
            outputs = model(inputs.to(device=device).view(-1, c, h, w))
            outputs = outputs.view(bs, ncrops, -1).mean(1).max(dim=1).indices.flatten()
            labels = labels.to(device=device)
            accuracy += (outputs == labels).sum() / total
        print(f"Top1 accuracy: {accuracy}")
        return accuracy


def run_eval(
    module: nn.Module, name: str, epoch: Optional[int], data: datasets.DatasetFolder
):
    if epoch is None or model.load(module, name, epoch, storage_dir=storage_dir) is not None:
        model.write_record(
            name,
            "eval",
            f"{epoch}\t{eval(module, data)}",
            storage_dir=storage_dir
        )
