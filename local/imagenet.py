from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from . import model as model
import local.device as device


def train_data(data_root: str):
    print(f"Loading train data `{data_root}`... ", flush=True, end="")
    r = datasets.ImageFolder(
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
    print("DONE")
    return r


def val_data(data_root: str):
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


def train(
    network: nn.Module,
    name: str,
    data: datasets.DatasetFolder,
    batch_size=256,
    num_epochs=2048,
    num_workers=4,
    init_fn: Optional[Callable[[nn.Module], Any]] = None,
    force: bool = False,
):
    args = {"batch_size": 256}
    network = network.to(device.device())
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            data, num_replicas=device.world_size(), rank=device.ordinal(), shuffle=True
        )
        if device.world_size() > 1
        else None
    )
    data_loader = device.loader(
        torch.utils.data.DataLoader2(
            data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            shuffle=False if train_sampler else True,
        )
    )
    optimizer = optim.AdamW(network.parameters())
    criterion = nn.CrossEntropyLoss().to(device.device())

    save_epoch, save_state = model.load(name, device=device.device())
    if save_epoch is not None:
        print(f"Resuming from epoch {save_epoch}")
        network.load_state_dict(save_state["model"])
        optimizer.load_state_dict(save_state["optim"])
        if not force:
            for arg, arg_val in args.items():
                save_arg = save_state["args"][arg]
                if save_arg != arg_val:
                    raise Exception(
                        f"Mismatched {arg}: {save_arg} != {arg_val}\n  Override this error with --force"
                    )
    else:
        if init_fn is not None:
            init_fn(network)
        print("Saving initial state as epoch 0")
        if device.is_main():
            model.save(
                name,
                0,
                {
                    "loss": None,
                    "model": network.state_dict(),
                    "optim": optimizer.state_dict(),
                    "args": args,
                },
            )

    total = len(data_loader.dataset)
    print(f"Iterating {total} samples")

    network.train()

    for epoch in range(save_epoch + 1 if save_epoch else 1, num_epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in tqdm(data_loader):
            inputs = inputs.to(device.device())
            labels = labels.to(device.device())
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            losses = device.mesh_reduce("loss", loss.item(), lambda x: sum(x))
            epoch_loss += losses / total
            optimizer.zero_grad()
            loss.backward()
            device.optim_step(optimizer)
        print(f"[epoch {epoch}]: loss: {epoch_loss}")
        if device.is_main():
            model.save(
                name,
                epoch,
                {
                    "loss": epoch_loss,
                    "model": network.state_dict(),
                    "optim": optimizer.state_dict(),
                    "args": args,
                },
            )
    device.rendezvous("end")


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


@torch.no_grad()
def divergence(
    network0: nn.Module,
    network1: nn.Module,
    data: datasets.DatasetFolder,
    batch_size=256,
    num_workers=8,
):
    data_loader = torch.utils.data.DataLoader2(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    log_softmax = nn.LogSoftmax(dim=1).to(device.device())
    criterion = nn.KLDivLoss(reduction="sum", log_target=True).to(device.device())
    network0.eval()
    network0.to(device.device())
    network1.eval()
    network1.to(device.device())

    total_loss = 0.0
    total = len(data_loader.dataset)
    print(f"Iterating {total} samples")
    for inputs, _ in tqdm(data_loader):
        inputs = inputs.to(device.device())
        outputs0 = network0(inputs)
        outputs1 = network1(inputs)
        loss = criterion(log_softmax(outputs0), log_softmax(outputs1))
        total_loss += loss.item() / total
        device.step()
    print(f"Divergence: {total_loss}")
    return total_loss
