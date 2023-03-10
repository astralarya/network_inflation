import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader2
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from tqdm import tqdm

from local import data as _data
from local import device


def divergence(
    network0: nn.Module,
    network1: nn.Module,
    data: datasets.DatasetFolder,
    num_epochs=8,
    batch_size=256,
    num_workers=8,
    nprocs=8,
):
    print(f"Spawning {nprocs} processes")
    device.spawn(
        _worker,
        (
            {
                "network0": network0,
                "network1": network1,
                "data": data,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _divergence(**_args)


@torch.no_grad()
def _divergence(
    network0: nn.Module,
    network1: nn.Module,
    data: datasets.DatasetFolder,
    num_epochs=8,
    batch_size=256,
    num_workers=4,
):
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
            shuffle=False if data_sampler else True,
            collate_fn=_data.train_collate_fn(data),
        )
    )
    log_softmax = nn.LogSoftmax(dim=1).to(device.device())
    criterion = nn.KLDivLoss(reduction="sum", log_target=True).to(device.device())
    network0.eval()
    network0.to(device.device())
    network1.eval()
    network1.to(device.device())

    total = len(data)
    total_loss = 0.0
    if device.is_main():
        print(f"Iterating {total} samples")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, _ in tqdm(data_loader, disable=not device.is_main()):
            inputs = inputs.to(device.device())
            outputs0 = network0(inputs)
            outputs1 = network1(inputs)
            loss = criterion(log_softmax(outputs0), log_softmax(outputs1))
            epoch_loss += device.mesh_reduce("loss", loss.item(), lambda x: sum(x))
            device.step()
        total_loss += epoch_loss / total
        if device.is_main():
            print(f"Divergence (epoch {epoch}): {epoch_loss / total}")
            print(f"Divergence (total): {total_loss / (epoch + 1)}")
    device.rendezvous("end")
    return total_loss / num_epochs
