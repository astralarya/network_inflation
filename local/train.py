from os import environ
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader2
from tqdm import tqdm


from local import data
from local import device
from local import model
from local import resnet


def train(
    name: str,
    finetune: bool = False,
    inflate: Optional[str] = None,
    nprocs: int = 8,
    num_workers: int = 4,
    batch_size: int = 256,
    num_epochs: int = 600,
    opt: str = "sgd",
    momentum: float = 0.9,
    lr: float = 0.5,
    lr_scheduler: str = "cosineannealinglr",
    lr_warmup_epochs: int = 5,
    lr_warmup_method: str = "linear",
    lr_warmup_decay: int = 0.01,
    weight_decay: float = 2e-05,
    norm_weight_decay: float = 0.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    random_erase: float = 0.1,
    ra_sampler=True,
    ra_reps=4,
    model_ema=True,
    model_ema_steps=32,
    modeal_ema_decay=0.9998,
    train_crop_size=224,
    val_crop_size=224,
    model_path: Optional[Path] = None,
    imagenet_path: Optional[Union[Path, str]] = None,
):
    lr_step = 10
    lr_gamma = 0.5

    if model_path is None:
        model_path = Path("models")
    elif type(imagenet_path) == str:
        model_path = Path(model_path)

    if imagenet_path is None:
        imagenet_path = Path(environ.get("IMAGENET_PATH", "/mnt/imagenet/imagenet-1k"))
    elif type(imagenet_path) == str:
        imagenet_path = Path(imagenet_path)

    model_name = model_path / name
    if finetune is True:
        model_name = f"{model_name}--finetune"
    if inflate is not None:
        model_name = f"{model_name}--inflate-{inflate}"

    args = {"batch_size": batch_size}

    network = resnet.network_load(name, inflate, reset=not finetune)

    train_dataset = data.train_dataset(
        imagenet_path / "train",
        transform=data.train_transform(
            crop_size=train_crop_size, random_erase=random_erase
        ),
    )
    train_collate_fn = data.train_collate_fn(
        dataset=train_dataset,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
    )

    save_epoch, save_state = model.load(model_name)
    if save_epoch is not None:
        print(f"Resuming from epoch: {save_epoch}")
        optimizer = optim.SGD(
            network.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=lr_step, gamma=lr_gamma
        )
        network.load_state_dict(save_state["model"])
        optimizer.load_state_dict(save_state["optimizer"])
        scheduler.load_state_dict(save_state["scheduler"])

    else:
        optimizer = optim.SGD(
            network.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=lr_step, gamma=lr_gamma
        )
        model.save(
            model_name,
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

    print(f"Spawning {nprocs} processes")
    device.spawn(
        _worker,
        (
            {
                "name": model_name,
                "network": network,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "train_dataset": train_dataset,
                "train_collate_fn": train_collate_fn,
                "init_epoch": save_epoch + 1 if save_epoch else 1,
                "num_epochs": num_epochs,
                "num_workers": num_workers,
                "batch_size": batch_size,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _train(
        name=_args["name"],
        network=_args["network"],
        optimizer=_args["optimizer"],
        scheduler=_args["scheduler"],
        train_dataset=_args["train_dataset"],
        train_collate_fn=_args["train_collate_fn"],
        init_epoch=_args["init_epoch"],
        num_epochs=_args["num_epochs"],
        num_workers=_args["num_workers"],
        batch_size=_args["batch_size"],
    )


def _train(
    name: str,
    network: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    train_dataset: datasets.DatasetFolder,
    train_collate_fn: Optional[Callable],
    init_epoch: int,
    num_epochs: int,
    num_workers: int,
    batch_size: int,
):
    device.sync_seed()

    args = {"batch_size": batch_size, "nprocs": device.world_size()}

    data_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=device.world_size(),
            rank=device.ordinal(),
            shuffle=True,
        )
        if device.world_size() > 1
        else None
    )
    data_loader = device.loader(
        DataLoader2(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=data_sampler,
            shuffle=False if data_sampler else True,
            collate_fn=train_collate_fn,
        )
    )

    network = network.to(device.device())
    network.train()
    criterion = nn.CrossEntropyLoss().to(device.device())

    total = len(train_dataset)
    if device.is_main():
        print(f"Iterating {total} samples")

    for epoch in range(init_epoch, num_epochs + 1):
        epoch_loss = 0.0
        for inputs, labels in tqdm(data_loader, disable=not device.is_main()):
            inputs = inputs.to(device.device())
            labels = labels.to(device.device())
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            losses = device.mesh_reduce("loss", loss.item(), lambda x: sum(x))
            epoch_loss += losses / total
            optimizer.zero_grad()
            loss.backward()
            device.optim_step(optimizer)
            scheduler.step()
        if device.is_main():
            print(f"[epoch {epoch}]: loss: {epoch_loss}")
            model.save(
                name,
                epoch,
                {
                    "loss": epoch_loss,
                    "model": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": args,
                },
            )
    device.rendezvous("end")
