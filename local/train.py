from os import environ
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm


from local import inflate as _inflate
from local import model
from local import resnet


from . import model as model
import local.device as device


def data(data_root: str):
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


def train(
    name: str,
    num_epochs: int = 2048,
    finetune: bool = False,
    inflate: Optional[str] = None,
    lr: float = 0.1,
    lr_step: int = 10,
    lr_gamma: float = 0.5,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
    batch_size: int = 256,
    nprocs: int = 8,
    num_workers: int = 4,
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

    model_name = model_path / name
    if finetune is True:
        model_name = f"{model_name}--finetune"
    if inflate is not None:
        model_name = f"{model_name}--inflate-{inflate}"

    args = {"batch_size": batch_size, "nprocs": device.world_size()}

    network = getattr(resnet, name, lambda: None)()
    if network is None:
        print(f"Unknown network: {name}")
        exit(1)

    train_data = data(imagenet_path / "train")

    save_epoch, save_state = model.load(model_name)
    if save_epoch is not None:
        print(f"Resuming from epoch: {save_epoch}")
        optimizer = optim.SGD(
            network.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=lr_step, gamma=lr_gamma
        )
        network.load_state_dict(save_state["model"])
        optimizer.load_state_dict(save_state["optimizer"])
        scheduler.load_state_dict(save_state["scheduler"])

    else:
        if inflate:
            inflate_source = getattr(resnet, inflate, lambda: None)()
            if inflate_source is None:
                print(f"Unknown network: {inflate}")
                exit(1)
            print(f"Inflating network: {name} from {inflate}")
            _inflate.resnet(inflate_source, network)
        else:
            print(f"Reset network: {name}")
            model.reset(network)

        optimizer = optim.SGD(
            network.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
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
                "network": network,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "name": model_name,
                "data": train_data,
                "init_epoch": save_epoch + 1 if save_epoch else 1,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _loop(
        name=_args["name"],
        network=_args["network"],
        optimizer=_args["optimizer"],
        scheduler=_args["scheduler"],
        data=_args["data"],
        init_epoch=_args["init_epoch"],
        num_epochs=_args["num_epochs"],
        batch_size=_args["batch_size"],
        num_workers=_args["num_workers"],
    )


def _loop(
    name: str,
    network: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    data: datasets.DatasetFolder,
    init_epoch=1,
    num_epochs=2048,
    batch_size=256,
    num_workers=4,
):
    device.sync_seed()

    network = network.to(device.device())
    args = {"batch_size": batch_size, "nprocs": device.world_size()}
    total = len(data)

    data_sampler = (
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
            sampler=data_sampler,
            num_workers=num_workers,
            shuffle=False if data_sampler else True,
        )
    )
    criterion = nn.CrossEntropyLoss().to(device.device())

    network.train()

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
