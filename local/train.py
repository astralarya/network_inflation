from os import environ
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader2
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from tqdm import tqdm


from local import data
from local import device
from local import optim as _optim
from local import checkpoint
from local import resnet
from local.inflate import SequenceInflate
from local.extern.weight_decay import set_weight_decay
from local.extern.model_ema import ExponentialMovingAverage


def train(
    name: str,
    modifier: Optional[str] = None,
    finetune: bool = False,
    inflate: Optional[str] = None,
    inflate_strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    mask_inflate: bool = True,
    nprocs: int = 8,
    num_workers: int = 4,
    batch_size: int = 128,
    num_epochs: int = 600,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    random_erase: float = 0.1,
    opt: str = "sgd",
    momentum: float = 0.9,
    lr: float = 0.5,
    lr_scheduler: str = "cosineannealinglr",
    lr_step_size: int = 30,
    lr_gamma: float = 0.1,
    lr_min: float = 0.0,
    lr_warmup_epochs: int = 5,
    lr_warmup_method: str = "linear",
    lr_warmup_decay: float = 0.01,
    weight_decay: float = 2e-05,
    norm_weight_decay: float = 0.0,
    guide_alpha=1.0,
    guide_epochs=32,
    label_smoothing: float = 0.1,
    model_ema_steps: int = 32,
    model_ema_decay: float = 0.9998,
    model_path: Optional[Path] = None,
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

    train_dataset = data.load_dataset(
        imagenet_path / "train",
        transform=data.train_transform(random_erase=random_erase),
    )
    train_collate_fn = data.train_collate_fn(
        dataset=train_dataset,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
    )

    print(f"Spawning {nprocs} processes")
    device.spawn(
        _worker,
        (
            {
                "name": model_path / name,
                "modifier": modifier,
                "finetune": finetune,
                "inflate": inflate,
                "inflate_strategy": inflate_strategy,
                "mask_inflate": mask_inflate,
                "train_dataset": train_dataset,
                "train_collate_fn": train_collate_fn,
                "num_epochs": num_epochs,
                "nprocs": nprocs,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "opt": opt,
                "momentum": momentum,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "lr_step_size": lr_step_size,
                "lr_gamma": lr_gamma,
                "lr_min": lr_min,
                "lr_warmup_epochs": lr_warmup_epochs,
                "lr_warmup_method": lr_warmup_method,
                "lr_warmup_decay": lr_warmup_decay,
                "weight_decay": weight_decay,
                "norm_weight_decay": norm_weight_decay,
                "guide_alpha": guide_alpha,
                "guide_epochs": guide_epochs,
                "label_smoothing": label_smoothing,
                "model_ema_steps": model_ema_steps,
                "model_ema_decay": model_ema_decay,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _train(**_args)


def _train(
    name: str,
    modifier: Optional[str],
    finetune: bool,
    inflate: Optional[str],
    inflate_strategy: SequenceInflate,
    mask_inflate: bool,
    train_dataset: datasets.DatasetFolder,
    train_collate_fn: Optional[Callable],
    num_epochs: int,
    nprocs: int,
    num_workers: int,
    batch_size: int,
    opt: str,
    momentum: float,
    lr: float,
    lr_scheduler: str,
    lr_step_size: int,
    lr_gamma: float,
    lr_min: float,
    lr_warmup_epochs: int,
    lr_warmup_method: str,
    lr_warmup_decay: float,
    weight_decay: float,
    norm_weight_decay: float,
    guide_alpha: float,
    guide_epochs: int,
    label_smoothing: float,
    model_ema_steps: int,
    model_ema_decay: float,
):
    device.sync_seed()

    data_sampler = (
        DistributedSampler(
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

    state = resnet.network_load(
        name,
        modifier,
        inflate=inflate,
        reset=not finetune,
        inflate_strategy=inflate_strategy,
        mask_inflate=mask_inflate,
        device=device.device(),
        print_output=device.is_main(),
    )
    model = state.network
    model_name = state.name
    guide = state.inflate_network
    save_epoch = state.save_epoch
    save_state = state.save_state

    if device.is_main() and state.save_state is not None:
        print(f"Resuming from epoch: {state.save_epoch}")
    init_epoch = save_epoch + 1 if save_epoch else 1

    model.train()

    parameters = set_weight_decay(
        model,
        weight_decay=weight_decay,
        norm_weight_decay=norm_weight_decay,
    )
    optimizer = _optim.optimizer(
        parameters=parameters,
        guide=guide,
        optimizer=opt,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        guide_alpha=guide_alpha,
        guide_epochs=guide_epochs,
    )
    scheduler = _optim.lr_scheduler(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        lr_min=lr_min,
        lr_warmup_method=lr_warmup_method,
        lr_warmup_epochs=lr_warmup_epochs,
        lr_warmup_decay=lr_warmup_decay,
    )

    model_ema = None
    if model_ema_steps > 1:
        adjust = nprocs * batch_size * model_ema_steps / num_epochs
        alpha = 1.0 - model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(
            model, decay=1.0 - alpha, device=device.device()
        )

    if save_state is not None:
        if model_ema:
            model_ema.load_state_dict(save_state["model_ema"])
        optimizer.load_state_dict(save_state["optimizer"])
        scheduler.load_state_dict(save_state["scheduler"])
    elif device.is_main():
        checkpoint.save(
            model_name,
            0,
            {
                "loss": None,
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict() if model_ema else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )

    model_ema = model_ema.to(device.device())
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device.device())

    total = len(train_dataset)
    if device.is_main():
        print(f"Iterating {total} samples")

    for epoch in range(init_epoch, num_epochs + 1):
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(
            tqdm(data_loader, disable=not device.is_main())
        ):
            inputs = inputs.to(device.device())
            labels = labels.to(device.device())
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            device.optim_step(optimizer)
            if model_ema and i % model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < lr_warmup_epochs:
                    model_ema.n_averaged.fill_(0)
            losses = device.mesh_reduce("loss", loss.item(), lambda x: sum(x))
            epoch_loss += losses / total
        scheduler.step()
        if device.is_main():
            print(f"[epoch {epoch}]: loss: {epoch_loss}")
            checkpoint.save(
                model_name,
                epoch,
                {
                    "loss": epoch_loss,
                    "model": model.state_dict(),
                    "model_ema": model_ema.state_dict() if model_ema else None,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
            )
    device.rendezvous("end")
