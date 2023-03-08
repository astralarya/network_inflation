from os import environ
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
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
    finetune: bool = False,
    inflate: Optional[str] = None,
    inflate_strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    mask_inflate: bool = True,
    nprocs: int = 8,
    num_workers: int = 4,
    batch_size: int = 128,
    num_epochs: int = 600,
    opt: str = "sgd",
    momentum: float = 0.9,
    lr: float = 0.5,
    lr_scheduler: str = "cosineannealinglr",
    lr_step_size=30,
    lr_gamma=0.1,
    lr_min=0.0,
    lr_warmup_epochs: int = 5,
    lr_warmup_method: str = "linear",
    lr_warmup_decay: int = 0.01,
    weight_decay: float = 2e-05,
    norm_weight_decay: float = 0.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    random_erase: float = 0.1,
    model_ema_steps=32,
    model_ema_decay=0.9998,
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

    (model_name, model, save_epoch, save_state) = resnet.network_load(
        model_path / name,
        inflate=inflate,
        reset=not finetune,
        inflate_strategy=inflate_strategy,
        mask_inflate=mask_inflate,
    )

    parameters = set_weight_decay(
        model,
        weight_decay=weight_decay,
        norm_weight_decay=norm_weight_decay,
    )
    optimizer = _optim.optimizer(
        parameters=parameters,
        optimizer=opt,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
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
        model_ema = ExponentialMovingAverage(model, decay=1.0 - alpha)

    if save_state is not None:
        print(f"Resuming from epoch: {save_epoch}")
        if model_ema:
            model_ema.load_state_dict(save_state["model_ema"])
        optimizer.load_state_dict(save_state["optimizer"])
        scheduler.load_state_dict(save_state["scheduler"])

    else:
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

    print(f"Spawning {nprocs} processes")
    device.spawn(
        _worker,
        (
            {
                "name": model_name,
                "model": device.model(model),
                "model_ema": device.model(model_ema),
                "optimizer": optimizer,
                "scheduler": scheduler,
                "train_dataset": train_dataset,
                "train_collate_fn": train_collate_fn,
                "init_epoch": save_epoch + 1 if save_epoch else 1,
                "num_epochs": num_epochs,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "label_smoothing": label_smoothing,
                "lr_warmup_epochs": lr_warmup_epochs,
                "model_ema_steps": model_ema_steps,
            },
        ),
        nprocs=nprocs,
        start_method="fork",
    )


def _worker(idx: int, _args: dict):
    _train(**_args)


def _train(
    name: str,
    model: nn.Module,
    model_ema: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    train_dataset: datasets.DatasetFolder,
    train_collate_fn: Optional[Callable],
    init_epoch: int,
    num_epochs: int,
    num_workers: int,
    batch_size: int,
    label_smoothing: float,
    lr_warmup_epochs: int,
    model_ema_steps: int,
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

    model = model.to(device.device())
    model.train()
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
            losses = device.mesh_reduce("loss", loss.item(), lambda x: sum(x))
            epoch_loss += losses / total
            optimizer.zero_grad()
            loss.backward()
            device.optim_step(optimizer)
            scheduler.step()
            if model_ema and i % model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < lr_warmup_epochs:
                    model_ema.n_averaged.fill_(0)
        if device.is_main():
            print(f"[epoch {epoch}]: loss: {epoch_loss}")
            checkpoint.save(
                name,
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
