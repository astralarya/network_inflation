from typing import Sequence

import torch.nn as nn
import torch.optim as optim


def optimizer(
    parameters,
    optimizer: str,
    lr: float,
    momentum: float,
    weight_decay: float,
):
    optimizer = optimizer.lower()
    if optimizer.startswith("sgd"):
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov="nesterov" in optimizer,
        )
    elif optimizer == "rmsprop":
        return optim.RMSprop(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif optimizer == "adamw":
        return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError(
            f"Invalid optimizer {optimizer}. Only SGD, RMSprop and AdamW are supported."
        )


def lr_scheduler(
    optimizer: optim.Optimizer,
    lr_scheduler: str,
    num_epochs: int,
    lr_step_size: int,
    lr_gamma: float,
    lr_min: float,
    lr_warmup_method: str,
    lr_warmup_epochs: int,
    lr_warmup_decay: float,
):
    lr_scheduler = lr_scheduler.lower()
    if lr_scheduler == "steplr":
        main_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
    elif lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - lr_warmup_epochs, eta_min=lr_min
        )
    elif lr_scheduler == "exponentiallr":
        main_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if lr_warmup_epochs > 0:
        if lr_warmup_method == "linear":
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=lr_warmup_decay,
                total_iters=lr_warmup_epochs,
            )
        elif lr_warmup_method == "constant":
            warmup_lr_scheduler = optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=lr_warmup_decay,
                total_iters=lr_warmup_epochs,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[lr_warmup_epochs],
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler
