from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim

from local.guided_sgd import GuidedSGD


def optimizer(
    parameters: Sequence[nn.Parameter],
    optimizer="sgd",
    lr=0.5,
    momentum=0.9,
    weight_decay=2e-05,
    guide: Optional[Sequence[torch.Tensor]] = None,
    guide_alpha=1.0,
):
    optimizer = optimizer.lower()
    if optimizer.startswith("sgd"):
        if guide is not None:
            return GuidedSGD(
                parameters,
                lr=lr,
                guide_alpha=guide_alpha,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov="nesterov" in optimizer,
            )
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
    lr_scheduler="cosineannealinglr",
    num_epochs=600,
    lr_step_size=30,
    lr_gamma=0.1,
    lr_min=0.0,
    lr_warmup_method="linear",
    lr_warmup_epochs=5,
    lr_warmup_decay=0.01,
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
