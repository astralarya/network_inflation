from collections import namedtuple
from pathlib import Path
from typing import Optional

import torch
from torch.hub import load
from torchvision.models import resnet

from local import checkpoint
from local.inflate import resnet as inflate_resnet, guide_resnet, SequenceInflate


def network_pre(name: Optional[str]):
    if name is None:
        return None
    name = Path(name).name
    if name == "resnet18":
        return load(
            "pytorch/vision:v0.10.0",
            "resnet18",
            weights=resnet.ResNet18_Weights.IMAGENET1K_V1,
        )
    elif name == "resnet34":
        return load(
            "pytorch/vision:v0.10.0",
            "resnet34",
            weights=resnet.ResNet34_Weights.IMAGENET1K_V1,
        )
    elif name == "resnet50":
        return load(
            "pytorch/vision:v0.10.0",
            "resnet50",
            weights=resnet.ResNet50_Weights.IMAGENET1K_V2,
        )
    elif name == "resnet101":
        return load(
            "pytorch/vision:v0.10.0",
            "resnet101",
            weights=resnet.ResNet101_Weights.IMAGENET1K_V2,
        )
    elif name == "resnet152":
        return load(
            "pytorch/vision:v0.10.0",
            "resnet152",
            weights=resnet.ResNet152_Weights.IMAGENET1K_V2,
        )
    else:
        print(f"Unknown network: {name}")
        exit(1)


def network_type(name: str):
    name = Path(name).name
    network = getattr(resnet, name, lambda: None)
    if network is None:
        print(f"Unknown network: {name}")
        exit(1)
    return network


def network_sequence(name: Optional[str]):
    if name is None:
        return None
    name = Path(name).name
    if name == "resnet50":
        return [3, 4, 6, 3]
    elif name == "resnet101":
        return [3, 4, 23, 3]
    elif name == "resnet152":
        return [3, 8, 36, 3]
    else:
        print(f"Unknown network: {name}")
        exit(1)


NetworkInfo = namedtuple("NetworkInfo", "name network save_epoch save_state guide")


def network_load(
    name: str,
    suffix: Optional[str] = None,
    inflate: Optional[str] = None,
    epoch: Optional[int] = None,
    reset: bool = False,
    inflate_strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    init_only: bool = False,
    mask_inflate: bool = True,
    device: Optional[torch.device] = None,
    print_output: bool = True,
):
    basename = Path(name).name
    name = network_name(
        name,
        suffix,
        inflate,
        reset=reset,
        inflate_strategy=inflate_strategy,
        mask_inflate=mask_inflate,
    )

    save_epoch = None
    save_state = None
    guide = lambda _: None
    if inflate is None and reset is False and type(epoch) != int:
        model = network_pre(basename)
    else:
        model = network_type(basename)()
        model.to(device)
        save_epoch, save_state = checkpoint.load(
            name, epoch, device=device, print_output=print_output
        )
        if type(epoch) == int and save_epoch is None:
            raise Exception(f"Epoch not found for {name}: {epoch}")
        if save_epoch is not None:
            model.load_state_dict(save_state["model"])
        elif inflate is not None:
            if init_only:
                inflate_network = network_pre(inflate)
                model = inflate_resnet(
                    inflate_network, model, strategy=inflate_strategy, mask=mask_inflate
                )
            else:
                guide = lambda param: guide_resnet(
                    inflate_network,
                    param,
                    network_sequence(basename),
                    strategy=inflate_strategy,
                )

    return NetworkInfo(
        name=name,
        network=model,
        save_epoch=save_epoch,
        save_state=save_state,
        guide=guide,
    )


def network_name(
    name: str,
    suffix: Optional[str] = None,
    inflate: Optional[str] = None,
    reset: bool = False,
    inflate_strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    mask_inflate: bool = True,
):
    if inflate is None and reset is False:
        name = f"{name}-pretrained"

    if inflate is not None:
        name = f"{name}--inflate-{inflate}"

        if inflate_strategy == SequenceInflate.ALIGN_START:
            name = f"{name}-align-start"
        elif inflate_strategy == SequenceInflate.ALIGN_END:
            name = f"{name}-align-end"
        elif inflate_strategy == SequenceInflate.CENTER:
            name = f"{name}-center"
        elif inflate_strategy == SequenceInflate.SPACE_EVENLY:
            name = f"{name}-space-evenly"

        if mask_inflate is False:
            name = f"{name}-unmasked"

    if suffix is not None:
        name = f"{name}--{suffix}"

    return name
