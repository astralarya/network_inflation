from typing import Optional

from torch.hub import load
from torchvision.models import resnet

from local import inflate as _inflate


def network_pre(name: str):
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
    network = getattr(resnet, name, lambda: None)
    if network is None:
        print(f"Unknown network: {name}")
        exit(1)
    return network


def network_load(name: str, inflate: Optional[str] = None, reset: bool = False):
    if inflate is None and reset is False:
        network = network_pre(name)
    else:
        network = network_type(name)()

    inflate_network = None
    if inflate is not None:
        inflate_network = network_pre(inflate)
        network = _inflate.resnet(inflate_network, network)

    return network


def network_name(name, inflate, reset):
    if reset is False:
        name = f"{name}--finetune"
    if inflate is not None:
        name = f"{name}--inflate-{inflate}"
    return name
