from itertools import zip_longest

import torch
import torch.nn as nn
from torchvision.models import ResNet

from . import model


@torch.no_grad()
def resnet(network0: ResNet, network1: ResNet):
    """Initialize network1 via inflating network0

    Args:
        network0: Network to inflate.  Not mutated.
        network1: Network to initialize.  Mutated.
    """
    # Random initialization
    model.reset(network1)

    # Copy and mask
    for child in ["conv1", "bn1", "fc"]:
        copy(network0.get_submodule(child), network1.get_submodule(child))
    
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        children0 = network0.get_submodule(layer).children()
        children1 = network1.get_submodule(layer).children()

        for child0, child1 in zip_longest(children0, children1):
            if child1 is None:
                raise Exception("Inflate destination is smaller than source!")
            elif child0 is None:
                child1.get_parameter("conv3.weight").zero_()
                child1.get_parameter("bn3.bias").zero_()
            else:
                copy(child0, child1)


@torch.no_grad()
def copy(network0: nn.Module, network1: nn.Module):
    """Copy all parameters from network0 to network1 by name"""

    for name, param in network0.named_parameters():
        network1.get_parameter(name).copy_(param)