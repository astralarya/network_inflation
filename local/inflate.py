from enum import Enum
from typing import Union

import torch
import torch.nn as nn
from torchvision.models import ResNet


class SequenceInflate(Enum):
    ALIGN_START = 0
    ALIGN_END = 1
    CENTER = 2
    SPACE_EVENLY = 3


@torch.no_grad()
def resnet(
    network0: ResNet,
    network1: ResNet,
    strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    mask: Union[str, None] = "conv3.weight",
):
    """Initialize network1 via inflating network0

    Args:
        network0: Network to inflate.  Not mutated.
        network1: Network to initialize.  Mutated.
    """
    # Copy and mask
    for child in ["conv1", "bn1", "fc"]:
        copy(network0.get_submodule(child), network1.get_submodule(child))

    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        inflate_sequence(
            network0.get_submodule(layer),
            network1.get_submodule(layer),
            strategy=strategy,
            mask=mask,
        )
    return network1


@torch.no_grad()
def inflate_sequence(
    sequence0: nn.Module,
    sequence1: nn.Module,
    strategy: SequenceInflate,
    mask: Union[str, None],
):
    children0 = list(sequence0.children())
    children1 = list(sequence1.children())
    diff = len(children1) - len(children0)

    if strategy == SequenceInflate.ALIGN_START:
        children0.extend([None] * diff)
    elif strategy == SequenceInflate.ALIGN_END:
        children0[1:1] = [None] * diff
    elif strategy == SequenceInflate.CENTER:
        children0.extend([None] * (diff // 2 + diff % 2))
        children0[1:1] = [None] * (diff // 2)
    elif strategy == SequenceInflate.SPACE_EVENLY and diff > 0:
        step = (len(children0) - 1) / diff
        for i in range(1, diff):
            idx = 1 + round(i * step)
            children0[idx:idx] = [None]

    for child0, child1 in zip(children0, children1):
        if child1 is None:
            raise Exception("Inflate destination is smaller than source!")
        elif child0 is None and mask:
            child1.get_parameter(mask).zero_()
        else:
            copy(child0, child1)


@torch.no_grad()
def copy(network0: nn.Module, network1: nn.Module):
    """Copy all parameters from network0 to network1 by name"""

    network1.load_state_dict(network0.state_dict())
