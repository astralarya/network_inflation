from enum import Enum
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet


class SequenceInflate(Enum):
    ALIGN_START = "align-start"
    ALIGN_END = "align-end"
    CENTER = "center"
    SPACE_EVENLY = "space-evenly"

    def __str__(self):
        return self.value


@torch.no_grad()
def guide_resnet(
    network: ResNet,
    parameter_name: str,
    inflate_sequence: Sequence[int],
    strategy: SequenceInflate = SequenceInflate.ALIGN_START,
):
    """Return the guide for a Resnet parameter"""
    parameter_path = parameter_name.split(".")
    if parameter_path[0] in ["conv1", "bn1", "fc1"]:
        return network.get_parameter(parameter_name)
    elif parameter_path[0] in ["layer1", "layer2", "layer3", "layer4"]:
        layer_idx = int(parameter_path[0][5]) - 1
        sequence_idx = int(parameter_path[1])
        guide_module = guide_sequence(
            network.get_submodule(parameter_path[0]),
            index=sequence_idx,
            inflate_size=inflate_sequence[layer_idx],
            strategy=strategy,
        )
        if guide_module is None:
            return None
        return guide_module.get_parameter(".".join(parameter_path[1:]))


@torch.no_grad()
def guide_sequence(
    sequence: nn.Module,
    index: int,
    inflate_size: int,
    strategy: SequenceInflate,
):
    children = list(sequence.children())
    diff = len(inflate_size) - len(children)
    if diff < 0:
        raise Exception("Inflate destination is smaller than source!")

    if strategy == SequenceInflate.ALIGN_START:
        if index < len(sequence):
            return sequence[index]
        else:
            return None
    elif strategy == SequenceInflate.ALIGN_END:
        if index < diff:
            return None
        else:
            return sequence[index - diff]
    elif strategy == SequenceInflate.CENTER:
        start = diff // 2 + diff % 2
        if index < start:
            return None
        else:
            return sequence[start - index]
    elif strategy == SequenceInflate.SPACE_EVENLY and diff > 0:
        for idx in np.round(
            np.linspace(0.5 + np.finfo(float).eps, len(children - 0.5, diff))
        ).astype(int):
            children[idx:idx] = [None]
        return children[index]


@torch.no_grad()
def resnet(
    network0: ResNet,
    network1: ResNet,
    strategy: SequenceInflate = SequenceInflate.ALIGN_START,
    mask=True,
):
    """Initialize network1 via inflating network0

    Args:
        network0: Network to inflate.  Not mutated.
        network1: Network to initialize.  Mutated.
    """
    # Copy shared submodules
    for child in ["conv1", "bn1", "fc"]:
        copy(network0.get_submodule(child), network1.get_submodule(child))

    # Inflate sequences
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        inflate_sequence(
            network0.get_submodule(layer),
            network1.get_submodule(layer),
            strategy=strategy,
            mask="conv3.weight" if mask else None,
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
    if diff < 0:
        raise Exception("Inflate destination is smaller than source!")

    if strategy == SequenceInflate.ALIGN_START:
        children0.extend([None] * diff)
    elif strategy == SequenceInflate.ALIGN_END:
        children0[1:1] = [None] * diff
    elif strategy == SequenceInflate.CENTER:
        children0.extend([None] * (diff // 2 + diff % 2))
        children0[1:1] = [None] * (diff // 2)
    elif strategy == SequenceInflate.SPACE_EVENLY and diff > 0:
        for idx in np.round(
            np.linspace(0.5 + np.finfo(float).eps, len(children1) - 0.5, diff)
        ).astype(int):
            children0[idx:idx] = [None]

    for child0, child1 in zip(children0, children1):
        if child0 is None:
            if mask:
                child1.get_parameter(mask).zero_()
        else:
            copy(child0, child1)


@torch.no_grad()
def copy(network0: nn.Module, network1: nn.Module):
    """Copy all parameters from network0 to network1 by name"""

    network1.load_state_dict(network0.state_dict())
