from enum import Enum
from typing import Mapping

import torch
import torch.nn as nn

from . import model


class InflateAction(Enum):
    SET_ZERO = Enum.auto()


@torch.no_grad()
def inflate(network0: nn.Module, network1: nn.Module, mapping: Mapping[str, str]):
    """Initialize network1 via inflating network0

    Args:
        network0: Network to inflate.  Not mutated.
        network1: Network to initialize.  Mutated.
    """

    def action(input: InflateAction, output: str):
        match input:
            case InflateAction.SET_ZERO:
                network1.get_parameter(output).zero_()

    # Random initialization
    model.reset(network1)

    for output, input in mapping.items():
        if isinstance(input, str):
            # Copy parameters
            network1.get_parameter(output).copy_(network0.get_parameter(input))
        elif isinstance(input, InflateAction):
            action(input, output)


@torch.no_grad()
def copy(network0: nn.Module, network1: nn.Module):
    """Copy all parameters from network0 to network1 by name"""

    for name, param in network0.named_parameters():
        network1.get_parameter(name).copy_(param)