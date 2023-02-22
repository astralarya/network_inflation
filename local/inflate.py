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


def inflate_resnet50_resnet152(resnet50: resnet.ResNet, resnet152: resnet.ResNet):
    inflate(
        resnet50,
        resnet152,
        {
            # Input Layer
            "conv1.weight": "conv1.weight",
            "bn1.weight": "bn1.weight",
            "bn1.bias": "bn1.bias",
            # 'layer1.0.conv1.weight',
            # 'layer1.0.bn1.weight',
            # 'layer1.0.bn1.bias',
            # 'layer1.0.conv2.weight',
            # 'layer1.0.bn2.weight',
            # 'layer1.0.bn2.bias',
            # 'layer1.0.conv3.weight',
            # 'layer1.0.bn3.weight',
            # 'layer1.0.bn3.bias',
            # 'layer1.0.downsample.0.weight',
            # 'layer1.0.downsample.1.weight',
            # 'layer1.0.downsample.1.bias',
            # 'layer1.1.conv1.weight',
            # 'layer1.1.bn1.weight',
            # 'layer1.1.bn1.bias',
            # 'layer1.1.conv2.weight',
            # 'layer1.1.bn2.weight',
            # 'layer1.1.bn2.bias',
            # 'layer1.1.conv3.weight',
            # 'layer1.1.bn3.weight',
            # 'layer1.1.bn3.bias',
            # 'layer1.2.conv1.weight',
            # 'layer1.2.bn1.weight',
            # 'layer1.2.bn1.bias',
            # 'layer1.2.conv2.weight',
            # 'layer1.2.bn2.weight',
            # 'layer1.2.bn2.bias',
            # 'layer1.2.conv3.weight',
            # 'layer1.2.bn3.weight',
            # 'layer1.2.bn3.bias',
            # 'layer2.0.conv1.weight',
            # 'layer2.0.bn1.weight',
            # 'layer2.0.bn1.bias',
            # 'layer2.0.conv2.weight',
            # 'layer2.0.bn2.weight',
            # 'layer2.0.bn2.bias',
            # 'layer2.0.conv3.weight',
            # 'layer2.0.bn3.weight',
            # 'layer2.0.bn3.bias',
            # 'layer2.0.downsample.0.weight',
            # 'layer2.0.downsample.1.weight',
            # 'layer2.0.downsample.1.bias',
            # 'layer2.1.conv1.weight',
            # 'layer2.1.bn1.weight',
            # 'layer2.1.bn1.bias',
            # 'layer2.1.conv2.weight',
            # 'layer2.1.bn2.weight',
            # 'layer2.1.bn2.bias',
            # 'layer2.1.conv3.weight',
            # 'layer2.1.bn3.weight',
            # 'layer2.1.bn3.bias',
            # 'layer2.2.conv1.weight',
            # 'layer2.2.bn1.weight',
            # 'layer2.2.bn1.bias',
            # 'layer2.2.conv2.weight',
            # 'layer2.2.bn2.weight',
            # 'layer2.2.bn2.bias',
            # 'layer2.2.conv3.weight',
            # 'layer2.2.bn3.weight',
            # 'layer2.2.bn3.bias',
            # 'layer2.3.conv1.weight',
            # 'layer2.3.bn1.weight',
            # 'layer2.3.bn1.bias',
            # 'layer2.3.conv2.weight',
            # 'layer2.3.bn2.weight',
            # 'layer2.3.bn2.bias',
            # 'layer2.3.conv3.weight',
            # 'layer2.3.bn3.weight',
            # 'layer2.3.bn3.bias',
            # Output Layer
            "fc.weight": "fc.weight",
            "fc.bias": "fc.bias",
        },
    )
