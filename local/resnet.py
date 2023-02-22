from torch.hub import load
import torch.nn as nn
from torchvision.models import resnet


resnet18 = lambda: load('pytorch/vision:v0.10.0', 'resnet18', weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
resnet34 = lambda: load('pytorch/vision:v0.10.0', 'resnet34', weights=resnet.ResNet34_Weights.IMAGENET1K_V1)
resnet50 = lambda: load('pytorch/vision:v0.10.0', 'resnet50', weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
resnet101 = lambda: load('pytorch/vision:v0.10.0', 'resnet101', weights=resnet.ResNet101_Weights.IMAGENET1K_V1)
resnet152 = lambda: load('pytorch/vision:v0.10.0', 'resnet152', weights=resnet.ResNet152_Weights.IMAGENET1K_V1)

def inflate_resnet50_resnet152(resnet50: resnet.ResNet, resnet152: resnet.ResNet):
    """Initialize Resnet152 via inflating pretrained Resnet50

    Args:
        resnet50: Resnet to inflate.  Not mutated.
        resnet152: Resnet to initialize.  Mutated.
    """
    return