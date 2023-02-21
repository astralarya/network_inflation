from torch.hub import load
from torchvision.models import resnet


resnet18 = load('pytorch/vision:v0.10.0', 'resnet18', weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
resnet34 = load('pytorch/vision:v0.10.0', 'resnet34', weights=resnet.ResNet34_Weights.IMAGENET1K_V1)
resnet50 = load('pytorch/vision:v0.10.0', 'resnet50', weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
resnet101 = load('pytorch/vision:v0.10.0', 'resnet101', weights=resnet.ResNet101_Weights.IMAGENET1K_V1)
resnet152 = load('pytorch/vision:v0.10.0', 'resnet152', weights=resnet.ResNet152_Weights.IMAGENET1K_V1)

resnets = [
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
]
