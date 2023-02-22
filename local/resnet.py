from torch.hub import load
from torchvision.models import resnet


resnet18 = lambda: load('pytorch/vision:v0.10.0', 'resnet18', weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
resnet34 = lambda: load('pytorch/vision:v0.10.0', 'resnet34', weights=resnet.ResNet34_Weights.IMAGENET1K_V1)
resnet50 = lambda: load('pytorch/vision:v0.10.0', 'resnet50', weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
resnet101 = lambda: load('pytorch/vision:v0.10.0', 'resnet101', weights=resnet.ResNet101_Weights.IMAGENET1K_V1)
resnet152 = lambda: load('pytorch/vision:v0.10.0', 'resnet152', weights=resnet.ResNet152_Weights.IMAGENET1K_V1)
