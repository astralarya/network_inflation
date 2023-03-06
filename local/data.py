import functools

import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode

from local.extern import transforms as extern_transforms


def load_dataset(
    root: str,
    transform: nn.Module,
):
    print(f"Loading data `{root}`... ", flush=True, end="")
    r = ImageFolder(root, transform=transform)
    print("DONE")
    return r


def train_transform(
    crop_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    interpolation=InterpolationMode.BILINEAR,
    hflip_prob=0.5,
    random_erase=0.1,
    auto_augment="ta_wide",
    ra_magnitude=9,
    augmix_severity=3,
):
    interpolation = InterpolationMode(interpolation)
    transform = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]

    if hflip_prob > 0:
        transform.append(transforms.RandomHorizontalFlip(hflip_prob))

    if auto_augment is not None:
        if auto_augment == "ra":
            transform.append(
                transforms.autoaugment.RandAugment(
                    interpolation=interpolation, magnitude=ra_magnitude
                )
            )
        elif auto_augment == "ta_wide":
            transform.append(
                transforms.autoaugment.TrivialAugmentWide(interpolation=interpolation)
            )
        elif auto_augment == "augmix":
            transform.append(
                transforms.autoaugment.AugMix(
                    interpolation=interpolation, severity=augmix_severity
                )
            )
        else:
            aa_policy = transforms.autoaugment.AutoAugmentPolicy(auto_augment)
            transform.append(
                transforms.autoaugment.AutoAugment(
                    policy=aa_policy, interpolation=interpolation
                )
            )
    transform.extend(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    if random_erase > 0:
        transform.append(transforms.RandomErasing(p=random_erase))
    transform = transforms.Compose(transform)

    return transform


def val_transform(
    crop_size=224,
    resize_size=256,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    interpolation=InterpolationMode.BILINEAR,
):
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
            transforms.TenCrop(crop_size),
            transforms.Lambda(_stack_crops),
        ]
    )


def _stack_crops(crops):
    return torch.stack([crop for crop in crops])


def train_collate_fn(
    dataset: ImageFolder,
    mixup_alpha=0.2,
    cutmix_alpha=1.0,
):
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if mixup_alpha > 0.0:
        mixup_transforms.append(
            extern_transforms.RandomMixup(num_classes, p=1.0, alpha=mixup_alpha)
        )
    if cutmix_alpha > 0.0:
        mixup_transforms.append(
            extern_transforms.RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha)
        )
    if mixup_transforms:
        mixupcutmix = transforms.RandomChoice(mixup_transforms)

        return functools.partial(_collate_fn, mixupcutmix=mixupcutmix)

    return None


def _collate_fn(x, mixupcutmix):
    return mixupcutmix(*default_collate(x))
