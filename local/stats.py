import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets
import tqdm

from local import device


def cosine_similarity(weight0, weight1):
    cos = nn.functional.cosine_similarity
    return torch.stack(
        tuple(
            cos(
                weight0[idx].expand(weight1.shape),
                weight1,
            )
            for idx in range(weight0.shape[0])
        )
    )


def histogram(input, bins=100, range=(0, 1), width=1):
    histogram = input.histogram(bins=bins, range=range)
    plt.bar(
        x=histogram.bin_edges[:-1].detach(),
        height=histogram.hist.detach(),
        width=width * (range[1] - range[0]) / bins,
        align="edge",
    )


def random_unit(shape):
    return torch.normal(torch.zeros(shape), torch.ones(shape))


def self_similarity(weight0: torch.Tensor):
    weight0 = weight0.detach().flatten(start_dim=1)
    measure = (
        cosine_similarity(weight0, weight0).fill_diagonal_(0).abs().max(dim=1).values
    )
    histogram(measure)


def similarity(weight0: torch.Tensor, weight1: torch.Tensor):
    measure = (
        cosine_similarity(
            weight0.detach().flatten(start_dim=1),
            weight1.detach().flatten(start_dim=1),
        )
        .abs()
        .max(dim=1)
        .values
    )
    histogram(measure)


def null_similarity(weight0: torch.Tensor):
    weight0 = weight0.detach().flatten(start_dim=1)
    measure = torch.stack(
        [
            cosine_similarity(
                weight0,
                random_unit(weight0.shape),
            )
            .abs()
            .max(dim=1)
            .values
            for _ in range(2048)
        ]
    )
    histogram(measure, bins=1000)


@torch.no_grad()
def divergence(
    network0: nn.Module,
    network1: nn.Module,
    data: datasets.DatasetFolder,
    batch_size=256,
    num_workers=8,
):
    data_loader = torch.utils.data.DataLoader2(
        data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    log_softmax = nn.LogSoftmax(dim=1).to(device.device())
    criterion = nn.KLDivLoss(reduction="sum", log_target=True).to(device.device())
    network0.eval()
    network0.to(device.device())
    network1.eval()
    network1.to(device.device())

    total_loss = 0.0
    total = len(data_loader.dataset)
    print(f"Iterating {total} samples")
    for inputs, _ in tqdm(data_loader):
        inputs = inputs.to(device.device())
        outputs0 = network0(inputs)
        outputs1 = network1(inputs)
        loss = criterion(log_softmax(outputs0), log_softmax(outputs1))
        total_loss += loss.item() / total
        device.step()
    print(f"Divergence: {total_loss}")
    return total_loss
