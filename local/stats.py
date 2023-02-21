import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def cosine_similarity(weight0, weight1):
    cos = nn.functional.cosine_similarity
    return torch.stack(tuple(
        cos(
            weight0[idx].expand(weight1.shape),
            weight1,
        )
        for idx in range(weight0.shape[0])
    ))

def histogram(input, bins=100, range=(0,1), width=1):
    histogram = input.histogram(bins=bins,range=range)
    plt.bar(
        x=histogram.bin_edges[:-1].detach(),
        height=histogram.hist.detach(),
        width=width*(range[1]-range[0])/bins,
        align='edge')

def random_unit(shape):
    return torch.normal(torch.zeros(shape), torch.ones(shape))

def self_similarity(weight0: torch.Tensor):
    weight0 = weight0.detach().flatten(start_dim=1)
    measure = cosine_similarity(weight0, weight0).fill_diagonal_(0).abs().max(dim=1).values
    histogram(measure)

def similarity(weight0: torch.Tensor, weight1: torch.Tensor):
    measure = cosine_similarity(
        weight0.detach().flatten(start_dim=1),
        weight1.detach().flatten(start_dim=1),
    ).abs().max(dim=1).values
    histogram(measure)

def null_similarity(weight0: torch.Tensor):
    weight0 = weight0.detach().flatten(start_dim=1)
    measure = torch.stack([
        cosine_similarity(
            weight0,
            random_unit(weight0.shape),
        ).abs().max(dim=1).values
        for _ in range(2048)
    ])
    histogram(measure, bins=1000)