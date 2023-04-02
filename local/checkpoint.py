from collections import OrderedDict
import glob
import math
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from . import device as _device
from . import storage


def get_epoch(name: str, epoch: int = None, latest=True):
    if type(epoch) == int:
        save_path = Path(f"{name}/{epoch:08}.pkl")
        if storage.path_exists(save_path):
            return epoch
        else:
            return None
    else:
        save_paths = list(storage.path_iter(name))
        save_paths.sort(reverse=latest)
        save_path = next(iter(save_paths), None)
        return (
            int(save_path[len(f"{name}.") :].split(".")[0])
            if save_path is not None
            else None
        )


def prune_epochs(name: str, keep: Optional[int] = None):
    print(f"Pruning epochs for {name}")
    save_paths = list(storage.path_iter(name))
    save_paths.sort(reverse=True)
    for save_path in save_paths[len(save_paths) if keep is None else keep :]:
        print(f"Removing {save_path}")
        storage.path_unlink(save_path)


def iter_epochs(name: str, from_epoch: int = 0):
    epoch = get_epoch(name, latest=False)
    i = max(from_epoch, epoch or -math.inf)
    while storage.path_exists(f"{name}/{i:08}.pkl"):
        yield i
        i += 1


def write_log(name: str, data: str):
    with storage.path_open(name, "a") as logfile:
        logfile.write(data)


def log_epoch(name: str, increment: int = 0):
    try:
        with storage.path_open(name).open() as f:
            for line in f:
                pass
            return int(line.split("\t")[0]) + increment
    except:
        return 0


@torch.no_grad()
def to(state: Any, device: torch.device):
    if type(state) == dict or type(state) == OrderedDict:
        r = type(state)()
        for key, val in state.items():
            r[key] = to(val, device)
        return r
    elif type(state) == torch.Tensor:
        if device == _device.cpu:
            return _device.to_cpu(state)
        else:
            return state.to(device)
    else:
        return state


@torch.no_grad()
def save(name: str, epoch: int, state: Any):
    path = f"{name}/{epoch:08}.pkl"
    print(f"Saving `{path}`... ", flush=True, end="")
    state = to(state, _device.cpu)
    with storage.path_open(path, "wb") as save_file:
        torch.save(state, save_file)
    print("DONE")


@torch.no_grad()
def load(
    name: str,
    epoch: int = None,
    device: Optional[torch.device] = None,
    print_output=True,
):
    epoch = get_epoch(name, epoch)
    if epoch is None:
        return (None, None)
    path = f"{name}/{epoch:08}.pkl"
    if print_output:
        print(
            f"Loading `{path}`{f' to {device}' if device is not None else ''}... ",
            flush=True,
            end="",
        )
    try:
        with storage.path_open(path, "rb") as save_path:
            state = torch.load(save_path, map_location=device)
    except RuntimeError:
        with storage.path_open(path, "rb") as save_path:
            state = torch.load(save_path, map_location=_device.cpu)
            state = to(state, device)
    if print_output:
        print("DONE")
    return (epoch, state)


@torch.no_grad()
def reset(module: torch.nn.Module):
    def reset(module: torch.nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    module.apply(fn=reset)
