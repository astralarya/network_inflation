from collections import OrderedDict
import copy
import glob
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla = None

from . import device as _device


def get_epoch(name: str, epoch: int = None):
    if epoch is not None:
        save_path = Path(f"{name}/{epoch:08}.pkl")
        if save_path.exists():
            return epoch
        else:
            return None
    else:
        save_paths = glob.glob(f"{name}/{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=True)
        save_path = next(iter(save_paths), None)
        return (
            int(save_path[len(f"{name}.") :].split(".")[0])
            if save_path is not None
            else None
        )


def iter_epochs(name: str):
    i = 0
    p = Path(f"{name}/{i:08}.pkl")
    while p.exists():
        yield i
        i += 1
        p = Path(f"{name}/{i:08}.pkl")


def write_log(name: str, data: str):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{name}.log").open("a") as logfile:
        logfile.write(data)


@torch.no_grad()
def state_to(state: Any, device: torch.device):
    to = _device.to_cpu if device == _device.cpu else lambda x: x.to(device)
    if type(state) == dict or type(state) == OrderedDict:
        r = type(state)()
        for key, val in state.items():
            r[key] = state_to(val, device)
        return r
    elif type(state) == torch.Tensor:
        return to(state)
    else:
        return state


@torch.no_grad()
def save(name: str, epoch: int, state: Any):
    save_path = Path(f"{name}/{epoch:08}.pkl")
    print(f"Saving `{save_path}`... ", flush=True, end="")
    state = state_to(state, _device.cpu)
    Path(name).mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as save_file:
        torch.save(state, save_file)
    print("DONE")


@torch.no_grad()
def load(name: str, epoch: int = None, device: torch.device = None):
    epoch = get_epoch(name, epoch)
    if epoch is None:
        return (None, None)
    save_path = Path(f"{name}/{epoch:08}.pkl")
    print(
        f"Loading `{save_path}`{f' to {device}' if device is not None else ''}... ",
        flush=True,
        end="",
    )
    try:
        state = torch.load(save_path, map_location=device)
    except RuntimeError:
        state = torch.load(save_path, map_location=_device.cpu)
        state = state_to(state, device)
    print("DONE")
    return (epoch, state)


@torch.no_grad()
def reset(module: nn.Module):
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    module.apply(fn=reset)


@torch.no_grad()
def clone(module: nn.Module):
    return copy.deepcopy(module)
