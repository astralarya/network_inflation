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
        save_paths = glob.glob(f"{name}/{epoch:08}.pkl")
    else:
        save_paths = glob.glob(f"{name}/{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=True)
    save_path = next(iter(save_paths), None)
    return (
        int(save_path[len(f"{name}.") :].split(".")[0])
        if save_path is not None
        else None
    )


def list_epochs(name: str):
    save_paths = glob.glob(f"{name}/{'[0-9]'*8}.pkl")
    save_paths.sort()
    return [int(save_path[len(f"{name}/") :].split(".")[0]) for save_path in save_paths]


def write_log(name: str, data: str):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    with Path(f"{name}.log").open("a") as logfile:
        logfile.write(data)


def state_to(state: Any, device: torch.device):
    if type(state) == dict or type(state) == OrderedDict:
        r = type(state)()
        for key, val in state.items():
            r[key] = state_to(val, device)
        return r
    elif type(state) == torch.Tensor:
        return state.detach().clone().to(device)
    else:
        return state


def save(name: str, epoch: int, state: Any):
    if _device.is_main():
        state = state_to(state, _device.cpu)
        Path(name).mkdir(parents=True, exist_ok=True)
        save_path = Path(f"{name}/{epoch:08}.pkl")
        with save_path.open("wb") as save_file:
            print(f"Saving `{save_path}`... ", flush=True, end="")
            torch.save(state, save_file)
        print("DONE")


def load(name: str, epoch: int = None, device=None):
    epoch = get_epoch(name, epoch)
    if epoch is None:
        return (None, None)
    save_path = Path(f"{name}/{epoch:08}.pkl")
    print(
        f"Loading `{save_path}`{f' to {device}' if device is not None else ''}... ",
        flush=True,
        end="",
    )
    state = None
    if _device.is_main():
        try:
            state = torch.load(save_path, map_location=device)
        except RuntimeError:
            state = torch.load(save_path, map_location=_device.cpu)
    state = _device.mesh_reduce("load", state, lambda x: x[0])
    if _device.is_main():
        print(state.keys())
        print("DONE")
    state = state_to(state, device)
    return (epoch, state)


def reset(module: nn.Module):
    @torch.no_grad()
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    module.apply(fn=reset)

    if _device.world_size() > 1:
        y = module.state_dict() if _device.is_main() else None
        y, *_ = _device.rendezvous("reset", y)
        module.load_state_dict(y)


def clone(module: nn.Module):
    return copy.deepcopy(module)
