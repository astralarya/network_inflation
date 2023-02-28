from collections import OrderedDict
import copy
import glob
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn as nn
try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla = None

from .device import cpu


default_dir = "models"
Path(default_dir).mkdir(exist_ok=True)


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
    return [
        int(save_path[len(f"{name}/"):].split(".")[0])
        for save_path in save_paths
    ]


def write_log(name: str, data: Optional[str] = None):
    if data is not None:
        with Path(f"{name}.log").open("a") as logfile:
            logfile.write(data)


def port_state(state: Any):
    if type(state) == dict or type(state) == OrderedDict:
        r = type(state)()
        for key, val in state.items():
            r[key] = port_state(val)
        return r
    elif type(state) == torch.Tensor:
        return state.detach().clone().to(cpu)
    else:
        return state


def save(name: str, epoch: int, state: Any):
    state = port_state(state)
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
    print(f"Loading `{save_path}` to {device}... ", flush=True, end="")
    state = torch.load(save_path, map_location=device)
    print("DONE")
    return (epoch, state)


def reset(module: nn.Module):
    @torch.no_grad()
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    module.apply(fn=reset)


def clone(module: nn.Module):
    return copy.deepcopy(module)
