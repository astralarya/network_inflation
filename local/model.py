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

from .device import device, cpu


default_dir = "models"
Path(default_dir).mkdir(exist_ok=True)


def get_epoch(name: str, epoch: int = None):
    if epoch is not None:
        save_paths = glob.glob(f"{name}.{epoch:08}.pkl")
    else:
        save_paths = glob.glob(f"{name}.{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=True)
    save_path = next(iter(save_paths), None)
    return (
        int(save_path[len(f"{name}.") :].split(".")[0])
        if save_path is not None
        else None
    )


def list_epochs(name: str):
    save_paths = glob.glob(f"{name}.{'[0-9]'*8}.pkl")
    save_paths.sort()
    return [
        int(save_path[len(f"{name}."):].split(".")[0])
        for save_path in save_paths
    ]


def write_log(name: str, data: Optional[str] = None):
    if data is not None:
        with Path(f"{name}.log").open("a") as logfile:
            logfile.write(data)


def write_record(name: str, part: str, data: Optional[str] = None):
    write_log(f"{name}.__{part}__", data)


def save(module: nn.Module, name: str, epoch: int):
    save_path = Path(f"{name}.{epoch:08}.pkl")
    with save_path.open("wb") as save_file:
        print(f"Saving `{save_path}`")
        (xla.save if xla is not None else torch.save)(module.state_dict(), save_file)


def save_state(
    modules: Mapping[Optional[str], nn.Module], name: str, epoch: int, log: str = None
):
    write_log(name, log)
    for key, value in modules.items():
        save(value, name if key is None else f"{name}.__{key}__", epoch)


def load(module: nn.Module, name: str, epoch: int = None):
    epoch = get_epoch(name, epoch)
    if epoch is None:
        return None
    save_path = Path(f"{name}.{epoch:08}.pkl")
    print(f"Loading `{save_path}`")
    module.load_state_dict(torch.load(save_path))
    return epoch


def load_state(
    modules: Mapping[Optional[str], nn.Module],
    name: str,
    epoch: int = None,
    init_fn: Optional[Callable[[nn.Module], Any]] = None,
):
    if None in modules:
        epoch = get_epoch(name, epoch)
    for key, value in modules.items():
        load_epoch = load(value, name if key is None else f"{name}.__{key}__", epoch)
        if epoch is not None and load_epoch != epoch:
            raise Exception(f"Missing state `{name}.__{key}__.{epoch}.pkl`")
    if epoch is not None:
        print(f"Resuming from epoch {epoch}")
    else:
        if init_fn is not None and None in modules:
            init_fn(modules[None])
        print("Saving initial state as epoch 0")
        save_state(modules, name, 0)
    return 0 if epoch is None else epoch


def reset(module: nn.Module):
    @torch.no_grad()
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    module.apply(fn=reset)


def clone(module: nn.Module):
    return copy.deepcopy(module)


def clean(name: str):
    save_paths = [
        *glob.glob(f"{name}.{'[0-9]'*8}.pkl"),
        *glob.glob(f"{name}.__*__.{'[0-9]'*8}.pkl"),
        *glob.glob(f"{name}.log"),
        *glob.glob(f"{name}.__*__.log"),
    ]
    for path in save_paths:
        print(f"Deleting `{path}`")
        os.remove(path)
