import copy
import glob
import os
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from .device import device


out_dir = Path("models")
out_dir.mkdir(exist_ok=True)


def get_epoch(name: str, epoch: int = None):
    if epoch is not None:
        save_paths = glob.glob(f"{out_dir}/{name}.{epoch:08}.pkl")
    else:
        save_paths = glob.glob(f"{out_dir}/{name}.{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=True)
    save_path = next(iter(save_paths), None)
    return (
        int(save_path[len(f"{out_dir}/{name}.") :].split(".")[0])
        if save_path is not None
        else None
    )


def write_log(name: str, data: Optional[str] = None):
    if data is not None:
        with out_dir.joinpath(f"{name}.log").open("a") as logfile:
            logfile.write(data)


def write_record(name: str, part: str, data: Optional[str] = None):
    write_log(f"{name}.__{part}__", data)


def save(module: nn.Module, name: str, epoch: int):
    save_path = out_dir.joinpath(f"{name}.{epoch:08}.pkl")
    with save_path.open("wb") as save_file:
        print(f"Saving `{save_path}`")
        torch.save(module.state_dict(), save_file)


def save_state(
    modules: Mapping[Optional[str], nn.Module], name: str, epoch: int, log: str = None
):
    write_log(log)
    for key, value in modules.items():
        save(value, name if key is None else f"{name}.__{key}__", epoch)


def load(module: nn.Module, name: str, epoch: int = None):
    epoch = get_epoch(name, epoch)
    if epoch is None:
        return None
    save_path = f"{out_dir}/{name}.{epoch:08}.pkl"
    print(f"Loading `{save_path}`")
    module.load_state_dict(torch.load(save_path, map_location=device))
    return epoch


def load_state(
    modules: Mapping[Optional[str], nn.Module], name: str, epoch: int = None
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
        print("Saving initial state as epoch 0")
        save_state(modules, name, 0)
    return epoch


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
        *glob.glob(f"{out_dir}/{name}.{'[0-9]'*8}.pkl"),
        *glob.glob(f"{out_dir}/{name}.__*__.{'[0-9]'*8}.pkl"),
        *glob.glob(f"{out_dir}/{name}.log"),
        *glob.glob(f"{out_dir}/{name}.__*__.log"),
    ]
    for path in save_paths:
        print(f"Deleting `{path}`")
        os.remove(path)
