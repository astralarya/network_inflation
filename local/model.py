import copy
import glob
import os
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from .device import device


default_dir = "models"
Path(default_dir).mkdir(exist_ok=True)


def get_epoch(name: str, epoch: int = None, model_dir: str = default_dir):
    if epoch is not None:
        save_paths = glob.glob(f"{model_dir}/{name}.{epoch:08}.pkl")
    else:
        save_paths = glob.glob(f"{model_dir}/{name}.{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=True)
    save_path = next(iter(save_paths), None)
    return (
        int(save_path[len(f"{model_dir}/{name}.") :].split(".")[0])
        if save_path is not None
        else None
    )


def write_log(name: str, data: Optional[str] = None, model_dir: str = default_dir):
    if data is not None:
        with (Path(model_dir) / f"{name}.log").open("a") as logfile:
            logfile.write(data)


def write_record(name: str, part: str, data: Optional[str] = None, model_dir: str = default_dir):
    write_log(f"{name}.__{part}__", data, model_dir)


def save(module: nn.Module, name: str, epoch: int, model_dir: str = default_dir):
    save_path = Path(model_dir) / f"{name}.{epoch:08}.pkl"
    with save_path.open("wb") as save_file:
        print(f"Saving `{save_path}`")
        torch.save(module.state_dict(), save_file)


def save_state(
    modules: Mapping[Optional[str], nn.Module], name: str, epoch: int, log: str = None, model_dir: str = default_dir
):
    write_log(log, model_dir)
    for key, value in modules.items():
        save(value, name if key is None else f"{name}.__{key}__", epoch, model_dir)


def load(module: nn.Module, name: str, epoch: int = None, model_dir: str = default_dir):
    epoch = get_epoch(name, epoch, model_dir)
    if epoch is None:
        return None
    save_path = Path(model_dir) / f"{name}.{epoch:08}.pkl"
    print(f"Loading `{save_path}`")
    module.load_state_dict(torch.load(save_path, map_location=device))
    return epoch


def load_state(
    modules: Mapping[Optional[str], nn.Module], name: str, epoch: int = None, model_dir: str = default_dir
):
    if None in modules:
        epoch = get_epoch(name, epoch, model_dir)
    for key, value in modules.items():
        load_epoch = load(value, name if key is None else f"{name}.__{key}__", epoch, model_dir)
        if epoch is not None and load_epoch != epoch:
            raise Exception(f"Missing state `{name}.__{key}__.{epoch}.pkl`")
    if epoch is not None:
        print(f"Resuming from epoch {epoch}")
    else:
        print("Saving initial state as epoch 0")
        save_state(modules, name, 0, model_dir)
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


def clean(name: str, model_dir: str = default_dir):
    model_dir = Path(model_dir)
    save_paths = [
        *glob.glob(f"{model_dir}/{name}.{'[0-9]'*8}.pkl"),
        *glob.glob(f"{model_dir}/{name}.__*__.{'[0-9]'*8}.pkl"),
        *glob.glob(f"{model_dir}/{name}.log"),
        *glob.glob(f"{model_dir}/{name}.__*__.log"),
    ]
    for path in save_paths:
        print(f"Deleting `{path}`")
        os.remove(path)
