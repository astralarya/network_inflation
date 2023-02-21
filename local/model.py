import copy
import glob
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from .device import device


out_dir = "models"


def save(model: nn.Module, name: str, epoch: int):
    save_path = Path(out_dir).joinpath(f"{name}.{epoch:08}.pkl")
    with save_path.open("wb") as save_file:
        print(f"Saving `{save_path}`")
        torch.save(model.state_dict(), save_file)


def save_state(model_dict: Mapping[Optional[str], nn.Module], name: str, epoch: int, log: str = None):
    with open(Path(out_dir).joinpath(f"{name}.log"), "a") as logfile:
        logfile.write(log)
    for key, value in model_dict.items():
        save(value, name if key is None else f"{name}.__{key}__", epoch)


def load(model: nn.Module, name: str, epoch: int = None):
    epoch = get_epoch(name, epoch)
    if epoch is None:
        return None
    save_path = f"{out_dir}/{name}.{epoch:08}.pkl"
    print(f"Loading `{save_path}`")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return epoch

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


def load_state(model_dict: Mapping[Optional[str], nn.Module], name: str, epoch: int = None):
    if None in model_dict:
        epoch = get_epoch(name, epoch)
    for key, value in model_dict.items():
        load_epoch = load(value, name if key is None else f"{name}.__{key}__", epoch)
        if epoch is not None and load_epoch != epoch:
            raise Exception(f"Missing state `{name}.__{key}__.{epoch}.pkl`")
    return epoch


def reset(model: nn.Module):
    @torch.no_grad()
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    model.apply(fn=reset)


def clone(model: nn.Module):
    return copy.deepcopy(model)
    
