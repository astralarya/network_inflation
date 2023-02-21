import copy
import glob
from pathlib import Path

import torch
import torch.nn as nn


models_dir = "models"


def save(model: nn.Module, name: str, epoch: int):
    save_path = Path(models_dir).joinpath(f"{name}.{epoch:08}.pkl")
    with save_path.open("wb") as save_file:
        print(f"Saving `{save_path}`")
        torch.save(model.state_dict(), save_file)


def load(model: nn.Module, name: str, epoch: int = None):
    save_paths = None
    if epoch is not None:
        save_paths = glob.glob(f"{models_dir}/{name}.{epoch:08}.pkl")
    else:
        save_paths = glob.glob(f"{models_dir}/{name}.{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=True)
    save_path = next(iter(save_paths), None)
    if save_path is not None:
        print(f"Loading `{save_path}`")
        epoch = int(save_path[len(f"{models_dir}/{name}.") :].split(".")[0])
        model.load_state_dict(torch.load(save_path))
        return epoch
    else:
        return None


def reset(model: nn.Module):
    @torch.no_grad()
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    model.apply(fn=reset)


def clone(model: nn.Module):
    return copy.deepcopy(model)

