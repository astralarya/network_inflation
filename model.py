import glob
from pathlib import Path

import torch
import torch.nn as nn


models_dir = "models"


def save_model(model, name, epoch):
    save_path = Path(models_dir).joinpath(f"{name}.{epoch:03}.pkl")
    with save_path.open("wb") as save_file:
        torch.save(model.state_dict(), save_file)


def load_model(model, name):
    save_paths = glob.glob(f"{models_dir}/{name}.*.pkl")
    save_paths.sort(reverse=True)
    save_path = next(iter(save_paths), None)
    if save_path is not None:
        epoch = int(save_path[len(f"{models_dir}/{name}.") :].split(".")[0])
        model.load_state_dict(torch.load(save_path))
        return epoch
    else:
        return None


def reset_model(model: nn.Module):
    @torch.no_grad()
    def reset(module: nn.Module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            module.reset_parameters()

    model.apply(fn=reset)
