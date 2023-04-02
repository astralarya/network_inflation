from collections import OrderedDict
import glob
import math
from pathlib import Path
from typing import Any, Optional
from os import environ

import torch
import torch.nn as nn

from . import device as _device

try:
    from google.cloud import storage as gcloud_storage
except ImportError:
    gcloud_storage = None


GCLOUD_BUCKET = environ.get("GCLOUD_BUCKET_KEY", None)


def file__get_epoch(name: str, epoch: int = None, latest=True):
    if type(epoch) == int:
        save_path = Path(f"{name}/{epoch:08}.pkl")
        if save_path.exists():
            return epoch
        else:
            return None
    else:
        save_paths = glob.glob(f"{name}/{'[0-9]'*8}.pkl")
        save_paths.sort(reverse=latest)
        save_path = next(iter(save_paths), None)
        return (
            int(save_path[len(f"{name}/") :].split(".")[0])
            if save_path is not None
            else None
        )


def gcloud__get_epoch(name: str, epoch: int = None, latest=True):
    storage_client = gcloud_storage.Client()
    bucket = storage_client.bucket(GCLOUD_BUCKET)
    if type(epoch) == int:
        save_path = f"{name}/{epoch:08}.pkl"
        if bucket.Blob(save_path).exists():
            return epoch
        else:
            return None
    else:
        save_paths = list(
            storage_client.list_blobs(GCLOUD_BUCKET, prefix=f"{name}/")
        ).sort(reverse=latest)
        save_path = next(iter(save_paths), None)
        return (
            int(save_path.name[len(f"{name}/") :].split(".")[0])
            if save_path is not None
            else None
        )


def _get_epoch():
    if gcloud_storage and GCLOUD_BUCKET:
        return gcloud__get_epoch
    else:
        return file__get_epoch


get_epoch = _get_epoch()


def prune_epochs(name: str, keep: int = 32):
    print(f"Pruning epochs for {name}")
    save_paths = glob.glob(f"{name}/{'[0-9]'*8}.pkl")
    save_paths.sort(reverse=True)
    for save_path in save_paths[keep:]:
        print(f"Removing {save_path}")
        Path(save_path).unlink()


def iter_epochs(name: str, from_epoch: int = 0):
    epoch = get_epoch(name, latest=False)
    i = max(from_epoch, epoch or -math.inf)
    p = Path(f"{name}/{i:08}.pkl")
    while p.exists():
        yield i
        i += 1
        p = Path(f"{name}/{i:08}.pkl")


def write_log(name: str, data: str):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    with Path(name).open("a") as logfile:
        logfile.write(data)


def log_epoch(name: str, increment: int = 0):
    try:
        with Path(name).open() as f:
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
    save_path = Path(f"{name}/{epoch:08}.pkl")
    print(f"Saving `{save_path}`... ", flush=True, end="")
    state = to(state, _device.cpu)
    Path(name).mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as save_file:
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
    save_path = Path(f"{name}/{epoch:08}.pkl")
    if print_output:
        print(
            f"Loading `{save_path}`{f' to {device}' if device is not None else ''}... ",
            flush=True,
            end="",
        )
    try:
        state = torch.load(save_path, map_location=device)
    except RuntimeError:
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
