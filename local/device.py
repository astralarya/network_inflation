from typing import Optional

import torch

try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla = None

try:
    from torch_xla.distributed.parallel_loader import MpDeviceLoader as XlaLoader
except ImportError:
    XlaLoader = None

try:
    from torch_xla.distributed.xla_multiprocessing import (
        spawn as xla_spawn,
        MpModelWrapper as XlaModel,
    )
except ImportError:
    xla_spawn = None


_device = None


def device():
    global _device
    if _device:
        return _device
    else:
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif xla is not None and xla.xla_device() is not None:
            _device = xla.xla_device(get_ordinal())
        elif torch.cuda.is_available():
            _device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Device: {_device}")
        return _device


device_type = None

if torch.backends.mps.is_available():
    device_type = "mps"
elif torch.cuda.is_available():
    device_type = "cuda"
elif xla is not None:
    device_type = "xla"

cpu = torch.device("cpu")


def _step():
    if device_type == "xla":
        return xla.mark_step
    else:
        return lambda: None


step = _step()


def _optim_step():
    if device_type == "xla":
        return xla.optimizer_step
    else:
        return lambda x: x.step()


optim_step = _optim_step()


def _loader():
    if device_type == "xla":
        return lambda x: XlaLoader(x, device())
    else:
        return lambda x: x


loader = _loader()


def _model():
    if device_type == "xla":
        return lambda x: XlaModel(x)
    else:
        return lambda x: x


model = _model()


def _is_main():
    if device_type == "xla":
        return lambda: xla.is_master_ordinal()
    else:
        return lambda: True


is_main = _is_main()


def _ordinal():
    if device_type == "xla":
        return lambda: xla.get_ordinal()
    else:
        return lambda: 0


ordinal = _ordinal()


def _world_size():
    if device_type == "xla":
        return lambda: xla.xrt_world_size()
    else:
        return lambda: 1


world_size = _world_size()


def _spawn():
    if device_type == "xla":
        return xla_spawn
    else:
        return lambda x, args, **_: x(0, *args)


spawn = _spawn()


def _rendezvous():
    if device_type == "xla":
        return xla.rendezvous
    else:
        return lambda _, x=None: (x,)


rendezvous = _rendezvous()


def _mesh_reduce():
    if device_type == "xla":
        return xla.mesh_reduce
    else:
        return lambda _, data, reduce_fn: reduce_fn([data])


mesh_reduce = _mesh_reduce()
