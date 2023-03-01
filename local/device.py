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
    from torch_xla.distributed.xla_multiprocessing import spawn as xla_spawn
except ImportError:
    xla_spawn = None


device_type = None

if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_type = "mps"
elif xla is not None and xla.xla_device() is not None:
    device = xla.xla_device()
    device_type = "xla"
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    device_type = "cuda"
else:
    device = None

cpu = "cpu"

print(f"Device: {device}")


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
        return lambda x: XlaLoader(x, device)
    else:
        return lambda x: x


loader = _loader()


def _is_main():
    if device_type == "xla":
        return xla.is_master_ordinal()
    else:
        return True


is_main = _is_main()


def _spawn():
    if device_type == "xla":
        return xla_spawn
    else:
        return lambda x, args: x(0, *args)


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
