import torch

try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla = None

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


def _device_step():
    if device_type == "xla":
        return xla.mark_step
    else:
        return lambda: None


device_step = _device_step()
