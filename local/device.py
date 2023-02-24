import torch

try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla = None


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif xla is not None and xla.xla_device() is not None:
    device = xla.xla_device()
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
else:
    device = "cpu"

cpu = "cpu"
