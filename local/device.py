import torch

try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif xla is not None and xla.xla_device("TPU") is not None:
    device = xla.xla_device("TPU")
elif xla is not None and xla.xla_device("GPU") is not None:
    device = xla.xla_device("GPU")
elif torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
else:
    device = "cpu"

cpu = "cpu"
