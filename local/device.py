import torch

try:
    import torch_xla.core.xla_model as xla
except ImportError:
    xla = None

device_type = "cpu"

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
    device = "cpu"

cpu = "cpu"

print(f"Device: {device}")


def device_step():
    match device_type:
        case "xla":
            xla.mark_step()