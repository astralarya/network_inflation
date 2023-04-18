from collections import defaultdict, namedtuple
from typing import Any, Callable, Mapping, List, Iterable, Optional, Tuple

import torch


Guide = Callable[[str], Optional[torch.TupleType]]


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    guide: Optional[Guide] = None,
) -> list[dict]:
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    return build_params(
        model,
        guide=guide,
        group_classes={"norm": norm_classes},
        group_params={
            "norm": {"weight_decay": norm_weight_decay},
            "__default__": {"weight_decay": weight_decay},
        },
    )


def build_params(
    model: torch.nn.Module,
    guide: Optional[Guide] = None,
    group_classes: Mapping[str, Tuple[Any]] = {},
    group_keys: Mapping[str, Iterable[str]] = {},
    group_params: Mapping[str, Mapping] = {},
    default_group: str = "__default__",
) -> list[dict]:
    params = defaultdict(list)
    guides = defaultdict(list)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            g = guide(name) if guide else None
            if not p.requires_grad:
                continue
            for group, keys in group_keys.items():
                if any(
                    key == (f"{prefix}.{name}" if prefix != "" and "." in key else name)
                    for key in keys
                ):
                    params[group].append(p)
                    guides[group].append(g)
                    break
            else:
                for group, classes in group_classes.items():
                    if isinstance(module, classes):
                        params[group].append(p)
                        guides[group].append(g)
                        break
                else:
                    params[default_group].append(p)
                    guides[default_group].append(g)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for group in params:
        if len(params[group]) > 0:
            param_groups.append(
                {
                    "params": params[group],
                    "guide": guides[group],
                    **(group_params[group] if group in group_params else {}),
                }
            )
    return param_groups
