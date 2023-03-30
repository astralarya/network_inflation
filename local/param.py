from collections import defaultdict
from typing import Any, Mapping, List, Iterable, Optional, Tuple

import torch


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
):
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
        group_classes={"norm": norm_classes},
        group_params={"norm": norm_weight_decay, "__default__": weight_decay},
    )


def build_params(
    model: torch.nn.Module,
    group_classes: Mapping[str, Tuple[Any]] = {},
    group_keys: Mapping[str, Iterable[str]] = {},
    group_params: Mapping[str, Mapping] = {},
    default_group: str = "__default__",
):
    params = defaultdict(list)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for group, keys in group_keys.items():
                if any(
                    key == (f"{prefix}.{name}" if prefix != "" and "." in key else name)
                    for key in keys
                ):
                    params[group].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                for group, classes in group_classes.items():
                    if isinstance(module, classes):
                        params[group].append(p)
                        break
                params[default_group].append(p)

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
                    **(group_params[group] if group in group_params else {}),
                }
            )
    return param_groups
