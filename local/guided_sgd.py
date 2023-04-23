from typing import List, Optional
from itertools import zip_longest

import torch
from torch import Tensor
from torch.nn.functional import softmax
from torch.optim import SGD
from torch.optim.optimizer import _use_grad_for_differentiable


__all__ = ["GuidedSGD", "guided_sgd"]


class GuidedSGD(SGD):
    def __init__(
        self,
        params,
        guide_alpha=1.0,
        guide_epochs=32,
        **kwargs,
    ) -> None:
        self.guide_alpha = guide_alpha
        self.guide_epochs = guide_epochs
        super().__init__(params, **kwargs)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        guide_distance = [0.0] * len(self.param_groups[0]["guide"][0])
        for group in self.param_groups:
            for p in group["params"]:
                for idx, pos in enumerate(group["guide"]):
                    guide_distance[idx] += p.sub(pos).pow(2)
        guide_p = softmax(torch.stack([d.sum().sqrt() for d in guide_distance]))

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            guide = []
            momentum_buffer_list = []
            has_sparse_grad = False

            for idx, p in enumerate(group["params"]):
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    guide.append(group["guide"][idx])
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state["momentum_buffer"])

            guided_sgd(
                params_with_grad,
                d_p_list,
                momentum_buffer_list,
                guide=guide,
                guide_p=guide_p,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
            )

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def guided_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = None,
    foreach: bool = None,
    *,
    guide: List[Tensor],
    guide_p: Optional[List[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_guided_sgd
    else:
        func = _single_tensor_guided_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        guide_p=guide_p,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
    )


def _single_tensor_guided_sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    guide: List[Tensor],
    guide_p: Optional[List[Tensor]],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):

    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)


def _multi_tensor_guided_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    guide: List[Tensor],
    guide_p=List[Tensor],
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)
