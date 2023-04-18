import types
import math
from torch._six import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim import Optimizer

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class Scheduler(object):
    def __init__(self, optimizer, parameter: str, last_epoch=-1, verbose=False):
        self.parameter = parameter
        init_param = "init__{}".format(init_param)

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base parameter values
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault(init_param, group[parameter])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if init_param not in group:
                    raise KeyError(
                        "param '{}' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(
                            init_param, i
                        )
                    )
        self.base_values = [group[init_param] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.verbose = verbose

        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_value

    def get_value(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_value(self, is_verbose, group, value, epoch=None):
        """Display the current parameter value."""
        if is_verbose:
            if epoch is None:
                print(
                    "Adjusting parameter value"
                    " of group {} to {:.4e}.".format(group, value)
                )
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    "Epoch {}: adjusting parameter value"
                    " of group {} to {:.4e}.".format(epoch_str, group, value)
                )

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the paramter schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        class _enable_get_value_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_value_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_value_called_within_step = False

        with _enable_get_value_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_value()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_value"):
                    values = self._get_closed_form_value()
                else:
                    values = self.get_value()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, value = data
            param_group[self.parameter] = value
            self.print_value(self.verbose, i, value, epoch)

        self._last_value = [
            group[self.parameter] for group in self.optimizer.param_groups
        ]


class LinearScheduler(Scheduler):
    """Decays the value of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial value as value.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        parameter (str): Parameter name.
        start_factor (float): The number we multiply the parameter in the first epoch.
            The multiplication factor changes towards end_factor in the following epochs.
            Default: 1.0.
        end_factor (float): The number we multiply the parameter the end of linear changing
            process. Default: 0.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to `end_factor`.
            Default: 16.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses param = 0.05 for all groups
        >>> # param = 0.025    if epoch == 0
        >>> # param = 0.03125  if epoch == 1
        >>> # param = 0.0375   if epoch == 2
        >>> # param = 0.04375  if epoch == 3
        >>> # param = 0.05    if epoch >= 4
        >>> # xdoctest: +SKIP
        >>> scheduler = LinearScheduler(self.opt, "param", start_factor=0.5, end_factor=1.0, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        parameter,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=16,
        last_epoch=-1,
        verbose=False,
    ):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                "Starting multiplicative factor expected to be between 0 and 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super(LinearScheduler, self).__init__(optimizer, parameter, last_epoch, verbose)

    def get_value(self):
        if not self._get_value_called_within_step:
            warnings.warn(
                "To get the last parameter value computed by the scheduler, "
                "please use `get_last_value()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [
                group[self.parameter] * self.start_factor
                for group in self.optimizer.param_groups
            ]

        if self.last_epoch > self.total_iters:
            return [group[self.parameter] for group in self.optimizer.param_groups]

        return [
            group[self.parameter]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_value(self):
        return [
            base_value
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_value in self.base_values
        ]
