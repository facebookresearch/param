import gc
from typing import Callable

from ..init_helper import get_logger

logger = get_logger()

import torch

from ..operator import OperatorInterface


class UnaryOp(OperatorInterface):
    """
    UnaryOp is called in the form of tensor_obj.op(args), we convert it
    to a regular function call with "getattr(tensor_obj, op)(args)". So the
    first arg is assumed to be the `tensor_obj`.
    """

    def __init__(
        self,
        func_name: str,
    ):
        super(UnaryOp, self).__init__()
        self.func_name: str = func_name
        self.fwd_out: torch.tensor = None
        self.grad_in: torch.tensor = None

    def forward(self, *args, **kwargs):
        # The first arg is assume to be the inplace value, pass on the rest of
        # the args to the callable.
        # Unary op also does not support backward() because they are in-place.
        with torch.no_grad():
            getattr(args[0], self.func_name)(*args[1:], **kwargs)

    def create_grad(self):
        pass

    def backward(self):
        pass


class CallableOp(OperatorInterface):
    """
    Callable ops are ops can be called in the form of op(*args, **kwargs)
    """

    def __init__(
        self,
        func: Callable,
    ):
        super(CallableOp, self).__init__()
        self.func: Callable = func
        self.fwd_out: torch.tensor = None
        self.grad_in = None

    def cleanup(self):
        self.fwd_out = None
        self.grad_in = None
        gc.collect()

    def forward(self, *args, **kwargs):
        self.fwd_out = self.func(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        if not self.fwd_out.is_leaf:
            self.grad_in = torch.ones_like(self.fwd_out)
        else:
            logger.debug(f"{self.constructor.__name__}: skipping create_grad() due to forward result is leaf tensor.")

    def backward(self):
        if not self.fwd_out.is_leaf:
            self.fwd_out.backward(self.grad_in)
        else:
            logger.debug(f"{self.constructor.__name__}: skipping backward() due to forward result is leaf tensor.")


class BuildableOp(OperatorInterface):
    """
    BuildableOp are ops needs to be constructed first, before running with inputs.
    """

    def __init__(
        self,
        constructor: Callable,
    ):
        super(BuildableOp, self).__init__()
        self.constructor: Callable = constructor
        self.func: Callable = None
        self.fwd_out: torch.tensor = None
        self.grad_in = None

    # Construct and initialize the operator.
    def build(self, *args, **kwargs):
        # Use `to` to make sure weights are on device.
        self.func = self.constructor(*args, **kwargs).to(torch.device(self.device))

    def cleanup(self):
        self.fwd_out = None
        self.grad_in = None
        gc.collect()

    def forward(self, *args, **kwargs):
        self.fwd_out = self.func(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        if not self.fwd_out.is_leaf:
            self.grad_in = torch.ones_like(self.fwd_out)
        else:
            logger.debug(f"{self.constructor.__name__}: skipping create_grad() due to forward result is leaf tensor.")

    def backward(self):
        if not self.fwd_out.is_leaf:
            self.fwd_out.backward(self.grad_in)
        else:
            logger.debug(f"{self.constructor.__name__}: skipping backward() due to forward result is leaf tensor.")
