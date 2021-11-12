from typing import Callable

import torch

from ..operator import OperatorInterface


class InPlaceOpByName(OperatorInterface):
    """
    Inplace ops is called in the form of tensor.op(args), we convert it
    to a regular function call with "getattr(tensor, op)(args)"
    """

    def __init__(
        self,
        func_name: str,
    ):
        super(InPlaceOpByName, self).__init__()
        self.func_name: str = func_name
        self.fwd_out: torch.tensor = None
        self.grad_in: torch.tensor = None

    def forward(self, *args, **kwargs):
        # The first arg is assume to be the inplace value, pass on the rest of
        # the args to the callable.
        self.fwd_out = getattr(args[0], self.func_name)(*args[1:], **kwargs)

    def create_grad(self):
        self.grad_in = torch.ones_like(self.fwd_out, device=torch.device(self.device))

    def backward(self):
        self.fwd_out.backward(self.grad_in)


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

    def forward(self, *args, **kwargs):
        self.fwd_out = self.func(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        self.grad_in = torch.ones_like(self.fwd_out, device=torch.device(self.device))

    def backward(self):
        self.fwd_out.backward(self.grad_in)
