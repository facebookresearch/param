from typing import Dict

import torch

from ...lib.operator import (
    OperatorInterface,
    register_operator,
    register_operators,
)
from ...lib.pytorch.operator_impl import CallableOp, InPlaceOpByName


# Unary
unary_ops: Dict[str, OperatorInterface] = {
    "torch.add_": InPlaceOpByName("add_"),
    "torch.clamp_": InPlaceOpByName("clamp_"),
}
register_operators(unary_ops)

callable_ops: Dict[str, OperatorInterface] = {
    "torch.add": CallableOp(torch.add),
    "torch.baddbmm": CallableOp(torch.baddbmm),
    "torch.bmm": CallableOp(torch.bmm),
    "torch.cat": CallableOp(torch.cat),
    "torch.matmul": CallableOp(torch.matmul),
    "torch.mean": CallableOp(torch.mean),
    "torch.mm": CallableOp(torch.mm),
    "torch.mul": CallableOp(torch.mul),
    "torch.nn.functional.relu": CallableOp(torch.nn.functional.relu),
    "torch.reshape": CallableOp(torch.reshape),
}
register_operators(callable_ops)
