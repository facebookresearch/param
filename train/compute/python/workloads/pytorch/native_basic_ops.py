from typing import Dict

import torch

from ...lib.operator import OperatorInterface, register_operators
from ...lib.pytorch.operator_impl import BuildableOp, CallableOp, TorchScriptOp, UnaryOp


# Unary
unary_ops: Dict[str, OperatorInterface] = {
    "torch.add_": UnaryOp("add_"),
    "torch.clamp_": UnaryOp("clamp_"),
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


buildable_ops: Dict[str, OperatorInterface] = {
    "torch.nn.AdaptiveAvgPool2d": BuildableOp(torch.nn.AdaptiveAvgPool2d),
    "torch.nn.Conv2d": BuildableOp(torch.nn.Conv2d),
    "torch.nn.Dropout": BuildableOp(torch.nn.Dropout),
    "torch.nn.MaxPool2d": BuildableOp(torch.nn.MaxPool2d),
    "torch.nn.ReLU": BuildableOp(torch.nn.ReLU),
    "torch.nn.Linear": BuildableOp(torch.nn.Linear),
}
register_operators(buildable_ops)


# Operator schema: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml
torchscript_ops: Dict[str, OperatorInterface] = {
    "aten::add": TorchScriptOp("aten::add"),
    "aten::add_": TorchScriptOp("aten::add_"),
    "aten::matmul": TorchScriptOp("aten::matmul"),
    "aten::mul": TorchScriptOp("aten::mul"),
    "aten::sum": TorchScriptOp("aten::sum"),
    "aten::linear": TorchScriptOp("aten::linear"),
}
register_operators(torchscript_ops)
