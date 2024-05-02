from typing import Callable, List

from ..init_helper import get_logger

logger = get_logger()

import re

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

    def forward(self, *args, **kwargs):
        self.fwd_out = self.func(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        if not self.fwd_out.is_leaf:
            self.grad_in = torch.ones_like(self.fwd_out)
        else:
            logger.debug(
                f"{self.constructor.__name__}: skipping create_grad() due to forward result is leaf tensor."
            )

    def backward(self):
        if not self.fwd_out.is_leaf:
            self.fwd_out.backward(self.grad_in)
        else:
            logger.debug(
                f"{self.constructor.__name__}: skipping backward() due to forward result is leaf tensor."
            )


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

    def forward(self, *args, **kwargs):
        self.fwd_out = self.func(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        if not self.fwd_out.is_leaf:
            self.grad_in = torch.ones_like(self.fwd_out)
        else:
            logger.debug(
                f"{self.constructor.__name__}: skipping create_grad() due to forward result is leaf tensor."
            )

    def backward(self):
        if not self.fwd_out.is_leaf:
            self.fwd_out.backward(self.grad_in)
        else:
            logger.debug(
                f"{self.constructor.__name__}: skipping backward() due to forward result is leaf tensor."
            )


class TorchScriptOp(OperatorInterface):
    """
    TorchScriptOp generates a graph IR that runs a specific PyTorch function in
    the SSA form.
    """

    def __init__(
        self,
        func_name: str,
    ):
        super(TorchScriptOp, self).__init__()
        self.func_name: str = func_name
        self.func: Callable = None
        self.fwd_out: torch.tensor = None
        self.grad_in: torch.tensor = None

    def build(self, op_schema: str):
        """
        Because TorchScript is in SSA form, we expect at least one element in
        types for the output. An example is:
        ```
        graph(%0 : Tensor,
              %1 : Tensor,
              %2 : int):
          %3 : Tensor = aten::add(%0, %1, %2)
          return (%3)
        ```
        """

        def _extract_types(types_str: str):
            # split into args in the form of "type_name var_name".
            types = [item for item in types_str.split(",")]
            # separate betwen type and var name, keep types, skip *.
            types = [item.strip().split(" ")[0] for item in types if "*" not in item]
            # remove list length, e.g. int[2] -> int[].
            types = [re.sub(r"\[[0-9]\]", "[]", t) for t in types]
            # remove elem type for tensors, e.g. Tensor(float) -> Tensor
            var_types = [item if "Tensor" not in item else "Tensor" for item in types]
            return var_types

        assert (
            op_schema
        ), f"TorchScriptOp {self.func_name} should have at non-empty op schema."

        func_name, func_signature = op_schema.split("(", 1)
        arg_str, output_str = func_signature.split("->", 1)
        arg_str = arg_str.strip("() ")
        output_str = output_str.strip("() ")
        arg_types = _extract_types(arg_str)
        output_types = _extract_types(output_str)

        graph_args = []
        func_args = []

        func_schema = torch._C.parse_schema(op_schema)
        register_id = 0
        for data_type in arg_types:
            graph_args.append(f"%{register_id} : {data_type}")
            func_args.append(f"%{register_id}")
            register_id += 1

        func_outputs = []
        func_output_vars = []
        func_output_types = []
        for data_type in output_types:
            func_outputs.append(f"%{register_id} : {data_type}")
            func_output_vars.append(f"%{register_id}")
            func_output_types.append(data_type)
            output_var = f"%{register_id}"
            register_id += 1
        return_construct = ""
        if len(func_outputs) > 1:
            return_construct = f"%{register_id}: ({','.join(func_output_types)}) = prim::TupleConstruct({','.join(func_output_vars)})"
            output_var = f"%{register_id}"
        actual_func_name = func_schema.name

        ts_ir = f"""
            graph({",".join(graph_args)}):
                {",".join(func_outputs)} = {actual_func_name}({",".join(func_args)})
                {return_construct}
                return ({output_var})
        """
        ts_graph = torch._C.parse_ir(ts_ir)
        logger.debug(f"{self.func_name} TorchScript IR Graph: \n{ts_graph}")
        cu = torch._C.CompilationUnit()
        self.func = cu.create_function(self.func_name, ts_graph)

    def cleanup(self):
        self.fwd_out = None
        self.grad_in = None

    def forward(self, *args, **kwargs):
        self.fwd_out = self.func(*args, **kwargs)
        return self.fwd_out

    def create_grad(self):
        if not self.fwd_out.is_leaf:
            self.grad_in = torch.ones_like(self.fwd_out)
        else:
            logger.debug(
                f"{self.func_name}: skipping create_grad() due to forward result is leaf tensor."
            )

    def backward(self):
        if not self.fwd_out.is_leaf:
            self.fwd_out.backward(self.grad_in)
        else:
            logger.debug(
                f"{self.func_name}: skipping backward() due to forward result is leaf tensor."
            )
