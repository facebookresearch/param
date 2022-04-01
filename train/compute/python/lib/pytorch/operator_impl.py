from typing import Callable, List

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

    def build(self, types: List[str]):
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
        assert len(types) > 0, f"TorchScriptOp {self.func_name} should have at least one type definition for output."
        var_id = 0
        graph_args = []
        func_args = []
        input_types = types[:-1]
        output_type = types[-1]
        for var_type in input_types:
            graph_args.append(f"%{var_id} : {var_type}")
            func_args.append(f"%{var_id}")
            var_id += 1

        output_var = f"%{var_id}"

        ts_ir = f"""
            graph({",".join(graph_args)}):
                {output_var} : {output_type} = {self.func_name}({",".join(func_args)})
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
            logger.debug(f"{self.func_name}: skipping create_grad() due to forward result is leaf tensor.")

    def backward(self):
        if not self.fwd_out.is_leaf:
            self.fwd_out.backward(self.grad_in)
        else:
            logger.debug(f"{self.func_name}: skipping backward() due to forward result is leaf tensor.")