import abc
from typing import Dict, Set, Tuple, List, Any, Callable, Iterable, Type


class OperatorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            or NotImplemented
        )

    def __init__(self):
        pass

    # Construct and initialize the operator.
    def build(self, *args, **kwargs):
        pass

    # Reset any state and remove allocated resources.
    def cleanup(self):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Inplace ops is called in the form of tensor.op(args), we convert it
# to a regular function call with "getattr(tensor, op)(args)"
class InPlaceOpByName(OperatorInterface):
    def __init__(
        self,
        func_name: str,
    ):
        super(InPlaceOpByName, self).__init__()
        self.func_name: str = func_name

    def forward(self, *args, **kwargs):
        # The first arg is assume to be the inplace value, pass on the rest of
        # the args to the callable.
        getattr(args[0], self.func_name)(*args[1:], **kwargs)


# Callable ops are ops can be called in the form of op(*args, **kwargs)
class CallableOp(OperatorInterface):
    def __init__(
        self,
        func: Callable,
    ):
        super(CallableOp, self).__init__()
        self.func: Callable = func

    def forward(self, *args, **kwargs):
        self.func(*args, **kwargs)


def register_operator(name: str, operator_class: Type[OperatorInterface]):
    global op_map
    if name not in op_map:
        op_map[name] = operator_class
    else:
        raise ValueError(f'Duplicate operator registration name: "{name}"')


def register_operators(op_dict: Dict[str, Type[OperatorInterface]]):
    global op_map
    for name, operator_class in op_dict.items():
        if name not in op_map:
            op_map[name] = operator_class
        else:
            raise ValueError(f'Duplicate operator registration name: "{name}"')


# Global operator registry, a mapping of name to operator object
op_map: Dict[str, Type[OperatorInterface]] = {}
