import abc
from typing import Dict, Set, Tuple, List, Any, Callable, Iterable, Type


class OperatorInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "__call__")
            and callable(subclass.__call__)
            or NotImplemented
        )

    def __init__(self):
        self.build_iterator = None
        self.input_iterator = None
        self.build_data_generator = None
        self.input_data_generator = None

    def build(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def set_iterator(self, build_iterator, input_iterator):
        self.build_iterator = build_iterator
        self.input_iterator = input_iterator

    def set_data_generator(self, build_data_generator, input_data_generator):
        self.build_data_generator = build_data_generator
        self.input_data_generator = input_data_generator

    def get_build_iterator(self):
        return self.build_iterator

    def get_input_iterator(self):
        return self.input_iterator

    def get_build_data_generator(self):
        return self.build_data_generator

    def get_input_data_generator(self):
        return self.input_data_generator

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
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

    def __call__(self, *args, **kwargs):
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

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)


def register_operator(name: str, operator: OperatorInterface):
    if name not in operator_map:
        operator_map[name] = operator
    else:
        raise ValueError(f'Duplicate operator registration name: "{name}"')


def register_operators(op_dict: Dict[str, OperatorInterface]):
    for name, operator in op_dict.items():
        if name not in operator_map:
            operator_map[name] = operator
        else:
            raise ValueError(f'Duplicate operator registration name: "{name}"')


# Global operator registry, a mapping of name to operator object
operator_map: Dict[str, OperatorInterface] = {}
