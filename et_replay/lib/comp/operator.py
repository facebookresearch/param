import abc
from typing import Dict, Type

from .init_helper import get_logger

logger = get_logger()


class OperatorInterface(metaclass=abc.ABCMeta):
    """
    The OperatorInterface assumes the following operations:

    - An operator may require a build/initialization step.
    - Forward operation is always required.
    - Backward may require a gradient input, and create_grad should not be part
      of the benchmark measurement.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "forward")
            and callable(subclass.forward)
            or NotImplemented
        )

    def __init__(self):
        self.device = None

    # Construct and initialize the operator.
    def build(self, *args, **kwargs):
        pass

    # Reset any state and remove allocated resources.
    def cleanup(self):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def create_grad(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


def register_operator(name: str, operator_class: Type[OperatorInterface]):
    global op_map
    logger.debug(f"register op: {name}")
    if name not in op_map:
        op_map[name] = operator_class
    else:
        raise ValueError(f"Duplicate operator registration name: {name}")


def register_operators(op_dict: Dict[str, Type[OperatorInterface]]):
    global op_map
    for name, operator_class in op_dict.items():
        logger.debug(f"register op: {name}")
        if name not in op_map:
            op_map[name] = operator_class
        else:
            raise ValueError(f"Duplicate operator registration name: {name}")


# Global operator registry, a mapping of name to operator object
op_map: Dict[str, Type[OperatorInterface]] = {}
