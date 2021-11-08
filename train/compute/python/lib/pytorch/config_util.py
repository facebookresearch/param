import copy
from typing import Any
from typing import Dict


def create_operator_config(name: str):
    op_config = {
        name: {
            "build_iterator": None,
            "input_iterator": None,
            "build_data_generator": None,
            "input_data_generator": "PyTorch::DefaultDataGenerator",
            "config": [{"build": [], "input": [{"args": [], "kwargs": {}}]}],
        }
    }
    return op_config


_pytorch_data: Dict[str, Any] = {
    "int": {"type": "int", "value": None},
    "int_range": {"type": "int", "value_range": None},
    "long": {"type": "long", "value": None},
    "long_range": {"type": "long", "value_range": None},
    "float": {"type": "float", "value": 1.2},
    "float_range": {"type": "float", "value_range": None},
    "double": {"type": "double", "value": 3.4},
    "double_range": {"type": "double", "value_range": None},
    "bool": {"type": "bool", "value": None},
    "device": {"type": "device", "value": "cpu"},
    "str": {"type": "str", "value": "a string value"},
    "genericlist": {"type": "genericlist", "value": None},
    "tuple": {"type": "tuple", "value": None},
    "tensor": {"type": "tensor", "dtype": "float", "shape": None},
}


def create_data(type):
    return copy.deepcopy(_pytorch_data[type])
