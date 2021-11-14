import copy
import enum
from typing import Any
from typing import Dict


@enum.unique
class ExecutionPass(enum.Enum):
    # Forward pass will always run (also required for backward pass).
    FORWARD = "forward"
    # Run backward pass in addition to forward pass.
    BACKWARD = "backward"


def get_benchmark_options():
    options = {
        "device": "cpu",
        "pass_type": ExecutionPass.FORWARD,
        "warmup": 1,
        "iteration": 1,
        "out_stream": None,
        "resume_op_run_id": None,
    }

    return options


def create_bench_config(name: str):
    bench_config = {
        name: {
            "build_iterator": None,
            "input_iterator": None,
            "build_data_generator": None,
            "input_data_generator": "PyTorch::DefaultDataGenerator",
            "config": [{"build": [], "input": [{"args": [], "kwargs": {}}]}],
        }
    }
    return bench_config


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
