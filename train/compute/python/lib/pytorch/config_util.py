import copy
import enum
import platform
import socket
from datetime import datetime
from importlib.metadata import version as package_version
from typing import Any
from typing import Dict
from typing import List

import torch
from torch.utils.collect_env import get_nvidia_driver_version
from torch.utils.collect_env import run as run_cmd


@enum.unique
class ExecutionPass(enum.Enum):
    # Forward pass will always run (also required for backward pass).
    FORWARD = "forward"
    # Run backward pass in addition to forward pass.
    BACKWARD = "backward"


def get_op_run_id(op_name: str, run_id: str) -> str:
    return f"{op_name}:{run_id}"


def get_benchmark_options() -> Dict[str, Any]:
    options = {
        "device": "cpu",
        "pass_type": ExecutionPass.FORWARD,
        "warmup": 1,
        "iteration": 1,
        "out_file_prefix": None,
        "out_stream": None,
        "run_ncu": False,
        "ncu_args": "",
        "ncu_batch": 50,
        "resume_op_run_id": None,
    }

    return options


def create_bench_config(name: str) -> Dict[str, Any]:
    return {name: create_op_info()}


def create_op_info() -> Dict[str, Any]:
    return {
        "build_iterator": None,
        "input_iterator": None,
        "build_data_generator": None,
        "input_data_generator": "PyTorch::DefaultDataGenerator",
        "config": [{"build": [], "input": []}],
    }


def create_op_args(args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {"args": args, "kwargs": kwargs}


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


def create_data(type) -> Dict[str, Any]:
    return copy.deepcopy(_pytorch_data[type])


def get_sys_info():
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "param_bench_version": package_version("parambench_train_compute"),
        "pytorch_version": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cuda_device_driver": get_nvidia_driver_version(run_cmd),
        "cuda": torch.version.cuda,
        "cuda_gencode": torch.cuda.get_gencode_flags(),
        "cuda_device_id": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(),
        "python_version": platform.python_version(),
        "pytorch_debug_build": torch.version.debug,
        "pytorch_build_config": torch._C._show_config(),
    }
