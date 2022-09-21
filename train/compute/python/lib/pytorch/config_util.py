import copy
import enum
import os
import platform
import random
import socket
from typing import Any, Dict, List

import torch
from torch.utils.collect_env import get_nvidia_driver_version, run as run_cmd

from ...lib import __version__


@enum.unique
class ExecutionPass(enum.Enum):
    # Forward pass will always run (also required for backward pass).
    FORWARD = "forward"

    # Run backward pass in addition to forward pass.
    BACKWARD = "backward"


@enum.unique
class OpExecutionMode(enum.Enum):
    # Run operator seprately and clear cache between each call.
    DISCRETE = "discrete"

    # Run operator back to back without clear cache, etc.
    CONTINUOUS = "continuous"

    # Run operator back to back but record indivisual events.
    CONTINUOUS_EVENTS = "continuous_events"


def get_op_run_id(op_name: str, run_id: str) -> str:
    return f"{op_name}:{run_id}"


def get_benchmark_options() -> Dict[str, Any]:
    options = {
        "device": "cpu",
        "pass_type": ExecutionPass.FORWARD,
        "warmup": 1,
        "iteration": 1,
        "op_exec_mode": OpExecutionMode.DISCRETE,
        "cuda_l2_cache": False,
        "time_unit": "millisecond",
        "out_file_prefix": None,
        "out_stream": None,
        "run_ncu": False,
        "ncu_bin": "/usr/local/NVIDIA-Nsight-Compute-2021.2/ncu",
        "ncu_args": "",
        "ncu_warmup": 5,
        "ncu_iteration": 1,
        "run_nsys": False,
        "nsys_bin": "/opt/nvidia/nsight-systems/2021.4.1/bin/nsys",
        "nsys_args": "",
        "nsys_warmup": 5,
        "nsys_iteration": 10,
        "run_batch_size": 50,
        "batch_cuda_device": 1,
        "batch_cmd": "python -m param_bench.train.compute.python.pytorch.run_batch",
        "resume_op_run_id": None,
        "stop_op_run_id": None,
    }

    return options


def create_bench_config(name: str) -> Dict[str, Any]:
    return {name: create_op_info()}


def create_op_info() -> Dict[str, Any]:
    return {
        "build_iterator": None,
        "input_iterator": None,
        "build_data_generator": None,
        "input_data_generator": "PyTorch:DefaultDataGenerator",
        "config": [{"build": [], "input": []}],
    }


def create_op_args(args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {"args": args, "kwargs": kwargs}


_pytorch_type: Dict[str, Any] = {
    "int": {"type": "int", "value": None},
    "int_range": {"type": "int", "value_range": None},
    "long": {"type": "long", "value": None},
    "long_range": {"type": "long", "value_range": None},
    "float": {"type": "float", "value": None},
    "float_range": {"type": "float", "value_range": None},
    "double": {"type": "double", "value": None},
    "double_range": {"type": "double", "value_range": None},
    "bool": {"type": "bool", "value": None},
    "device": {"type": "device", "value": None},
    "str": {"type": "str", "value": None},
    "genericlist": {"type": "genericlist", "value": None},
    "tuple": {"type": "tuple", "value": None},
    "tensor": {"type": "tensor", "dtype": "float", "shape": None},
}


def create_type(type) -> Dict[str, Any]:
    return copy.deepcopy(_pytorch_type[type])


def get_sys_info():
    cuda_available = torch.cuda.is_available()
    cuda_info = {}
    if cuda_available:
        cuda_device_id = torch.cuda.current_device()
        cuda_device_property = torch.cuda.get_device_properties(cuda_device_id)
        cuda_info = {
            "cuda": torch.version.cuda,
            "cuda_device_driver": get_nvidia_driver_version(run_cmd),
            "cuda_gencode": torch.cuda.get_gencode_flags(),
            "cuda_device_id": cuda_device_id,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_property": cuda_device_property,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
        }

    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python_version": platform.python_version(),
        "param_train_compute_version": __version__,
        "cuda_available": cuda_available,
        **cuda_info,
        "pytorch_version": torch.__version__,
        "pytorch_debug_build": torch.version.debug,
        "pytorch_build_config": torch._C._show_config(),
    }


def init_pytorch(run_options: Dict[str, Any]):
    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    # Fix random number generator seeds.
    torch.manual_seed(0)
    random.seed(0)
