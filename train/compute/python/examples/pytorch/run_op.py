from ...lib.init_helper import init_logging, load_modules

# Initialize logging format before loading all other modules
init_logging()

import io
import json

import torch

from ...lib import pytorch as lib_pytorch
from ...lib.config import BenchmarkConfig
from ...lib.pytorch.benchmark import run_op, ExecutionPass
from ...lib.pytorch.config_util import create_operator_config, create_data
from ...workloads import pytorch as workloads_pytorch


def main():

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    op_name = "torch.mm"
    op_config = create_operator_config(op_name)
    tensor_1 = create_data("tensor")
    tensor_1["shape"] = [128, 128]
    tensor_2 = create_data("tensor")
    tensor_2["shape"] = [128, 128]

    # Add the two tensors as first and second positional args for the operator.
    op_config[op_name]["config"][0]["input"][0]["args"] = [tensor_1, tensor_2]
    print(op_config)

    device = "cuda"
    # Set the target device where this benchmark will run
    bench_config = BenchmarkConfig(device)

    # Load and initialize the operator configuration.
    bench_config.load(op_config)

    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    # By default, benchmark will run the forward pass.
    # By setting backward (which requires running forward pass), the benchmark
    # will run both forward and backward pass.
    pass_type = ExecutionPass.BACKWARD

    # Store result in this in-memory string buffer
    out_stream = io.StringIO()

    # Iterate through the operator configs, and run each operator benchmark.
    for config in bench_config.op_configs:
        run_op(
            config,
            1,
            3,
            device,
            pass_type,
            out_stream,
        )

    # Parse the benchmark result string in JSON format and print out.
    print("### Benchmark Results ###")
    for line in out_stream.getvalue().splitlines():
        result = json.loads(line)
        print(result)


if __name__ == "__main__":
    main()
