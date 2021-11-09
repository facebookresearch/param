import logging

from .lib.init_helper import init_logging, load_modules

# Initialize logging format before loading all other modules
logger = init_logging(logging.INFO)

import argparse

import torch

from .lib import pytorch as lib_pytorch
from .lib.config import BenchmarkConfig
from .lib.pytorch.benchmark import run_op, ExecutionPass
from .workloads import pytorch as workloads_pytorch


def main():
    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="The benchmark config file."
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=5, help="Number of warm up iterations."
    )
    parser.add_argument(
        "-i", "--iteration", type=int, default=1, help="Number of benchmark iterations."
    )
    parser.add_argument(
        "-b", "--backward", action="store_true", help="Include backward pass."
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="Target device for benchmark."
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        default="benchmark_result",
        help="File name prefix to write benchmark results.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase log output verbosity."
    )

    args = parser.parse_args()

    if args.verbose:
        init_logging(logging.DEBUG)

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    bench_config = BenchmarkConfig(args.device)
    bench_config.load_json_file(args.config)

    out_file_name = f"{args.output_prefix}.json"

    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    if args.backward:
        pass_type = ExecutionPass.BACKWARD
        logger.info(f"Pass: FORWARD and BACKWARD")
    else:
        pass_type = ExecutionPass.FORWARD
        logger.info(f"Pass: FORWARD")

    with open(out_file_name, "w") as out_file:
        for op_config in bench_config.op_configs:
            run_op(
                op_config,
                args.warmup,
                args.iteration,
                args.device,
                pass_type,
                out_file,
            )
        logger.info(f"Log written to {out_file_name}")


if __name__ == "__main__":
    main()
