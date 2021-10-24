import logging

from .lib.init_helper import init_logging, load_modules

# Initialize logging format before loading all other modules
init_logging()

import argparse
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import torch
from caffe2.python import core
from torch.autograd.profiler import record_function

from .lib import pytorch as lib_pytorch
from .lib.config import BenchmarkConfig
from .lib.pytorch.benchmark import run_op
from .workloads import pytorch as workloads_pytorch


def main():
    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument("--config", type=str, required=True, help="The op config file.")
    parser.add_argument("--warmup", type=int, default=5, help="number of iterations.")
    parser.add_argument("--iter", type=int, default=1, help="number of iterations.")
    parser.add_argument(
        "--metric", action="store_true", help="The metric collection mode."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device of execution."
    )
    parser.add_argument(
        "--out-file-name",
        type=str,
        default="op_bench_log.json",
        help="json file to write log info.",
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    bench_config = BenchmarkConfig(args.device)
    bench_config.load_json_file(args.config)

    out_file_name = args.out_file_name
    if args.metric:
        out_file_name = args.out_file_name + ".metric_config"

    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    with open(out_file_name, "w") as out_file:
        for op_config in bench_config.op_configs:
            run_op(
                op_config,
                args.warmup,
                args.iter,
                args.device,
                out_file,
                args.metric,
            )
        logging.info(f"Log written to {args.out_file_name}")


if __name__ == "__main__":
    main()
