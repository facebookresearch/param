from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    annotations,
)

import argparse, json, sys
import logging
import random

from enum import Enum
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import torch
from caffe2.python import core
from torch.autograd.profiler import record_function
from param.lib.timer import Timer
from param.lib.init_helper import init_logging, load_modules
from param.lib.config import BenchmarkConfig, OperatorConfig
from param.lib.operator import op_map
from param.lib.benchmark import run_op

import param.workloads


def main():
    init_logging()
    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument("--config", type=str, required=True, help="The op config file.")
    parser.add_argument(
        "--range", action="store_true", help="The config file has config range."
    )
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
    parser.add_argument(
        "--execution-graph",
        help="enable execution graph observer (high perfermance overhead, not for benchmarking)",
        action="store_true",
    )

    args = parser.parse_args()

    if args.execution_graph:
        core.GlobalInit(
            [
                "python",
                "--pytorch_enable_execution_graph_observer=true",
                "--pytorch_execution_graph_observer_iter_label=## BENCHMARK ##",
            ]
        )

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    load_modules(param.workloads)

    bench_configs = BenchmarkConfig(args.config, args.device)

    out_file_name = args.out_file_name
    if args.metric:
        out_file_name = args.out_file_name + ".metric_config"

    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    with open(out_file_name, "w") as out_file:
        with record_function("## BENCHMARK ##"):
            for op_config in bench_configs.op_configs:
                print(op_config.name)
                run_op(
                    op_config,
                    args.warmup,
                    args.iter,
                    args.device,
                    out_file,
                    args.metric,
                )
        # TODO lofe: repeating the record_function for execution graph only.
        with record_function("## BENCHMARK ##"):
            logging.info(f"Log written to {args.out_file_name}")


if __name__ == "__main__":
    main()
