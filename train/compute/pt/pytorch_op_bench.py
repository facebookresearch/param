from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    annotations,
)

import argparse, json, sys
import importlib
import logging
import pkgutil
import random
import time
from enum import Enum
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import pydot
import torch
from caffe2.python import core
from .pytorch_op_config import ConfigIterator, config_iterator_map, DummyConfigIterator
from .pytorch_op_def import OperatorConfig
from .pytorch_op_interface import OperatorInterface, operator_map
from .pytorch_op_util import DefaultDataGenerator
from torch.autograd.profiler import record_function

FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

import cea.ml_perf_model.gpu.microbench.pytorch.benchmark as benchmark


def load_benchmarks(package):
    # See https://packaging.python.org/guides/creating-and-discovering-plugins/
    benchmark_modules = pkgutil.iter_modules(package.__path__, package.__name__ + ".")
    for _, name, _ in benchmark_modules:
        logging.debug(f"Loading benchmark module: {name}")
        importlib.import_module(name)


# Timer in seconds
class Timer:
    def __init__(self, device: str):
        self.device: str = device
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.start_time = 0
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cpu":
            self.end_time = time.perf_counter()
        else:
            self.end_event.record()
            torch.cuda.synchronize()
            self.end_time = self.start_event.elapsed_time(self.end_event) * 1.0e-3

    # returns time in seconds
    def elapsed_time(self):
        return self.end_time - self.start_time


def benchmark_op(
    op_id: str, op: Callable, args: Any, kwargs: Any, device: str, num_iter: int
):
    time_records = []
    for _ in range(num_iter):
        # flush cache
        if device.startswith("cuda"):
            _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
            torch.cuda.empty_cache()

        with Timer(device) as timer:
            op(*args, **kwargs)
        time_records.append(timer.elapsed_time())
    return time_records


def collect_metric(
    op_name: str,
    id: str,
    op: Callable,
    args: Any,
    kwargs: Any,
    device: str,
    num_iter: int,
    config: Dict[str, Any],
    out_file: TextIO,
):
    if device.startswith("cuda"):
        # use nvtx allows us to collect only this part of kernel executions
        # and match op and arg variants to metrics.
        logging.info(f"Running {op_name}[{id}] for {num_iter} CUDA metric iterations")
        torch.cuda.nvtx.range_push("op_bench")
        for _ in range(num_iter):
            # flush cache
            _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
            torch.cuda.empty_cache()

            torch.cuda.nvtx.range_push(f"{op_name}[{id}]")
            op(*args, **kwargs)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        stats = {"name": op_name, "id": id, "iter": num_iter, "config": config}
        out_file.write(json.dumps(stats) + "\n")
        out_file.flush()
    else:
        raise Exception("Non-GPU metric mode is not supported.")


def warmup(
    op_name: str,
    id: str,
    op: Callable,
    args: Any,
    kwargs: Any,
    device: str,
    num_iter: int,
):
    logging.debug(f"Running {op_name}[{id}] for {num_iter} warm up iterations")
    # warm up
    time_records = benchmark_op(f"{op_name}[{id}]", op, args, kwargs, device, num_iter)
    logging.info(f"  warmup: {time_records}")


def measure_latency(
    op_name: str,
    id: str,
    op: Callable,
    args: Any,
    kwargs: Any,
    device: str,
    num_iter: int,
    config: Dict[str, Any],
    out_file: TextIO,
):
    logging.debug(f"Running {op_name}[{id}] for {num_iter} measured iterations")
    torch.cuda.nvtx.range_push("op_bench")
    time_records = benchmark_op(f"{op_name}[{id}]", op, args, kwargs, device, num_iter)
    torch.cuda.nvtx.range_pop()
    tot = sum(time_records)
    logging.info(f"  rec: {time_records}")
    logging.info(f"  avg: {tot/num_iter:.6f} sec")
    logging.info(f"  tot: {tot:.6f} sec")
    stats = {
        "name": op_name,
        "id": id,
        "time": time_records,
        "iter": num_iter,
        "config": config,
    }
    out_file.write(json.dumps(stats) + "\n")
    out_file.flush()


def run_op(
    op_name: str,
    op: OperatorInterface,
    configs: List[Dict[str, Any]],
    warmup_iter: int,
    num_iter: int,
    device: str,
    out_file: TextIO,
    metric_mode: bool,
):
    config_id = 0
    for config in configs:
        if "input" not in config:
            logging.error(f"{op_name} has no input configureations defined, skipped.")
            return

        # build op
        build_config = []
        build_iterator = op.get_build_iterator()
        input_iterator = op.get_input_iterator()
        if build_iterator:
            if "build" not in config:
                logging.error(
                    f"{op_name} has build iterator, but no build configureations defined, skipped."
                )
                return
            generate_build_config: ConfigIterator = build_iterator(
                config, "build", device
            )
        else:
            generate_build_config: ConfigIterator = DummyConfigIterator(
                config, "build", device
            )

        op_config = {}
        for (build_id, build_config) in generate_build_config:
            logging.info(f"{config_id}:{build_id} {build_config}")
            build_data_gen = op.get_build_data_generator()()
            (build_args, build_kwargs) = build_data_gen.get_data(build_config, device)
            logging.info(f"{build_args} {build_kwargs}")
            # reset operator to clear memory before new build
            op.reset()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            op.build(*build_args, **build_kwargs)
            op_config["build"] = build_config
            op_config["input"] = config["input"]
            generate_input_config: ConfigIterator = input_iterator(
                op_config, "input", device
            )
            for (input_id, input_config) in generate_input_config:
                logging.info(f"{op_name}[{config_id}:{build_id}:{input_id}]:")
                logging.info(f"  {input_config}")
                # generate data

                input_data_gen = op.get_input_data_generator()()
                (input_args, input_kwargs) = input_data_gen.get_data(
                    input_config, device
                )
                id = f"{config_id}:{build_id}:{input_id}"
                warmup(op_name, id, op, input_args, input_kwargs, device, warmup_iter)

                final_config = {"build": build_config, "input": input_config}

                # collect CUDA metrics
                if metric_mode:
                    collect_metric(
                        op_name,
                        id,
                        op,
                        input_args,
                        input_kwargs,
                        device,
                        num_iter,
                        final_config,
                        out_file,
                    )
                else:
                    measure_latency(
                        op_name,
                        id,
                        op,
                        input_args,
                        input_kwargs,
                        device,
                        num_iter,
                        final_config,
                        out_file,
                    )

                logging.debug(f"Finished running {op_name}[{id}].")

        config_id += 1


def main():

    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument("--config", type=str, required=True, help="The op config file.")
    parser.add_argument(
        "--range", action="store_true", help="The config file has config range."
    )
    parser.add_argument(
        "--filter", type=str, default="", help="The input op config file."
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

    load_benchmarks(benchmark)

    print(operator_map)

    op_filter = {x.strip() for x in args.filter.split(",") if x.strip()}

    op_configs = OperatorConfig(args.config, args.device, op_filter)

    out_file_name = args.out_file_name
    if args.metric:
        out_file_name = args.out_file_name + ".metric_config"

    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    with open(out_file_name, "w") as out_file:
        with record_function("## BENCHMARK ##"):
            for (op_name, op, configs) in op_configs.get_selected_ops():
                run_op(
                    op_name,
                    op,
                    configs,
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
