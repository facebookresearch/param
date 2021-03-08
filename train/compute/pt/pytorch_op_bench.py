
from __future__ import absolute_import, division, print_function, unicode_literals, annotations
import argparse, json, sys
from pprint import pprint
import pydot
from enum import Enum
import logging
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO
import torch
from torch.autograd.profiler import record_function
import time
import random
from pytorch_op_def import get_pytorch_ops, pytorch_dtype_map
from pytorch_op_util import OperatorConfig, OpDataIter, OpConfigType

FORMAT = '[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

# Timer in seconds
class Timer():
    def __init__(self, device:str):
        self.device:str = device
        self.start_time:float = 0
        self.end_time:float = 0
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

def benchmark_op(op_id:str, num_iter:int, device:str, op:Callable, *args, **kwargs):
    time_records = []
    for _ in range(num_iter):
        # flush cache
        _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
        torch.cuda.empty_cache()

        with Timer(device) as timer:
            op(*args, **kwargs)
        time_records.append(timer.elapsed_time())
    return time_records

def collect_metric(op_id:str, num_iter:int, device:str, op:Callable, *args, **kwargs):
    for _ in range(num_iter):
        # flush cache
        _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
        torch.cuda.empty_cache()

        torch.cuda.nvtx.range_push(op_id)
        op(*args, **kwargs)
        torch.cuda.nvtx.range_pop()

def run_op(op:Dict[str, Any], num_iter:int, device:str, json_file:TextIO, metric_mode:bool):
    ops_map = get_pytorch_ops()
    op_name = op["name"]
    (id, args, kwargs, arg_config) = op["args"]

    logging.info(f"{op_name}[{id}]:")
    logging.info(f"  {arg_config}")
    warmup_iter = 5
    logging.debug(f"Running {op_name}[{id}] for {warmup_iter} warm up iterations")
    # warm up
    time_records = benchmark_op(f"{op_name}[{id}]", warmup_iter, device, ops_map[op_name], *args, **kwargs)
    logging.info(f"  warmup: {time_records}")

    # collect CUDA metrics
    if metric_mode:
        if device.startswith("cuda"):
            # use nvtx allows us to collect only this part of kernel executions
            # and match op and arg variants to metrics.
            logging.info(f"Running {op_name}[{id}] for {num_iter} CUDA metric iterations")
            torch.cuda.nvtx.range_push("op_bench")
            collect_metric(f"{op_name}[{id}]", num_iter, device, ops_map[op_name], *args, **kwargs)
            torch.cuda.nvtx.range_pop()
            stats = {"name": op_name, "id": id, "iter": num_iter, "config": arg_config}
            json_file.write(json.dumps(stats) + "\n")
            json_file.flush()
        else:
            raise Exception('Non-GPU metric mode is not supported.')
    else:
        # actual timing
        logging.debug(f"Running {op_name}[{id}] for {num_iter} measured iterations")
        torch.cuda.nvtx.range_push("op_bench")
        time_records = benchmark_op(f"{op_name}[{id}]", num_iter, device, ops_map[op_name], *args, **kwargs)
        torch.cuda.nvtx.range_pop()
        tot = sum(time_records) 
        logging.info(f"  rec: {time_records}")
        logging.info(f"  avg: {tot/num_iter:.6f} sec")
        logging.info(f"  tot: {tot:.6f} sec")
        stats = {"name": op_name, "id": id, "time": time_records, "iter": num_iter, "config": arg_config}
        json_file.write(json.dumps(stats) + "\n")
        json_file.flush()

    logging.debug(f"Finished running {op_name}[{id}].")


def main():

    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument(
        "--input", type=str, required=True, help="The input op config file."
        )
    parser.add_argument(
        "--range", action='store_true', help="The config file has config range."
        )
    parser.add_argument(
        "--filter", type=str, default="", help="The input op config file."
        )
    parser.add_argument(
        "--iter", type=int, default=1, help="number of iterations."
    )
    parser.add_argument(
        "--metric", action='store_true', help="The metric collection mode."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="device of execution."
    )
    parser.add_argument(
        "--out_json", type=str, default="op_bench_log.json", help="json file to write log info."
    )
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    op_filter = {x.strip() for x in args.filter.split(",") if x.strip()}

    if args.range:
        op_configs = OperatorConfig(args.input, OpConfigType.RANGE, args.device, op_filter)
    else:
        op_configs = OperatorConfig(args.input, OpConfigType.SAMPLE, args.device, op_filter)

    out_json = args.out_json
    if args.metric:
        out_json = args.out_json + ".metric_config"
        
    # We don't want too many threads for stable benchmarks
    torch.set_num_threads(1)

    with open(out_json, "w") as json_file:
        with record_function("## BENCHMARK ##"):
            for op in op_configs.get_selected_ops():
                data_iter = OpDataIter(op_configs, op)
                for op_data in data_iter:
                    run_op(op_data, args.iter, args.device, json_file, args.metric)

        logging.info(f"Log written to {args.out_json}.")

if __name__ == "__main__":
    main()
