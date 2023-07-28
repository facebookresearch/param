#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import json
import logging
import time
from typing import Any, Dict, Optional

import torch

from param_bench.train.compute.python.lib import __version__, pytorch as lib_pytorch
from param_bench.train.compute.python.lib.config import BenchmarkConfig
from param_bench.train.compute.python.lib.init_helper import load_modules
from param_bench.train.compute.python.lib.pytorch.benchmark import (
    make_default_benchmark,
)
from param_bench.train.compute.python.lib.pytorch.config_util import (
    ExecutionPass,
    get_benchmark_options,
    get_sys_info,
    OpExecutionMode,
)
from param_bench.train.compute.python.workloads import pytorch as workloads_pytorch
from torch.autograd.profiler import record_function
from torch.profiler import _ExperimentalConfig, ExecutionTraceObserver


class BenchmarkHelper:
    def __init__(
        self,
        config: Any,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self.logger = logger or logging.getLogger("foo").addHandler(
            logging.NullHandler()
        )

        if getattr(config, "version", True):
            logger.info(f"PARAM train compute version: {__version__}")

        # Load PyTorch implementations for data generator and operators.
        load_modules(lib_pytorch)

        # Load PyTorch operator workloads.
        load_modules(workloads_pytorch)

        self.run_options: Dict[str, Any] = get_benchmark_options()
        self.run_options["warmup"] = getattr(config, "warmup", 1)
        self.run_options["iteration"] = getattr(config, "iteration", 1)
        self.run_options["device"] = getattr(config, "device", "cpu")
        self.run_options["cuda_l2_cache"] = (
            getattr(config, "cuda_l2_cache", "off") == "on"
        )
        self.run_options["resume_op_run_id"] = getattr(config, "resume_id", None)
        self.run_options["stop_op_run_id"] = getattr(config, "stop_id", None)
        self.run_options["run_batch_size"] = getattr(config, "run_batch_size", 50)
        self.run_options["batch_cuda_device"] = getattr(config, "batch_cuda_device", 1)

        if getattr(config, "backward", False):
            self.run_options["pass_type"] = ExecutionPass.BACKWARD
        else:
            self.run_options["pass_type"] = ExecutionPass.FORWARD

        self.run_options["op_exec_mode"] = OpExecutionMode(
            getattr(config, "exec_mode", "discrete")
        )

        if getattr(config, "batch_cmd", None):
            self.run_options["batch_cmd"] = config.batch_cmd

        # NCU
        self.run_ncu = getattr(config, "run_ncu", False)
        self.run_options["run_ncu"] = self.run_ncu
        if getattr(config, "ncu_bin", None):
            self.run_options["ncu_bin"] = config.ncu_bin
        if getattr(config, "ncu_warmup", None):
            self.run_options["ncu_warmup"] = config.ncu_warmup
        if getattr(config, "ncu_iteration", None):
            self.run_options["ncu_iteration"] = config.ncu_iteration
        if getattr(config, "ncu_args_file", None):
            with open(config.ncu_args_file, "r") as ncu_file:
                self.run_options["ncu_args"] = ncu_file.read().strip()

        # NSys
        self.run_nsys = getattr(config, "run_nsys", False)
        self.run_options["run_nsys"] = self.run_nsys
        if getattr(config, "nsys_bin", None):
            self.run_options["nsys_bin"] = config.nsys_bin
        if getattr(config, "nsys_warmup", None):
            self.run_options["nsys_warmup"] = config.nsys_warmup
        if getattr(config, "nsys_iteration", None):
            self.run_options["nsys_iteration"] = config.nsys_iteration
        if getattr(config, "nsys_args_file", None):
            with open(config.nsys_args_file, "r") as nsys_file:
                self.run_options["nsys_args"] = nsys_file.read().strip()

        self.run_options["cmd_args"] = getattr(config, "__dict__", None)
        self.run_options["out_file_prefix"] = getattr(config, "output_prefix", "")

        self.cupti_profiler: bool = getattr(config, "cupti_profiler", False)
        self.profile: bool = getattr(config, "profile", True)

        self.write_option: str = "a" if getattr(config, "append", True) else "w"

        self.et: bool = getattr(config, "et", False)
        self.et_file: str = ""

        self.trace_file: str = ""

        if self.cupti_profiler and not self.run_options["device"].startswith("cuda"):
            self.logger.warning(
                "Cannot use --cupti_profiler when not running on cuda device"
            )
            self.cupti_profiler = False

        if self.cupti_profiler and (self.run_ncu or self.run_nsys):
            self.logger.warning("Cannot use --cupti_profiler when running with NCU")
            self.cupti_profiler = False

        if self.cupti_profiler and not self.profile:
            self.logger.warning(
                "Enabling pytorch profiler as --cupti_profiler was added"
            )
            self.profile = True

        self.cupti_profiler_metrics: str = getattr(
            config, "cupti_profiler_metrics", "kineto__cuda_core_flops"
        )
        self.cupti_profiler_measure_per_kernel: bool = getattr(
            config, "cupti_profiler_measure_per_kernel", True
        )

    def eval(self, evaluate_file_path: str) -> Dict:
        self.logger.info("Microbenchmarking started")

        self.out_file_prefix = (
            self.run_options["out_file_prefix"]
            + "_"
            + (evaluate_file_path.split("/")[-1]).split(".")[0]
        )

        ret_files = {}

        self.out_file_name = f"{self.out_file_prefix}_benchmark.json"

        with open(self.out_file_name, self.write_option) as out_file:
            self.run_options["out_stream"] = out_file
            benchmark_setup = {
                "run_options": self.run_options,
                "sys_info": get_sys_info(),
                "start_time": time.process_time(),
            }
            print(json.dumps(benchmark_setup, default=str), file=out_file)

            # This hack is necessary for Kineto profiler library to be initialized
            # and thus be able to track active CUDA contexts.
            if self.cupti_profiler:
                with torch.autograd.profiler.profile(
                    enabled=True,
                    use_cuda=True,
                    use_kineto=True,
                ) as _:
                    self.logger.info("Running dummy profiler warmup for CUPTI.")

            bench_config = BenchmarkConfig(self.run_options)
            bench_config.load_json_file(evaluate_file_path)
            benchmark = make_default_benchmark(bench_config)
            use_cuda = False
            if self.run_options["device"].startswith("cuda"):
                use_cuda = True

            et = None
            if self.et:
                self.et_file = f"{self.out_file_prefix}_et.json"
                et = ExecutionTraceObserver()
                et.register_callback(self.et_file)
                et.start()

            cupti_profiler_config = (
                _ExperimentalConfig(
                    profiler_metrics=self.cupti_profiler_metrics.split(","),
                    profiler_measure_per_kernel=self.cupti_profiler_measure_per_kernel,
                )
                if self.cupti_profiler
                else None
            )

            with torch.autograd.profiler.profile(
                self.profile,
                use_cuda=use_cuda,
                use_kineto=True,
                record_shapes=False,
                experimental_config=cupti_profiler_config,
                use_cpu=(not self.cupti_profiler)
                or self.cupti_profiler_measure_per_kernel,
            ) as prof:
                with record_function(f"[param|{self.run_options['device']}]"):
                    benchmark.run()

            if et:
                et.stop()
                et.unregister_callback()
                self.logger.info(f"execution trace: {self.et_file}")
                ret_files["et_file"] = self.et_file

            print(
                json.dumps({"finish_time": time.process_time()}),
                file=out_file,
            )
            if self.profile and prof:
                self.trace_file = f"{self.out_file_prefix}_trace.json"
                self.logger.info(f"trace: {self.trace_file}")
                prof.export_chrome_trace(self.trace_file)
                ret_files["trace_file"] = self.trace_file

            ret_files["benchmark_file"] = self.out_file_name

        self.logger.info(f"benchmark result: {self.out_file_name}")

        return ret_files
