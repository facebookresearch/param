import abc
import copy
import enum
import json
import logging
import os
import subprocess
from multiprocessing import shared_memory, resource_tracker
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import TextIO
from typing import Type

import torch

from ..config import OperatorConfig, BenchmarkConfig
from ..init_helper import get_logger, load_package
from ..operator import OperatorInterface
from .config_util import create_op_info, get_benchmark_options, ExecutionPass
from .op_executor import OpExecutor
from .timer import Timer, format_float_val_list


logger = get_logger()


class BuildExecutor(metaclass=abc.ABCMeta):
    """
    An abstract base class for build executor. The build executor is responsible
    for materialize build and input data, proper initialize/reset the operator,
    and call OpExecutor to collect metrics.

    Expected parameters:

    build_input_config: A dictionary of "build" and "input" configs. The build config
    is expected to be in its final form, while the input configs may or may not be.

    op_config: Operator configurations.

    run_options: Benchmark run options.

    op_run_id: A unique string id that identifies the current build configuration.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "run") and callable(subclass.run) or NotImplemented

    def __init__(self):
        pass

    # Loads arg configurations and generates the arg data for an op.
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError


class OpBuildExecutor(BuildExecutor):
    def __init__(
        self,
        build_input_config: Dict[str, Any],
        op_config: OperatorConfig,
        run_options: Dict[str, Any],
        build_id: str,
    ):
        super(OpBuildExecutor, self).__init__()
        self.build_input_config = build_input_config
        self.op_config = op_config
        self.run_options = run_options
        self.build_id = build_id
        self.out_stream = run_options["out_stream"]
        self.input_config_queue = []

    def run(self):
        # Reset operator to clear memory before new build
        self.op_config.op.cleanup()
        build_config = self.build_input_config["build"]
        logger.info(f"build_id [{self.build_id}]")
        logger.debug(f"build_config: {build_config}")
        if build_config:
            build_data_gen = self.op_config.build_data_generator()
            (build_args, build_kwargs) = build_data_gen.get_data(
                build_config, self.run_options["device"]
            )
            logger.debug(f"{build_args} {build_kwargs}")
            self.op_config.op.build(*build_args, **build_kwargs)

        generate_input_config = self.op_config.input_iterator(
            self.build_input_config, "input", self.run_options["device"]
        )

        for (input_id, input_config) in generate_input_config:
            self._run_for_input(input_id, input_config)

        if self.run_options["run_ncu"] and self.input_config_queue:
            self._run_ncu()
            self.input_config_queue.clear()


    def _run_for_input(self, input_id: str, input_config: Dict[str, Any]):
        op_run_id = self.build_id + f":{input_id}"
        logger.info(f"input_id: [{input_id}]")
        logger.debug(f"input_config: {input_config}")
        # generate input data
        input_data_gen = self.op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(
            input_config, self.run_options["device"]
        )

        op_exe = OpExecutor(self.op_config.name, self.op_config.op, self.run_options)

        metrics = op_exe.run(input_args, input_kwargs, op_run_id)
        # print(result)
        final_config = {
            "build": self.build_input_config["build"],
            "input": input_config,
        }

        output_stats(
            self.out_stream, self.op_config.name, op_run_id, metrics, final_config
        )
        logger.debug(f"Finished running [{op_run_id}].")

        if self.run_options["run_ncu"]:
            input_config["id"] = input_id
            self.input_config_queue.append(input_config)
            if len(self.input_config_queue) == 100:
                self._run_ncu()
                self.input_config_queue.clear()


    def _run_ncu(self):
        final_config = {
            "build": self.build_input_config["build"],
            "input": self.input_config_queue,
        }
        op_name = self.op_config.name
        NCU_BIN = "/usr/local/NVIDIA-Nsight-Compute-2021.2/ncu"
        ncu_bin = os.getenv("NCU_BIN")
        if not ncu_bin:
            ncu_bin = NCU_BIN

        param_bench_range = "param_bench@measure"
        input_id = self.input_config_queue[0]["id"]
        out_prefix = f"benchmark.ncu.{self.build_id}:{input_id}"
        out_prefix = out_prefix.replace(":", "-")
        metrics = "dram__bytes.sum"
        ncu_options = (
            f"--log-file {out_prefix}.log --csv --app-replay-buffer file --target-processes all "
            f"--metrics {metrics} --nvtx --nvtx-include {param_bench_range}"
        )

        op_info = create_op_info()
        op_info["build_iterator"] = (
            self.op_config.info["build_iterator"]
            if "build_iterator" in self.op_config.info
            else None
        )
        op_info["input_iterator"] = (
            self.op_config.info["input_iterator"]
            if "input_iterator" in self.op_config.info
            else None
        )
        op_info["build_data_generator"] = (
            self.op_config.info["build_data_generator"]
            if "build_data_generator" in self.op_config.info
            else None
        )
        op_info["input_data_generator"] = (
            self.op_config.info["input_data_generator"]
            if "input_data_generator" in self.op_config.info
            else None
        )

        op_info["config"][0]["build"] = final_config["build"]
        op_info["config"][0]["input"] = final_config["input"]
        run_options = get_benchmark_options()
        run_options["device"] = self.run_options["device"]
        run_options["pass_type"] = self.run_options["pass_type"].value
        run_options["warmup"] = 1
        run_options["iteration"] = 1
        config = {
            "op_name": op_name,
            "build_id": self.build_id,
            "op_info": op_info,
            "run_options": run_options,
        }
        config_str = json.dumps(config)

        """
        BUG: Python shared memory bug workaround.
        Shared memory has a bug to proper track and release memory, see
        https://bugs.python.org/issue39959
        Fixed PR: https://github.com/python/cpython/pull/20136
        Workaround: unregister(shm._name, "shared_memory") from resource_tracker
        in other processes which access this shm.
        """
        shm = shared_memory.SharedMemory(create=True, size=len(config_str))

        shm.buf[:] = config_str.encode("utf-8")
        logger.debug(f"shared memory buffer: {shm.name}")
        benchmark_cmd = f"python -m param_bench.train.compute.python.pytorch.run_batch -s {shm.name}"
        if logger.getEffectiveLevel() == logging.DEBUG:
            benchmark_cmd += " -v"
        cmd = [ncu_bin] + ncu_options.split(" ") + benchmark_cmd.split(" ")
        cmd_str = " ".join(cmd)
        logger.info(f"Running: {cmd_str}")
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            for line in proc.stdout:
                if line.strip():
                    print(line, end="")
        shm.close()
        shm.unlink()


class MaterializedBuildExecutor(BuildExecutor):
    def __init__(
        self,
        build_input_config: Dict[str, Any],
        op_config: OperatorConfig,
        run_options: Dict[str, Any],
        build_id: str,
    ):
        super(MaterializedBuildExecutor, self).__init__()
        self.build_input_config = build_input_config
        self.op_config = op_config
        self.run_options = run_options
        self.build_id = build_id
        self.out_stream = run_options["out_stream"]
        if "build" not in self.build_input_config:
            self.build_input_config["build"] = None

    def run(self):
        # Reset operator to clear memory before new build
        self.op_config.op.cleanup()
        build_config = self.build_input_config["build"]
        logger.debug(build_config)
        if build_config:
            build_data_gen = self.op_config.build_data_generator()
            (build_args, build_kwargs) = build_data_gen.get_data(
                build_config, self.run_options["device"]
            )
            logger.debug(f"{build_args} {build_kwargs}")
            self.op_config.op.build(*build_args, **build_kwargs)

        materialized_input_configs = self.build_input_config["input"]
        counter = 0
        for input_config in materialized_input_configs:
            # Override input_id if one exists in the config.
            if "id" in input_config:
                input_id = input_config["id"]
            else:
                input_id = counter
            op_run_id = self.build_id + f":{input_id}"
            logger.info(f"input_id: [{op_run_id}]")
            logger.debug(f"input_config: {input_config}")
            # generate data
            self._run_for_input(op_run_id, input_config)

            counter += 1
            logger.debug(f"Finished running [{op_run_id}].")

    def _run_for_input(self, op_run_id: str, input_config: Dict[str, Any]):
        # generate input data
        input_data_gen = self.op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(
            input_config, self.run_options["device"]
        )

        op_exe = OpExecutor(self.op_config.name, self.op_config.op, self.run_options)

        metrics = op_exe.run(input_args, input_kwargs, op_run_id)
        # print(result)
        final_config = {
            "build": self.build_input_config["build"],
            "input": input_config,
        }

        output_stats(
            self.out_stream, self.op_config.name, op_run_id, metrics, final_config
        )


def output_stats(
    out_stream: TextIO,
    name: str,
    op_run_id: str,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
):
    for pass_name, metric in metrics.items():
        logger.info(f"pass: {pass_name}")
        for metric_name, records in metric.items():
            total = sum(records)
            avg = total / len(records)
            logger.info(
                f"metric: {metric_name}, avg: {avg:.6f} sec, tot: {total:.6f} sec"
            )
            logger.info(f"{format_float_val_list(records, 6)}")
    stats = {
        "name": name,
        "id": op_run_id,
        "metric": metrics,
        "config": config,
    }
    out_stream.write(json.dumps(stats) + "\n")
    out_stream.flush()
