from ..init_helper import get_logger

logger = get_logger()

import abc
import json
import logging
import os
import subprocess
from multiprocessing import shared_memory
from typing import Any
from typing import Dict
from typing import List
from typing import TextIO

from ..config import OperatorConfig
from .config_util import create_op_info, get_benchmark_options
from .op_executor import OpExecutor


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

    run_id: A unique string id that identifies the current build configuration.
    """

    _skip_run = False
    _resume_op_run_id = None

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "run") and callable(subclass.run) or NotImplemented

    def __init__(self):
        pass

    # Loads arg configurations and generates the arg data for an op.
    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @staticmethod
    def set_resume_op_run_id(resume_op_run_id: str):
        logger.debug(f"resume_op_run_id: {resume_op_run_id}")
        BuildExecutor._resume_op_run_id = resume_op_run_id
        # if a valid resume_op_run_id is defined, skip runs by default until
        # a match op run id is found.
        if BuildExecutor._resume_op_run_id:
            BuildExecutor._skip_run = True
        logger.debug(f"_resume_op_run_id: {BuildExecutor._resume_op_run_id}")
        logger.debug(f"skip_run: {BuildExecutor._skip_run}")

    @staticmethod
    def should_skip(op_run_id: str):
        # Check if we should skip the run, check if a matching run id is found.
        if BuildExecutor._skip_run:
            if op_run_id == BuildExecutor._resume_op_run_id:
                BuildExecutor._skip_run = False
        logger.debug(f"op_run_id: {op_run_id}")
        logger.debug(f"resume_op_run_id: {BuildExecutor._resume_op_run_id}")
        logger.debug(f"skip_run: {BuildExecutor._skip_run}")
        return BuildExecutor._skip_run


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
            logger.debug(f"build args: {build_args} {build_kwargs}")
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
        run_id = self.build_id + f":{input_id}"

        if BuildExecutor.should_skip(f"{self.op_config.name}:{run_id}"):
            return

        logger.info(f"input_id: [{input_id}]")
        logger.debug(f"input_config: {input_config}")

        # generate input data
        input_data_gen = self.op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(
            input_config, self.run_options["device"]
        )

        op_exe = OpExecutor(self.op_config.name, self.op_config.op, self.run_options)

        metrics = op_exe.run(input_args, input_kwargs, run_id)
        # print(result)
        final_config = {
            "build": self.build_input_config["build"],
            "input": input_config,
        }

        output_stats(
            self.out_stream, self.op_config.name, run_id, metrics, final_config
        )
        logger.debug(f"Finished running [{run_id}].")

        if self.run_options["run_ncu"]:
            input_config["id"] = input_id
            self.input_config_queue.append(input_config)
            if len(self.input_config_queue) == self.run_options["ncu_batch"]:
                self._run_ncu()
                self.input_config_queue.clear()

    def _run_ncu(self):
        NCU_BIN = "/usr/local/NVIDIA-Nsight-Compute-2021.2/ncu"
        ncu_bin = os.getenv("NCU_BIN")
        if not ncu_bin:
            ncu_bin = NCU_BIN

        param_bench_range = "param_bench@measure"
        input_id = self.input_config_queue[0]["id"]
        ncu_log_file = f"benchmark.ncu.{self.build_id}:{input_id}.log"
        ncu_log_file = ncu_log_file.replace(":", "-")
        ncu_extra_args = self.run_options["ncu_args"]
        ncu_options = (
            f"--log-file {ncu_log_file} --csv --app-replay-buffer file --nvtx "
            f"--nvtx-include {param_bench_range} --target-processes all"
        )
        if ncu_extra_args:
            ncu_options += f" {ncu_extra_args}"

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

        op_info["config"][0]["build"] = self.build_input_config["build"]
        op_info["config"][0]["input"] = self.input_config_queue
        run_options = get_benchmark_options()
        run_options["device"] = self.run_options["device"]
        run_options["pass_type"] = self.run_options["pass_type"].value
        run_options["warmup"] = 1
        run_options["iteration"] = 1
        config = {
            "op_name": self.op_config.name,
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
        logger.info(f"ncu log file: {ncu_log_file}")


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
            self._run_for_input(input_id, input_config)

            counter += 1

    def _run_for_input(self, input_id: str, input_config: Dict[str, Any]):
        run_id = self.build_id + f":{input_id}"

        if BuildExecutor.should_skip(f"{self.op_config.name}:{run_id}"):
            return

        logger.info(f"input_id: [{run_id}]")
        logger.debug(f"input_config: {input_config}")

        # generate input data
        input_data_gen = self.op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(
            input_config, self.run_options["device"]
        )

        op_exe = OpExecutor(self.op_config.name, self.op_config.op, self.run_options)

        metrics = op_exe.run(input_args, input_kwargs, run_id)
        # print(result)
        final_config = {
            "build": self.build_input_config["build"],
            "input": input_config,
        }

        output_stats(
            self.out_stream, self.op_config.name, run_id, metrics, final_config
        )

        logger.debug(f"Finished running [{run_id}].")


def output_stats(
    out_stream: TextIO,
    name: str,
    run_id: str,
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
        "id": run_id,
        "metric": metrics,
        "config": config,
    }
    out_stream.write(json.dumps(stats) + "\n")
    out_stream.flush()


def format_float_val_list(time_records: List[float], decimals: int = 6):
    format_str = f"{{0:.{decimals}f}}"
    return f"[{', '.join([format_str.format(i) for i in time_records])}]"
