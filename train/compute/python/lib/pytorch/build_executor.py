from ..init_helper import get_logger

logger = get_logger()

import abc
import enum
import gc
import json
import logging
import os
import shlex
import subprocess
from datetime import datetime
from multiprocessing import shared_memory
from typing import Any, Dict, List, TextIO

from ..config import OperatorConfig
from .config_util import create_op_info, get_benchmark_options
from .cuda_util import free_torch_cuda_memory, log_cuda_memory_usage
from .op_executor import OpExecutor


@enum.unique
class BenchmarkTransitionState(enum.Enum):
    # Forward pass will always run (also required for backward pass).
    RUN = "Run"
    SKIP = "Skip"
    STOP = "Stop"


class StopBenchmarkException(Exception):
    pass


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

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "run") and callable(subclass.run) or NotImplemented

    def __init__(self):
        self._transition_state = BenchmarkTransitionState.RUN
        self._resume_op_run_id = None
        self._stop_op_run_id = None

    # Loads arg configurations and generates the arg data for an op.
    @abc.abstractmethod
    def run(
        self,
        op_config: OperatorConfig,
        build_input_config: Dict[str, Any],
        config_build_id: str,
    ):
        raise NotImplementedError

    def set_resume_op_run_id(self, resume_op_run_id: str):
        logger.debug(f"resume_op_run_id: {resume_op_run_id}")
        self._resume_op_run_id = resume_op_run_id
        # If a valid resume_op_run_id is defined, skip runs by default until
        # a match op run id is found.
        if self._resume_op_run_id:
            self._transition_state = BenchmarkTransitionState.SKIP
        logger.debug(f"_resume_op_run_id: {self._resume_op_run_id}")
        logger.debug(f"_transition_state: {self._transition_state}")

    def set_stop_op_run_id(self, stop_op_run_id: str):
        logger.debug(f"stop_op_run_id: {stop_op_run_id}")
        self._stop_op_run_id = stop_op_run_id
        logger.debug(f"_stop_op_run_id: {self._stop_op_run_id}")
        logger.debug(f"_transition_state: {self._transition_state}")

    def get_transition_state(self, op_run_id: str):
        # Check if we should skip the run, check if a matching run id is found.
        if self._transition_state == BenchmarkTransitionState.SKIP:
            if op_run_id == self._resume_op_run_id:
                self._transition_state = BenchmarkTransitionState.RUN
                logger.info(f"Resume benchmark check matched [{op_run_id}]")
        # If a valid stop_op_run_id is defined, check a match op run id is found
        # to stop the run
        if op_run_id == self._stop_op_run_id:
            self._transition_state = BenchmarkTransitionState.STOP
            logger.info(f"Stop benchmark check matched [{op_run_id}]")

        logger.debug(f"op_run_id: {op_run_id}")
        logger.debug(f"_transition_state: {self._transition_state}")
        return self._transition_state


class OpBuildExecutor(BuildExecutor):
    """
    OpBuildExecutor is the default BuildExecutor that supports most features.
    It will take a materialized build config and a list of input configs (which
    may contain macros and need to be materialized through iterators). It also
    supports queueing a batch of input configs and collecting NCU metrics. If
    a resume op_run_id is defined in the run_options, it will skip running
    benchmarks till the matching op_run_id is found.
    """

    def __init__(self, run_options: Dict[str, Any]):
        super(OpBuildExecutor, self).__init__()
        self.run_options = run_options
        self.out_stream = run_options["out_stream"]

        self.input_config_queue = []
        self.op_config = None
        self.build_input_config = None
        self.config_build_id = None

    def _should_run_batch_input_config(self):
        return self.run_options["run_ncu"] or self.run_options["run_nsys"]

    def run(
        self,
        op_config: OperatorConfig,
        build_input_config: Dict[str, Any],
        config_build_id: str,
    ):
        def _run_nsight():
            if self.run_options["run_ncu"]:
                shm, config = self._create_shm_config(
                    warmup=self.run_options["ncu_warmup"],
                    iteration=self.run_options["ncu_iteration"],
                )
                self._run_ncu(shm, config)
                shm.close()
                shm.unlink()
            if self.run_options["run_nsys"]:
                shm, config = self._create_shm_config(
                    warmup=self.run_options["nsys_warmup"],
                    iteration=self.run_options["nsys_iteration"],
                )
                self._run_nsys(shm, config)
                shm.close()
                shm.unlink()

        self.input_config_queue.clear()
        self.op_config = op_config
        self.build_input_config = build_input_config
        self.config_build_id = config_build_id

        # Reset operator to clear memory before new build
        self.op_config.op.cleanup()
        build_config = self.build_input_config["build"]
        logger.debug(f"config_build_id: [{self.config_build_id}]")
        logger.debug(f"build_config: {build_config}")
        if build_config is not None:
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
            config_run_id = f"{self.op_config.name}|{self.config_build_id}|{input_id}"
            transition_state = self.get_transition_state(config_run_id)
            if transition_state == BenchmarkTransitionState.SKIP:
                continue
            elif transition_state == BenchmarkTransitionState.STOP:
                # break here to finsh the remaining work before raising the stop
                # exception below
                break
            # run for input, also fills the input_config_queue if there is a
            # need to run in nsight batch.
            self._run_for_input(input_id, input_config)

            if self._should_run_batch_input_config():
                # Check if the queue has enough for a nsight batch to run.
                if len(self.input_config_queue) == self.run_options["run_batch_size"]:
                    _run_nsight()
                    self.input_config_queue.clear()

        # If any input_config remains in the queue, run them with NSight.
        if self._should_run_batch_input_config() and self.input_config_queue:
            _run_nsight()
            self.input_config_queue.clear()

        # Raise exception if we need to stop running.
        if transition_state == BenchmarkTransitionState.STOP:
            raise StopBenchmarkException(f"Benchmark stopped at [{config_run_id}]")

    def _run_for_input(self, input_id: str, input_config: Dict[str, Any]):
        run_id = f"{self.config_build_id}|{input_id}"

        logger.info(f"input_id: [{input_id}]")
        if self.run_options["device"].startswith("cuda"):
            log_cuda_memory_usage()
        logger.debug(f"input_config: {input_config}")

        # generate input data only if we actually need to run it
        if self.run_options["warmup"] != 0 or self.run_options["iteration"] != 0:
            input_data_gen = self.op_config.input_data_generator()
            (input_args, input_kwargs) = input_data_gen.get_data(
                input_config, self.run_options["device"]
            )
            op_exe = OpExecutor(
                self.op_config.name, self.op_config.op, self.run_options
            )

            metrics = op_exe.run(input_args, input_kwargs, run_id)

            # clean up inputs
            input_args = None
            input_kwargs = None
            # Reset operator outputs to clear memory
            self.op_config.op.cleanup()
            free_torch_cuda_memory()
        else:
            metrics = {}

        final_config = {
            "build": self.build_input_config["build"],
            "input": input_config,
        }

        output_stats(
            self.out_stream, self.op_config.name, run_id, metrics, final_config
        )
        logger.debug(f"finished running [{run_id}].")

        # add this input config to the queue if we want to run it in a batch.
        if self._should_run_batch_input_config():
            # Record the current input_id so the NSight run can reuse this id.
            input_config["id"] = input_id
            self.input_config_queue.append(input_config)

    def _create_shm_config(self, warmup=1, iteration=1):
        op_info = create_op_info()
        op_info["build_iterator"] = self.op_config.info.get("build_iterator", None)
        op_info["input_iterator"] = self.op_config.info.get("input_iterator", None)
        op_info["build_data_generator"] = self.op_config.info.get(
            "build_data_generator", None
        )
        op_info["input_data_generator"] = self.op_config.info.get(
            "input_data_generator", None
        )

        op_info["config"][0]["build"] = self.build_input_config["build"]
        op_info["config"][0]["input"] = self.input_config_queue
        run_options = get_benchmark_options()
        run_options["cuda_l2_cache"] = self.run_options["cuda_l2_cache"]
        run_options["device"] = self.run_options["device"]
        run_options["iteration"] = iteration
        run_options["op_exec_mode"] = self.run_options["op_exec_mode"].value
        run_options["pass_type"] = self.run_options["pass_type"].value
        run_options["warmup"] = warmup
        config = {
            "op_name": self.op_config.name,
            "config_build_id": self.config_build_id,
            "op_info": op_info,
            "run_options": run_options,
        }
        config_str = json.dumps(config)

        # debug
        # with open("batch_config.json", "w") as batch_config:
        #     batch_config.write(config_str)

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
        return shm, config

    def _run_ncu(self, shm, config):
        if not self.run_options["run_ncu"]:
            return

        ncu_bin = self.run_options["ncu_bin"]

        param_bench_range = "param_bench:measure"
        start_input_id = self.input_config_queue[0]["id"]
        out_file_prefix = self.run_options["out_file_prefix"]
        timestamp = int(datetime.timestamp(datetime.now()))
        ncu_log_file = f"{out_file_prefix}_{os.getpid()}_{timestamp}_ncu.log"
        ncu_extra_args = self.run_options["ncu_args"]
        ncu_options = (
            f"--log-file {ncu_log_file} --csv --app-replay-buffer file --nvtx "
            f"--nvtx-include {param_bench_range} --target-processes all"
        )
        if ncu_extra_args:
            ncu_options += f" {ncu_extra_args}"

        batch_cmd = self.run_options["batch_cmd"]
        benchmark_cmd = f"{batch_cmd} -s {shm.name}"
        if logger.getEffectiveLevel() == logging.DEBUG:
            benchmark_cmd += " -v"
        cmd = [ncu_bin] + shlex.split(ncu_options) + shlex.split(benchmark_cmd)
        cmd_str = " ".join(cmd)
        logger.info(f"running: {cmd_str}")
        batch_env = os.environ.copy()
        batch_env["CUDA_VISIBLE_DEVICES"] = str(self.run_options["batch_cuda_device"])
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=batch_env,
        ) as proc:
            for line in proc.stdout:
                if line.strip():
                    print(line, end="")

        end_input_id = self.input_config_queue[-1]["id"]
        print(
            json.dumps(
                {
                    "ncu_file": ncu_log_file,
                    "ncu_cmd_str": cmd_str,
                    "config": config,
                    "start_run_id": f"{self.config_build_id}|{start_input_id}",
                    "end_run_id": f"{self.config_build_id}|{end_input_id}",
                }
            ),
            file=self.out_stream,
        )
        logger.info(f"ncu result: {ncu_log_file}")

    def _run_nsys(self, shm, config):
        if not self.run_options["run_nsys"]:
            return

        nsys_bin = self.run_options["nsys_bin"]

        start_input_id = self.input_config_queue[0]["id"]
        out_file_prefix = self.run_options["out_file_prefix"]
        timestamp = int(datetime.timestamp(datetime.now()))
        nsys_output_file = f"{out_file_prefix}_{os.getpid()}_{timestamp}_nsys.nsys-rep"
        nsys_extra_args = self.run_options["nsys_args"]
        nsys_options = f"profile -t cuda,nvtx -o {nsys_output_file} "
        if nsys_extra_args:
            nsys_options += f" {nsys_extra_args}"

        batch_cmd = self.run_options["batch_cmd"]
        benchmark_cmd = f"{batch_cmd} -s {shm.name}"
        if logger.getEffectiveLevel() == logging.DEBUG:
            benchmark_cmd += " -v"
        cmd = [nsys_bin] + shlex.split(nsys_options) + shlex.split(benchmark_cmd)
        cmd_str = " ".join(cmd)
        batch_env = os.environ.copy()
        batch_env["CUDA_VISIBLE_DEVICES"] = str(self.run_options["batch_cuda_device"])
        logger.info(f"running: {cmd_str}")
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=batch_env,
        ) as proc:
            for line in proc.stdout:
                if line.strip():
                    print(line, end="")
        end_input_id = self.input_config_queue[-1]["id"]
        print(
            json.dumps(
                {
                    "nsys_file": nsys_output_file,
                    "nsys_cmd_str": cmd_str,
                    "config": config,
                    "start_run_id": f"{self.config_build_id}|{start_input_id}",
                    "end_run_id": f"{self.config_build_id}|{end_input_id}",
                }
            ),
            file=self.out_stream,
        )
        logger.info(f"nsys result: {nsys_output_file}")


class MaterializedBuildExecutor(BuildExecutor):
    """
    MaterializedBuildExecutor is a simple BuildExecutor that runs a single
    materialized (all the macros are expanded) build config with a list of
    materialized input configs. It simply iterate through them and run
    OpExecutor on each config.
    """

    def __init__(self, run_options: Dict[str, Any]):
        super(MaterializedBuildExecutor, self).__init__()
        self.run_options = run_options
        self.out_stream = run_options["out_stream"]

        self.build_input_config = None
        self.op_config = None
        self.config_build_id = None

    def run(
        self,
        op_config: OperatorConfig,
        build_input_config: Dict[str, Any],
        config_build_id: str,
    ):
        self.build_input_config = build_input_config
        self.op_config = op_config
        self.config_build_id = config_build_id
        if "build" not in self.build_input_config:
            self.build_input_config["build"] = None

        # Reset operator to clear memory before new build
        self.op_config.op.cleanup()
        build_config = self.build_input_config["build"]
        logger.debug(build_config)
        if build_config is not None:
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
            counter += 1

            config_run_id = f"{self.op_config.name}|{self.config_build_id}|{input_id}"
            transition_state = self.get_transition_state(config_run_id)
            if transition_state == BenchmarkTransitionState.SKIP:
                continue
            elif transition_state == BenchmarkTransitionState.STOP:
                # break here to finsh the remaining work before raising the stop
                # exception below
                break

            self._run_for_input(input_id, input_config)

        # Raise exception if we need to stop running.
        if transition_state == BenchmarkTransitionState.STOP:
            raise StopBenchmarkException(f"Benchmark stopped at [{config_run_id}]")

    def _run_for_input(self, input_id: str, input_config: Dict[str, Any]):
        run_id = f"{self.config_build_id}|{input_id}"

        logger.info(f"run_id: [{run_id}]")
        if self.run_options["device"].startswith("cuda"):
            log_cuda_memory_usage()
        logger.debug(f"input_config: {input_config}")

        # generate input data only if we actually need to run it
        if self.run_options["warmup"] != 0 or self.run_options["iteration"] != 0:
            input_data_gen = self.op_config.input_data_generator()
            (input_args, input_kwargs) = input_data_gen.get_data(
                input_config, self.run_options["device"]
            )

            op_exe = OpExecutor(
                self.op_config.name, self.op_config.op, self.run_options
            )

            metrics = op_exe.run(input_args, input_kwargs, run_id)
        else:
            metrics = {}

        # clean up inputs
        input_args = None
        input_kwargs = None
        # Reset operator outputs to clear memory
        self.op_config.op.cleanup()
        free_torch_cuda_memory()

        # print(result)
        final_config = {
            "build": self.build_input_config["build"],
            "input": input_config,
        }

        output_stats(
            self.out_stream, self.op_config.name, run_id, metrics, final_config
        )

        logger.debug(f"finished running [{run_id}].")


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
            total = 0
            avg = 0
            if records:
                total = sum(records)
                avg = total / len(records)
            if metric_name.endswith(".time"):
                logger.info(
                    f"metric: {metric_name}, average: {avg:.3f} ms, total: {total:.3f} ms"
                )
                logger.info(f"{format_float_val_list(records, 3)}")
            elif metric_name.endswith(".memory"):
                logger.info(
                    f"metric: {metric_name}, average: {avg:.3f} MB, total: {total:.3f} MB"
                )
                logger.info(f"{format_float_val_list(records, 3)}")

    stats = {
        "op_name": name,
        "id": run_id,
        "metric": metrics,
        "config": config,
    }
    out_stream.write(json.dumps(stats) + "\n")
    out_stream.flush()


def format_float_val_list(time_records: List[float], decimals: int = 3):
    format_str = f"{{0:.{decimals}f}}"
    return f"[{', '.join([format_str.format(i) for i in time_records])}]"
