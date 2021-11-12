import abc
import enum
import json
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Type

import torch

from ..config import OperatorConfig, BenchmarkConfig
from ..init_helper import get_logger
from ..iterator import ConfigIterator
from ..operator import OperatorInterface
from .timer import Timer, format_float_val_list

logger = get_logger()

try:
    import nvtx

    USE_NVTX = True
except ModuleNotFoundError as error:
    USE_NVTX = False
    logger.warning(f"Failed to import NVTX, will not emit NVTX range info.")

# USE_NVTX = load_package("nvtx")


@enum.unique
class ExecutionPass(enum.Enum):
    # Forward pass will always run (also required for backward pass).
    FORWARD = 0
    # Run backward pass in addition to forward pass.
    BACKWARD = 1


def get_benchmark_options():
    options = {
        "device": "cpu",
        "pass_type": ExecutionPass.FORWARD,
        "warmup": 5,
        "iteration": 10,
        "out_stream": None,
        "resume_op_run_id": None,
    }

    return options


def _clear_cache():
    L2_cache_size = {
        70: 6 * 1024 * 1024,  # V100 6 MB L2 cache
        80: 40 * 1024 * 1024,  # A100 40 MB L2 cache
    }
    capability = torch.cuda.get_device_capability()
    device_type = capability[0] * 10 + capability[1]
    _ = torch.zeros(L2_cache_size[device_type] // 4).float() * 2
    del _
    torch.cuda.empty_cache()


class OpExecutor:
    """
    OpExecutor takes an operator and run options (such as warmups, number of
    iteration etc.) and execute the actual operator benchmark. It will return
    a dictionary of collected metric results.
    """

    def __init__(self, name: str, op: OperatorInterface, run_options: Dict[str, Any]):
        self.name = name
        self.op = op
        self.device = run_options["device"]
        self.iteration = run_options["iteration"]
        self.warmup = run_options["warmup"]
        self.pass_type = run_options["pass_type"]

    def run(
        self, input_args: List, input_kwargs: Dict[str, Any], op_run_id: str
    ) -> Dict[str, Any]:
        result = {}
        result[str(ExecutionPass.FORWARD)] = {}
        if self.pass_type == ExecutionPass.BACKWARD:
            result[str(ExecutionPass.BACKWARD)] = {}

        # Warm up forward (and maybe backward depending on pass_type).
        self._measure(
            input_args, input_kwargs, self.warmup, "warmup", op_run_id, result
        )
        # Actual measurements.
        self._measure(
            input_args, input_kwargs, self.iteration, "measure", op_run_id, result
        )

        return result

    def _benchmark_op(
        self, op: Callable, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ):
        logger.debug(f"benchmarking {self.name} {tag} {op_run_id}")
        # flush cache
        if self.device.startswith("cuda"):
            _clear_cache()
            if USE_NVTX:
                tag_rng = nvtx.start_range(domain="param_bench", message=tag)
                op_run_id_rng = nvtx.start_range(domain=self.name, message=op_run_id)

        with Timer(self.device) as timer:
            op(*args, **kwargs)

        if self.device.startswith("cuda") and USE_NVTX:
            nvtx.end_range(op_run_id_rng)
            nvtx.end_range(tag_rng)

        return timer.elapsed_time()

    def _benchmark_loop(
        self, count: int, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ):
        fw_time_records = []
        bw_time_records = []
        for _ in range(count):
            op_run_pass = f"{op_run_id}:{ExecutionPass.FORWARD}"
            latency = self._benchmark_op(
                self.op.forward, args, kwargs, tag, op_run_pass
            )
            fw_time_records.append(latency)
            if self.pass_type == ExecutionPass.BACKWARD:
                self.op.create_grad()
                op_run_pass = f"{op_run_id}:{ExecutionPass.BACKWARD}"
                latency = self._benchmark_op(self.op.backward, [], {}, tag, op_run_pass)
                bw_time_records.append(latency)
        return (fw_time_records, bw_time_records)

    def _measure(
        self,
        args: List,
        kwargs: Dict[str, Any],
        iteration: int,
        tag: str,
        op_run_id: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info(f"Running [{op_run_id}] for {iteration} {tag} iteration")

        (fw_time_records, bw_time_records) = self._benchmark_loop(
            iteration, args, kwargs, tag, op_run_id
        )

        metric_name = tag + ".time"
        pass_name = str(ExecutionPass.FORWARD)
        result[pass_name][metric_name] = fw_time_records
        if self.pass_type == ExecutionPass.BACKWARD:
            pass_name = str(ExecutionPass.BACKWARD)
            result[pass_name][metric_name] = bw_time_records


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
        op_run_id: str,
    ):
        super(OpBuildExecutor, self).__init__()
        self.build_input_config = build_input_config
        self.op_config = op_config
        self.run_options = run_options
        self.op_run_id = op_run_id
        self.out_stream = run_options["out_stream"]

    def run(self):
        # Reset operator to clear memory before new build
        self.op_config.op.cleanup()
        build_config = self.build_input_config["build"]
        if build_config:
            build_data_gen = self.op_config.build_data_generator()
            (build_args, build_kwargs) = build_data_gen.get_data(
                build_config, self.op_config.device
            )
            logger.debug(f"{build_args} {build_kwargs}")
            self.op_config.op.build(*build_args, **build_kwargs)

        generate_input_config: ConfigIterator = self.op_config.input_iterator(
            self.build_input_config, "input", self.run_options["device"]
        )

        for (input_id, input_config) in generate_input_config:
            logger.info(f"input_id: [{input_id}]")
            logger.debug(f"input_config: {input_config}")
            # generate data
            op_run_id = f"{self.op_run_id}:{input_id}"
            self._run_for_input(op_run_id, input_config)

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

        self._output_stats(op_run_id, metrics, final_config)

    def _output_stats(
        self, op_run_id: str, metrics: Dict[str, Any], config: Dict[str, Any]
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
            "name": self.op_config.name,
            "id": op_run_id,
            "metric": metrics,
            "device": self.run_options["device"],
            "config": config,
        }
        self.out_stream.write(json.dumps(stats) + "\n")
        self.out_stream.flush()


class Benchmark:
    """
    Benchmark is the high level interface to collect metrics for a large number
    of workloads with many configurations. This class does not execute the
    benchmark directly. It takes a BuildExecutor class to create an executor.
    The build executor a build config and operator inputs. The actual execution
    of the workloads are implemented in OpExecutor.

    The reason to have this flexibility is to provide a library interface that
    allows use cases where a small number of build and input configurations can
    be created on the fly and the user simply wants to get the metrics directly
    from OpExecutor.

    In other cases, a derived class of BuildExecutor may implement a parallel
    (multiprocess) way to run the benchmarks, or run additional tool chains.
    All this can be done by implementing a new BuildExecutor and without
    modifying the benchmark logics in the OpExecutor.

    bench_config: contains all the benchmark configurations for the workloads.

    build_executor: a BuildExecutor that takes a concrete build config and operator
    inputs, op configuration, to run and collect benchmark metrics.
    """

    def __init__(
        self, bench_config: BenchmarkConfig, build_executor: Type[BuildExecutor]
    ):
        # We don't want too many threads for stable benchmarks
        torch.set_num_threads(1)

        self.bench_config = bench_config
        self.build_executor = build_executor
        self.run_options = bench_config.run_options

    def run(self):
        for op_config in self.bench_config.op_configs:
            self.run_op(op_config)

    def run_op(self, op_config: OperatorConfig) -> List[str]:
        config_id = 0
        for config in op_config.config:
            op_run_id = str(config_id)
            if "input" not in config:
                logger.error(
                    f"{op_config.name} has no input configurations defined, skipped."
                )
                return

            logger.info(f"{op_config.name}:")
            # build op
            build_config = []
            build_input_config = {}
            generate_build_config = None
            if op_config.build_iterator and "build" in config:
                if config["build"]:
                    generate_build_config: ConfigIterator = op_config.build_iterator(
                        config, "build", self.run_options["device"]
                    )

            if generate_build_config:
                for (build_id, build_config) in generate_build_config:
                    logger.info(f"build_id [{config_id}:{build_id}]")
                    logger.debug(f"build_config: {build_config}")
                    build_input_config["build"] = build_config
                    build_input_config["input"] = config["input"]
                    op_run_id += f":{build_id}"
                    build_exe = self.build_executor(
                        build_input_config,
                        op_config,
                        self.bench_config.run_options,
                        op_run_id,
                    )
                    build_exe.run()
            else:
                logger.info(f"build_id: [{config_id}:{0}]")
                logger.debug(f"build_config: {build_config}")
                build_input_config["build"] = build_config
                build_input_config["input"] = config["input"]
                op_run_id += f":0"
                build_exe = self.build_executor(
                    build_input_config,
                    op_config,
                    self.bench_config.run_options,
                    op_run_id,
                )
                build_exe.run()

            config_id += 1


def make_default_benchmark(bench_config: BenchmarkConfig):
    return Benchmark(bench_config, OpBuildExecutor)
