import abc
import enum
import json
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import TextIO
from typing import Type

import torch

from ..config import OperatorConfig, BenchmarkConfig
from ..init_helper import get_logger, load_package
from ..iterator import ConfigIterator
from ..operator import OperatorInterface
from .build_executor import BuildExecutor, OpBuildExecutor
from .config_util import ExecutionPass
from .op_executor import OpExecutor
from .timer import Timer, format_float_val_list

logger = get_logger()


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
        BuildExecutor.set_resume_op_run_id(self.run_options["resume_op_run_id"])

    def run(self):
        for op_config in self.bench_config.op_configs:
            self.run_op(op_config)

    def run_op(self, op_config: OperatorConfig) -> List[str]:
        config_id = 0
        for config in op_config.info["config"]:
            op_run_id = str(config_id)
            if "input" not in config:
                logger.error(
                    f"{op_config.name} has no input configurations defined, skipped."
                )
                return

            logger.info(f"{op_config.name}:")
            generate_build_config = None
            if op_config.build_iterator and "build" in config:
                if config["build"]:
                    generate_build_config = op_config.build_iterator(
                        config, "build", self.run_options["device"]
                    )

            build_input_config = {}
            if generate_build_config:
                for (build_id, build_config) in generate_build_config:
                    op_run_id += f":{build_id}"
                    build_input_config["build"] = build_config
                    build_input_config["input"] = config["input"]
                    build_exe = self.build_executor(
                        build_input_config,
                        op_config,
                        self.bench_config.run_options,
                        op_run_id,
                    )
                    build_exe.run()
            else:
                op_run_id += ":0"
                build_input_config["build"] = []
                build_input_config["input"] = config["input"]
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
