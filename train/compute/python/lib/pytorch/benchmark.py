from ..init_helper import get_logger

logger = get_logger()

from typing import List, Type

from ..config import BenchmarkConfig, OperatorConfig
from .build_executor import BuildExecutor, OpBuildExecutor, StopBenchmarkException
from .config_util import init_pytorch
from .operator_impl import TorchScriptOp

UNSUPPORTED_OPS = [
    "aten::record_stream",
    "aten::to",
    "aten::select",
    "aten::item",
    "aten::cat",
    "aten::split_with_sizes",
]


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
        init_pytorch(bench_config.run_options)
        self.bench_config = bench_config
        self.build_executor = build_executor
        self.run_options = bench_config.run_options

        # Construct a BuildExecutor
        self.build_executor = build_executor(self.run_options)
        self.build_executor.set_resume_op_run_id(self.run_options["resume_op_run_id"])
        self.build_executor.set_stop_op_run_id(self.run_options["stop_op_run_id"])

    def run(self):
        try:
            for op_config in self.bench_config.op_configs:
                if op_config.op is None:
                    if (
                        op_config.name not in UNSUPPORTED_OPS
                        and op_config.name.startswith("aten::")
                    ):
                        logger.info(f"register torchscript op: {op_config.name}")
                        op_config.op = TorchScriptOp(op_config.name)
                        op_config.op.device = self.run_options["device"]

                if op_config.op is not None:
                    self.run_op(op_config)
        except StopBenchmarkException as stop_event:
            logger.info(stop_event)

    def run_op(self, op_config: OperatorConfig) -> List[str]:
        logger.info(f"### op: {op_config.name}")
        config_id = 0
        for config in op_config.info["config"]:
            op_run_id = str(config_id)
            logger.info(f"config_id: [{op_run_id}]")
            if "input" not in config:
                logger.error(
                    f"{op_config.name} has no input configurations defined, skipped."
                )
                return

            generate_build_config = None
            if op_config.build_iterator and "build" in config:
                logger.debug(f"build_config: {config['build']}")
                if config["build"]:
                    generate_build_config = op_config.build_iterator(
                        config, "build", self.run_options["device"]
                    )

            build_input_config = {}
            if generate_build_config:
                logger.debug("generating build config")
                for (build_id, build_config) in generate_build_config:
                    logger.info(f"build_id: [{build_id}]")
                    logger.debug(f"build_config: {build_config}")
                    op_run_id = f"{op_run_id}|{build_id}"
                    build_input_config["build"] = build_config
                    build_input_config["input"] = config["input"]
                    self.build_executor.run(op_config, build_input_config, op_run_id)
            else:
                build_id = "0"
                build_config = config.get("build", None)
                logger.info(f"build_id: [{build_id}]")
                logger.debug(f"build_config: {build_config}")
                op_run_id = f"{op_run_id}|{build_id}"
                build_input_config["build"] = build_config
                build_input_config["input"] = config["input"]
                self.build_executor.run(op_config, build_input_config, op_run_id)

            config_id += 1


def make_default_benchmark(bench_config: BenchmarkConfig):
    return Benchmark(bench_config, OpBuildExecutor)
