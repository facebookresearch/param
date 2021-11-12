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
from .timer import Timer, format_time_list

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
        "iterations": 1,
        "resume": None
    }

    return options


def _clear_cache(self):
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
    iterations etc.) and execute the actual operator benchmark. It will return
    a dictionary of collected metric results.
    """
    def __init__(self, op: OperatorInterface, run_options: Dict[str, Any]):
        self.op = op
        self.device = run_options["device"]
        self.iterations = run_options["iterations"]
        self.warmup = run_options["warmup"]
        self.pass_type = run_options["pass_type"]

    def run(self, input_args: List, input_kwargs: Dict[str, Any], op_run_id: str) -> Dict[str, Any]:
        # Warm up forward (and maybe backward depending on pass_type).
        self._warmup(input_args, input_kwargs, op_run_id)

        return self._collect_metric(input_args, input_kwargs, op_run_id)

    def _benchmark_op(
        self, op: Callable, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ):
        logger.debug(f"benchmarking {tag} {op_run_id}")
        # flush cache
        if self.device.startswith("cuda"):
            _clear_cache()
            if USE_NVTX:
                tag_rng = nvtx.start_range(domain="param_bench", message=tag)
                op_run_id_rng = nvtx.start_range(
                    domain="param_bench", message=op_run_id
                )

        with Timer(self.device) as timer:
            op(*args, **kwargs)

        if self.device.startswith("cuda") and USE_NVTX:
            nvtx.end_range(op_run_id_rng)
            nvtx.end_range(tag_rng)

        return timer.elapsed_time()

    def _benchmark_loop(
        self, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ):
        fw_time_records = []
        bw_time_records = []
        for _ in range(self.iterations):
            op_run_id += f":{ExecutionPass.FORWARD}"
            latency = self._benchmark_op(self.op.forward, args, kwargs, tag, op_run_id)
            fw_time_records.append(latency)
            if self.pass_type == ExecutionPass.BACKWARD:
                self.op.create_grad()
                op_run_id += ":{ExecutionPass.BACKWARD}"
                latency = self._benchmark_op(self.op.backward, [], {}, tag, op_run_id)
                bw_time_records.append(latency)
        return (fw_time_records, bw_time_records)

    def _warmup(self, args: List, kwargs: Dict[str, Any], op_run_id: str):
        logger.info(f"  Running {op_run_id} for {self.warmup} warm up iterations")

        # warm up
        fw_time_records, bw_time_records = self._benchmark_loop(
            args, kwargs, "warmup", op_run_id
        )

        logger.info(f"    fw_warmup: {format_time_list(fw_time_records, 6)}")
        if self.pass_type == ExecutionPass.BACKWARD:
            logger.info(f"    bw_warmup: {format_time_list(bw_time_records, 6)}")

    def _collect_metric(self, args: List, kwargs: Dict[str, Any], op_run_id: str) -> Dict[str, Any]:
        results = {}
        logger.info(f"  Running {op_run_id} for {self.iterations} measured iterations")

        (fw_time_records, bw_time_records) = self._benchmark_loop(
            args, kwargs, "metric", op_run_id
        )

        pass_name = str(ExecutionPass.FORWARD)
        logger.info(f"    pass: {pass_name}")
        results[pass_name] = fw_time_records
        if self.pass_type == ExecutionPass.BACKWARD:
            pass_name = str(ExecutionPass.BACKWARD)
            logger.info(f"    pass: {pass_name}")
            results[pass_name] = bw_time_records

        return results


class BuildExecutor(metaclass=abc.ABCMeta):
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
        self.build_input_config = build_input_config
        self.op_config = op_config
        self.run_options = run_options
        self.op_run_id = op_run_id

    def run_for_input(self, op_run_id: str, input_config: Dict[str, Any]):
        # generate input data
        input_data_gen = self.op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(
            input_config, self.run_options["device"]
        )

        pt_instance = OpExecutor(self.op_config.op, self.run_options)

        result = pt_instance.run(input_args, input_kwargs, op_run_id)
        # print(result)

    def run(self):
        # reset operator to clear memory before new build
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
            logger.info(f"  input_id: [{input_id}]")
            logger.debug(f"  input_config: {input_config}")
            # generate data
            op_run_id = f"{self.op_run_id}:{input_id}"
            self.run_for_input(op_run_id, input_config)

            logger.debug(f"Finished running {self.op_config.name}[{id}].")

    def output_stats(self, time_records, pass_name, iterations, config):
        total = sum(time_records)
        logger.info(f"    time: {format_time_list(time_records, 6)}")
        logger.info(f"    avg: {total/iterations:.6f} sec")
        logger.info(f"    tot: {total:.6f} sec")
        stats = {
            "name": self.op_name,
            "id": self.id,
            "pass": pass_name,
            "device": self.device,
            "time": time_records,
            "iter": iterations,
            "config": config,
        }
        return json.dumps(stats)


class Benchmark:
    def __init__(
        self, bench_config: BenchmarkConfig, executor_class: Type[BuildExecutor]
    ):
        # We don't want too many threads for stable benchmarks
        torch.set_num_threads(1)

        self.bench_config = bench_config
        self.executor_class = executor_class
        self.run_options = bench_config.run_options

    def run(self):
        for op_config in self.bench_config.op_configs:
            self.run_op(op_config)

    def run_op(self, op_config: OperatorConfig) -> List[str]:
        """
        Run an operator based on its op_config.

        """
        config_id = 0
        for config in op_config.config:
            op_run_id = f"{op_config.name}:{config_id}"
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
                    logger.info(f"  build_id [{config_id}:{build_id}]")
                    logger.debug(f"  build_config: {build_config}")
                    build_input_config["build"] = build_config
                    build_input_config["input"] = config["input"]
                    op_run_id += f":{build_id}"
                    exe = self.executor_class(
                        build_input_config,
                        op_config,
                        self.bench_config.run_options,
                        op_run_id,
                    )
                    exe.run()
            else:
                logger.info(f"  build_id: [{config_id}:{0}]")
                logger.debug(f"  build_config: {build_config}")
                build_input_config["build"] = build_config
                build_input_config["input"] = config["input"]
                op_run_id += f":0"
                exe = self.executor_class(
                    build_input_config,
                    op_config,
                    self.bench_config.run_options,
                    op_run_id,
                )
                exe.run()

            config_id += 1


def make_default_benchmark(bench_config: BenchmarkConfig):
    return Benchmark(bench_config, OpBuildExecutor)
