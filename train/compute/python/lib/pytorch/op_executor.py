from ..init_helper import get_logger, load_package

logger = get_logger()

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import torch
from torch.autograd.profiler import record_function

from ..operator import OperatorInterface
from .config_util import ExecutionPass
from .timer import Timer


# NVTX is used to mark ranges for benchmark GPU kernels.
# It's use to correlate operator configurations and metrics collected from
# NSight tools.
USE_NVTX = load_package("nvtx")
if USE_NVTX:
    import nvtx


def _clear_cache():
    L2_cache_size = {
        70: 6 * 1024 * 1024,  # V100 6 MB L2 cache
        80: 40 * 1024 * 1024,  # A100 40 MB L2 cache
    }
    capability = torch.cuda.get_device_capability()
    device_type = capability[0] * 10 + capability[1]

    with record_function("__param_bench__:_clear_cache"):
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
        result[ExecutionPass.FORWARD.value] = {}
        if self.pass_type == ExecutionPass.BACKWARD:
            result[ExecutionPass.BACKWARD.value] = {}

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
    ) -> float:
        logger.debug(f"benchmarking {self.name} {tag} {op_run_id}")
        # flush cache
        if self.device.startswith("cuda"):
            _clear_cache()
            if USE_NVTX:
                tag_rng = nvtx.start_range(domain="param_bench", message=tag)
                op_run_id_rng = nvtx.start_range(domain=self.name, message=op_run_id)

        with record_function("__param_bench__:_benchmark_op"):
            with Timer(self.device) as timer:
                op(*args, **kwargs)

        if self.device.startswith("cuda") and USE_NVTX:
            nvtx.end_range(op_run_id_rng)
            nvtx.end_range(tag_rng)

        # Return result in milliseconds.
        return timer.elapsed_time()

    def _benchmark_loop(
        self, count: int, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ) -> Tuple[List[float], List[float]]:
        fw_time_records = []
        bw_time_records = []
        for _ in range(count):
            op_run_pass = f"{op_run_id}:{ExecutionPass.FORWARD.value}"
            latency = self._benchmark_op(
                self.op.forward, args, kwargs, tag, op_run_pass
            )
            fw_time_records.append(latency)
            if self.pass_type == ExecutionPass.BACKWARD:
                self.op.create_grad()
                op_run_pass = f"{op_run_id}:{ExecutionPass.BACKWARD.value}"
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
        logger.info(f"running [{op_run_id}] for {iteration} {tag} iteration")

        (fw_time_records, bw_time_records) = self._benchmark_loop(
            iteration, args, kwargs, tag, op_run_id
        )

        metric_name = tag + ".time"
        pass_name = ExecutionPass.FORWARD.value
        result[pass_name][metric_name] = fw_time_records
        if self.pass_type == ExecutionPass.BACKWARD:
            pass_name = ExecutionPass.BACKWARD.value
            result[pass_name][metric_name] = bw_time_records
