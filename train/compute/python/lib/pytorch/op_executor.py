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
from .config_util import ExecutionPass, OpExecutionMode
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
        self.exec_mode = run_options["op_exec_mode"]
        self.benchmark_func = {
            OpExecutionMode.DISCRETE: self._benchmark_discrete,
            OpExecutionMode.CONTINUOUS: self._benchmark_continuous,
            OpExecutionMode.CONTINUOUS_EVENTS: self._benchmark_continuous,
        }

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
    ) -> Tuple[float, float]:
        logger.debug(f"benchmarking {self.name} {tag} {op_run_id}")
        gpu_memory = 0
        # flush cache
        if self.device.startswith("cuda"):
            _clear_cache()
            # Reset to measure peak memory usage
            torch.cuda.reset_peak_memory_stats()
            if USE_NVTX:
                tag_range = nvtx.start_range(domain="param_bench", message=tag)
                op_run_id_range = nvtx.start_range(domain=self.name, message=op_run_id)

        timer = Timer(self.device)
        with record_function("__param_bench__:_benchmark_op"):
            timer.start()
            op(*args, **kwargs)
            timer.stop()

        if self.device.startswith("cuda") and USE_NVTX:
            nvtx.end_range(op_run_id_range)
            nvtx.end_range(tag_range)
            # Memory size in MB
            gpu_memory = torch.cuda.max_memory_allocated() / (1048576)

        # Return result in milliseconds.
        return timer.elapsed_time_ms(), gpu_memory

    def _benchmark_discrete(
        self, count: int, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        fw_time_records = []
        bw_time_records = []
        fw_gpu_mem_records = []
        bw_gpu_mem_records = []
        for _ in range(count):
            op_run_pass = f"{op_run_id}:{ExecutionPass.FORWARD.value}"
            latency, peak_memory = self._benchmark_op(
                self.op.forward, args, kwargs, tag, op_run_pass
            )
            fw_time_records.append(latency)
            fw_gpu_mem_records.append(peak_memory)
            if self.pass_type == ExecutionPass.BACKWARD:
                self.op.create_grad()
                op_run_pass = f"{op_run_id}:{ExecutionPass.BACKWARD.value}"
                latency, peak_memory = self._benchmark_op(
                    self.op.backward, [], {}, tag, op_run_pass
                )
                bw_time_records.append(latency)
                bw_gpu_mem_records.append(peak_memory)
        return (
            fw_time_records,
            fw_gpu_mem_records,
            bw_time_records,
            bw_gpu_mem_records,
        )

    def _benchmark_loop_cuda_events(
        self,
        count: int,
        args: List,
        kwargs: Dict[str, Any],
        tag: str,
        op_run_id: str,
    ) -> float:
        """
        Using CUDA events to record is making the assumptions that we are running single stream.
        In this mode, we do not flush cache, assuming benefit from data in warmup.
        """

        def create_cuda_start_stop_events(count: int):
            return [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for i in range(count)
            ]

        def compute_cuda_event_delta(events: List[Tuple[Any]]):
            deltas = []
            for event_pair in events:
                deltas.append(event_pair[0].elapsed_time(event_pair[1]))

            return deltas

        fw_time_records = []
        bw_time_records = []
        fw_gpu_mem_records = []
        bw_gpu_mem_records = []

        if USE_NVTX:
            tag_range = nvtx.start_range(domain="param_bench", message=tag)

        with record_function("__param_bench__:_benchmark_op"):
            fw_events = create_cuda_start_stop_events(count)
            torch.cuda.reset_peak_memory_stats()
            if USE_NVTX:
                op_run_id_range = nvtx.start_range(
                    domain=self.name,
                    message=f"{op_run_id}:{ExecutionPass.FORWARD.value}",
                )
            for i in range(count):
                fw_events[i][0].record()
                self.op.forward(*args, **kwargs)
                fw_events[i][1].record()

            torch.cuda.synchronize()
            if USE_NVTX:
                nvtx.end_range(op_run_id_range)
            fw_time_records = compute_cuda_event_delta(fw_events)
            fw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

            if self.pass_type == ExecutionPass.BACKWARD:
                self.op.create_grad()

                bw_events = create_cuda_start_stop_events(count)
                torch.cuda.reset_peak_memory_stats()
                if USE_NVTX:
                    op_run_id_range = nvtx.start_range(
                        domain=self.name,
                        message=f"{op_run_id}:{ExecutionPass.FORWARD.value}_{ExecutionPass.BACKWARD.value}",
                    )
                for i in range(count):
                    self.op.forward(*args, **kwargs)
                    bw_events[i][0].record()
                    self.op.backward()
                    bw_events[i][1].record()

                torch.cuda.synchronize()
                if USE_NVTX:
                    nvtx.end_range(op_run_id_range)
                bw_time_records = compute_cuda_event_delta(bw_events)
                bw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

        if USE_NVTX:
            nvtx.end_range(tag_range)

        # Return result in milliseconds.
        return fw_time_records, fw_gpu_mem_records, bw_time_records, bw_gpu_mem_records

    def _benchmark_loop_cuda(
        self,
        count: int,
        args: List,
        kwargs: Dict[str, Any],
        tag: str,
        op_run_id: str,
    ) -> float:
        fw_time_records = []
        bw_time_records = []
        fw_gpu_mem_records = []
        bw_gpu_mem_records = []
        logger.debug(f"benchmarking {self.name} {tag} {op_run_id}")
        if USE_NVTX:
            tag_range = nvtx.start_range(domain="param_bench", message=tag)

        with record_function("__param_bench__:_benchmark_op"):
            fw_time = 0
            bw_time = 0
            if USE_NVTX:
                op_run_id_range = nvtx.start_range(
                    domain=self.name,
                    message=f"{op_run_id}:{ExecutionPass.FORWARD.value}",
                )

            # Always run forward.
            torch.cuda.reset_peak_memory_stats()
            timer = Timer(self.device)
            timer.start()
            for i in range(count):
                self.op.forward(*args, **kwargs)
            timer.stop()
            fw_time = timer.elapsed_time_ms() / count

            if USE_NVTX:
                nvtx.end_range(op_run_id_range)

            fw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

            if self.pass_type == ExecutionPass.BACKWARD:
                self.op.create_grad()
                torch.cuda.reset_peak_memory_stats()
                if USE_NVTX:
                    op_run_id_range = nvtx.start_range(
                        domain=self.name,
                        message=f"{op_run_id}:{ExecutionPass.FORWARD.value}_{ExecutionPass.BACKWARD.value}",
                    )
                timer.start()
                for i in range(count):
                    self.op.forward(*args, **kwargs)
                    self.op.backward()
                timer.stop()

                # Subtract forward time to get backward time.
                bw_time = timer.elapsed_time_ms() / count - fw_time

                if USE_NVTX:
                    nvtx.end_range(op_run_id_range)

                bw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

        if USE_NVTX:
            nvtx.end_range(tag_range)

        fw_time_records.append(fw_time)
        bw_time_records.append(bw_time)

        # Return result in milliseconds.
        return fw_time_records, fw_gpu_mem_records, bw_time_records, bw_gpu_mem_records

    def _benchmark_loop_cpu(
        self,
        count: int,
        args: List,
        kwargs: Dict[str, Any],
        tag: str,
        op_run_id: str,
    ) -> float:
        logger.debug(f"benchmarking {self.name} {tag} {op_run_id}")

        fw_time_records = []
        fw_gpu_mem_records = []
        bw_time_records = []
        bw_gpu_mem_records = []
        timer = Timer(self.device)
        with record_function("__param_bench__:_benchmark_op"):
            if self.pass_type == ExecutionPass.FORWARD:
                for i in range(count):
                    timer.start()
                    self.op.forward(*args, **kwargs)
                    timer.stop()
                    fw_time_records.append(timer.elapsed_time_ms())

            elif self.pass_type == ExecutionPass.BACKWARD:
                for i in range(count):
                    self.op.forward(*args, **kwargs)
                    self.op.create_grad()
                    timer.start()
                    self.op.backward()
                    timer.stop()
                    bw_time_records.append(timer.elapsed_time_ms())

        # Return result in milliseconds.
        return fw_time_records, fw_gpu_mem_records, bw_time_records, bw_gpu_mem_records

    def _benchmark_continuous(
        self, count: int, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        if self.device.startswith("cpu"):
            return self._benchmark_loop_cpu(count, args, kwargs, tag, op_run_id)
        elif self.device.startswith("cuda"):
            if self.exec_mode == OpExecutionMode.CONTINUOUS:
                return self._benchmark_loop_cuda(count, args, kwargs, tag, op_run_id)
            elif self.exec_mode == OpExecutionMode.CONTINUOUS_EVENTS:
                return self._benchmark_loop_cuda_events(
                    count, args, kwargs, tag, op_run_id
                )
        return [], [], [], []

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

        (
            fw_time_records,
            fw_mem_records,
            bw_time_records,
            bw_mem_records,
        ) = self.benchmark_func[self.exec_mode](iteration, args, kwargs, tag, op_run_id)

        metric_name = tag + ".time"
        pass_name = ExecutionPass.FORWARD.value
        result[pass_name][metric_name] = fw_time_records
        if self.pass_type == ExecutionPass.BACKWARD:
            pass_name = ExecutionPass.BACKWARD.value
            result[pass_name][metric_name] = bw_time_records

        metric_name = tag + ".gpu.memory"
        pass_name = ExecutionPass.FORWARD.value
        result[pass_name][metric_name] = fw_mem_records
        if self.pass_type == ExecutionPass.BACKWARD:
            pass_name = ExecutionPass.BACKWARD.value
            result[pass_name][metric_name] = bw_mem_records
