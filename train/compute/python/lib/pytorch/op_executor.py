from ..init_helper import get_logger, load_package

logger = get_logger()

from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.autograd.profiler import record_function

from ..operator import OperatorInterface
from .config_util import ExecutionPass, OpExecutionMode
from .timer import Timer


def _clear_cache():
    L2_cache_size = {
        70: 6 * 1024 * 1024,  # V100 6 MB L2 cache
        80: 40 * 1024 * 1024,  # A100 40 MB L2 cache
    }
    capability = torch.cuda.get_device_capability()
    device_type = capability[0] * 10 + capability[1]

    with record_function("[param|clear_cache]"):
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
        self.cuda_l2_cache = run_options["cuda_l2_cache"]
        self.pass_type = run_options["pass_type"]
        self.exec_mode = run_options["op_exec_mode"]
        self.use_cuda = self.device.startswith("cuda")
        self.benchmark_func = {
            OpExecutionMode.DISCRETE: self._benchmark_discrete,
            OpExecutionMode.CONTINUOUS: self._benchmark_continuous,
            OpExecutionMode.CONTINUOUS_EVENTS: self._benchmark_continuous,
        }
        self._label_template_fwd = (
            f"[param|{self.name}|{{op_run_id}}|{{tag}}|{ExecutionPass.FORWARD.value}]"
        )
        self._label_template_bwd = (
            f"[param|{self.name}|{{op_run_id}}|{{tag}}|{ExecutionPass.BACKWARD.value}]"
        )
        self._label_template_fwd_bwd = f"[param|{self.name}|{{op_run_id}}|{{tag}}|{ExecutionPass.FORWARD.value}_{ExecutionPass.BACKWARD.value}]"

    def run(
        self, input_args: List, input_kwargs: Dict[str, Any], op_run_id: str
    ) -> Dict[str, Any]:
        result = {}
        result[ExecutionPass.FORWARD.value] = {}
        if self.pass_type == ExecutionPass.BACKWARD:
            result[ExecutionPass.BACKWARD.value] = {}

        with record_function(f"[param|{self.name}|{op_run_id}]"):
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
        self, op: Callable, args: List, kwargs: Dict[str, Any], tag: str, label_str: str
    ) -> Tuple[float, float]:
        logger.debug(f"benchmarking {label_str}")
        gpu_memory = 0
        timer = Timer(self.device)

        # flush cache
        if self.use_cuda:
            if not self.cuda_l2_cache:
                _clear_cache()
            # Reset to measure peak memory usage
            torch.cuda.reset_peak_memory_stats()

        with record_function(label_str):
            timer.start()
            if self.use_cuda:
                op_run_id_range = torch.cuda.nvtx.range_start(label_str)
            op(*args, **kwargs)
            timer.stop()
            if self.use_cuda:
                torch.cuda.nvtx.range_end(op_run_id_range)
                # Memory size in MB
                gpu_memory = torch.cuda.max_memory_allocated() / (1048576)

        # Return result in milliseconds.
        return timer.elapsed_time_ms(), gpu_memory

    def _benchmark_discrete(
        self, count: int, args: List, kwargs: Dict[str, Any], tag: str, op_run_id: str
    ) -> Tuple[List[float], List[float], List[float], List[float]]:

        logger.debug(f"benchmarking [{self.name}|{op_run_id}|{tag}]")

        fw_time_records = []
        bw_time_records = []
        fw_gpu_mem_records = []
        bw_gpu_mem_records = []

        if self.pass_type == ExecutionPass.BACKWARD:
            label_str = self._label_template_fwd_bwd.format(
                tag=tag, op_run_id=op_run_id
            )
        else:
            label_str = self._label_template_fwd.format(tag=tag, op_run_id=op_run_id)

        if self.use_cuda:
            tag_range = torch.cuda.nvtx.range_start(f"[param|{tag}]")

        with record_function(label_str):
            for _ in range(count):
                label_str = self._label_template_fwd.format(
                    tag=tag, op_run_id=op_run_id
                )
                latency, peak_memory = self._benchmark_op(
                    self.op.forward, args, kwargs, tag, label_str
                )
                fw_time_records.append(latency)
                fw_gpu_mem_records.append(peak_memory)
                if self.pass_type == ExecutionPass.BACKWARD:
                    self.op.create_grad()
                    label_str = self._label_template_bwd.format(
                        tag=tag, op_run_id=op_run_id
                    )
                    latency, peak_memory = self._benchmark_op(
                        self.op.backward, [], {}, tag, label_str
                    )
                    bw_time_records.append(latency)
                    bw_gpu_mem_records.append(peak_memory)

        if self.use_cuda:
            torch.cuda.nvtx.range_end(tag_range)
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
        logger.debug(f"benchmarking [{self.name}|{op_run_id}|{tag}]")

        def create_cuda_start_stop_events(count: int):
            return [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _i in range(count)
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

        tag_range = torch.cuda.nvtx.range_start(f"[param|{tag}]")

        fw_events = create_cuda_start_stop_events(count)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        label_str = self._label_template_fwd.format(tag=tag, op_run_id=op_run_id)
        with record_function(label_str):
            op_run_id_range = torch.cuda.nvtx.range_start(label_str)
            for i in range(count):
                fw_events[i][0].record()
                self.op.forward(*args, **kwargs)
                fw_events[i][1].record()

            torch.cuda.synchronize()
            torch.cuda.nvtx.range_end(op_run_id_range)

        fw_time_records = compute_cuda_event_delta(fw_events)
        fw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

        if self.pass_type == ExecutionPass.BACKWARD:
            self.op.create_grad()

            bw_events = create_cuda_start_stop_events(count)
            torch.cuda.reset_peak_memory_stats()
            label_str = self._label_template_fwd_bwd.format(
                tag=tag, op_run_id=op_run_id
            )
            with record_function(label_str):
                torch.cuda.synchronize()
                op_run_id_range = torch.cuda.nvtx.range_start(label_str)
                for i in range(count):
                    self.op.forward(*args, **kwargs)
                    bw_events[i][0].record()
                    self.op.backward()
                    bw_events[i][1].record()

                torch.cuda.synchronize()
                torch.cuda.nvtx.range_end(op_run_id_range)
            bw_time_records = compute_cuda_event_delta(bw_events)
            bw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

        torch.cuda.nvtx.range_end(tag_range)

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

        logger.debug(f"benchmarking {self.name}|{op_run_id}|{tag}")

        fw_time_records = []
        bw_time_records = []
        fw_gpu_mem_records = []
        bw_gpu_mem_records = []

        tag_range = torch.cuda.nvtx.range_start(f"[param|{tag}]")

        fw_time = 0
        bw_time = 0
        # Always run forward.
        torch.cuda.reset_peak_memory_stats()
        label_str = self._label_template_fwd.format(tag=tag, op_run_id=op_run_id)
        with record_function(label_str):
            timer = Timer(self.device)
            timer.start()
            op_run_id_range = torch.cuda.nvtx.range_start(label_str)
            for _i in range(count):
                self.op.forward(*args, **kwargs)
            timer.stop()
            torch.cuda.nvtx.range_end(op_run_id_range)

        fw_time = timer.elapsed_time_ms() / count

        fw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

        if self.pass_type == ExecutionPass.BACKWARD:
            self.op.create_grad()
            torch.cuda.reset_peak_memory_stats()
            label_str = self._label_template_fwd_bwd.format(
                tag=tag, op_run_id=op_run_id
            )
            with record_function(label_str):
                timer.start()
                op_run_id_range = torch.cuda.nvtx.range_start(label_str)
                for _i in range(count):
                    self.op.forward(*args, **kwargs)
                    self.op.backward()
                timer.stop()
                torch.cuda.nvtx.range_end(op_run_id_range)
            # Subtract forward time to get backward time.
            bw_time = timer.elapsed_time_ms() / count - fw_time
            bw_gpu_mem_records.append(torch.cuda.max_memory_allocated() / (1048576))

        torch.cuda.nvtx.range_end(tag_range)

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
        logger.debug(f"benchmarking [{self.name}|{op_run_id}|{tag}]")

        fw_time_records = []
        fw_gpu_mem_records = []
        bw_time_records = []
        bw_gpu_mem_records = []
        timer = Timer(self.device)
        if self.pass_type == ExecutionPass.FORWARD:
            label_str = self._label_template_fwd.format(tag=tag, op_run_id=op_run_id)
            with record_function(label_str):
                for _i in range(count):
                    timer.start()
                    self.op.forward(*args, **kwargs)
                    timer.stop()
                    fw_time_records.append(timer.elapsed_time_ms())

        elif self.pass_type == ExecutionPass.BACKWARD:
            label_str = self._label_template_fwd_bwd.format(
                tag=tag, op_run_id=op_run_id
            )
            with record_function(label_str):
                for _i in range(count):
                    timer.start()
                    self.op.forward(*args, **kwargs)
                    timer.stop()
                    fw_time_records.append(timer.elapsed_time_ms())
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
        fw_time_records = []
        fw_mem_records = []
        bw_time_records = []
        bw_mem_records = []
        if iteration > 0:
            (
                fw_time_records,
                fw_mem_records,
                bw_time_records,
                bw_mem_records,
            ) = self.benchmark_func[self.exec_mode](
                iteration, args, kwargs, tag, op_run_id
            )

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
