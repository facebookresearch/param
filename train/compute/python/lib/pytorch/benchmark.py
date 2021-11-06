import enum
import json
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import TextIO

import torch

from ..config import OperatorConfig
from ..iterator import ConfigIterator
from ..operator import OperatorInterface
from .timer import Timer


@enum.unique
class ExecutionPass(enum.Enum):
    FORWARD = 0
    BACKWARD = 1


def _clear_cache():
    # TODO lofe: update L2 cache size depending on GPU
    L2_cache_size = {
        70: 6 * 1024 * 1024,  # V100 6 MB L2 cache
        80: 40 * 1024 * 1024,  # A100 40 MB L2 cache
    }
    capability = torch.cuda.get_device_capability()
    device_type = capability[0] * 10 + capability[1]
    _ = torch.zeros(L2_cache_size[device_type] // 4).float() * 2
    del _
    torch.cuda.empty_cache()


def benchmark_op(
    op_id: str, op: Callable, args: Any, kwargs: Any, device: str, iterations: int
):
    time_records = []
    for _ in range(iterations):
        # flush cache
        # TODO lofe: update cache size depending on GPU
        if device.startswith("cuda"):
            _clear_cache()

        logging.debug(f"running {op_id}")
        with Timer(device) as timer:
            op(*args, **kwargs)

        time_records.append(timer.elapsed_time())
    return time_records


def collect_metric(
    op_name: str,
    pass_name: str,
    id: str,
    op: Callable,
    args: Any,
    kwargs: Any,
    device: str,
    iterations: int,
    config: Dict[str, Any],
    out_stream: TextIO,
):
    if device.startswith("cuda"):
        # use nvtx allows us to collect only this part of kernel executions
        # and match op and arg variants to metrics.
        logging.info(f"  Running {op_name}[{id}] ({pass_name}) for {iterations} metric collection iterations")
        torch.cuda.nvtx.range_push("op_bench")
        for _ in range(iterations):
            # flush cache
            _clear_cache()

            torch.cuda.nvtx.range_push(f"{op_name}[{id}]")
            op(*args, **kwargs)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        stats = {
            "name": op_name,
            "id": id,
            "pass": pass_name,
            "device": device,
            "iter": iterations,
            "config": config,
        }
        out_stream.write(json.dumps(stats) + "\n")
        out_stream.flush()
    else:
        raise Exception("Non-GPU metric mode is not supported.")


def warmup(
    op_name: str,
    id: str,
    op: Callable,
    args: Any,
    kwargs: Any,
    device: str,
    warmup_count: int,
):
    logging.info(f"  Running {op_name}[{id}] for {warmup_count} warm up iterations")
    # warm up
    time_records = benchmark_op(
        f"{op_name}[{id}]", op, args, kwargs, device, warmup_count
    )
    logging.info(f"    warm: {time_records}")


def measure_latency(
    op_name: str,
    pass_name: str,
    id: str,
    op: OperatorInterface,
    args: Any,
    kwargs: Any,
    device: str,
    iterations: int,
    config: Dict[str, Any],
    out_stream: TextIO,
):
    logging.info(f"  Running {op_name}[{id}] ({pass_name}) for {iterations} measured iterations")
    torch.cuda.nvtx.range_push("op_bench")
    time_records = benchmark_op(
        f"{op_name}[{id}]", op, args, kwargs, device, iterations
    )
    torch.cuda.nvtx.range_pop()
    total = sum(time_records)
    logging.info(f"    pass: {pass_name}")
    logging.info(f"    time: {time_records}")
    logging.info(f"    avg: {total/iterations:.6f} sec")
    logging.info(f"    tot: {total:.6f} sec")
    stats = {
        "name": op_name,
        "id": id,
        "pass": pass_name,
        "device": device,
        "time": time_records,
        "iter": iterations,
        "config": config,
    }
    out_stream.write(json.dumps(stats) + "\n")
    out_stream.flush()


def run_op_for_inputs(
    config: Dict[str, Any],
    op_config,
    device: str,
    config_id: str,
    build_id: str,
    warmup_count: int,
    iterations: int,
    pass_type: ExecutionPass,
    metric_mode: bool,
    out_stream: TextIO,
):
    generate_input_config: ConfigIterator = op_config.input_iterator(
        config, "input", device
    )

    for (input_id, input_config) in generate_input_config:
        logging.info(
            f"    input_config [{config_id}:{build_id}:{input_id}]: {input_config}"
        )
        # generate data

        input_data_gen = op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(input_config, device)
        id = f"{config_id}:{build_id}:{input_id}"

        warmup(
            op_config.name,
            id,
            op_config.op.forward,
            input_args,
            input_kwargs,
            device,
            warmup_count,
        )

        final_config = {"build": config["build"], "input": input_config}

        if metric_mode:
            # collect CUDA metrics
            benchmark_func = collect_metric
        else:
            benchmark_func = measure_latency

        benchmark_func(
            op_config.name,
            str(ExecutionPass.FORWARD),
            id,
            op_config.op.forward,
            input_args,
            input_kwargs,
            device,
            iterations,
            final_config,
            out_stream,
        )

        if pass_type == ExecutionPass.BACKWARD:
            op_config.op.create_grad()
            benchmark_func(
                op_config.name,
                str(ExecutionPass.BACKWARD),
                id,
                op_config.op.backward,
                [],
                {},
                device,
                iterations,
                final_config,
                out_stream,
            )

        logging.debug(f"Finished running {op_config.name}[{id}].")


def run_op(
    op_config: OperatorConfig,
    warmup_count: int,
    iterations: int,
    device: str,
    pass_type: ExecutionPass,
    metric_mode: bool,
    out_stream: TextIO,
):
    """
    Run an operator based on its op_config.

    """

    config_id = 0
    for config in op_config.config:
        if "input" not in config:
            logging.error(
                f"{op_config.name} has no input configureations defined, skipped."
            )
            return

        logging.info(f"{op_config.name}:")
        # build op
        build_config = []
        build_input_config = {}
        if op_config.build_iterator and "build" in config:
            generate_build_config: ConfigIterator = op_config.build_iterator(
                config, "build", device
            )

            for (build_id, build_config) in generate_build_config:
                logging.info(f"  build_config [{config_id}:{build_id}]: {build_config}")
                build_data_gen = op_config.build_data_generator()
                (build_args, build_kwargs) = build_data_gen.get_data(
                    build_config, device
                )
                logging.debug(f"{build_args} {build_kwargs}")
                # reset operator to clear memory before new build
                op_config.op.cleanup()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(build_args)
                op_config.op.build(*build_args, **build_kwargs)
                build_input_config["build"] = build_config
                build_input_config["input"] = config["input"]
                run_op_for_inputs(
                    build_input_config,
                    op_config,
                    device,
                    config_id,
                    build_id,
                    warmup_count,
                    iterations,
                    pass_type,
                    metric_mode,
                    out_stream,
                )
        else:
            logging.info(f"  build_config [{config_id}:{0}]: {build_config}")
            build_input_config["build"] = build_config
            build_input_config["input"] = config["input"]
            run_op_for_inputs(
                build_input_config,
                op_config,
                device,
                config_id,
                0,
                warmup_count,
                iterations,
                pass_type,
                metric_mode,
                out_stream,
            )

        config_id += 1
