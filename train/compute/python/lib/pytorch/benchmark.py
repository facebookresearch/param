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
from .timer import Timer, format_time_list

try:
    import nvtx

    HAS_NVTX = True
except ModuleNotFoundError as error:
    HAS_NVTX = False
    logging.warn(f"Failed to import NVTX, will not emit NVTX range info.")


@enum.unique
class ExecutionPass(enum.Enum):
    # Forward pass will always run (also require for backward pass).
    FORWARD = 0
    # Run backward pass in addition to forward pass.
    BACKWARD = 1


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


def benchmark_op(tag: str, id: str, op: Callable, args: Any, kwargs: Any, device: str):
    logging.debug(f"running {tag}")

    # flush cache
    if device.startswith("cuda"):
        _clear_cache()
        if HAS_NVTX:
            tag_rng = nvtx.start_range(domain="param_bench", message=tag)
            id_rng = nvtx.start_range(domain="param_bench", message=id)

    with Timer(device) as timer:
        op(*args, **kwargs)

    if device.startswith("cuda") and HAS_NVTX:
        nvtx.end_range(id_rng)
        nvtx.end_range(tag_rng)

    return timer.elapsed_time()


def benchmark_loop(
    tag: str, iterations: int, op_id: str, pass_type, op, args, kwargs, device
):
    fw_time_records = []
    bw_time_records = []
    for _ in range(iterations):
        latency = benchmark_op(
            tag, f"{op_id}[{ExecutionPass.FORWARD}]", op.forward, args, kwargs, device
        )
        fw_time_records.append(latency)
        if pass_type == ExecutionPass.BACKWARD:
            op.create_grad()
            latency = benchmark_op(
                tag, f"{op_id}[{ExecutionPass.BACKWARD}]", op.backward, [], {}, device
            )
            bw_time_records.append(latency)
    return (fw_time_records, bw_time_records)


def warmup(
    op_name: str,
    pass_type: ExecutionPass,
    id: str,
    op: OperatorInterface,
    args: Any,
    kwargs: Any,
    device: str,
    warmup_count: int,
):
    op_id = f"[{op_name}][{id}]"
    logging.info(f"  Running {op_id} for {warmup_count} warm up iterations")

    # warm up
    fw_time_records, bw_time_records = benchmark_loop(
        "warmup", warmup_count, op_id, pass_type, op, args, kwargs, device
    )

    logging.info(f"    fw_warm: {format_time_list(fw_time_records, 6)}")
    logging.info(f"    bw_warm: {format_time_list(bw_time_records, 6)}")


def collect_metric(
    op_name: str,
    pass_type: ExecutionPass,
    id: str,
    op: OperatorInterface,
    args: Any,
    kwargs: Any,
    device: str,
    iterations: int,
    config: Dict[str, Any],
    out_stream: TextIO,
):
    def output_stats(time_records, pass_name):
        nonlocal op_name, id, device, iterations, config, out_stream
        total = sum(time_records)
        logging.info(f"    time: {format_time_list(time_records, 6)}")
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

    op_id = f"[{op_name}][{id}]"
    logging.info(f"  Running {op_id} for {iterations} measured iterations")

    fw_time_records, bw_time_records = benchmark_loop(
        "metric", iterations, op_id, pass_type, op, args, kwargs, device
    )

    pass_name = str(ExecutionPass.FORWARD)
    logging.info(f"    pass: {pass_name}")
    output_stats(fw_time_records, pass_name)
    if pass_type == ExecutionPass.BACKWARD:
        pass_name = str(ExecutionPass.BACKWARD)
        logging.info(f"    pass: {pass_name}")
        output_stats(bw_time_records, pass_name)


def run_op_for_inputs(
    config: Dict[str, Any],
    op_config,
    device: str,
    config_id: str,
    build_id: str,
    warmup_count: int,
    iterations: int,
    pass_type: ExecutionPass,
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

        # Warm up forward (and maybe backward depending on pass_type).
        warmup(
            op_config.name,
            pass_type,
            id,
            op_config.op,
            input_args,
            input_kwargs,
            device,
            warmup_count,
        )

        final_config = {"build": config["build"], "input": input_config}

        collect_metric(
            op_config.name,
            pass_type,
            id,
            op_config.op,
            input_args,
            input_kwargs,
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
    out_stream: TextIO,
):
    """
    Run an operator based on its op_config.

    """

    config_id = 0
    for config in op_config.config:
        if "input" not in config:
            logging.error(
                f"{op_config.name} has no input configurations defined, skipped."
            )
            return

        logging.info(f"{op_config.name}:")
        # build op
        build_config = []
        build_input_config = {}
        generate_build_config = None
        if op_config.build_iterator and "build" in config:
            if config["build"]:
                generate_build_config: ConfigIterator = op_config.build_iterator(
                    config, "build", device
                )

        if generate_build_config:
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
                out_stream,
            )

        config_id += 1
