import json
import logging
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

import torch

from ..config import OperatorConfig
from ..iterator import ConfigIterator
from ..operator import OperatorInterface
from ..timer import Timer


def benchmark_op(
    op_id: str, op: Callable, args: Any, kwargs: Any, device: str, num_iter: int
):
    time_records = []
    for _ in range(num_iter):
        # flush cache
        if device.startswith("cuda"):
            _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
            torch.cuda.empty_cache()

        logging.debug(f"running {op_id}")
        with Timer(device) as timer:
            op.forward(*args, **kwargs)
        time_records.append(timer.elapsed_time())
    return time_records


def collect_metric(
    op_name: str,
    id: str,
    op: Callable,
    args: Any,
    kwargs: Any,
    device: str,
    num_iter: int,
    config: Dict[str, Any],
    out_file: TextIO,
):
    if device.startswith("cuda"):
        # use nvtx allows us to collect only this part of kernel executions
        # and match op and arg variants to metrics.
        logging.info(f"Running {op_name}[{id}] for {num_iter} CUDA metric iterations")
        torch.cuda.nvtx.range_push("op_bench")
        for _ in range(num_iter):
            # flush cache
            _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
            torch.cuda.empty_cache()

            torch.cuda.nvtx.range_push(f"{op_name}[{id}]")
            op(*args, **kwargs)
            torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()
        stats = {"name": op_name, "id": id, "iter": num_iter, "config": config}
        out_file.write(json.dumps(stats) + "\n")
        out_file.flush()
    else:
        raise Exception("Non-GPU metric mode is not supported.")


def warmup(
    op_name: str,
    id: str,
    op: OperatorInterface,
    args: Any,
    kwargs: Any,
    device: str,
    num_iter: int,
):
    logging.debug(f"Running {op_name}[{id}] for {num_iter} warm up iterations")
    # warm up
    time_records = benchmark_op(f"{op_name}[{id}]", op, args, kwargs, device, num_iter)
    logging.info(f"    warm: {time_records}")


def measure_latency(
    op_name: str,
    id: str,
    op: OperatorInterface,
    args: Any,
    kwargs: Any,
    device: str,
    num_iter: int,
    config: Dict[str, Any],
    out_file: TextIO,
):
    logging.debug(f"Running {op_name}[{id}] for {num_iter} measured iterations")
    torch.cuda.nvtx.range_push("op_bench")
    time_records = benchmark_op(f"{op_name}[{id}]", op, args, kwargs, device, num_iter)
    torch.cuda.nvtx.range_pop()
    tot = sum(time_records)
    logging.info(f"    time: {time_records}")
    logging.info(f"    avg: {tot/num_iter:.6f} sec")
    logging.info(f"    tot: {tot:.6f} sec")
    stats = {
        "name": op_name,
        "id": id,
        "time": time_records,
        "iter": num_iter,
        "config": config,
    }
    out_file.write(json.dumps(stats) + "\n")
    out_file.flush()


def run_op_for_inputs(
    config: Dict[str, Any],
    op_config,
    device: str,
    config_id,
    build_id,
    warmup_iter,
    run_iter,
    metric_mode,
    out_file,
):
    generate_input_config: ConfigIterator = op_config.input_iterator(
        config, "input", device
    )

    for (input_id, input_config) in generate_input_config:
        logging.info(f"    input_config [{config_id}:{build_id}:{input_id}]: {input_config}")
        # generate data

        input_data_gen = op_config.input_data_generator()
        (input_args, input_kwargs) = input_data_gen.get_data(input_config, device)
        id = f"{config_id}:{build_id}:{input_id}"
        warmup(
            op_config.name,
            id,
            op_config.op,
            input_args,
            input_kwargs,
            device,
            warmup_iter,
        )

        final_config = {"build": config["build"], "input": input_config}

        # collect CUDA metrics
        if metric_mode:
            collect_metric(
                op_config.name,
                id,
                op_config.op,
                input_args,
                input_kwargs,
                device,
                run_iter,
                final_config,
                out_file,
            )
        else:
            measure_latency(
                op_config.name,
                id,
                op_config.op,
                input_args,
                input_kwargs,
                device,
                run_iter,
                final_config,
                out_file,
            )

        logging.debug(f"Finished running {op_config.name}[{id}].")


def run_op(
    op_config: OperatorConfig,
    warmup_iter: int,
    run_iter: int,
    device: str,
    out_file: TextIO,
    metric_mode: bool,
):
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
                op_config.op.build(*build_args, **build_kwargs)
                build_input_config["build"] = build_config
                build_input_config["input"] = config["input"]
                run_op_for_inputs(
                    build_input_config,
                    op_config,
                    device,
                    config_id,
                    build_id,
                    warmup_iter,
                    run_iter,
                    metric_mode,
                    out_file,
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
                warmup_iter,
                run_iter,
                metric_mode,
                out_file,
            )

        config_id += 1
