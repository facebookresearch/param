import logging

from ..lib.init_helper import init_logging, load_modules

# Initialize logging format before loading all other modules
logger = init_logging(logging.INFO)

import argparse
import json
import os
from multiprocessing import shared_memory
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import TextIO
from typing import Type

from ..lib import pytorch as lib_pytorch
from ..lib.config import OperatorConfig, make_op_config
from ..lib.pytorch.build_executor import MaterializedBuildExecutor
from ..lib.pytorch.config_util import (
    ExecutionPass,
    get_benchmark_options,
    create_bench_config,
    create_data,
)
from ..lib.pytorch.op_executor import OpExecutor
from ..workloads import pytorch as workloads_pytorch


def main():
    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument(
        "-s",
        "--shm",
        type=str,
        required=False,
        help="The shared memory buffer name for the config.",
    )
    parser.add_argument(
        "-f", "--file", type=str, required=False, help="The file name for the config."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase log output verbosity."
    )

    args = parser.parse_args()

    if args.verbose:
        init_logging(logging.DEBUG)

    if args.shm:
        print(args.shm)
        shm = shared_memory.SharedMemory(args.shm)
        config = json.loads(bytes(shm.buf[:]).decode("utf-8", "strict"))
        shm.close()
    elif args.file:
        with open(args.file) as config_file:
            config = json.load(config_file)
    else:
        logger.info("No inputs provided.")
        return

    op_name = config["op_name"]
    build_id = config["build_id"]
    op_info = config["op_info"]
    run_options = config["run_options"]

    logger.debug(f"op_name: {op_name}")
    logger.debug(f"build_id: {build_id}")
    logger.debug(f"op_info: {op_info}")
    logger.debug(f"run_options: {run_options}")

    run_options["pass_type"] = ExecutionPass(run_options["pass_type"])

    op_config = make_op_config(op_name, op_info, run_options["device"])
    build_input_config = op_info["config"][0]

    # Don't need to write out anything.
    with open(os.devnull, "w") as out_stream:
        run_options["out_stream"] = out_stream
        build_exe = MaterializedBuildExecutor(
            build_input_config, op_config, run_options, build_id
        )
        build_exe.run()


if __name__ == "__main__":
    main()
