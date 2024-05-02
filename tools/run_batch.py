import logging

from ..lib.init_helper import init_logging, load_modules

# Initialize logging format before loading all other modules
logger = init_logging(logging.INFO)

import argparse
import json
import os
from multiprocessing import resource_tracker, shared_memory

from ..lib import pytorch as lib_pytorch
from ..lib.config import make_op_config
from ..lib.pytorch.build_executor import MaterializedBuildExecutor
from ..lib.pytorch.config_util import ExecutionPass, OpExecutionMode
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
        """
        Shared memory has a bug to proper track and release memory, see
        https://bugs.python.org/issue39959
        Fixed PR: https://github.com/python/cpython/pull/20136
        Workaround: unregister from resource_tracker.
        """
        shm = shared_memory.SharedMemory(args.shm)
        logger.debug(f"shared memory: {shm.name}")
        resource_tracker.unregister(shm._name, "shared_memory")
        config = json.loads(bytes(shm.buf[:]).decode("utf-8", "strict"))
        shm.close()
    elif args.file:
        with open(args.file) as config_file:
            config = json.load(config_file)
    else:
        logger.info("no inputs provided.")
        return

    op_name = config["op_name"]
    config_build_id = config["config_build_id"]
    op_info = config["op_info"]
    run_options = config["run_options"]

    logger.debug(f"op_name: {op_name}")
    logger.debug(f"config_build_id: {config_build_id}")
    logger.debug(f"op_info: {op_info}")
    logger.debug(f"run_options: {run_options}")

    run_options["pass_type"] = ExecutionPass(run_options["pass_type"])
    run_options["op_exec_mode"] = OpExecutionMode(run_options["op_exec_mode"])

    op_config = make_op_config(op_name, op_info, run_options["device"])
    build_input_config = op_info["config"][0]

    # Don't need to write out anything.
    with open(os.devnull, "w") as out_stream:
        run_options["out_stream"] = out_stream
        build_exe = MaterializedBuildExecutor(run_options)
        build_exe.run(op_config, build_input_config, config_build_id)


if __name__ == "__main__":
    main()
