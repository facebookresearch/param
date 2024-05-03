import argparse
import json
import logging
import os
from multiprocessing import resource_tracker, shared_memory

from param.comp import pytorch as lib_pytorch
from param.comp.config import make_op_config
from param.comp.init_helper import init_logging, load_modules
from param.comp.pytorch.build_executor import MaterializedBuildExecutor
from param.comp.pytorch.config_util import ExecutionPass, OpExecutionMode
from param.comp.workloads import pytorch as workloads_pytorch


def main():
    """
    Entry point for the microbenchmark application. This function handles command-line
    arguments, initializes necessary components, and controls the execution flow
    based on provided input (shared memory or file).
    """
    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)
    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument("-s", "--shm", type=str, required=False, help="The shared memory buffer name for the config.")
    parser.add_argument("-f", "--file", type=str, required=False, help="The file name for the config.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase log output verbosity.")

    args = parser.parse_args()

    if args.verbose:
        init_logging(logging.DEBUG)

    if args.shm:
        shm = shared_memory.SharedMemory(name=args.shm)
        logging.debug(f"Shared memory: {shm.name}")
        resource_tracker.unregister(shm._name, "shared_memory")
        config = json.loads(bytes(shm.buf[:]).decode("utf-8"))
        shm.close()
    elif args.file:
        with open(args.file) as config_file:
            config = json.load(config_file)
    else:
        logging.info("No inputs provided.")
        return

    op_name = config["op_name"]
    config_build_id = config["config_build_id"]
    op_info = config["op_info"]
    run_options = config["run_options"]

    logging.debug(f"Op name: {op_name}")
    logging.debug(f"Config build ID: {config_build_id}")
    logging.debug(f"Op info: {op_info}")
    logging.debug(f"Run options: {run_options}")

    run_options["pass_type"] = ExecutionPass(run_options["pass_type"])
    run_options["op_exec_mode"] = OpExecutionMode(run_options["op_exec_mode"])

    op_config = make_op_config(op_name, op_info, run_options["device"])
    build_input_config = op_info["config"][0]

    with open(os.devnull, "w") as out_stream:
        run_options["out_stream"] = out_stream
        build_exe = MaterializedBuildExecutor(run_options)
        build_exe.run(op_config, build_input_config, config_build_id)


if __name__ == "__main__":
    main()
