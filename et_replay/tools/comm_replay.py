# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from typing import Any, List, Namespace

import torch
from et_replay.comm import CommTraceReplayer


def parse_arguments() -> Namespace:
    """
    Parse command line arguments for commTraceReplayBench.

    This function consolidates common and specific arguments required
    for setting up the communication trace replay environment.

    Returns:
        Namespace: A namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="PARAM-Comms Trace Replay Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # Common/Base arguments for all PARAM-Comm benchmarks
    parser.add_argument(
        "--master-ip",
        type=str,
        default=(
            "127.0.0.1"
            if "MASTER_ADDR" not in os.environ
            else os.environ["MASTER_ADDR"]
        ),
        help="The master-IP to coordinate for Pytorch distributed stack",
    )
    parser.add_argument(
        "--master-port",
        type=str,
        default=(
            "29500" if "MASTER_PORT" not in os.environ else os.environ["MASTER_PORT"]
        ),
        help="The master-port to coordinate for Pytorch distributed stack",
    )
    parser.add_argument(
        "--nw-stack",
        type=str,
        default="pytorch-dist",
        help="network stack to be used, supports pytorch-dist",
    )
    parser.add_argument(
        "--dtype",
        type=lambda dtype: getattr(torch, dtype),
        default=torch.float32,
        help="data type for operations",
    )
    parser.add_argument(
        "--log",
        "--log-level",
        type=str,
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="device to place data for collective benchmarking",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl" if torch.cuda.is_available() else "gloo",
        choices=["nccl", "gloo"],
        help="The backend to be used in PyTorch distributed process group",
    )
    parser.add_argument(
        "--z",
        "--blocking",
        type=int,
        default=0,
        choices=[0, 1],
        help="use blocking/non-blocking mode for collectives",
    )
    parser.add_argument(
        "--c", type=int, default=0, choices=[0, 1], help="enable data validation check"
    )
    parser.add_argument(
        "--use-ext-dist",
        "--use-ext-pg",
        action="store_true",
        default=False,
        help="use extend_distributed wrapper",
    )
    parser.add_argument(
        "--init-method",
        "--pg-init-method",
        type=str,
        default=None,
        help="URL specifying how to initialize the process group",
    )
    parser.add_argument(
        "--enable-local-report",
        action="store_true",
        default=False,
        help="Toggle to enable all nodes' local rank report the output",
    )
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        default=False,
        help="toggle to enable pytorch profiler",
    )
    parser.add_argument(
        "--use-perf-logger",
        "--use-custom-perf-logger",
        nargs="+",
        type=str,
        default=None,
        help="add name of custom performer loggers to use them in addition to text output",
    )
    parser.add_argument(
        "--ibv-devices",
        type=str,
        default="",
        help="list of ib devices to use for distributed communication",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        default=False,
        help="Toggle to skip running collectives and only do initialization",
    )

    # Specific arguments for commTraceReplayBench
    parser.add_argument(
        "--trace-path",
        type=str,
        default="./",
        help="File path to read the trace. All rank read their own trace file unless `--use-one-trace` is used.",
    )
    parser.add_argument(
        "--trace-type",
        type=str,
        default="basic",
        help="Trace type used for replay. Supported trace types: basic. By default use basic trace.",
    )
    parser.add_argument(
        "--use-one-trace",
        action="store_true",
        default=False,
        help="Toggle to use only one trace for all ranks",
    )
    parser.add_argument(
        "--disable-parallel-read",
        action="store_true",
        default=False,
        help="Disable parallel read from input trace path. Instead, rank 0 will read and broadcast to other ranks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Toggle to only analyze trace without actually replaying collectives",
    )
    parser.add_argument(
        "--auto-shrink",
        action="store_true",
        default=False,
        help="Toggle to shrink message size when it does not match with the current scale (only for debug purpose)",
    )
    parser.add_argument(
        "--max-msg-cnt",
        type=int,
        default=0,
        help="Only replay first N operations (0 means no limit)",
    )
    parser.add_argument(
        "--do-warm-up",
        action="store_true",
        default=False,
        help="Toggle to disable performing extra replaying for warm-up",
    )
    parser.add_argument(
        "--reuse-tensors",
        action="store_true",
        default=False,
        help="Toggle to cache and reuse the same input/output for each compute kernel",
    )
    parser.add_argument(
        "--allow-ops",
        "--allow-list",
        type=str,
        default="all",
        help="List of desired collectives (separate by comma) to be replayed",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        nargs="?",
        const="",
        help="Output path to write the replayed trace for post performance analysis. Set as empty string to skip output",
    )
    parser.add_argument(
        "--output-ranks",
        type=str,
        default="all",
        help="List of ranks separated by comma or a range specified by start:end to generate replayed trace for post performance analysis. Default including all ranks.",
    )
    parser.add_argument(
        "--colls-per-batch",
        type=int,
        default=1,
        help="Toggle to set number of consecutive collectives in a batch. This also enables per batch latency stats.",
    )
    parser.add_argument(
        "--use-timestamp",
        action="store_true",
        default=False,
        help="Toggle to use time-based replay.",
    )
    parser.add_argument(
        "--num-replays",
        type=int,
        default=1,
        help="Number of times to replay the given trace, used to get more accurate replay for small traces.",
    )
    parser.add_argument(
        "--profiler-num-replays-start",
        type=int,
        default=1,
        help="Replay iteration to start collecting profiler after warmup. Default start from 1 replay if --enables-profiler is True",
    )
    parser.add_argument(
        "--profiler-num-replays",
        type=int,
        default=1,
        help="Number of replay iterations to collect profiler. Default profile 1 replays if --enables-profiler is True.",
    )

    args, _ = parser.parse_known_args()
    return args


def graceful_exit(message: Any = None) -> None:
    """
    Gracefully exits the program if a fatal error is encountered.

    Args:
        message: Optional message to log before exiting.

    Returns:
        None: Exits the program.
    """
    if message:
        logging.error(message)
    sys.exit(1)


def validate_arguments(
    args: Namespace, valid_trace_types: List[str] = ["basic", "et", "kineto"]
) -> None:
    """
    Validate command line arguments for commTraceReplayBench.

    This function checks the validity of the provided command line arguments
    and raises errors or logs warnings if any discrepancies are found.

    Args:
        args: Namespace containing collection of args to validate.
        valid_trace_types: List of valid trace types supported by the system.

    Returns:
        None: This function raises exceptions or logs errors and terminates
              the program on invalid arguments.
    """
    # Validate network stack
    if args.nw_stack not in args.supportedNwstacks:
        graceful_exit(
            f"Specified backend: {args.nw_stack} is not one of the supported backends: {args.supportedNwstacks}."
        )

    # Set logging level
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        graceful_exit(f"Invalid log level: {args.log}")

    # Set logging configuration
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    # Validate master IP and Port
    check_master_env("MASTER_ADDR", args.master_ip, "127.0.0.1")
    check_master_env("MASTER_PORT", args.master_port, "29500")

    # Specific checks for trace replay
    if not (os.path.isfile(args.trace_file) or os.path.isdir(args.trace_file)):
        graceful_exit(
            f"The specified trace path '{args.trace_file}' is neither a file nor a directory."
        )

    if args.trace_type not in valid_trace_types:
        graceful_exit(
            f"Trace type {args.trace_type} is not valid! Supported trace types: {valid_trace_types}."
        )


def check_master_env(env_var: str, arg_value: str, default_value: str) -> None:
    """
    Check and log master IP or Port based on environment variables and arguments.

    Args:
        env_var: Name of the environment variable (e.g., 'MASTER_ADDR' or 'MASTER_PORT').
        arg_value: Command line argument value for the master IP or Port.
        default_value: Default value if neither environment variable nor argument is set.

    Returns:
        None: Adjusts environment variables and logs as needed.
    """
    if env_var in os.environ and arg_value not in (default_value, os.environ[env_var]):
        logging.warning(
            f"--{env_var.lower()}={arg_value} while {env_var}={os.environ[env_var]}. Using --{env_var.lower()}={arg_value} and continue..."
        )
        os.environ[env_var] = arg_value
    elif env_var not in os.environ:
        os.environ[env_var] = arg_value


def main() -> None:
    """
    1) Read environment variables.
    2) Parse commmand line arguments.
    3) Read and analyze trace file.
    4) Run replay.
    """
    args = parse_arguments()
    validate_arguments(args)

    comm_replayer = CommTraceReplayer()
    comm_replayer.run(comm_trace_replay_args)


if __name__ == "__main__":
    main()
