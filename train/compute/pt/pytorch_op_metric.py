from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    annotations,
)

import argparse, json, os, sys
import logging
import subprocess
import time
from enum import Enum
from pprint import pprint
from typing import Dict, Set, List, Tuple, Any, Callable, Iterable, Type, TextIO

FORMAT = "[%(asctime)s] %(filename)s [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

# default ncu path
NCU_BIN: str = "/usr/local/NVIDIA-Nsight-Compute-2020.3/ncu"


def run_ncu(args: str, device: str, metrics: str, out_prefix: str):
    ncu_bin = os.getenv("NCU_BIN")
    if not ncu_bin:
        ncu_bin = NCU_BIN
    ncu_options = (
        f"--log-file {out_prefix}.ncu.log --csv --target-processes all "
        f"--metrics {metrics} --nvtx --nvtx-include op_bench/+"
    )

    cmd = (
        [ncu_bin]
        + ncu_options.split(" ")
        + args.split(" ")
        + ["--device", device, "--out_json", f"{out_prefix}.json"]
    )
    logging.info("Running: " + " ".join(cmd))
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")


def main():

    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument(
        "--bench_args",
        type=str,
        required=True,
        help="Args to pass to benchmark script.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="The target device.")
    parser.add_argument("--metrics", type=str, default="", help="The  config file.")
    parser.add_argument(
        "--metrics_file", type=str, default=None, help="The metrics config file."
    )
    parser.add_argument(
        "--output_prefix", type=str, default="op_metrics", help="output file prefix"
    )

    args = parser.parse_args()

    metrics = {x.strip() for x in args.metrics.split(",") if x.strip()}
    if args.metrics_file:
        with open(args.metrics_file, "r") as metrics_file:
            metrics.update(
                {x.strip() for x in metrics_file.read().split(",") if x.strip()}
            )
    # combine all metrics
    metrics_str = ",".join(str(s) for s in metrics)
    if args.device.startswith("cuda"):
        run_ncu(args.bench_args, args.device, metrics_str, args.output_prefix)
    else:
        logging.warning("Only GPU metrics are supported right now.")


if __name__ == "__main__":
    main()
