import argparse
import logging
import os
import subprocess

from ..lib.init_helper import init_logging

# Initialize logging format before loading all other modules
logger = init_logging(logging.INFO)


# default ncu path
NCU_BIN: str = "/usr/local/NVIDIA-Nsight-Compute-2021.2/ncu"


def run_ncu(args: str, metrics: str, out_prefix: str):
    ncu_bin = os.getenv("NCU_BIN")
    param_bench_range = "param_bench@measure"
    if not ncu_bin:
        ncu_bin = NCU_BIN
    ncu_options = (
        f"--log-file {out_prefix}.ncu.log --csv --app-replay-buffer file --target-processes all "
        f"--metrics {metrics} --nvtx --nvtx-include {param_bench_range}"
    )

    cmd = [ncu_bin] + ncu_options.split(" ") + args.split(" ")
    logger.info("running: " + " ".join(cmd))
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
        "-b",
        "--benchmark",
        type=str,
        required=True,
        help="Args to pass to benchmark script.",
    )
    parser.add_argument("-m", "--metrics", type=str, default="", help="The metric ids.")
    parser.add_argument(
        "-f", "--metrics_file", type=str, default=None, help="The metrics config file."
    )
    parser.add_argument(
        "-o",
        "--output_prefix",
        type=str,
        default="benchmark",
        help="output file prefix",
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
    run_ncu(args.benchmark, metrics_str, args.output_prefix)


if __name__ == "__main__":
    main()
