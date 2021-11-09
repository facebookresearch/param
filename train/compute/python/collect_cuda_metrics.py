import argparse, os
import logging
import subprocess

from .lib.init_helper import init_logging

# Initialize logging format before loading all other modules
logger = init_logging(logging.INFO)


# default ncu path
NCU_BIN: str = "/usr/local/NVIDIA-Nsight-Compute-2021.2/ncu"


def run_ncu(args: str, metrics: str, out_prefix: str):
    ncu_bin = os.getenv("NCU_BIN")
    if not ncu_bin:
        ncu_bin = NCU_BIN
    ncu_options = (
        f"--log-file {out_prefix}.ncu.log --csv --target-processes all "
        f"--metrics {metrics} --nvtx --nvtx-include param_bench@metric"
    )

    cmd = [ncu_bin] + ncu_options.split(" ") + args.split(" ")
    logger.info("Running: " + " ".join(cmd))
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
    parser.add_argument("--metrics", type=str, default="", help="The metric ids.")
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
    run_ncu(args.bench_args, metrics_str, args.output_prefix)


if __name__ == "__main__":
    main()
