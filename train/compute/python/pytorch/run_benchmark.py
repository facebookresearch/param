import argparse
import json
import logging

from ..lib import pytorch as lib_pytorch
from ..lib.config import BenchmarkConfig
from ..lib.init_helper import init_logging, load_modules
from ..lib.pytorch.benchmark import make_default_benchmark
from ..lib.pytorch.config_util import get_benchmark_options, ExecutionPass, get_sys_info
from ..workloads import pytorch as workloads_pytorch


def main():
    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="The benchmark config file."
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=1, help="Number of warm up iterations."
    )
    parser.add_argument(
        "-i", "--iteration", type=int, default=1, help="Number of benchmark iterations."
    )
    parser.add_argument(
        "-b", "--backward", action="store_true", help="Include backward pass."
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="Target device for benchmark."
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        default="benchmark_result",
        help="File name prefix to write benchmark results.",
    )
    parser.add_argument(
        "-r",
        "--resume-id",
        type=str,
        default=None,
        help="Define a resume op_run_id to continue benchmark, skip all previous configs.",
    )
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file, rather than overwrite.",
    )
    parser.add_argument(
        "--ncu", action="store_true", help="Run NSight Compute to collect metrics."
    )
    parser.add_argument(
        "--ncu-args-file",
        type=str,
        default=None,
        help="NSight Compute extra command line options (metrics etc.).",
    )
    parser.add_argument(
        "--ncu-batch",
        type=int,
        default=50,
        help="NSight Compute input batch size (number of input configs to run in one launch).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase log output verbosity."
    )

    args = parser.parse_args()

    if args.verbose:
        logger = init_logging(logging.DEBUG)
    else:
        logger = init_logging(logging.INFO)

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    run_options = get_benchmark_options()
    run_options["warmup"] = args.warmup
    run_options["iteration"] = args.iteration
    run_options["device"] = args.device
    run_options["resume_op_run_id"] = args.resume_id
    run_options["ncu_batch"] = args.ncu_batch

    if args.backward:
        run_options["pass_type"] = ExecutionPass.BACKWARD
    else:
        run_options["pass_type"] = ExecutionPass.FORWARD

    if args.ncu:
        run_options["run_ncu"] = True

    out_file_name = f"{args.output_prefix}.json"

    write_option = "a" if args.append else "w"

    if args.ncu_args_file:
        with open(args.ncu_args_file, "r") as ncu_file:
            run_options["ncu_args"] = ncu_file.read().strip()

    with open(out_file_name, write_option) as out_file:
        run_options["out_file_prefix"] = args.output_prefix
        run_options["out_stream"] = out_file
        benchmark_setup = {"run_options": run_options, "sys_info": get_sys_info()}
        out_file.write(json.dumps(benchmark_setup, default=str) + "\n")

        bench_config = BenchmarkConfig(run_options)
        bench_config.load_json_file(args.config)
        benchmark = make_default_benchmark(bench_config)
        benchmark.run()

    logger.info(f"benchmark result: {out_file_name}")


if __name__ == "__main__":
    main()
