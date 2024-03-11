import argparse
import json
import logging
import os
from datetime import datetime

import torch
from torch.autograd.profiler import record_function
from torch.profiler import _ExperimentalConfig, ExecutionTraceObserver

from ..lib import __version__, pytorch as lib_pytorch
from ..lib.config import BenchmarkConfig
from ..lib.init_helper import init_logging, load_modules
from ..lib.pytorch.benchmark import make_default_benchmark
from ..lib.pytorch.config_util import (
    ExecutionPass,
    get_benchmark_options,
    get_sys_info,
    OpExecutionMode,
)
from ..workloads import pytorch as workloads_pytorch


def main():
    parser = argparse.ArgumentParser(description="PyTorch Microbenchmarks")
    parser.add_argument("-c", "--config", type=str, help="The benchmark config file.")
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
        "-s",
        "--stop_id",
        type=str,
        default=None,
        help="Define a stop op_run_id (exclusive) to stop benchmark, skip remaining configs.",
    )
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append to output file, rather than overwrite.",
    )
    parser.add_argument(
        "--cuda-l2-cache",
        default="on",
        nargs="?",
        choices=["on", "off"],
        help="Set option for CUDA GPU L2 cache between iterations in discrete mode.",
    )
    parser.add_argument(
        "--ncu", action="store_true", help="Run NSight Compute to collect metrics."
    )
    parser.add_argument(
        "--ncu-bin",
        type=str,
        default=None,
        help="Path to the NSight Compute (ncu) binary.",
    )
    parser.add_argument(
        "--ncu-args-file",
        type=str,
        default=None,
        help="NSight Compute extra command line options (metrics etc.).",
    )
    parser.add_argument(
        "--ncu-warmup",
        type=int,
        default=None,
        help="NSight Systems number of warmup runs.",
    )
    parser.add_argument(
        "--ncu-iteration",
        type=int,
        default=None,
        help="NSight Systems number of measured iteration runs.",
    )
    parser.add_argument(
        "--nsys", action="store_true", help="Run NSight Systems to collect metrics."
    )
    parser.add_argument(
        "--nsys-bin",
        type=str,
        default=None,
        help="Path to the NSight Systems (nsys) binary.",
    )
    parser.add_argument(
        "--nsys-args-file",
        type=str,
        default=None,
        help="NSight Systems extra command line options (metrics etc.).",
    )
    parser.add_argument(
        "--nsys-warmup",
        type=int,
        default=None,
        help="NSight Systems number of warmup runs.",
    )
    parser.add_argument(
        "--nsys-iteration",
        type=int,
        default=None,
        help="NSight Systems number of measured iteration runs.",
    )
    parser.add_argument(
        "--run-batch-size",
        type=int,
        default=50,
        help="Batch run input size (number of input configs to run in one launch), used by both NCU and NSYS.",
    )
    parser.add_argument(
        "--batch-cuda-device",
        type=int,
        default=1,
        help="CUDA GPU device ID to run batch job.",
    )
    parser.add_argument(
        "--batch-cmd",
        type=str,
        default=None,
        help="Run batch job command.",
    )
    parser.add_argument(
        "--exec-mode",
        type=str,
        default="discrete",
        nargs="?",
        choices=["discrete", "continuous", "continuous_events"],
        help="Set execution mode of the operators (discrete, continuous, continuous_events). Default=discrete",
    )
    parser.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help="Enable profiler and tracing.",
    )
    parser.add_argument(
        "--cupti-profiler",
        action="store_true",
        help="Run CUPTI Profiler to measure performance events directly,"
        "The measurements will be written to the profile trace file."
        "See --cupti_profiler_metrics for supported metrics.",
    )
    parser.add_argument(
        "--cupti-profiler-metrics",
        type=str,
        default="kineto__cuda_core_flops",
        help="Comma separated list of metrics to measure on the CUDA device"
        "You can use any metrics available here: "
        "https://docs.nvidia.com/cupti/r_main.html#r_host_metrics_api\n"
        " eg: L2 misses, L1 bank conflicts.\n "
        "Additionally, Two special metrics are useful for measuring FLOPS\n"
        "-  kineto__cuda_core_flops = CUDA floating point op counts\n"
        "-  kineto__tensor_core_insts = Tensor core op counts\n",
    )
    parser.add_argument(
        "--cupti-profiler-measure-per-kernel",
        action="store_true",
        help="Run CUPTI Profiler measurements for every GPU kernel"
        "Warning : this can be slow",
    )
    parser.add_argument(
        "--et",
        action="store_true",
        help="Collect execution trace.",
    )

    parser.add_argument(
        "-l", "--log-level", default="INFO", help="Log output verbosity."
    )
    parser.add_argument("--version", action="store_true", help="Print version.")

    args = parser.parse_args()

    logger = init_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    if args.version:
        logger.info(f"PARAM train compute version: {__version__}")
        return
    elif not args.config:
        parser.print_usage()
        return

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    run_options = get_benchmark_options()
    run_options["warmup"] = args.warmup
    run_options["iteration"] = args.iteration
    run_options["device"] = args.device
    run_options["cuda_l2_cache"] = args.cuda_l2_cache == "on"
    run_options["resume_op_run_id"] = args.resume_id
    run_options["stop_op_run_id"] = args.stop_id
    run_options["run_batch_size"] = args.run_batch_size
    run_options["batch_cuda_device"] = args.batch_cuda_device

    if args.backward:
        run_options["pass_type"] = ExecutionPass.BACKWARD
    else:
        run_options["pass_type"] = ExecutionPass.FORWARD

    run_options["op_exec_mode"] = OpExecutionMode(args.exec_mode)
    run_options["run_ncu"] = args.ncu
    run_options["run_nsys"] = args.nsys

    pid = os.getpid()

    start_time = datetime.now()
    timestamp = int(datetime.timestamp(start_time))

    out_file_prefix = f"{args.output_prefix}_{pid}_{timestamp}"
    out_file_name = f"{out_file_prefix}.json"

    write_option = "a" if args.append else "w"

    if args.batch_cmd:
        run_options["batch_cmd"] = args.batch_cmd

    if args.ncu_bin:
        run_options["ncu_bin"] = args.ncu_bin
    if args.ncu_warmup:
        run_options["ncu_warmup"] = args.ncu_warmup
    if args.ncu_iteration:
        run_options["ncu_iteration"] = args.ncu_iteration
    if args.ncu_args_file:
        with open(args.ncu_args_file, "r") as ncu_file:
            run_options["ncu_args"] = ncu_file.read().strip()

    if args.nsys_bin:
        run_options["nsys_bin"] = args.nsys_bin
    if args.nsys_warmup:
        run_options["nsys_warmup"] = args.nsys_warmup
    if args.nsys_iteration:
        run_options["nsys_iteration"] = args.nsys_iteration
    if args.nsys_args_file:
        with open(args.nsys_args_file, "r") as nsys_file:
            run_options["nsys_args"] = nsys_file.read().strip()

    if args.cupti_profiler and not run_options["device"].startswith("cuda"):
        logger.warning("Cannot use --cupti_profiler when not running on cuda device")
        args.cupti_profiler = False
    if args.cupti_profiler and not args.profile:
        logger.warning("Enabling pytorch profiler as --cupti_profiler was added")
        args.profile = True

    run_options["cmd_args"] = args.__dict__

    with open(out_file_name, write_option) as out_file:
        run_options["out_file_prefix"] = args.output_prefix
        run_options["out_stream"] = out_file
        benchmark_setup = {
            "run_options": run_options,
            "sys_info": get_sys_info(),
            "start_time": start_time.isoformat(timespec="seconds"),
        }
        print(json.dumps(benchmark_setup, default=str), file=out_file)
        # This hack is necessary for Kineto profiler library to be initialized
        # and thus be able to track active CUDA contexts.
        if args.cupti_profiler:
            with torch.autograd.profiler.profile(
                enabled=True,
                use_cuda=True,
                use_kineto=True,
            ) as _:
                logger.info("Running dummy profiler warmup for CUPTI.")

        bench_config = BenchmarkConfig(run_options)
        bench_config.load_json_file(args.config)
        benchmark = make_default_benchmark(bench_config)
        use_cuda = False
        if run_options["device"].startswith("cuda"):
            use_cuda = True

        et = None
        if args.et:
            et_file = f"{out_file_prefix}_et.json"
            et = ExecutionTraceObserver()
            et.register_callback(et_file)
            et.start()

        cupti_profiler_config = (
            _ExperimentalConfig(
                profiler_metrics=args.cupti_profiler_metrics.split(","),
                profiler_measure_per_kernel=args.cupti_profiler_measure_per_kernel,
            )
            if args.cupti_profiler
            else None
        )

        with torch.autograd.profiler.profile(
            args.profile,
            use_cuda=use_cuda,
            use_kineto=True,
            record_shapes=False,
            experimental_config=cupti_profiler_config,
            # use_cpu enables profiling and recodring of CPU pytorch operators.
            # This is useful in CUPTI profiler mode if we are measuring per GPU kernel metrics.
            use_cpu=(not args.cupti_profiler) or args.cupti_profiler_measure_per_kernel,
        ) as prof:
            with record_function(f"[param|{run_options['device']}]"):
                benchmark.run()

        if et:
            et.stop()
            et.unregister_callback()
            logger.info(f"Exeution trace: {et_file}")

        print(
            json.dumps({"finish_time": datetime.now().isoformat(timespec="seconds")}),
            file=out_file,
        )
        if args.profile and prof:
            trace_file = f"{out_file_prefix}_trace.json"
            logger.info(f"Kineto trace: {trace_file}")
            prof.export_chrome_trace(trace_file)
            print(json.dumps({"trace_file": trace_file}), file=out_file)

    logger.info(f"Benchmark result: {out_file_name}")


if __name__ == "__main__":
    main()
