#!/usr/bin/env python3

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import logging
import time

import numpy as np

# pytorch
import torch
from param_bench.train.comms.pt import comms_utils

from param_bench.train.comms.pt.comms import commsCollBench
from param_bench.train.comms.pt.comms_utils import (
    ensureTensorFlush,
    MultilineFormatter,
    paramDeviceTimer,
    paramStreamGuard,
)
from param_bench.train.comms.pt.logger_utils import (
    benchType,
    commsCollPerfMetrics,
    customized_perf_loggers,
)

logger = logging.getLogger(__name__)


# define the collective benchmark
class commsComputeBench(commsCollBench):
    # def readCollArgs(self, parser):
    def readArgs(self, parser):
        # read the common/basic arguments
        super().readArgs(parser)
        # experiment related parameters
        parser.add_argument(
            "--mode",
            type=str,
            default="comms-compute",
            help="benchmark mode",
            choices=["compute", "comms-compute"],
        )  # compute-only or comms-compute mode
        # For comm-compute or compute mode
        parser.add_argument(
            "--kernel",
            type=str,
            default="gemm",
            help="Compute kernel, used for comms-compute or compute mode",
            choices=[
                "gemm",
                "emb_lookup",
                "add",
                "sub",
                "add_num",
                "sub_num",
                "copy",
                "d2h",
                "h2d",
            ],
        )  # Compute kernel: "gemm"
        parser.add_argument(
            "--use-triton",
            action="store_true",
            default=False,
            help="To use triton kernel for gemm",
        )  # Enable triton kernels option
        parser.add_argument(
            "--num-compute",
            "--num-compute-per-iteration",
            type=int,
            default=100,
            help="number of compute kernels to execute for every iteration",
        )  # Launch one coll for every n compute kernels
        # For GEMM
        parser.add_argument(
            "--mm-dim",
            "--comp-dim",
            type=int,
            nargs="*",
            default=[100],
            help="dimension size of matrices for GEMM or other compute kernels except emb_lookup, "
            "For gemm, '--mm-dim m n p' uses two input tensors A[m,n] * B[n,p]"
            "For add or sub, '--mm-dim m n' uses the dimension of input annd output tensors are (m x n)"
            "If only one value is provided, it uses the dimension of input and output tensors are (n x n), i.e., square tensors",
        )  # Matrix multiplication dim n, A[m,n] * B [n,p]
        parser.add_argument(
            "--comp-data-type",
            type=str,
            default="float32",
            help="datatype for GEMM or other compute kernels except emb_lookup"
            + str(self.supportedDtype),
        )
        # For emb lookup
        parser.add_argument(
            "--emb-dim",
            type=int,
            default=128,
            help="dimension size for Embedding table compute kernel",
        )  # Embedding table dimension
        parser.add_argument(
            "--num-embs",
            type=int,
            default=100000,
            help="Embedding table hash size for Embedding table compute kernel",
        )  # Embedding table hash size
        parser.add_argument(
            "--batch-size",
            type=int,
            default=512,
            help="number of samples reading the table concurrently",
        )  # #Samples reading the table concurrently
        parser.add_argument(
            "--num-emb-tables-per-device",
            "--ntables",
            "--num-emb-tables",
            type=int,
            default=8,
            help="Number of embedding tables (per device) for embedding table compute kernel",
        )  # number of Embedding table
        parser.add_argument(
            "--num-emb-tables-batched",
            type=int,
            default=-1,
            help="Number of embedding tables to batch together when doing embedding lookups and communication (-1 means to do no batching)",
        )  # number of Embedding table batched
        parser.add_argument(
            "--bag-size",
            type=int,
            default=20,
            help="bag size for Embedding table compute kernel",
        )  # number of Embedding table
        return parser.parse_known_args()

    # Check arguments that may be custmized per benchmark in a single run
    # does not depend on data type
    def checkArgs(self, args):  # noqa: C901
        if (
            len(args.mm_dim) > 3
            or len(args.mm_dim) == 0
            or (args.kernel == "gemm" and len(args.mm_dim) == 2)
            or (args.kernel in ("add", "sub") and len(args.mm_dim) == 3)
        ):
            logger.error(
                "mm_dim should have either 1 input argument, "
                "or 3 input arguments for gemm, or 2 input arguments for add/sub, "
                f"but got {len(args.mm_dim)} for {args.kernel}"
            )
            comms_utils.gracefulExit()

    def runColl(self, comm_fn=None, compute_fn=None, dcheck=False):
        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_begin")

        elapsedTimeNS = 0.0
        is_blocking = not self.collectiveArgs.asyncOp
        enable_comms = (
            False if (comm_fn is None or comm_fn == self.backendFuncs.noop) else True
        )

        for nIter in range(
            self.collectiveArgs.numWarmupIters + self.collectiveArgs.numIters
        ):
            if self.collectiveArgs.enable_profiler:
                comms_utils.sampleProfiler()
            if nIter == self.collectiveArgs.numWarmupIters:
                # Flush non-blocking ops to ensure warmup is really complete
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
                ensureTensorFlush(self.collectiveArgs.opTensor)
                # Start measuring time after warmup iterations
                elapsedTimeNS = 0.0
                self.collectiveArgs.quant_time.reset()
                self.collectiveArgs.dequant_time.reset()
                if self.collectiveArgs.comm_dev_time:
                    self.collectiveArgs.comm_dev_time.reset()
                if self.collectiveArgs.compute_dev_time:
                    self.collectiveArgs.compute_dev_time.reset()
            # reset tensor values for data validation check
            if enable_comms and dcheck:
                self.setTensorVal(self.collectiveArgs.opTensor)
            # for blocking mode, do barrier before starting collective
            if is_blocking:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            start = time.monotonic()  # available only in py3
            with paramStreamGuard(
                stream=self.backendFuncs.get_current_stream(
                    device=self.collectiveArgs.device
                ),
                curDevice=self.collectiveArgs.device,
                backendFuncs=self.backendFuncs,
                timer=self.collectiveArgs.comm_dev_time,
                is_blocking=False,
            ):
                self.collectiveArgs.group = self.collectiveArgs.groups[
                    self.collectiveArgs.pgId
                ]
                for _ in range(self.collectiveArgs.numCollPerIter):
                    comm_fn(self.collectiveArgs)

            with paramStreamGuard(
                stream=self.collectiveArgs.compute_stream,
                curDevice=self.collectiveArgs.device,
                backendFuncs=self.backendFuncs,
                timer=self.collectiveArgs.compute_dev_time,
                is_blocking=False,
            ):
                for _ in range(self.collectiveArgs.numComputePerIter):
                    # TODO: investigate the cache effect
                    # Flush the cache
                    # _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
                    compute_fn(self.collectiveArgs)
            if is_blocking:  # should be sychronous, wait for the collective
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
                # caputure per-op kernel time only for blocking case
                if self.collectiveArgs.comm_dev_time:
                    self.collectiveArgs.comm_dev_time.elapsedTime()
                if self.collectiveArgs.compute_dev_time:
                    self.collectiveArgs.compute_dev_time.elapsedTime()

            # Measuring time.
            elapsedTimeNS += (
                time.monotonic() - start
            ) * 1e9  # keeping time in NS, helps in divising data by nanosecond

        start = time.monotonic()  # available only in py3
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)
        end = time.monotonic()  # available only in py3

        ensureTensorFlush(self.collectiveArgs.opTensor)

        elapsedTimeNS += (
            end - start
        ) * 1e9  # keeping time in NS, helps in divising data by nanoseconds

        memSize = self.backendFuncs.get_mem_size(self.collectiveArgs)

        avgIterNS, algBW = comms_utils.getAlgBW(
            elapsedTimeNS,
            memSize,
            self.collectiveArgs.numIters * self.collectiveArgs.numCollPerIter,
        )
        busBW = self.backendFuncs.getBusBW(
            self.collectiveArgs.collective,
            algBW,
            self.collectiveArgs,
        )

        # reset group to sync among all global ranks
        self.collectiveArgs.group = self.backendFuncs.get_default_group()
        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_end")

        results = {
            "timeUS": avgIterNS / 1e3,
            "algBW": algBW,
            "busBW": busBW,
            "memSize": memSize,
        }
        return results

    def initCollectiveArgs(self, commsParams):
        # lint was complaining that benchTime was too complex!
        (
            local_rank,
            global_rank,
            world_size,
            group,
            curDevice,
            curHwDevice,
        ) = comms_utils.get_rank_details(
            self.backendFuncs
        )  # Getting ranks from backednFuncs object, since we cannot use MPI (e.g.: TPU) to launch all the processes.
        groups = self.backendFuncs.get_groups()
        num_pgs = len(groups)

        # global world size
        self.comm_size = world_size

        # Update world_size with the number of ranks in the group.
        # FIXME: only support one global group for comms-compute mode
        self.collectiveArgs.pgId = 0
        myGroup = groups[self.collectiveArgs.pgId]
        world_size = self.backendFuncs.get_group_size(myGroup)
        myGroupRanks = commsParams.groupRanks[self.collectiveArgs.pgId]

        self.global_rank = global_rank
        self.report = (
            True
            if global_rank == 0 or (commsParams.enable_local_report and local_rank == 0)
            else False
        )

        if commsParams.sizes is not None:
            allSizes = commsParams.sizes
            if self.report:
                logger.info(
                    f"Benchmarking with user-specified message sizes {allSizes}, --b and --e are ignored"
                )
        else:
            comms_utils.fixBeginSize(
                commsParams, world_size
            )  # Ensuring that all-reduce and all-to-all has atleast one member per rank.
            allSizes = comms_utils.getSizes(
                commsParams.beginSize,
                commsParams.endSize,
                commsParams.stepFactor,
                commsParams.stepBytes,
            )  # Given the begin-size, end-size, step-factor what are the message sizes to iterate on.

        self.collectiveArgs.group = group
        self.collectiveArgs.groups = groups
        self.collectiveArgs.num_pgs = num_pgs
        self.collectiveArgs.device = curDevice
        self.collectiveArgs.world_size = world_size
        self.collectiveArgs.numIters = commsParams.numIters
        self.collectiveArgs.numWarmupIters = commsParams.numWarmupIters
        self.collectiveArgs.global_rank = global_rank
        self.collectiveArgs.backendFuncs = self.backendFuncs
        self.collectiveArgs.collective = commsParams.collective
        op = self.backendFuncs.get_reduce_op("sum")
        self.collectiveArgs.op = op
        # Update root rank for current PG, as torch.dist requires the global rank
        self.collectiveArgs.srcOrDst = myGroupRanks[commsParams.srcOrDst]
        self.collectiveArgs.src_ranks = commsParams.src_ranks
        self.collectiveArgs.dst_ranks = commsParams.dst_ranks

        self.collectiveArgs.pt2pt = commsParams.pt2pt
        self.collectiveArgs.window = commsParams.window
        self.collectiveArgs.asyncOp = False if commsParams.blockingFlag == 1 else True
        self.collectiveArgs.numComputePerIter = commsParams.num_compute
        self.collectiveArgs.numCollPerIter = commsParams.num_coll
        self.collectiveArgs.use_triton = commsParams.use_triton
        self.collectiveArgs.include_0B = commsParams.include_0B

        if commsParams.bitwidth < 32:
            comms_utils.initQuantCommCtx(self.collectiveArgs, commsParams)

        computeFunc = self.backendFuncs.noop
        if (
            commsParams.mode != "comms"
        ):  # Compute mode related initialization if not in comms-only mode
            self.collectiveArgs.compute_stream = self.backendFuncs.get_new_stream()
            if commsParams.kernel == "gemm":
                if len(commsParams.mm_dim) == 1:
                    # duplicate dim to make them square tensors
                    commsParams.mm_dim = [
                        commsParams.mm_dim[0],
                        commsParams.mm_dim[0],
                        commsParams.mm_dim[0],
                    ]

                computeFunc = self.backendFuncs.gemm
                (
                    self.collectiveArgs.MMout,
                    self.collectiveArgs.MMin1,
                    self.collectiveArgs.MMin2,
                ) = self.prepGemmNotSquare(
                    commsParams.mm_dim[0],
                    commsParams.mm_dim[1],
                    commsParams.mm_dim[1],
                    commsParams.mm_dim[2],
                    commsParams.comp_data_type,
                    curDevice,
                )

                if self.report:
                    print(
                        f"[Rank {global_rank:>3}] mode: {commsParams.mode}, num_coll: {commsParams.num_coll}, collectives datatype: {commsParams.data_type}, kernel: {commsParams.kernel}, num_compute {commsParams.num_compute}, mm_dim {commsParams.mm_dim}, comp_datatype {self.collectiveArgs.MMout.dtype} "
                    )
            elif commsParams.kernel == "emb_lookup":
                comms_utils.init_emb_lookup(
                    self.collectiveArgs, commsParams, self.backendFuncs
                )
                computeFunc = self.backendFuncs.emb_lookup
                if self.report:
                    print(
                        f"[Rank {global_rank:>3}] mode: {commsParams.mode}, num_coll: {commsParams.num_coll}, kernel: {commsParams.kernel}, num_compute {commsParams.num_compute}, "
                        f"emb_dim {commsParams.emb_dim}, num_embs {commsParams.num_embs}, batch_size {commsParams.batch_size}"
                    )
            elif commsParams.kernel in [
                "add",
                "sub",
                "add_num",
                "sub_num",
                "copy",
                "d2h",
                "h2d",
            ]:
                if len(commsParams.mm_dim) == 2:
                    # make the third element be 1 to calculate BW correctly for add/sub
                    commsParams.mm_dim.append(1)
                (
                    self.collectiveArgs.compOut,
                    self.collectiveArgs.compIn1,
                    self.collectiveArgs.compIn2,
                ) = self.prepComp(
                    commsParams.mm_dim[0],
                    commsParams.mm_dim[1],
                    commsParams.comp_data_type,
                    curDevice,
                    commsParams.kernel,
                )
                computeFunc = self.backendFuncs.computeFunc[commsParams.kernel]
                if self.report:
                    print(
                        f"[Rank {global_rank:>3}] mode: {commsParams.mode}, num_coll: {commsParams.num_coll}, kernel: {commsParams.kernel}, num_compute {commsParams.num_compute}, mm_dim {commsParams.mm_dim}"
                    )

        # Enable device timer only for comms-compute mode
        # since CPU timer would capture individual time
        if commsParams.mode == "comms-compute":
            self.collectiveArgs.compute_dev_time = paramDeviceTimer(
                name="compute_timer", backendFuncs=self.backendFuncs
            )
            self.collectiveArgs.comm_dev_time = paramDeviceTimer(
                name="comm_timer", backendFuncs=self.backendFuncs
            )
        else:
            self.collectiveArgs.comm_dev_time = None
            self.collectiveArgs.compute_dev_time = None

        self.backendFuncs.sync_barrier(self.collectiveArgs)
        if self.report:
            print(
                f"[Rank {global_rank:>3}] allSizes: {allSizes} element_size: {commsParams.element_size}"
                + f" local_rank: {local_rank}, num_pg {self.collectiveArgs.num_pgs}, groupSize {self.collectiveArgs.world_size}"
            )
        if self.collectiveArgs.collective == "pt2pt":
            self.checkPt2PtRanks()
        else:
            self.checkCollectiveRanks()

        return (
            global_rank,
            world_size,
            allSizes,
            computeFunc,
        )

    def printPreamble(self, commsParams):
        logger.debug(f"\tcommsParams: {str(commsParams.__dict__)}")
        header = "\n\tCOMMS-RES"

        tflops_fmt = ""
        if commsParams.kernel == "gemm" and commsParams.mode != "comms":
            tflops_fmt = "{:>15}"
        dev_time_fmt = ""
        if commsParams.mode == "comms-compute":
            dev_time_fmt = "{:>20}{:>20}"
        if self.collectiveArgs.collective == "pt2pt":
            fmt = (
                "{:>40}{:>20}{:>10}{:>10}{:>25}{:>10}{:>10}{:>15}{:>15}{:>18}{:>18}"
                + dev_time_fmt
                + tflops_fmt
            )
            header += fmt.format(
                "size (B)",
                "pingLatency(us):p50",
                "p75",
                "p95",
                "pingPongLatency(us):p50",
                "p75",
                "p95",
                "avgUniBW(GB/s)",
                "avgBiBW(GB/s)",
                "totalUniBW(GB/s)",
                "totalBiBW(GB/s)",
                "TotalLatency(us):p50",
                "CompLatency(us):p50",
                "TFlops",
            )
        else:
            if commsParams.bitwidth < 32:
                fmt = (
                    "-QUANT\t{:>40}{:>18}{:>25}{:>15}{:>15}{:>15}"
                    + dev_time_fmt
                    + tflops_fmt
                )
                header += fmt.format(
                    "size (B)",
                    "nElementsPerRank",
                    "P95 Latency(us): Quant",
                    "Comms",
                    "De-Quant",
                    "Overall",
                    "TotalLatency(us):p50",
                    "CompLatency(us):p50",
                    "TFlops",
                )
            else:
                fmt = (
                    "{:>40}{:>18}{:>22}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
                    + dev_time_fmt
                    + tflops_fmt
                )
                header += fmt.format(
                    "total-size (B)",
                    "nElementsPerRank",
                    "nElementsPairPerRank",
                    "Latency(us):p50",
                    "p75",
                    "p95",
                    "Min",
                    "Max",
                    "AlgBW(GB/s)",
                    "BusBW(GB/s)",
                    "TotalLatency(us):p50",
                    "CompLatency(us):p50",
                    "TFlops",
                )

        print(header)

    def reportBenchTime(
        self,
        collectiveArgs,
        commsParams,
        results,
        tensorList,
        quantTimeTensorList,
        dequantTimeTensorList,
        commUsElapsedList,
        computeUsElapsedList,
    ):
        # convernt num_elements to # of elements per rank
        if commsParams.collective in (
            "all_to_all",
            "all_to_allv",
            "all_to_all_single",
            "reduce_scatter",
            "reduce_scatter_v",
            "reduce_scatter_base",
            "all_gather",
            "all_gather_v",
            "all_gather_base",
        ):
            results["numElements"] = int(
                results["numElements"] // collectiveArgs.world_size
            )

        perf_metrics = None

        if commsParams.collective == "pt2pt":
            perf_metrics = self.reportBenchTimePt2Pt(commsParams, tensorList, results)
        elif commsParams.bitwidth < 32:
            perf_metrics = self.reportBenchTimeCollWithQuant(
                commsParams,
                results,
                tensorList,
                quantTimeTensorList,
                dequantTimeTensorList,
            )
        else:
            perf_metrics = self.reportBenchTimeColl(
                commsParams,
                results,
                tensorList,
                commUsElapsedList,
                computeUsElapsedList,
            )

        # use custom perf_loggers if specified and registered
        if perf_metrics is not None and commsParams.use_perf_logger is not None:
            for perfLoggerName in commsParams.use_perf_logger:
                if perfLoggerName in customized_perf_loggers:
                    customized_perf_loggers[perfLoggerName].logPerf(
                        "comms",
                        perf_metrics,
                        self.backendFuncs,
                    )
                else:
                    logger.info(
                        f"Skipping logger '{perfLoggerName}' because it is not registered or implemented"
                    )

    def reportBenchTimeColl(
        self, commsParams, results, tensorList, commUsElapsedList, computeUsElapsedList
    ):
        latencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(tensorList)
        logger.debug(f"Latency across all ranks: {latencyAcrossRanks}")

        commLatencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(
            commUsElapsedList
        )
        computeLatencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(
            computeUsElapsedList
        )
        logger.debug(f"CommLatency across all ranks {commLatencyAcrossRanks}")
        logger.debug(f"ComputeLatency across all ranks {computeLatencyAcrossRanks}")

        # Include only communicating ranks
        if self.collectiveArgs.collective == "multicast":
            commRanks = [self.collectiveArgs.srcOrDst] + self.collectiveArgs.dst_ranks
        elif self.collectiveArgs.collective == "incast":
            commRanks = [self.collectiveArgs.srcOrDst] + self.collectiveArgs.src_ranks
        else:
            commRanks = range(self.collectiveArgs.world_size)

        latencyAcrossCommRanks = latencyAcrossRanks[commRanks]
        logger.debug(
            "Latency across communicating ranks (%s): %s"
            % (commRanks, latencyAcrossCommRanks)
        )

        nm = 0
        if commsParams.mode != "comms":
            nm = commsParams.mm_dim[0] * commsParams.mm_dim[1] * commsParams.mm_dim[2]

        tflop = (2 * nm) * 1e-12
        secs = results["timeUS"] * 1e-6
        # use compute-only time to compute tflops in comms-compute mode
        compute_p50 = 0.0
        if computeLatencyAcrossRanks.size:
            compute_p50 = np.percentile(computeLatencyAcrossRanks, 50)
            tflops = tflop / compute_p50 * 1e6  # US to sec
        else:
            tflops = tflop * self.collectiveArgs.numComputePerIter / secs

        # report comms-only time if comms-only time is captured;
        # report original cpu time as total time
        total_p50 = 0.0
        if commLatencyAcrossRanks.size:
            total_p50 = np.percentile(latencyAcrossCommRanks, 50)
            latencyAcrossCommRanks = commLatencyAcrossRanks

        # comms only time
        p50 = np.percentile(latencyAcrossCommRanks, 50)
        p75 = np.percentile(latencyAcrossCommRanks, 75)
        p95 = np.percentile(latencyAcrossCommRanks, 95)
        minlat = np.amin(latencyAcrossCommRanks)
        maxlat = np.amax(latencyAcrossCommRanks)

        # adjust algBW/busBW based on final comms p50 latency
        _, algBW = comms_utils.getAlgBW(p50 * 1e3, results["memSize"], 1)
        busBW = self.backendFuncs.getBusBW(
            self.collectiveArgs.collective,
            algBW,
            self.collectiveArgs,
        )

        # adjust busBW
        busBW *= commsParams.bitwidth / 32.0

        tflops_fmt = ""
        if commsParams.kernel == "gemm" and commsParams.mode != "comms":
            tflops_fmt = "{:>15}"
        dev_time_fmt = ""
        if commsParams.mode == "comms-compute":
            dev_time_fmt = "{:>20}{:>20}"

        fmt = (
            "\tCOMMS-RES-{}-{}{}{:>18}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
            + dev_time_fmt
            + tflops_fmt
        )
        print(
            fmt.format(
                self.collectiveArgs.collective,
                self.collectiveArgs.data_type,
                self.tag,
                results["memSize"],
                str("%d" % (results["numElements"])),
                str("%.1f" % (p50)),
                str("%.1f" % (p75)),
                str("%.1f" % (p95)),
                str("%.1f" % (minlat)),
                str("%.1f" % (maxlat)),
                str("%.3f" % (algBW)),
                str("%.3f" % (busBW)),
                str("%.1f" % (total_p50)),
                str("%.1f" % (compute_p50)),
                str("%.5f" % (tflops)),
            )
        )

        return commsCollPerfMetrics(
            self.collectiveArgs.collective,
            self.collectiveArgs.data_type,
            benchType.Collective,
            commsParams.backend,
            self.tag,
            results["memSize"],
            results["memSize"],
            results["numElements"],
            float(p50),
            float(p75),
            float(p95),
            float(minlat),
            float(maxlat),
            algBW,
            busBW,
            tflops,
        )

    def benchTime(self, index, commsParams, backendFuncs):
        for coll in commsParams.collective_list:
            commsParams.collective = coll
            self.benchComm(index, commsParams, backendFuncs)

    def benchComm(self, index, commsParams, backendFuncs):
        # Get NW stack specific parameters
        (
            global_rank,
            world_size,
            allSizes,
            computeFunc,
        ) = self.initCollectiveArgs(commsParams)

        backendFuncs.sync_barrier(self.collectiveArgs)
        if self.report:
            self.printPreamble(commsParams)

        for curSize in allSizes:
            results = {}
            timeUsElapsedList = []
            quantTimeElapsedList = []
            dequantTimeElapsedList = []
            numElements = int(curSize // commsParams.element_size)
            collectiveFunc = self.backendFuncs.noop
            commUsElapsedList = []
            computeUsElapsedList = []

            if (
                commsParams.mode != "compute"
            ):  # comms specific initializations if not in compute-only mode
                # set corresponding function pointers
                if commsParams.collective != "pt2pt":
                    collectiveFunc = backendFuncs.collectiveFunc[commsParams.collective]

                commsArgs = comms_utils.commsArgs()
                commsArgs.inMsgSize = numElements
                commsArgs.outMsgSize = numElements
                commsArgs.worldSize = world_size
                commsArgs.inSplit = commsParams.inSplit
                commsArgs.outSplit = commsParams.outSplit
                commsArgs.comms = commsParams.collective

                (
                    self.collectiveArgs.ipTensor,
                    self.collectiveArgs.opTensor,
                ) = self.prepComm(
                    curComm=commsArgs,
                    commsParams=commsParams,
                )

            # Setup the arguments.
            self.collectiveArgs.dataSize = curSize
            self.collectiveArgs.numElements = numElements
            self.collectiveArgs.waitObj = []
            results["numElements"] = numElements

            self.collectiveArgs.data_type = commsParams.data_type
            if commsParams.size_start_profiler == curSize:
                self.collectiveArgs.enable_profiler = comms_utils.startProfiler(
                    rank=self.backendFuncs.get_global_rank(),
                    device=self.collectiveArgs.device,
                    numWarmupIters=self.collectiveArgs.numWarmupIters,
                    numIters=self.collectiveArgs.numIters,
                )

            # self.collectiveArgs has all the information on the experiment.
            if commsParams.collective == "pt2pt":
                results.update(self.runPt2Pt())

                timeUsElapsedList = [
                    np.mean(np.array(results["pingPerIterNS"])) / 1e3,
                    np.mean(np.array(results["pingPongPerIterNS"])) / 1e3,
                    results["avgUniBW"],
                    results["avgBiBW"],
                ]  # time in US
                if (
                    global_rank in self.collectiveArgs.src_ranks
                    or global_rank in self.collectiveArgs.dst_ranks
                ):
                    logger.debug(timeUsElapsedList)
            else:
                results.update(
                    self.runColl(
                        comm_fn=collectiveFunc,
                        compute_fn=computeFunc,
                        dcheck=commsParams.dcheck,
                    )
                )
                timeUsElapsedList = [results["timeUS"]]

            # stop profiler if used
            if self.collectiveArgs.enable_profiler:
                comms_utils.sampleProfiler(stop=True)
                self.collectiveArgs.enable_profiler = False

            # perfom data validation check on the final opTensor
            if commsParams.dcheck == 1:
                self.dcheck(commsParams, curSize, self.collectiveArgs.opTensor)

            backendFuncs.clear_memory(self.collectiveArgs)

            # gather quantization overhead if enabled
            if commsParams.bitwidth < 32:
                # calculate average (de-)quantization overhead
                results["quantTimeUS"] = (
                    self.collectiveArgs.quant_time.getTimeUS()
                    / self.collectiveArgs.numIters
                )
                results["dequantTimeUS"] = (
                    self.collectiveArgs.dequant_time.getTimeUS()
                    / self.collectiveArgs.numIters
                )
                quantTimeElapsedList.append(results["quantTimeUS"])
                dequantTimeElapsedList.append(results["dequantTimeUS"])

                logger.debug(quantTimeElapsedList)
                quantTimeElapsedList = self.gatherBenchTime(
                    self.collectiveArgs, commsParams, quantTimeElapsedList
                )
                dequantTimeElapsedList = self.gatherBenchTime(
                    self.collectiveArgs, commsParams, dequantTimeElapsedList
                )

            # gather single compute and communication kernel time per iteration
            if self.collectiveArgs.comm_dev_time:
                results["commTimeUS"] = (
                    self.collectiveArgs.comm_dev_time.getTimeUS()
                    / self.collectiveArgs.numIters
                    / self.collectiveArgs.numCollPerIter
                )
                commUsElapsedList.append(results["commTimeUS"])
                commUsElapsedList = self.gatherBenchTime(
                    self.collectiveArgs, commsParams, commUsElapsedList
                )

            if self.collectiveArgs.compute_dev_time:
                results["computeTimeUS"] = (
                    self.collectiveArgs.compute_dev_time.getTimeUS()
                    / self.collectiveArgs.numIters
                    / self.collectiveArgs.numComputePerIter
                )
                computeUsElapsedList.append(results["computeTimeUS"])
                computeUsElapsedList = self.gatherBenchTime(
                    self.collectiveArgs, commsParams, computeUsElapsedList
                )

            # gather results from all global ranks and report performance to stdout
            tensorList = self.gatherBenchTime(
                self.collectiveArgs, commsParams, timeUsElapsedList
            )
            if self.report:
                self.reportBenchTime(
                    self.collectiveArgs,
                    commsParams,
                    results,
                    tensorList,
                    quantTimeElapsedList,
                    dequantTimeElapsedList,
                    commUsElapsedList,
                    computeUsElapsedList,
                )

            self.backendFuncs.sync_barrier(
                self.collectiveArgs, desc=f"curSize_{curSize}"
            )

        comms_utils.clearQuantCommCtx(self.collectiveArgs)

        # wait rank 0 reports results to avoid other ranks mess up the output
        self.backendFuncs.sync_barrier(self.collectiveArgs, "benchtime")


def main():
    collComputeBenchObj = commsComputeBench()

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="PARAM-Comm-Compute Benchmark",
        formatter_class=MultilineFormatter,
        allow_abbrev=False,
    )
    args, _ = collComputeBenchObj.readArgs(parser)

    comms_env_params = comms_utils.read_comms_env_vars()
    if comms_env_params["global_rank"] == 0 or (
        args.enable_local_report and comms_env_params["local_rank"] == 0
    ):
        print("\t PARAM COMM-COMPUTE environment: %s " % (str(comms_env_params)))
        print(
            "\t backend: %s nw-stack: %s mode: %s args.data_types: %s args.b: %s args.e: %s args.f: %s args.z: %s args.master_ip: %s "
            % (
                args.backend,
                args.nw_stack,
                args.mode,
                args.data_types,
                args.b,
                args.e,
                args.f,
                args.z,
                args.master_ip,
            )
        )

    collComputeBenchObj.checkBasicArgs(args)

    # Initialize backend
    bootstrap_info = comms_utils.bootstrap_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, comms_env_params
    )
    commsParamsBase = comms_utils.commsParamsHolderBase(args)
    collComputeBenchObj.initBackend(bootstrap_info, commsParamsBase)

    # Dedupes and syncs value for args.data_types based on args.data_type/args.dtype if not passed in args.
    collComputeBenchObj.syncCommBenchDataTypes(args)

    collComputeBenchObj.checkArgs(args)

    # FIXME: only support single global PG for comm-compute benchmark
    groupRanks = {}
    groupRanks[0] = []
    for rank in range(0, bootstrap_info.world_size):
        groupRanks[0].append(rank)

    for data_type in args.data_types:
        args.data_type = data_type.lower()

        collComputeBenchObj.checkArgsdataType(args)

        element_size = torch.ones([1], dtype=args.dtype).element_size()

        commsParams = comms_utils.commsComputParamsHolder(
            args,
            bootstrap_info,
            element_size,
            collComputeBenchObj.benchTime,
            groupRanks,
        )
        commsParams.comp_data_type = collComputeBenchObj.dtypeMap[
            commsParams.comp_data_type
        ]

        collComputeBenchObj.runBench(commsParams)


if __name__ == "__main__":
    main()
