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
    commsQuantCollPerfMetrics,
    customized_perf_loggers,
)

logger = logging.getLogger(__name__)


# define the collective benchmark
class commsOverlapBench(commsCollBench):
    def readArgs(self, parser):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--ssp",
            "--sizes-pair",
            type=lambda s: [int(item) for item in s.split(",") if item],
            default=None,
            help="benchmark only specified sizes, comma-separated",
        )  # COMMS mode, use specified sizes instead of increasing from small to large
        parser.add_argument(
            "--process-group",
            type=str,
            default=None,
            help="Set the process group to use for the collective."
            "For example, for a communication pattern like this: \n"
            "    0 - 1           0   1    "
            "                    |   |    "
            "    2 - 3           2   3    "
            "Collective 1     Collective 2"
            "                             "
            "Use the following arguments to specify the process group for each collective: \n"
            "Rank 0: --process-group [0,1] --process-group-pair [0,2]"
            "Rank 1: --process-group [0,1] --process-group-pair [1,3]"
            "Rank 2: --process-group [2,3] --process-group-pair [0,2]"
            "Rank 3: --process-group [2,3] --process-group-pair [1,3]",
        )
        parser.add_argument(
            "--pair",
            action="store_true",
            default=False,
            help="Toggle to enable collective pair mode",
        )
        parser.add_argument(
            "--collective-pair",
            "--collectives-pair",
            action="extend",
            nargs="+",
            type=str,
            default=[],
            help="Collective/s pair operation to be evaluated. It can be a single collective or a space-separated list of collectives. "
            "if there are multiple collectives, they will be executed in parallel. ",
        )  # collective op to pair with the other collective, --collective should be non-empty
        parser.add_argument(
            "--process-group-pair",
            action="extend",
            type=str,
            nargs="+",
            default=[],
            help="Set the process group to use for the pair collective/s.\n"
            "For example\n:"
            "--collective-pair all_gather --process-group-pair [0,1,2,3]\n"
            "--collective-pair all_gather,reduce_scatter --process-group-pair [0,1] [0,3]\n",
        )
        parser.add_argument(
            "--overlap-pair-pgs",
            action="store_true",
            default=False,
            help="Toggle to enable overlapping collective pair with two pgs",
        )  # overlap collective pair with two pgs

    # Check arguments that may be custmized per benchmark in a single run
    # does not depend on data type
    def checkArgs(self, args):  # noqa: C901
        super().checkArgs(args)
        world_size = self.backendFuncs.get_world_size()

        if args.multi_comms > 1 and args.overlap_pair_pgs:
            logger.error("--overlap-pair-pgs is not supported with --multi-comms > 1")
            comms_utils.gracefulExit()

        if args.process_group or args.process_group_pair:
            if not args.process_group or not args.process_group_pair:
                comms_utils.gracefulExit(
                    "--process-group and --process-group-pair should always be used together"
                )

            if args.multi_comms > 1:
                comms_utils.gracefulExit(
                    "--process-group is not supported with --multi-comms > 1"
                )

            if args.overlap_pair_pgs:
                comms_utils.gracefulExit(
                    "--process-group and --process-group-pair are not supported with --overlap-pair-pgs > 1"
                )

            if "[" not in args.process_group or "]" not in args.process_group:
                comms_utils.gracefulExit(
                    "--process-group should be a list of ranks separated by comma e.g. [0,1,2,3]"
                )

            if len(args.process_group.strip("[]").split(",")) > world_size:
                comms_utils.gracefulExit(
                    f"Number of ranks in --process-group {len(args.process_group.strip('[]').split(','))} cannot be greater than world size {world_size}"
                )

            if len(args.collective_pair) != len(args.process_group_pair):
                comms_utils.gracefulExit(
                    f"Number of pair collectives ({args.collective_pair}) and number of pair process groups ({len(args.process_group_pair)}) must be equal"
                )

            for pgIdx in range(len(args.process_group_pair)):
                if (
                    "[" not in args.process_group_pair[pgIdx]
                    or "]" not in args.process_group_pair[pgIdx]
                ):
                    comms_utils.gracefulExit(
                        "--process-group-pair should be composed of list of process groups separated by a space.\n"
                        "Each process groups is a list of ranks inside square brackets separated by comma\n"
                        "e.g. --process-group-pair [0,1] [2,3]\n"
                        f"process group {pgIdx}: {args.process_group_pair[pgIdx]} is missing the square brackets"
                    )
                number_of_ranks_in_pg = len(
                    args.process_group_pair[pgIdx].strip("[]").split(",")
                )

                if number_of_ranks_in_pg > world_size:
                    logger.error(
                        f"Number of ranks in --process-group-pair number {pgIdx}: {args.process_group_pair[pgIdx]} size:{number_of_ranks_in_pg}"
                        "cannot be greater than world size:{world_size}"
                    )
                    comms_utils.gracefulExit()

    def runColl(self, comm_fn=None, comm_fn_pair_list=None, dcheck=False):
        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_begin")

        elapsedCPUTimeNS = 0.0
        is_blocking = not self.collectiveArgs.asyncOp
        enable_comms_pair = (
            False if (comm_fn_pair_list is None or comm_fn_pair_list == []) else True
        )

        # for comms pair mode, force async comms for overlapping evaluation
        if enable_comms_pair:
            self.collectiveArgs.asyncOp = True
        for nIter in range(
            self.collectiveArgs.numWarmupIters + self.collectiveArgs.numIters
        ):
            if self.collectiveArgs.enable_profiler:
                comms_utils.sampleProfiler()
            if nIter == self.collectiveArgs.numWarmupIters:
                # Flush non-blocking ops to ensure warmup is really complete
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
                ensureTensorFlush(self.collectiveArgs.opTensor)
                if enable_comms_pair:
                    for opTensor_pair in self.collectiveArgs.opTensor_pair:
                        ensureTensorFlush(opTensor_pair)
                # Start measuring time after warmup iterations
                elapsedCPUTimeNS = 0.0
                if self.collectiveArgs.comm_dev_time:
                    self.collectiveArgs.comm_dev_time.reset()
                self.collectiveArgs.quant_time.reset()
                self.collectiveArgs.dequant_time.reset()
            # reset tensor values for data validation check
            if dcheck:
                self.setTensorVal(self.collectiveArgs.opTensor)
            # for blocking mode, do barrier before starting collective
            if is_blocking:
                # set default group to sync among all global ranks
                self.collectiveArgs.group = self.backendFuncs.get_default_group()
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            start = time.monotonic()  # available only in py3
            with paramStreamGuard(
                stream=self.backendFuncs.get_current_stream(
                    device=self.collectiveArgs.device
                ),
                curDevice=self.collectiveArgs.device,
                backendFuncs=self.backendFuncs,
                is_blocking=False,
                timer=self.collectiveArgs.comm_dev_time,
            ):
                self.collectiveArgs.group = self.collectiveArgs.groups[
                    self.collectiveArgs.pgId
                ]
                for _ in range(self.collectiveArgs.numCollPerIter):
                    comm_fn(self.collectiveArgs)
            if self.collectiveArgs.comm_dev_time:
                self.collectiveArgs.comm_dev_time.elapsedTime()
            if enable_comms_pair:
                for pairIdx in range(len(comm_fn_pair_list)):
                    comm_fn_pair = comm_fn_pair_list[pairIdx]

                    with paramStreamGuard(
                        stream=self.collectiveArgs.pair_stream_list[pairIdx],
                        curDevice=self.collectiveArgs.device,
                        backendFuncs=self.backendFuncs,
                        is_blocking=False,
                        timer=self.collectiveArgs.comm_dev_time,
                    ):
                        # post another collecitve if on comms pair mode, otherwise it's noop
                        self.collectiveArgs.group = self.collectiveArgs.groups[
                            self.collectiveArgs.pairPgId[pairIdx]
                        ]
                        for _ in range(self.collectiveArgs.numCollPerIter):
                            comm_fn_pair(
                                self.collectiveArgs,
                                pair=enable_comms_pair,
                                pairIdx=pairIdx,
                            )

            if is_blocking:  # should be sychronous, wait for the collective
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
                if self.collectiveArgs.comm_dev_time:
                    self.collectiveArgs.comm_dev_time.elapsedTime()

            # Measuring time.
            elapsedCPUTimeNS += (
                time.monotonic() - start
            ) * 1e9  # keeping time in NS, helps in divising data by nanosecond

        start = time.monotonic()  # available only in py3
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)
        end = time.monotonic()  # available only in py3

        ensureTensorFlush(self.collectiveArgs.opTensor)
        if enable_comms_pair:
            for opTensor_pair in self.collectiveArgs.opTensor_pair:
                ensureTensorFlush(opTensor_pair)

        elapsedCPUTimeNS += (
            end - start
        ) * 1e9  # keeping time in NS, helps in divising data by nanoseconds

        memSize = self.backendFuncs.get_mem_size(self.collectiveArgs)
        if self.collectiveArgs.comm_dev_time:
            elapsedTimeNS = self.collectiveArgs.comm_dev_time.elapsedTimeNS
            logger.debug(
                f"elapsedCPUTimeNS={elapsedCPUTimeNS/self.collectiveArgs.numIters}, elapsedDeviceTimeNS={elapsedTimeNS/self.collectiveArgs.numIters}."
            )
        else:
            elapsedTimeNS = elapsedCPUTimeNS

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
        if enable_comms_pair:
            memSize_pair = self.backendFuncs.get_mem_size(
                self.collectiveArgs, pair=enable_comms_pair
            )
            memSize += memSize_pair

            _, algBW_pair = comms_utils.getAlgBW(
                elapsedTimeNS,
                memSize_pair,
                self.collectiveArgs.numIters * self.collectiveArgs.numCollPerIter,
            )
            algBW += algBW_pair

            for pair_idx in range(len(self.collectiveArgs.collective_pair)):
                busBW += self.backendFuncs.getBusBW(
                    self.collectiveArgs.collective_pair[pair_idx],
                    algBW_pair,
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
        (
            global_rank,
            world_size,
            allSizes,
        ) = super().initCollectiveArgs(commsParams)

        if commsParams.sizes_pair is not None:
            allSizes_pair = commsParams.sizes_pair
            if self.report:
                logger.info(
                    f"Benchmarking with user-specified pair message sizes {allSizes_pair}"
                )
        else:
            allSizes_pair = allSizes
        self.collectiveArgs.pair = commsParams.pair
        self.collectiveArgs.collective_pair = commsParams.collective_pair
        if self.collectiveArgs.pair:
            self.collectiveArgs.pair_stream_list = [
                self.backendFuncs.get_new_stream()
                for _ in self.collectiveArgs.collective_pair
            ]

        return (
            global_rank,
            world_size,
            allSizes,
            allSizes_pair,
        )

    def gatherBenchTime(self, collectiveArgs, commsParams, timeUsElapsedList):
        # Push the list to device, then do an all-gather.
        timeElapsedTensor = torch.tensor(
            timeUsElapsedList,
            device=(
                self.backendFuncs.get_device()
                if commsParams.backend == "nccl"
                else torch.device("cpu")
            ),
        )
        collectiveArgs.opTensor = None
        if commsParams.backend != "xla":
            timeList = list(
                torch.ones(
                    (self.comm_size,) + timeElapsedTensor.shape,
                    dtype=timeElapsedTensor.dtype,
                    device=timeElapsedTensor.device,
                ).unbind(0)
            )
            collectiveArgs.opTensor = timeList

        collectiveArgs.ipTensor = timeElapsedTensor
        collectiveArgs.dataSize = (
            timeElapsedTensor.nelement() * timeElapsedTensor.element_size()
        )
        collectiveArgs.numElements = timeElapsedTensor.nelement()

        # use allgather as all process group should support it
        self.backendFuncs.all_gather(collectiveArgs)
        self.backendFuncs.complete_accel_ops(collectiveArgs)

        return timeList

    def printPreamble(self, commsParams):
        logger.debug(f"\tcommsParams: {str(commsParams.__dict__)}")
        header = "\n\tCOMMS-RES"

        if self.collectiveArgs.collective == "pt2pt":
            fmt = "{:>40}{:>20}{:>10}{:>10}{:>25}{:>10}{:>10}{:>15}{:>15}{:>18}{:>18}"
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
            )
        else:
            if commsParams.bitwidth < 32:
                fmt = "-QUANT\t{:>40}{:>18}{:>25}{:>15}{:>15}{:>15}"
                header += fmt.format(
                    "size (B)",
                    "nElementsPerRank",
                    "P95 Latency(us): Quant",
                    "Comms",
                    "De-Quant",
                    "Overall",
                    "TotalLatency(us):p50",
                )
            elif not self.collectiveArgs.pair:
                fmt = "{:>40}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
                header += fmt.format(
                    "size (B)",
                    "nElementsPerRank",
                    "Latency(us):p50",
                    "p75",
                    "p95",
                    "Min",
                    "Max",
                    "AlgBW(GB/s)",
                    "BusBW(GB/s)",
                    "TotalLatency(us):p50",
                )
            else:
                fmt = "{:>40}{:>18}{:>22}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
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
                )

        print(header)

    def reportBenchTimeCollWithQuant(
        self,
        commsParams,
        results,
        tensorList,
        quantTimeTensorList,
        dequantTimeTensorList,
    ):
        latencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(tensorList)
        # quant tensor
        quantLatencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(
            quantTimeTensorList
        )
        # dequant tensor
        dequantLatencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(
            dequantTimeTensorList
        )

        p95 = np.percentile(latencyAcrossRanks, 95)

        quant_p95 = np.percentile(quantLatencyAcrossRanks, 95)
        dequant_p95 = np.percentile(dequantLatencyAcrossRanks, 95)

        print(
            "\tCOMMS-RES-QUANT-{}-{}{}\t{:>15}{:>18}{:>25}{:>15}{:>15}{:>15}".format(
                self.collectiveArgs.collective,
                self.collectiveArgs.data_type,
                self.tag,
                results["memSize"],
                str("%d" % (results["numElements"])),
                str("%.1f" % (quant_p95)),
                str("%.1f" % (p95 - quant_p95 - dequant_p95)),
                str("%.1f" % (dequant_p95)),
                str("%.1f" % (p95)),
                # str("%.3f" % (algBW)),
                # str("%.3f" % (busBW)),
            )
        )

        return commsQuantCollPerfMetrics(
            self.collectiveArgs.collective,
            self.collectiveArgs.data_type,
            benchType.Collective,
            commsParams.backend,
            self.tag,
            results["memSize"],
            results["memSize"],
            results["numElements"],
            results["numElements_pair"] if "numElements_pair" in results else 0,
            float(quant_p95),
            float(p95 - quant_p95 - dequant_p95),
            float(dequant_p95),
            float(p95),
        )

    def reportBenchTime(
        self,
        collectiveArgs,
        commsParams,
        results,
        tensorList,
        quantTimeTensorList,
        dequantTimeTensorList,
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

    def reportBenchTimeColl(self, commsParams, results, tensorList):
        latencyAcrossRanks = self.backendFuncs.tensor_list_to_numpy(tensorList)
        logger.debug(f"Latency across all ranks: {latencyAcrossRanks}")

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

        # report original cpu time as total time
        total_p50 = 0.0

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

        if not self.collectiveArgs.pair:
            fmt = "\tCOMMS-RES-{}-{}{}{:>18}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
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
                )
            )
        else:
            # convernt to # of elements per rank
            if commsParams.collective_pair in (
                "all_to_all",
                "all_to_allv",
                "all_to_all_single",
            ):
                results["numElements_pair"] = int(
                    results["numElements_pair"] // self.backendFuncs.get_world_size()
                )
            fmt = "\tCOMMS-RES-{}-{}{}{:>18}{:>18}{:>22}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
            print(
                fmt.format(
                    self.collectiveArgs.collective,
                    self.collectiveArgs.data_type,
                    self.tag,
                    results["memSize"],
                    str("%d" % (results["numElements"])),
                    str("%d" % (results["numElements_pair"])),
                    str("%.1f" % (p50)),
                    str("%.1f" % (p75)),
                    str("%.1f" % (p95)),
                    str("%.1f" % (minlat)),
                    str("%.1f" % (maxlat)),
                    str("%.3f" % (algBW)),
                    str("%.3f" % (busBW)),
                    str("%.1f" % (total_p50)),
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
            results["numElements_pair"] if "numElements_pair" in results else 0,
            float(p50),
            float(p75),
            float(p95),
            float(minlat),
            float(maxlat),
            algBW,
            busBW,
        )

    def benchTime(self, index, commsParams, backendFuncs):
        for coll in commsParams.collective_list:
            logger.debug("Running collective: %s" % coll)
            commsParams.collective = coll
            self.benchComm(index, commsParams, backendFuncs)

    def benchComm(self, index, commsParams, backendFuncs):
        # Get NW stack specific parameters
        (
            global_rank,
            world_size,
            allSizes,
            allSizes_pair,
        ) = self.initCollectiveArgs(commsParams)

        backendFuncs.sync_barrier(self.collectiveArgs)
        if self.report:
            self.printPreamble(commsParams)

        for curSize, curSize_pair in zip(allSizes, allSizes_pair):
            results = {}
            timeUsElapsedList = []
            quantTimeElapsedList = []
            dequantTimeElapsedList = []
            numElements = int(curSize // commsParams.element_size)
            collectiveFunc = self.backendFuncs.noop
            collectiveFunc_pair_list = []

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

            # comms-pair specific initializations
            if commsParams.pair:
                # set corresponding function pointers
                for collective in commsParams.collective_pair:
                    collectiveFunc_pair_list.append(
                        backendFuncs.collectiveFunc[collective]
                    )

                # TODO: allow user to set specific size
                # Setup the arguments.
                self.collectiveArgs.dataSize_pair = curSize_pair
                self.collectiveArgs.numElements_pair = int(
                    self.collectiveArgs.dataSize_pair // commsParams.element_size
                )
                results["numElements_pair"] = self.collectiveArgs.numElements_pair
                commsArgs = comms_utils.commsArgs()
                commsArgs.inMsgSize = self.collectiveArgs.numElements_pair
                commsArgs.outMsgSize = self.collectiveArgs.numElements_pair

                for pairIdx in range(len(commsParams.collective_pair)):
                    commsArgs.worldSize = self.collectiveArgs.world_size_pair[
                        pairIdx + 1
                    ]
                    commsArgs.comms = commsParams.collective_pair[pairIdx]
                    (
                        ipTensor_pair,
                        opTensor_pair,
                    ) = self.prepComm(
                        curComm=commsArgs,
                        commsParams=commsParams,
                    )
                    if len(self.collectiveArgs.ipTensor_pair) < pairIdx + 1:
                        self.collectiveArgs.ipTensor_pair.append(ipTensor_pair)
                        self.collectiveArgs.opTensor_pair.append(opTensor_pair)
                    else:
                        self.collectiveArgs.ipTensor_pair[pairIdx] = ipTensor_pair
                        self.collectiveArgs.opTensor_pair[pairIdx] = opTensor_pair

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
                        comm_fn_pair_list=collectiveFunc_pair_list,
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
                )

            self.backendFuncs.sync_barrier(
                self.collectiveArgs, desc=f"curSize_{curSize}"
            )

        comms_utils.clearQuantCommCtx(self.collectiveArgs)

        # wait rank 0 reports results to avoid other ranks mess up the output
        self.backendFuncs.sync_barrier(self.collectiveArgs, "benchtime")

    def genMultiCommGroups(
        self,
        multi_comms,
        backend,
        pair,
        pair_collectives_list,
        overlap_pair_pgs,
        process_group,
        process_groups_pair,
    ):
        self.collectiveArgs.pgId = 0  # default group id
        self.collectiveArgs.pairPgId = []

        global_rank = self.backendFuncs.get_global_rank()
        world_size = self.backendFuncs.get_world_size()
        groupRanks = {}

        if process_group and process_groups_pair:
            groupRanks[0] = [int(p) for p in process_group.strip("[]").split(",")]

            groupRanks = groupRanks | {
                i + 1: list(map(int, val.replace("[", "").replace("]", "").split(",")))
                for i, val in enumerate(process_groups_pair)
            }
            self.collectiveArgs.world_size_pair = [len(v) for v in groupRanks.values()]

            self.collectiveArgs.pairPgId = [
                i + 1 for i in range(len(pair_collectives_list))
            ]

            print(f"groupRanks {groupRanks}")
            self.backendFuncs.groupRanks = groupRanks
            self.backendFuncs.initialize_groups(backend=backend, force_new_group=True)

        elif multi_comms > 1:
            self.collectiveArgs.pgId = global_rank % multi_comms
            for pgId in range(multi_comms):
                groupRanks[pgId] = []
            for rank in range(world_size):
                pgId = rank % multi_comms
                groupRanks[pgId].append(rank)
            for pgId in range(multi_comms):
                logger.info(
                    f"PARAM COMMS Rank {global_rank} created group {pgId} with ranks {groupRanks[pgId]}"
                )

            # FIXME: how to proper generate groupRanks before initializing backend?
            self.backendFuncs.groupRanks = groupRanks
            self.backendFuncs.initialize_groups(backend)

        elif pair and overlap_pair_pgs:
            # create two communicators each including all ranks
            num_pgs = 1 + len(pair_collectives_list)
            for pgId in range(0, num_pgs):
                if pgId > 0:
                    self.collectiveArgs.pairPgId.append(pgId)
                groupRanks[pgId] = []
                for rank in range(0, world_size):
                    groupRanks[pgId].append(rank)
                logger.info(
                    f"PARAM COMMS Rank {global_rank} created group {pgId} with ranks {groupRanks[pgId]}"
                )
            self.backendFuncs.groupRanks = groupRanks
            self.backendFuncs.initialize_groups(backend=backend, force_new_group=True)

        else:
            # default is single group including all ranks.
            # create the same groupRanks argument for simple
            # query in later logic no matter the group splitting
            groupRanks[0] = []
            for rank in range(0, world_size):
                groupRanks[0].append(rank)

        return groupRanks

    def runBench(self, commsParams):
        try:
            self.backendFuncs.benchmark_comms(self.benchTime, commsParams)
        except ValueError as ve:
            logger.critical(repr(ve))
            raise


def main():
    collBenchObj = commsOverlapBench()

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="PARAM-CommOverlap Benchmark",
        formatter_class=MultilineFormatter,
        allow_abbrev=False,
    )
    collBenchObj.readArgs(parser)
    args, _ = parser.parse_known_args()

    comms_env_params = comms_utils.read_comms_env_vars()
    if comms_env_params["global_rank"] == 0 or (
        args.enable_local_report and comms_env_params["local_rank"] == 0
    ):
        print("\t PARAM COMM-OVERLAP environment: %s " % (str(comms_env_params)))
        print(
            "\t backend: %s nw-stack: %s args.data_types: %s args.b: %s args.e: %s args.f: %s args.z: %s args.master_ip: %s "
            % (
                args.backend,
                args.nw_stack,
                args.data_types,
                args.b,
                args.e,
                args.f,
                args.z,
                args.master_ip,
            )
        )

    collBenchObj.checkBasicArgs(args)

    # Initialize backend
    bootstrap_info = comms_utils.bootstrap_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, comms_env_params
    )
    commsParamsBase = comms_utils.commsParamsHolderBase(args)
    collBenchObj.initBackend(bootstrap_info, commsParamsBase)

    # Dedupes and syncs value for args.data_types based on args.data_type/args.dtype if not passed in args.
    collBenchObj.syncCommBenchDataTypes(args)

    collBenchObj.checkArgs(args)

    groupRanks = collBenchObj.genMultiCommGroups(
        args.multi_comms,
        args.backend,
        args.pair,
        args.collective_pair,
        args.overlap_pair_pgs,
        args.process_group,
        args.process_group_pair,
    )

    for data_type in args.data_types:
        args.data_type = data_type.lower()

        collBenchObj.checkArgsdataType(args)
        element_size = torch.ones([1], dtype=args.dtype).element_size()

        commsParams = comms_utils.commsOverlapParamsHolder(
            args, bootstrap_info, element_size, collBenchObj.benchTime, groupRanks
        )

        collBenchObj.runBench(commsParams)


if __name__ == "__main__":
    main()
