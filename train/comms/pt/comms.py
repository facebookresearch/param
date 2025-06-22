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
from param_bench.train.comms.pt.comms_utils import (
    bootstrap_info_holder,
    commsParamsHolder,
    commsParamsHolderBase,
    ensureTensorFlush,
    MultilineFormatter,
    paramCommsBench,
    paramDeviceTimer,
    paramStreamGuard,
)
from param_bench.train.comms.pt.logger_utils import (
    benchType,
    commsCollPerfMetrics,
    commsPt2PtPerfMetrics,
    commsQuantCollPerfMetrics,
    customized_perf_loggers,
)

from param_bench.train.comms.pt.pytorch_backend_utils import (
    backendFunctions,
    pt2ptPatterns,
    supportedC10dBackends,
    supportedCollectives,
)

logger = logging.getLogger(__name__)


# define the collective benchmark
class commsCollBench(paramCommsBench):
    def __init__(self):
        super().__init__(supportedNwstacks=["pytorch-dist", "pytorch-xla-tpu"])
        self.tag = ""
        self.backendFuncs = None

    def readArgs(self, parser):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--w", type=int, default=5, help="number of warmup iterations"
        )  # number of warmup-iterations
        parser.add_argument(
            "--n", "--num_iters", type=int, default=5, help="number of iterations"
        )  # number of iterations
        parser.add_argument(
            "--num-coll",
            "--num-coll-per-iteration",
            type=int,
            default=1,
            help="number of collective operations to execute for every iteration",
        )  # number of collective operations to execute for every iteration
        parser.add_argument(
            "--b",
            "--begin-size",
            type=str,
            default="8",
            help="minimum size, in bytes, to start with",
        )  # COMMS mode, begin the sweep at.
        parser.add_argument(
            "--e",
            "--end-size",
            type=str,
            default="64",
            help="maximum size, in bytes, to end at",
        )  # COMMS mode, end the sweep at.
        parser.add_argument(
            "--f", type=int, default=2, help="multiplication factor between sizes"
        )  # COMMS mode, multiplication factor.
        parser.add_argument(
            "--sb",
            type=int,
            default=0,
            help="step bytes between sizes, 0 value disables step increment and uses multiplication factor instead",
        )  # COMMS mode, additive step bytes for sizes.
        parser.add_argument(
            "--i",
            "--in-split",
            type=lambda s: [int(item) for item in s.split(",") if item],
            default=None,
            help="comma-separated split of number of elements in input tensor",
        )  # COMMS mode, input tensor split, by number of elements. Overrides --b and --e.
        parser.add_argument(
            "--o",
            "--out-split",
            type=lambda s: [int(item) for item in s.split(",") if item],
            default=None,
            help="comma-separated split of number of elements in output tensor",
        )  # COMMS mode, output tensor split, by number of elements.
        parser.add_argument(
            "--ss",
            "--sizes",
            type=lambda s: [int(item) for item in s.split(",") if item],
            default=None,
            help="benchmark only specified sizes, comma-separated",
        )  # COMMS mode, use specified sizes instead of increasing from small to large
        parser.add_argument(
            "--data-types",
            "--data-type",
            type=lambda s: [str(item) for item in s.split(",") if item],
            default="float32",
            help="comma-separated list of datatypes, supports "
            + str(self.supportedDtype),
        )  # The comma-separated list of data-types
        parser.add_argument(
            "--collective",
            "--collectives",
            type=str,
            default="all_reduce",
            help="Collective operation(s) to be evaluated, separated by comma if multiple ops are provided. "
            "supportedCollectives: {}".format(supportedCollectives),
        )  # collective op to benchmark
        parser.add_argument(
            "--root", type=int, default=0, help="root process for reduce benchmark"
        )  # root process for reduce and bcast (and gather, scatter, etc., if support in the future)
        # TODO: check the correctness of root, should be between 0 to [world_size -1]
        parser.add_argument(
            "--src-ranks",
            type=str,
            nargs="?",
            help="R|src ranks for many-to-one incast pattern or pt2pt.\n"
            "List of ranks separated by comma or a range specified by start:end.\n"
            "Pt2pt one2one should set only one rank.\n"
            "The default value of incast includes all ranks, pt2pt includes rank 0.",
        )  # optional: group of src ranks in many-to-one incast or pt2pt
        parser.add_argument(
            "--dst-ranks",
            type=str,
            nargs="?",
            help="R|dst ranks for one-to-many multicast pattern or pt2pt.\n"
            "List of ranks separated by comma or a range specified by start:end.\n"
            "Pt2pt one2one should set only one rank\n"
            "The default value of multicast includes all ranks, pt2pt includes rank 1.",
        )  # optional: group of dst ranks in one-to-many multicast or pt2pt
        parser.add_argument(
            "--multi-comms",
            type=int,
            default=1,
            help="Set to enable multi-comm group mode, cannot use together with --overlap-pair-pgs mode. Default 1 comm group",
        )
        parser.add_argument(
            "--pt2pt",
            type=str,
            default=None,
            help="point to point pattern",
            choices=pt2ptPatterns,
        )  # point to point mode
        parser.add_argument(
            "--window",
            type=int,
            default=100,
            help="window size for pt2pt throughput test",
        )  # optional:  point to point throughput test window size
        parser.add_argument(
            "--size-start-profiler",
            type=str,
            default=None,
            help="execute pytorch profiler at specified size",
        )  # execute pytorch profiler at specified size if applicable
        parser.add_argument(
            "--tag",
            type=str,
            default=None,
            help="customized tag or keyword to be added into final output lines",
        )  # execute pytorch profiler at specified size if applicable
        parser.add_argument(
            "--include-0B",
            action="store_true",
            default=False,
            help="Select some ranks to send/receive 0B messages",
        )
        parser.add_argument(
            "--graph-launches",
            type=int,
            default=0,
            help="Number of graph launches for each data-size",
        )
        parser.add_argument(
            "--use-device-time",
            action="store_true",
            default=False,
            help="use device time measurement",
        )

    def _checkPt2Pt(self, args):
        if args.pt2pt is None:
            return args.collective
        if args.pt2pt not in pt2ptPatterns:
            logger.error(
                f"Specified pt2pt pattern: {args.pt2pt} is not one of the supported pt2pt patterns: {str(pt2ptPatterns)}"
            )
            comms_utils.gracefulExit()
        return "pt2pt"

    def _check_for_in_out_split(self, args, element_size):
        if args.i is None and args.o is None:
            return args.b, args.e

        if args.i is not None:
            supported_split_coll = ["reduce_scatter_v", "all_to_allv"]
            inout_len = sum(args.i)
        else:
            supported_split_coll = ["all_gather_v", "all_to_allv"]
            inout_len = sum(args.o)

        if not any(coll in args.collective.split(",") for coll in supported_split_coll):
            logger.error(
                "Collective does not support input-split argument (--i) or output-split argument (--o)"
            )
            comms_utils.gracefulExit()

        begin = inout_len * element_size
        end = begin
        logger.warning(
            f"Overwriting begin-size (--b {args.b}) with {begin} and end-size (--e {args.e}) with {end} to match requested input-split (--i) {args.i} or output-split (--o) {args.o}"
        )
        return begin, end

    def _check_device_type(self, args):
        if args.device == "cpu" and args.backend == "nccl":
            raise ValueError(f"NCCL is not supported for device type {args.device}")

        # Overwrite user-input rocm device as we internally use cuda for both GPUs
        if args.device == "rocm":
            return "cuda"
        return args.device

    def _check_bitwidth(self, args):
        if args.bitwidth >= 32:
            return
        if args.device != "cuda":
            logger.error(
                f"collective quantization may not be fully supported for {args.device}"
            )
        for coll in args.collective.split(","):
            comms_utils.checkQuantArgs(
                coll,
                args.dtype,
                args.b,
                args.quant_a2a_embedding_dim,
                args.z,
            )

    def syncCommBenchDataTypes(self, args):
        args.data_types = list(args.data_types)
        if args.data_types is None:
            # If args --data-types is missing, replace it with value passed for --data-type arg.
            if args.data_type is not None:
                args.data_types = [args.data_type]

            # If both --data-types and --data-type are not present, args.data_types is set to default value for dtype(ie; "float32")
            else:
                key = [
                    key for key, value in self.dtypeMap.items() if value == self.dtype
                ][0]
                args.data_types = [key]

    # Check basic arguments that are unchanged for all benchmarks in a single run
    def checkBasicArgs(self, args):
        super().checkArgs(args)

        args.collective = self._checkPt2Pt(args)
        args.device = self._check_device_type(args)

        if args.size_start_profiler:
            args.size_start_profiler = comms_utils.parsesize(args.size_start_profiler)

        self.tag = f"-{args.tag}" if args.tag is not None else ""

    # Check arguments that may be custmized per benchmark in a single run
    # does not depend on data type
    def checkArgs(self, args):  # noqa: C901
        reduce_ops = ["all_reduce", "reduce", "reduce_scatter", "reduce_scatter_v"]
        if (
            args.c == 1
            and args.z == 0
            and any(coll in args.collective.split(",") for coll in reduce_ops)
        ):
            logger.warning(
                f"Data validation is not supported for {reduce_ops} in non-blocking mode, disabled and continue"
            )
            args.c = 0

        world_size = self.backendFuncs.get_world_size()
        if args.i is not None and (world_size != len(args.i)):
            logger.error("An input split must be provided for all participating ranks")
            comms_utils.gracefulExit()

        if args.o is not None and (world_size != len(args.o)):
            logger.error("An output split must be provided for all participating ranks")
            comms_utils.gracefulExit()

        if args.src_ranks:
            args.src_ranks = comms_utils.parseRankList(args.src_ranks)
            if len(args.src_ranks) == 0 or any(
                r < 0 or r >= world_size for r in args.src_ranks
            ):
                logger.error(f"wrong src_ranks ({args.src_ranks})")
                comms_utils.gracefulExit()

        if args.dst_ranks:
            args.dst_ranks = comms_utils.parseRankList(args.dst_ranks)
            if len(args.dst_ranks) == 0 or any(
                r < 0 or r >= world_size for r in args.dst_ranks
            ):
                logger.error(f"wrong dst_ranks ({args.dst_ranks})")
                comms_utils.gracefulExit()

        if args.graph_launches > 0 and args.device != "cuda":
            logger.error("cuda graph is only supported for cuda or rocm device")
            comms_utils.gracefulExit()

    # depnds on data type
    def checkArgsdataType(self, args):  # noqa: C901
        args.b = comms_utils.parsesize(args.b)
        args.e = comms_utils.parsesize(args.e)

        if args.data_type not in self.supportedDtype:
            logger.error(
                f"Specified dtype: {args.data_type} is not one of the supported commstyle: {str(self.supportedDtype)}"
            )
            comms_utils.gracefulExit()
        if args.data_type == "bfloat16" and args.backend == "gloo":
            logger.error(
                f"Specified dtype: {args.data_type} does not work with gloo backend"
            )
            comms_utils.gracefulExit()

        args.dtype = self.dtypeMap[args.data_type]
        element_size = torch.ones([1], dtype=args.dtype).element_size()

        args.b, args.e = self._check_for_in_out_split(args, element_size)

        if args.b < 1:
            logger.warning(
                f"Starting size (--b {args.b}) should be greater than 1 byte...fix and continue"
            )
            args.b = 1

        if args.e < args.b:
            logger.warning(
                f"the begin-size (--b {args.b}) is larger than the end-size (--e {args.e})"
            )

        if args.sb % element_size != 0:
            logger.error("Step size bytes must be a multiple of element size")
            comms_utils.gracefulExit()

        # run a few sanity checks
        self._check_bitwidth(args)

    def run_coll_cuda_graph(self, comm_fn=None, dcheck=False):
        self.backendFuncs.sync_barrier(
            self.collectiveArgs, desc="run_coll_cuda_graph_begin"
        )
        elapsedCPUTimeNS = 0.0

        # 1. Warmup phase
        # launch collective on a separate stream and sync with current_stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.collectiveArgs.numWarmupIters):
                if self.collectiveArgs.enable_profiler:
                    comms_utils.sampleProfiler()
                comm_fn(self.collectiveArgs)
        torch.cuda.current_stream().wait_stream(s)

        # 2. capturing graph
        # in cuda graph, we need to use sync mode
        # TODO: this might need PTD fix (async_op=True won't work under cuda graph)
        self.collectiveArgs.asyncOp = False
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(self.collectiveArgs.numIters):
                if dcheck:
                    # reset input tensor for data validation
                    self.setTensorVal(self.collectiveArgs.ipTensor)
                comm_fn(self.collectiveArgs)

        # 3. Replay
        start = time.monotonic()  # available only in py3
        for _ in range(self.collectiveArgs.graph_launches):
            if self.collectiveArgs.enable_profiler:
                comms_utils.sampleProfiler()

            # [optional] we can feed new input data to ipTensor for each replay
            g.replay()

        self.backendFuncs.complete_accel_ops(self.collectiveArgs)

        end = time.monotonic()  # available only in py3

        ensureTensorFlush(self.collectiveArgs.opTensor)

        elapsedCPUTimeNS += (
            end - start
        ) * 1e9  # keeping time in NS, helps in divising data by nanoseconds
        elapsedTimeNS = elapsedCPUTimeNS
        logger.debug(f"elapsedCPUTimeNS={elapsedCPUTimeNS}")

        memSize = self.backendFuncs.get_mem_size(self.collectiveArgs)

        avgIterNS, algBW = comms_utils.getAlgBW(
            elapsedTimeNS,
            memSize,
            self.collectiveArgs.numIters
            * self.collectiveArgs.numCollPerIter
            * self.collectiveArgs.graph_launches,
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

    def run_coll_non_graph(self, comm_fn=None, dcheck=False):
        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_begin")

        elapsedCPUTimeNS = 0.0
        is_blocking = not self.collectiveArgs.asyncOp

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
                elapsedCPUTimeNS = 0.0
                if self.collectiveArgs.use_device_time:
                    self.collectiveArgs.comm_dev_time.reset()
                self.collectiveArgs.quant_time.reset()
                self.collectiveArgs.dequant_time.reset()
            # reset tensor values for data validation check
            if dcheck:
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
                is_blocking=False,
                timer=self.collectiveArgs.comm_dev_time,
            ):
                self.collectiveArgs.group = self.collectiveArgs.groups[
                    self.collectiveArgs.pgId
                ]
                for _ in range(self.collectiveArgs.numCollPerIter):
                    comm_fn(self.collectiveArgs)

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

        elapsedCPUTimeNS += (
            end - start
        ) * 1e9  # keeping time in NS, helps in divising data by nanoseconds

        memSize = self.backendFuncs.get_mem_size(self.collectiveArgs)
        if self.collectiveArgs.use_device_time:
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

    def runColl(self, comm_fn=None, dcheck=False):
        return (
            self.run_coll_non_graph(comm_fn, dcheck)
            if self.collectiveArgs.graph_launches == 0
            else self.run_coll_cuda_graph(comm_fn, dcheck)
        )

    def runPt2Pt(self):
        self.backendFuncs.sync_barrier(self.collectiveArgs)
        # warm-up
        memSize = self.backendFuncs.get_mem_size(self.collectiveArgs)
        self.getPingLatency(self.collectiveArgs.numWarmupIters)
        self.getPingPongLatency(self.collectiveArgs.numWarmupIters)
        self.getUniBW(self.collectiveArgs.numWarmupIters, memSize)
        self.getBiBW(self.collectiveArgs.numWarmupIters, memSize)
        self.backendFuncs.sync_barrier(self.collectiveArgs, "runpt2pt_begin")
        # pt2pt benchmark
        pingPerIterNS = self.getPingLatency(self.collectiveArgs.numIters)
        pingPongPerIterNS = self.getPingPongLatency(self.collectiveArgs.numIters)
        avgUniBW = self.getUniBW(self.collectiveArgs.numIters, memSize)
        avgBiBW = self.getBiBW(self.collectiveArgs.numIters, memSize)
        self.backendFuncs.sync_barrier(self.collectiveArgs, "runpt2pt")
        results = {
            "pingPerIterNS": pingPerIterNS,
            "pingPongPerIterNS": pingPongPerIterNS,
            "avgUniBW": avgUniBW,
            "avgBiBW": avgBiBW,
            "memSize": memSize,
        }
        return results

    def getPingLatency(self, numIters):
        logger.debug(
            "STATUS: begin ping test with src_ranks=%s, dst_ranks=%s."
            % (self.collectiveArgs.src_ranks, self.collectiveArgs.dst_ranks)
        )
        self.collectiveArgs.asyncOp = False
        # get one-way latency
        pingLatencyNS = []
        for _ in range(numIters):
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            start = time.monotonic()
            if self.collectiveArgs.global_rank in self.collectiveArgs.src_ranks:
                idx = self.collectiveArgs.src_ranks.index(
                    self.collectiveArgs.global_rank
                )
                self.collectiveArgs.dst_rank = self.collectiveArgs.dst_ranks[idx]
                self.backendFuncs.send(
                    collectiveArgs=self.collectiveArgs,
                )
            elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                idx = self.collectiveArgs.dst_ranks.index(
                    self.collectiveArgs.global_rank
                )
                self.collectiveArgs.src_rank = self.collectiveArgs.src_ranks[idx]
                self.backendFuncs.recv(
                    collectiveArgs=self.collectiveArgs,
                )
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            pingLatencyNS.append(
                (time.monotonic() - start) * 1e9
            )  # keeping time in NS, helps in divising data by nanosecond
        logger.debug("STATUS: end ping test.")
        return pingLatencyNS

    def getPingPongLatency(self, numIters):
        logger.debug(
            "STATUS: begin ping-pong with src_ranks=%s, dst_ranks=%s."
            % (self.collectiveArgs.src_ranks, self.collectiveArgs.dst_ranks)
        )
        self.collectiveArgs.asyncOp = False
        # get round-trip latency
        pingPongLatencyNS = []
        for _ in range(numIters):
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            start = time.monotonic()
            if self.collectiveArgs.global_rank in self.collectiveArgs.src_ranks:
                idx = self.collectiveArgs.src_ranks.index(
                    self.collectiveArgs.global_rank
                )
                self.collectiveArgs.dst_rank = self.collectiveArgs.dst_ranks[idx]
                self.backendFuncs.send(
                    collectiveArgs=self.collectiveArgs,
                )
                self.collectiveArgs.src_rank = self.collectiveArgs.dst_ranks[idx]
                self.backendFuncs.recv(
                    collectiveArgs=self.collectiveArgs,
                )
            elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                idx = self.collectiveArgs.dst_ranks.index(
                    self.collectiveArgs.global_rank
                )
                self.collectiveArgs.src_rank = self.collectiveArgs.src_ranks[idx]
                self.backendFuncs.recv(
                    collectiveArgs=self.collectiveArgs,
                )
                self.collectiveArgs.dst_rank = self.collectiveArgs.src_ranks[idx]
                self.backendFuncs.send(
                    collectiveArgs=self.collectiveArgs,
                )
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            pingPongLatencyNS.append(
                (time.monotonic() - start) * 1e9
            )  # keeping time in NS, helps in divising data by nanosecond
        logger.debug("STATUS: end ping-pong test.")
        return pingPongLatencyNS

    def getUniBW(self, numIters, memSize):
        logger.debug(
            "STATUS: begin UniBW test with src_ranks=%s, dst_ranks=%s."
            % (self.collectiveArgs.src_ranks, self.collectiveArgs.dst_ranks)
        )
        self.collectiveArgs.asyncOp = True
        # get unidirectional bandwidth
        uniLatencyNS = []
        for _ in range(numIters):
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            start = time.monotonic()
            for w in range(self.collectiveArgs.window):
                if self.collectiveArgs.global_rank in self.collectiveArgs.src_ranks:
                    idx = self.collectiveArgs.src_ranks.index(
                        self.collectiveArgs.global_rank
                    )
                    self.collectiveArgs.dst_rank = self.collectiveArgs.dst_ranks[idx]
                    self.collectiveArgs.collective = "send"
                    self.backendFuncs.P2POp(
                        collectiveArgs=self.collectiveArgs,
                        tag=w,
                    )
                elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                    idx = self.collectiveArgs.dst_ranks.index(
                        self.collectiveArgs.global_rank
                    )
                    self.collectiveArgs.src_rank = self.collectiveArgs.src_ranks[idx]
                    self.collectiveArgs.collective = "recv"
                    self.backendFuncs.P2POp(
                        collectiveArgs=self.collectiveArgs,
                        tag=w,
                    )
            self.backendFuncs.batch_isend_irecv(self.collectiveArgs)

            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            uniLatencyNS.append(
                (time.monotonic() - start) * 1e9
            )  # keeping time in NS, helps in divising data by nanosecond
        uniLatencyNS = [lat / self.collectiveArgs.window for lat in uniLatencyNS]
        uniLatencyNS = np.mean(np.array(uniLatencyNS))
        _, avgUniBW = comms_utils.getAlgBW(
            uniLatencyNS, memSize, self.collectiveArgs.numCollPerIter
        )
        logger.debug("STATUS: end UniBW test.")
        return avgUniBW

    def getBiBW(self, numIters, memSize):
        logger.debug(
            "STATUS: begin BiBW test with src_ranks=%s, dst_ranks=%s."
            % (self.collectiveArgs.src_ranks, self.collectiveArgs.dst_ranks)
        )
        self.collectiveArgs.asyncOp = True
        # get bidirectional bandwidth
        biLatencyNS = []
        for _ in range(numIters):
            self.backendFuncs.sync_barrier(self.collectiveArgs)
            start = time.monotonic()
            for w in range(self.collectiveArgs.window):
                if self.collectiveArgs.global_rank in self.collectiveArgs.src_ranks:
                    idx = self.collectiveArgs.src_ranks.index(
                        self.collectiveArgs.global_rank
                    )
                    self.collectiveArgs.collective = "send"
                    self.collectiveArgs.src_rank = self.collectiveArgs.dst_ranks[idx]
                    self.collectiveArgs.dst_rank = self.collectiveArgs.dst_ranks[idx]
                    self.backendFuncs.P2POp(
                        collectiveArgs=self.collectiveArgs,
                        tag=w,
                    )
                    self.collectiveArgs.collective = "recv"
                    self.backendFuncs.P2POp(
                        collectiveArgs=self.collectiveArgs,
                        tag=w + self.collectiveArgs.window,
                    )
                elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                    idx = self.collectiveArgs.dst_ranks.index(
                        self.collectiveArgs.global_rank
                    )
                    self.collectiveArgs.src_rank = self.collectiveArgs.src_ranks[idx]
                    self.collectiveArgs.dst_rank = self.collectiveArgs.src_ranks[idx]
                    self.collectiveArgs.collective = "recv"
                    self.backendFuncs.P2POp(
                        collectiveArgs=self.collectiveArgs,
                        tag=w,
                    )
                    self.collectiveArgs.collective = "send"
                    self.backendFuncs.P2POp(
                        collectiveArgs=self.collectiveArgs,
                        tag=w + self.collectiveArgs.window,
                    )
            self.backendFuncs.batch_isend_irecv(self.collectiveArgs)
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            biLatencyNS.append(
                (time.monotonic() - start) * 1e9
            )  # keeping time in NS, helps in divising data by nanosecond
        biLatencyNS = [lat / self.collectiveArgs.window for lat in biLatencyNS]
        biLatencyNS = np.mean(np.array(biLatencyNS))
        _, avgBiBW = comms_utils.getAlgBW(
            biLatencyNS, 2 * memSize, self.collectiveArgs.numCollPerIter
        )
        logger.debug("STATUS: end UniBW test.")
        return avgBiBW

    def checkPt2PtRanks(self):
        # set default values
        if not self.collectiveArgs.src_ranks:
            self.collectiveArgs.src_ranks = [0]
        if not self.collectiveArgs.dst_ranks:
            self.collectiveArgs.dst_ranks = [1]

        # sanity check
        if self.collectiveArgs.pt2pt == "one2one":
            if (
                len(self.collectiveArgs.src_ranks) > 1
                or len(self.collectiveArgs.dst_ranks) > 1
            ):
                if self.report:
                    logger.error(
                        "One2one Pt2Pt requires only a single rank is specified in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()
        elif self.collectiveArgs.pt2pt == "pairwise":
            # pairwise pt2pt requires identical number of ranks in src_ranks and dst_ranks.
            if len(self.collectiveArgs.src_ranks) != len(self.collectiveArgs.dst_ranks):
                if self.report:
                    logger.error(
                        "Pairwise Pt2Pt requires identical number of members in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()
            # pairwise pt2pt does not allow same rank to exist in both groups
            if bool(
                set(self.collectiveArgs.src_ranks).intersection(
                    self.collectiveArgs.dst_ranks
                )
            ):
                if self.report:
                    logger.error(
                        "Pairwise Pt2Pt requires distinct members in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()

        if self.report:
            print(
                f"\t collective={self.collectiveArgs.collective}\t{self.collectiveArgs.pt2pt}, src_ranks={self.collectiveArgs.src_ranks}, dst_ranks={self.collectiveArgs.dst_ranks}"
            )

    def checkCollectiveRanks(self):
        if self.collectiveArgs.collective == "incast":
            # incast: set default value and exclude root
            if not self.collectiveArgs.src_ranks:
                self.collectiveArgs.src_ranks = [*range(self.comm_size)]
            if self.collectiveArgs.srcOrDst in self.collectiveArgs.src_ranks:
                self.collectiveArgs.src_ranks.remove(self.collectiveArgs.srcOrDst)
        elif self.collectiveArgs.collective == "multicast":
            # multicast: set default value and exclude root
            if not self.collectiveArgs.dst_ranks:
                self.collectiveArgs.dst_ranks = [*range(self.comm_size)]
            if self.collectiveArgs.srcOrDst in self.collectiveArgs.dst_ranks:
                self.collectiveArgs.dst_ranks.remove(self.collectiveArgs.srcOrDst)

        if self.report:
            print(
                f"\t collective={self.collectiveArgs.collective}, src_ranks={self.collectiveArgs.src_ranks}, dst_ranks={self.collectiveArgs.dst_ranks}"
            )

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
        self.collectiveArgs.numCollPerIter = commsParams.num_coll
        self.collectiveArgs.include_0B = commsParams.include_0B
        self.collectiveArgs.graph_launches = commsParams.graph_launches
        self.collectiveArgs.use_device_time = commsParams.use_device_time

        if commsParams.bitwidth < 32:
            comms_utils.initQuantCommCtx(self.collectiveArgs, commsParams)

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

        if self.collectiveArgs.use_device_time:
            self.collectiveArgs.comm_dev_time = paramDeviceTimer(
                name="comm_timer", backendFuncs=self.backendFuncs
            )
        else:
            self.collectiveArgs.comm_dev_time = None

        return (
            global_rank,
            world_size,
            allSizes,
        )

    def gatherBenchTime(self, collectiveArgs, commsParams, timeUsElapsedList):
        # Push the list to device, then do an all-gather.
        timeElapsedTensor = torch.tensor(
            timeUsElapsedList,
            device=(self.backendFuncs.get_device()),
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
            else:
                fmt = "{:>40}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
                header += fmt.format(
                    "total-size (B)",
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
        if "all_to_all" in commsParams.collective or commsParams.collective in (
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

        fmt = (
            "\tCOMMS-RES-{}-{}{}{:>18}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
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
        )

    def reportBenchTimePt2Pt(self, commsParams, resultsAcrossRanks, results):
        pingLatencyAcrossRanks = []
        pingPongLatencyAcrossRanks = []
        uniBWAcrossRanks = []
        biBWAcrossRanks = []
        # idx = 0
        for curRankTensor in resultsAcrossRanks:
            pingLatencyAcrossRanks.append(curRankTensor[0].item())
            pingPongLatencyAcrossRanks.append(curRankTensor[1].item())
            uniBWAcrossRanks.append(curRankTensor[2].item())
            biBWAcrossRanks.append(curRankTensor[3].item())

        pingLatencyAcrossRanks = np.array(pingLatencyAcrossRanks)
        pingPongLatencyAcrossRanks = np.array(pingPongLatencyAcrossRanks)
        uniBWAcrossRanks = np.array(uniBWAcrossRanks)
        biBWAcrossRanks = np.array(biBWAcrossRanks)

        # Include only communicating ranks
        commRanks = self.collectiveArgs.src_ranks + self.collectiveArgs.dst_ranks
        pingLatencyAcrossCommRanks = pingLatencyAcrossRanks[commRanks]
        pingPongLatencyAcrossCommRanks = pingPongLatencyAcrossRanks[commRanks]
        uniBWAcrossCommRanks = uniBWAcrossRanks[commRanks]
        biBWAcrossCommRanks = biBWAcrossRanks[commRanks]

        logger.debug(
            "Ping latency across communicating ranks (%s): %s"
            % (commRanks, pingLatencyAcrossCommRanks)
        )
        logger.debug(
            "PingPong latency across communicating ranks (%s): %s"
            % (commRanks, pingPongLatencyAcrossCommRanks)
        )
        logger.debug(
            "UniBW across all communicating ranks (%s): %s"
            % (commRanks, uniBWAcrossCommRanks)
        )
        logger.debug(
            "BiBW across all communicating ranks (%s): %s"
            % (commRanks, biBWAcrossCommRanks)
        )

        avgUniBW = np.mean(uniBWAcrossCommRanks)
        avgBiBW = np.mean(biBWAcrossCommRanks)
        totalUniBW = np.sum(uniBWAcrossCommRanks) / 2
        totalBiBW = np.sum(biBWAcrossCommRanks) / 2

        ping_p50 = np.percentile(pingLatencyAcrossCommRanks, 50)
        ping_p75 = np.percentile(pingLatencyAcrossCommRanks, 75)
        ping_p95 = np.percentile(pingLatencyAcrossCommRanks, 95)

        ping_pong_p50 = np.percentile(pingPongLatencyAcrossCommRanks, 50)
        ping_pong_p75 = np.percentile(pingPongLatencyAcrossCommRanks, 75)
        ping_pong_p95 = np.percentile(pingPongLatencyAcrossCommRanks, 95)

        print(
            "\tCOMMS-RES-{}-{}{}{:>15}{:>20}{:>10}{:>10}{:>25}{:>10}{:>10}{:>15}{:>15}{:>18}{:>18}".format(
                self.collectiveArgs.collective,
                self.collectiveArgs.data_type,
                self.tag,
                results["memSize"],
                str("%.1f" % (ping_p50)),
                str("%.1f" % (ping_p75)),
                str("%.1f" % (ping_p95)),
                str("%.1f" % (ping_pong_p50)),
                str("%.1f" % (ping_pong_p75)),
                str("%.1f" % (ping_pong_p95)),
                str("%.3f" % (avgUniBW)),
                str("%.3f" % (avgBiBW)),
                str("%.3f" % (totalUniBW)),
                str("%.3f" % (totalBiBW)),
            )
        )

        return commsPt2PtPerfMetrics(
            self.collectiveArgs.collective,
            self.collectiveArgs.data_type,
            benchType.Collective,
            commsParams.backend,
            self.tag,
            results["memSize"],
            results["memSize"],
            results["numElements"],
            float(ping_p50),
            float(ping_p75),
            float(ping_p95),
            float(avgUniBW),
            float(avgBiBW),
            float(totalUniBW),
            float(totalBiBW),
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
        ) = self.initCollectiveArgs(commsParams)

        backendFuncs.sync_barrier(self.collectiveArgs)
        if self.report:
            self.printPreamble(commsParams)

        backendFuncs.set_up()
        self.tear_down_fns.append(backendFuncs.tear_down)

        for curSize in allSizes:
            results = {}
            timeUsElapsedList = []
            quantTimeElapsedList = []
            dequantTimeElapsedList = []
            numElements = int(curSize // commsParams.element_size)
            collectiveFunc = self.backendFuncs.noop

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
                    numIters=self.collectiveArgs.graph_launches
                    if self.collectiveArgs.graph_launches
                    else self.collectiveArgs.numIters,
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
    ):
        self.collectiveArgs.pgId = 0  # default group id

        global_rank = self.backendFuncs.get_global_rank()
        world_size = self.backendFuncs.get_world_size()
        groupRanks = {}

        if multi_comms > 1:
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
            self.backendFuncs.initialize_groups(backend=backend)

        else:
            # default is single group including all ranks.
            # create the same groupRanks argument for simple
            # query in later logic no matter the group splitting
            groupRanks[0] = []
            for rank in range(0, world_size):
                groupRanks[0].append(rank)

        return groupRanks

    def initBackend(
        self, bootstrap_info: bootstrap_info_holder, commsParams: commsParamsHolderBase
    ):
        # Init the desired backend
        backendObj = None
        if (
            commsParams.nw_stack == "pytorch-dist"
            and commsParams.backend in supportedC10dBackends
        ):
            from param_bench.train.comms.pt.pytorch_dist_backend import (
                PyTorchDistBackend,
            )
            from param_bench.train.comms.pt.pytorch_nvshmem_backend import (
                PyTorchNVShmemBackend,
            )

            backendObj = (
                PyTorchNVShmemBackend(bootstrap_info, commsParams)
                if commsParams.use_nvshmem
                else PyTorchDistBackend(bootstrap_info, commsParams)
            )
        elif commsParams.nw_stack == "pytorch-xla-tpu":
            from param_bench.train.comms.pt.pytorch_tpu_backend import PyTorchTPUBackend

            backendObj = PyTorchTPUBackend(bootstrap_info, commsParams)
        else:
            # check for customized backend
            try:
                logging.warning(
                    f"Attempt loading customized backend {commsParams.backend} if registered. Note that this is not officially supported. Use it with caution and at your own risk."
                )
                from param_bench.train.comms.pt.pytorch_backend_utils import (
                    customized_backend,
                )

                backendObj = customized_backend[commsParams.backend](
                    bootstrap_info, commsParams
                )
            except KeyError as e:
                logger.error(
                    f"Unsupported NW stack for backend {commsParams.backend}: {e}"
                )
                comms_utils.gracefulExit()

        self.backendFuncs = backendObj
        self.backendFuncs.initialize_backend(
            bootstrap_info.master_ip,
            bootstrap_info.master_port,
            backend=commsParams.backend,
            eager_mode=(commsParams.init_only or commsParams.eager_init),
        )
        self.backendFuncs.sayHello()  # Informs us where each process is running.

    def runBench(self, commsParams):
        try:
            self.backendFuncs.benchmark_comms(self.benchTime, commsParams)
        except ValueError as ve:
            logger.critical(repr(ve))
            raise
        finally:
            for fn in self.tear_down_fns:
                fn()


def main():
    collBenchObj = commsCollBench()

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="PARAM-Comm Benchmark",
        formatter_class=MultilineFormatter,
        allow_abbrev=False,
    )
    collBenchObj.readArgs(parser)
    args, _ = parser.parse_known_args()

    comms_env_params = comms_utils.read_comms_env_vars()
    if comms_env_params["global_rank"] == 0 or (
        args.enable_local_report and comms_env_params["local_rank"] == 0
    ):
        print("\t PARAM COMM environment: %s " % (str(comms_env_params)))
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
    )

    for data_type in args.data_types:
        args.data_type = data_type.lower()

        collBenchObj.checkArgsdataType(args)
        element_size = torch.ones([1], dtype=args.dtype).element_size()

        commsParams = comms_utils.commsParamsHolder(
            args, bootstrap_info, element_size, collBenchObj.benchTime, groupRanks
        )
        collBenchObj.backendFuncs.commsParams = commsParams
        collBenchObj.runBench(commsParams)


if __name__ == "__main__":
    main()
