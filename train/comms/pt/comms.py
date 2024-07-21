#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import time

import numpy as np

# pytorch
import torch

from param_bench.train.comms.pt import comms_utils
from param_bench.train.comms.pt.comms_utils import (
    bootstrap_info_holder,
    commsParamsHolderBase,
    ensureTensorFlush,
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
    pt2ptPatterns,
    supportedC10dBackends,
    supportedCollectives,
)

logger = logging.getLogger(__name__)


class MultilineFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.ArgumentDefaultsHelpFormatter._split_lines(self, text, width)


# define the collective benchmark
class commsCollBench(paramCommsBench):
    def __init__(self):
        super().__init__(supportedNwstacks=["pytorch-dist", "pytorch-xla-tpu"])
        self.tag = ""
        self.backendFuncs = None

    # def readCollArgs(self, parser):
    def readArgs(self, parser):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--w", type=int, default=5, help="number of warmup iterations"
        )  # number of warmup-iterations
        parser.add_argument(
            "--n", "--num_iters", type=int, default=5, help="number of iterations"
        )  # number of iterations
        # experiment related parameters
        parser.add_argument(
            "--mode",
            type=str,
            default="comms",
            help="benchmark mode",
            choices=["comms", "compute", "dlrm", "comms-compute"],
        )  # alternative is DLRM mode or comm-compute mode
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
        # For comm-compute or compute mode
        parser.add_argument(
            "--kernel",
            type=str,
            default="gemm",
            help="Compute kernel, used for comms-compute or compute mode",
            choices=["gemm", "emb_lookup", "add", "sub", "add_num", "sub_num", "copy"],
        )  # Compute kernel: "gemm"
        parser.add_argument(
            "--num-compute",
            "--num-compute-per-iteration",
            type=int,
            default=100,
            help="number of compute kernels to execute for every iteration",
        )  # Launch one coll for every n compute kernels
        parser.add_argument(
            "--num-coll",
            "--num-coll-per-iteration",
            type=int,
            default=1,
            help="number of collective operations to execute for every iteration",
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
            "--pair",
            action="store_true",
            default=False,
            help="Toggle to enable collective pair mode",
        )
        parser.add_argument(
            "--collective-pair",
            type=str,
            default="all_reduce",
            help="Collective pair operation to be evaluated",
            choices=supportedCollectives,
        )  # collective op to pair with the other collective, --collective should be non-empty
        parser.add_argument(
            "--multi-comms",
            type=int,
            default=1,
            help="Set to enable multi-comm group mode, cannot use together with --overlap-pair-pgs mode. Default 1 comm group",
        )
        parser.add_argument(
            "--overlap-pair-pgs",
            action="store_true",
            default=False,
            help="Toggle to enable overlapping collective pair with two pgs",
        )  # overlap collective pair with two pgs
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
        return parser.parse_known_args()

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

        if args.comp_data_type not in self.supportedDtype:
            logger.error(
                f"Specified comp datatype: {args.comp_data_type} is not one of the supported commstyle: {str(self.supportedDtype)}"
            )
            comms_utils.gracefulExit()
        args.comp_data_type = self.dtypeMap[args.comp_data_type]

        if args.size_start_profiler:
            args.size_start_profiler = comms_utils.parsesize(args.size_start_profiler)

        self.tag = f"-{args.tag}" if args.tag is not None else ""

    # Check arguments that may be custmized per benchmark in a single run
    def checkArgs(self, args):  # noqa: C901
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

        # run a few sanity checks
        self._check_bitwidth(args)

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
        if args.multi_comms > 1 and args.overlap_pair_pgs:
            logger.error("--overlap-pair-pgs is not supported with --multi-comms > 1")
            comms_utils.gracefulExit()

        if args.mode in ("compute", "comms-compute") and (
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

    def runColl(self, comm_fn=None, compute_fn=None, comm_fn_pair=None, dcheck=False):
        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_begin")

        elapsedTimeNS = 0.0
        is_blocking = not self.collectiveArgs.asyncOp
        enable_comms = (
            False if (comm_fn is None or comm_fn == self.backendFuncs.noop) else True
        )
        enable_compute = (
            False
            if (compute_fn is None or compute_fn == self.backendFuncs.noop)
            else True
        )
        enable_comms_pair = (
            False
            if (comm_fn_pair is None or comm_fn_pair == self.backendFuncs.noop)
            else True
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
                    ensureTensorFlush(self.collectiveArgs.opTensor_pair)
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
                for _ in range(self.collectiveArgs.numCollPerIter):
                    self.collectiveArgs.group = self.collectiveArgs.groups[
                        self.collectiveArgs.pgId
                    ]
                    comm_fn(self.collectiveArgs)
                    # post another collecitve if on comms pair mode, otherwise it's noop
                    self.collectiveArgs.group = self.collectiveArgs.groups[
                        self.collectiveArgs.pairPgId
                    ]
                    comm_fn_pair(self.collectiveArgs, pair=enable_comms_pair)

            if enable_compute:
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
                if enable_compute and self.collectiveArgs.compute_dev_time:
                    self.collectiveArgs.compute_dev_time.elapsedTime()

            # Measuring time.
            elapsedTimeNS += (
                time.monotonic() - start
            ) * 1e9  # keeping time in NS, helps in divising data by nanosecond

        start = time.monotonic()  # available only in py3
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)
        end = time.monotonic()  # available only in py3

        ensureTensorFlush(self.collectiveArgs.opTensor)
        if enable_comms_pair:
            ensureTensorFlush(self.collectiveArgs.opTensor_pair)

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

            busBW += self.backendFuncs.getBusBW(
                self.collectiveArgs.collective_pair,
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
        self.collectiveArgs.pair = commsParams.pair
        self.collectiveArgs.pairPgId = (
            1 if commsParams.overlap_pair_pgs else self.collectiveArgs.pgId
        )
        self.collectiveArgs.collective_pair = commsParams.collective_pair
        self.collectiveArgs.pt2pt = commsParams.pt2pt
        self.collectiveArgs.window = commsParams.window
        self.collectiveArgs.asyncOp = False if commsParams.blockingFlag == 1 else True
        self.collectiveArgs.numComputePerIter = commsParams.num_compute
        self.collectiveArgs.numCollPerIter = commsParams.num_coll

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
            elif commsParams.kernel in ["add", "sub", "add_num", "sub_num", "copy"]:
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
            local_rank,
            global_rank,
            world_size,
            group,
            curDevice,
            curHwDevice,
            allSizes,
            computeFunc,
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
            elif not self.collectiveArgs.pair:
                fmt = (
                    "{:>40}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
                    + dev_time_fmt
                    + tflops_fmt
                )
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

        if not self.collectiveArgs.pair:
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
            fmt = (
                "\tCOMMS-RES-{}-{}{}{:>18}{:>18}{:>22}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}"
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
                    str("%d" % (results["numElements_pair"])),
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
            results["numElements_pair"] if "numElements_pair" in results else 0,
            float(p50),
            float(p75),
            float(p95),
            float(minlat),
            float(maxlat),
            algBW,
            busBW,
            tflops,
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
            results["numElements_pair"] if "numElements_pair" in results else 0,
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
            commsParams.collective = coll
            self.benchComm(index, commsParams, backendFuncs)

    def benchComm(self, index, commsParams, backendFuncs):
        # Get NW stack specific parameters
        (
            local_rank,
            global_rank,
            world_size,
            group,
            curDevice,
            curHwDevice,
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
            collectiveFunc_pair = self.backendFuncs.noop
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

            if (
                commsParams.pair and commsParams.mode != "compute"
            ):  # comms-pair specific initializations if not in compute-only mode:
                # set corresponding function pointers
                collectiveFunc_pair = backendFuncs.collectiveFunc[
                    commsParams.collective_pair
                ]
                # TODO: allow user to set specific size
                # Setup the arguments.
                self.collectiveArgs.dataSize_pair = curSize
                self.collectiveArgs.numElements_pair = int(
                    self.collectiveArgs.dataSize_pair // commsParams.element_size
                )
                results["numElements_pair"] = self.collectiveArgs.numElements_pair
                commsArgs = comms_utils.commsArgs()
                commsArgs.inMsgSize = self.collectiveArgs.numElements_pair
                commsArgs.outMsgSize = self.collectiveArgs.numElements_pair
                commsArgs.worldSize = world_size
                commsArgs.comms = commsParams.collective_pair
                (
                    self.collectiveArgs.ipTensor_pair,
                    self.collectiveArgs.opTensor_pair,
                ) = self.prepComm(
                    curComm=commsArgs,
                    commsParams=commsParams,
                )

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
                        comm_fn_pair=collectiveFunc_pair,
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

    def genMultiCommGroups(self, commsParams: commsParamsHolderBase):
        self.collectiveArgs.pgId = 0  # default group id
        global_rank = self.backendFuncs.get_global_rank()
        world_size = self.backendFuncs.get_world_size()
        commsParams.groupRanks = {}
        if commsParams.multi_comms > 1:
            self.collectiveArgs.pgId = global_rank % commsParams.multi_comms
            for pgId in range(commsParams.multi_comms):
                commsParams.groupRanks[pgId] = []
            for rank in range(world_size):
                pgId = rank % commsParams.multi_comms
                commsParams.groupRanks[pgId].append(rank)
            for pgId in range(commsParams.multi_comms):
                logger.info(
                    f"PARAM COMMS Rank {global_rank} created group {pgId} with ranks {commsParams.groupRanks[pgId]}"
                )

            # FIXME: how to proper generate groupRanks before initializing backend?
            self.backendFuncs.commsParams.groupRanks = commsParams.groupRanks
            self.backendFuncs.initialize_groups(commsParams.backend)

        elif commsParams.pair and commsParams.overlap_pair_pgs:
            # create two communicators each including all ranks
            commsParams.num_pgs = 2
            for pgId in range(0, commsParams.num_pgs):
                commsParams.groupRanks[pgId] = []
                for rank in range(0, world_size):
                    commsParams.groupRanks[pgId].append(rank)
                logger.info(
                    f"PARAM COMMS Rank {global_rank} created group {pgId} with ranks {commsParams.groupRanks[pgId]}"
                )
            self.backendFuncs.commsParams.groupRanks = commsParams.groupRanks
            self.backendFuncs.initialize_groups(commsParams.backend)

        else:
            # default is single group including all ranks.
            # create the same groupRanks argument for simple
            # query in later logic no matter the group splitting
            commsParams.groupRanks[0] = []
            for rank in range(0, world_size):
                commsParams.groupRanks[0].append(rank)

    def initBackend(
        self, bootstrap_info: bootstrap_info_holder, commsParams: commsParamsHolderBase
    ):
        # Init the desired backend
        if (
            commsParams.nw_stack == "pytorch-dist"
            and commsParams.backend in supportedC10dBackends
        ):
            from param_bench.train.comms.pt.pytorch_dist_backend import (
                PyTorchDistBackend,
            )

            backendObj = PyTorchDistBackend(bootstrap_info, commsParams)
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


def main():
    collBenchObj = commsCollBench()

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="PARAM-Comm Benchmark",
        formatter_class=MultilineFormatter,
        allow_abbrev=False,
    )
    args, leftovers = collBenchObj.readArgs(parser)

    comms_env_params = comms_utils.read_comms_env_vars()
    if comms_env_params["global_rank"] == 0 or (
        args.enable_local_report and comms_env_params["local_rank"] == 0
    ):
        print("\t PARAM COMM environment: %s " % (str(comms_env_params)))
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

    collBenchObj.checkBasicArgs(args)

    # Initialize backend
    bootstrap_info = comms_utils.bootstrap_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, comms_env_params
    )
    commsParamsBase = comms_utils.commsParamsHolderBase(args)
    collBenchObj.initBackend(bootstrap_info, commsParamsBase)

    # Dedupes and syncs value for args.data_types based on args.data_type/args.dtype if not passed in args.
    collBenchObj.syncCommBenchDataTypes(args)

    for data_type in args.data_types:
        args.data_type = data_type.lower()

        collBenchObj.checkArgs(args)
        element_size = torch.ones([1], dtype=args.dtype).element_size()

        commsParams = comms_utils.commsParamsHolder(
            args, bootstrap_info, element_size, collBenchObj.benchTime
        )

        # FIXME: this should be outside dtype loop
        collBenchObj.genMultiCommGroups(commsParams)

        collBenchObj.runBench(commsParams)


if __name__ == "__main__":
    main()
