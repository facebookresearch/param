#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import time

import comms_utils
import numpy as np

# pytorch
import torch
from comms_utils import paramCommsBench, ensureTensorFlush

### TODO: add these to class variables?
supportedCollectives = [
    "reduce",
    "all_reduce",
    "all_to_all",
    "all_to_allv",
    "all_gather",
    "broadcast",
    "reduce_scatter",
    "all_gather_base",
    "incast",
    "multicast",
]  # , "scatter", "gather"]
pt2ptPatterns = [
    "one2one",
    "pairwise",
]

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

    # def readCollArgs(self, parser):
    def readArgs(self, parser):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--w", type=int, default=5, help="number of warmup iterations"
        )  # number of warmup-iterations
        parser.add_argument(
            "--n", type=int, default=5, help="number of iterations"
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
            "--b", type=str, default="8", help="minimum size, in bytes, to start with"
        )  # COMMS mode, begin the sweep at.
        parser.add_argument(
            "--e", type=str, default="64", help="maximum size, in bytes, to end at"
        )  # COMMS mode, end the sweep at.
        parser.add_argument(
            "--f", type=int, default=2, help="multiplication factor between sizes"
        )  # COMMS mode, multiplication factor.
        parser.add_argument(
            "--collective",
            type=str,
            default="all_reduce",
            help="Collective operation to be evaluated",
            choices=supportedCollectives,
        )  # collective op to benchmark
        # For comm-compute or compute mode
        parser.add_argument(
            "--kernel",
            type=str,
            default="gemm",
            help="Compute kernel, used for comms-compute or compute mode",
            choices=["gemm", "emb_lookup"],
        )  # Compute kernel: "gemm"
        parser.add_argument(
            "--num-compute",
            type=int,
            default=100,
            help="one collective for every NUM_COMPUTE compute kernels",
        )  # Launch one coll for every n compute kernels
        # For GEMM
        parser.add_argument(
            "--mm-dim",
            type=int,
            default=100,
            help="dimension size for GEMM compute kernel",
        )  # Matrix multiplication dim n, A[n,n] * B [n,n]
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
            "--avg-len",
            type=int,
            default=28,
            help="Average lookup operations per sample",
        )  # Average #lookup per sample
        parser.add_argument(
            "--batch-size",
            type=int,
            default=512,
            help="number of samples reading the table concurrently",
        )  # #Samples reading the table concurrently
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

        return parser.parse_known_args()

    def checkArgs(self, args):
        super().checkArgs(args)

        if args.pt2pt is not None:
            args.collective = "pt2pt"
            if args.pt2pt not in pt2ptPatterns:
                logger.error(
                    f"Specified pt2pt pattern: {args.pt2pt} is not one of the supported pt2pt patterns: {str(pt2ptPatterns)}"
                )
                comms_utils.gracefulExit()

        args.b = comms_utils.parsesize(args.b)
        args.e = comms_utils.parsesize(args.e)
        args.dtype = self.dtypeMap[args.data_type]

        if args.b < 1:
            logger.warn(
                f"Starting size (--b {args.b}) should be greater than 1 byte...fix and continue"
            )
            args.b = 1

        if args.e < args.b:
            logger.warn(
                f"the begin-size (--b {args.b}) is larger than the end-size (--e {args.e})"
            )

        if args.device == "cpu" and args.backend == "nccl":
            raise ValueError(f"NCCL is not supported for device type {args.device}")

        if args.c == 1 and args.z == 0:
            logger.warn(
                "Data validation may not be fully supported for non-blocking mode"
            )

        # run a few sanity checks
        if args.bitwidth < 32:
            if args.device != "cuda":
                logger.error(
                    f"collective quantization may not be fully supported for {args.device}"
                )
            comms_utils.checkQuantArgs(
                args.collective,
                args.dtype,
                args.b,
                args.quant_a2a_embedding_dim,
                args.z,
            )

    def runColl(self, comm_fn=None, compute_fn=None, comm_fn_pair=None):
        self.backendFuncs.complete_accel_ops(self.collectiveArgs, initOp=True)
        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_begin")

        elapsedTimeNS = 0.0
        is_blocking = not self.collectiveArgs.asyncOp
        enable_comms = False if (comm_fn is None or comm_fn == self.backendFuncs.noop) else True
        enable_compute = False if (compute_fn is None or compute_fn == self.backendFuncs.noop) else True
        enable_comms_pair = False if (comm_fn_pair is None or comm_fn_pair == self.backendFuncs.noop) else True

        # for comms pair mode, force async comms for overlapping evaluation
        if enable_comms_pair:
            self.collectiveArgs.asyncOp = True
        for nIter in range(
            self.collectiveArgs.numWarmupIters + self.collectiveArgs.numIters
        ):
            if nIter == self.collectiveArgs.numWarmupIters:
                # Start measuring time after warmup iterations
                elapsedTimeNS = 0.0
                self.collectiveArgs.quant_time.reset()
                self.collectiveArgs.dequant_time.reset()
            # for blocking mode, do barrier before starting collective
            if is_blocking:
                # reset tensor values for data validation check
                if enable_comms:
                    self.setTensorVal(self.collectiveArgs.opTensor)
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            start = time.monotonic()  # available only in py3
            self.collectiveArgs.group = self.backendFuncs.get_next_group()
            comm_fn(self.collectiveArgs)
            # post another collecitve if on comms pair mode, otherwise it's noop
            self.collectiveArgs.group = self.backendFuncs.get_next_group()
            comm_fn_pair(self.collectiveArgs, pair=enable_comms_pair)

            if enable_compute:
                for _ in range(self.collectiveArgs.numComputePerColl):
                    # TODO: investigate the cache effect
                    # Flush the cache
                    # _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
                    compute_fn(self.collectiveArgs)
            if is_blocking:  # should be sychronous, wait for the collective
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
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
            elapsedTimeNS, memSize, self.collectiveArgs.numIters
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
                elapsedTimeNS, memSize_pair, self.collectiveArgs.numIters
            )
            algBW += algBW_pair

            busBW += self.backendFuncs.getBusBW(
                self.collectiveArgs.collective_pair,
                algBW_pair,
                self.collectiveArgs,
            )

        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_end")

        results = {
            "timeUS": avgIterNS / 1e3,
            "algBW": algBW,
            "busBW": busBW,
            "memSize": memSize,
        }
        return results

    def runPt2Pt(self):
        self.backendFuncs.complete_accel_ops(self.collectiveArgs, initOp=True)
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
                self.backendFuncs.send(
                    self.collectiveArgs, self.collectiveArgs.dst_ranks[idx]
                )
            elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                idx = self.collectiveArgs.dst_ranks.index(
                    self.collectiveArgs.global_rank
                )
                self.backendFuncs.recv(
                    self.collectiveArgs, self.collectiveArgs.src_ranks[idx]
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
                self.backendFuncs.send(
                    self.collectiveArgs, self.collectiveArgs.dst_ranks[idx]
                )
                self.backendFuncs.recv(
                    self.collectiveArgs, self.collectiveArgs.dst_ranks[idx]
                )
            elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                idx = self.collectiveArgs.dst_ranks.index(
                    self.collectiveArgs.global_rank
                )
                self.backendFuncs.recv(
                    self.collectiveArgs, self.collectiveArgs.src_ranks[idx]
                )
                self.backendFuncs.send(
                    self.collectiveArgs, self.collectiveArgs.src_ranks[idx]
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
                    self.backendFuncs.isend(
                        self.collectiveArgs, self.collectiveArgs.dst_ranks[idx], tag=w
                    )
                elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                    idx = self.collectiveArgs.dst_ranks.index(
                        self.collectiveArgs.global_rank
                    )
                    self.backendFuncs.irecv(
                        self.collectiveArgs, self.collectiveArgs.src_ranks[idx], tag=w
                    )
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            uniLatencyNS.append(
                (time.monotonic() - start) * 1e9
            )  # keeping time in NS, helps in divising data by nanosecond
        uniLatencyNS = [lat / self.collectiveArgs.window for lat in uniLatencyNS]
        uniLatencyNS = np.mean(np.array(uniLatencyNS))
        _, avgUniBW = comms_utils.getAlgBW(uniLatencyNS, memSize, 1)
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
                    self.backendFuncs.isend(
                        self.collectiveArgs, self.collectiveArgs.dst_ranks[idx], tag=w
                    )
                    self.backendFuncs.irecv(
                        self.collectiveArgs,
                        self.collectiveArgs.dst_ranks[idx],
                        tag=w + self.collectiveArgs.window,
                    )
                elif self.collectiveArgs.global_rank in self.collectiveArgs.dst_ranks:
                    idx = self.collectiveArgs.dst_ranks.index(
                        self.collectiveArgs.global_rank
                    )
                    self.backendFuncs.irecv(
                        self.collectiveArgs, self.collectiveArgs.src_ranks[idx], tag=w
                    )
                    self.backendFuncs.isend(
                        self.collectiveArgs,
                        self.collectiveArgs.src_ranks[idx],
                        tag=w + self.collectiveArgs.window,
                    )
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            biLatencyNS.append(
                (time.monotonic() - start) * 1e9
            )  # keeping time in NS, helps in divising data by nanosecond
        biLatencyNS = [lat / self.collectiveArgs.window for lat in biLatencyNS]
        biLatencyNS = np.mean(np.array(biLatencyNS))
        _, avgBiBW = comms_utils.getAlgBW(biLatencyNS, 2 * memSize, 1)
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
                if self.global_rank == 0:
                    logger.error(
                        "One2one Pt2Pt requires only a single rank is specified in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()
        elif self.collectiveArgs.pt2pt == "pairwise":
            # pairwise pt2pt requires identical number of ranks in src_ranks and dst_ranks.
            if len(self.collectiveArgs.src_ranks) != len(self.collectiveArgs.dst_ranks):
                if self.global_rank == 0:
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
                if self.global_rank == 0:
                    logger.error(
                        "Pairwise Pt2Pt requires distinct members in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()

        if self.global_rank == 0:
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

        if self.global_rank == 0:
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
        self.backendFuncs.sayHello()  # Informs us where each process is running.
        groups = self.backendFuncs.get_groups()
        num_pgs = len(groups)

        self.comm_size = world_size
        self.global_rank = global_rank

        comms_utils.fixBeginSize(
            commsParams, world_size
        )  # Ensuring that all-reduce and all-to-all has atleast one member per rank.
        allSizes = comms_utils.getSizes(
            commsParams.beginSize, commsParams.endSize, commsParams.stepFactor
        )  # Given the begin-size, end-size, step-factor what are the message sizes to iterate on.

        if global_rank == 0:
            print(
                f"[Rank {global_rank:>3}] allSizes: {allSizes} local_rank: {local_rank} element_size: {commsParams.element_size}"
            )

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
        self.collectiveArgs.srcOrDst = commsParams.srcOrDst
        self.collectiveArgs.src_ranks = commsParams.src_ranks
        self.collectiveArgs.dst_ranks = commsParams.dst_ranks
        self.collectiveArgs.pair = commsParams.pair
        self.collectiveArgs.collective_pair = commsParams.collective_pair
        self.collectiveArgs.pt2pt = commsParams.pt2pt
        self.collectiveArgs.window = commsParams.window
        self.collectiveArgs.asyncOp = False if commsParams.blockingFlag == 1 else True

        if commsParams.bitwidth < 32:
            comms_utils.initQuantCommCtx(self.collectiveArgs, commsParams)

        if self.collectiveArgs.collective == "pt2pt":
            self.checkPt2PtRanks()
        else:
            self.checkCollectiveRanks()

        computeFunc = self.backendFuncs.noop
        if (
            commsParams.mode != "comms"
        ):  # Compute mode related initialization if not in comms-only mode
            if commsParams.kernel == "gemm":
                computeFunc = self.backendFuncs.gemm

                mm_dim = commsParams.mm_dim
                in1 = np.random.rand(mm_dim, mm_dim)
                MMin1 = torch.FloatTensor(in1).to(curDevice)
                in2 = np.random.rand(mm_dim, mm_dim)
                MMin2 = torch.FloatTensor(in2).to(curDevice)
                in3 = np.random.rand(mm_dim, mm_dim)
                MMin3 = torch.FloatTensor(in3).to(curDevice)
                MMout = self.backendFuncs.alloc_empty(
                    [mm_dim, mm_dim], commsParams.dtype, curDevice
                )
                self.collectiveArgs.MMout = MMout
                self.collectiveArgs.MMin1 = MMin1
                self.collectiveArgs.MMin2 = MMin2
                self.collectiveArgs.MMin3 = MMin3
                self.collectiveArgs.numComputePerColl = commsParams.num_compute
            elif commsParams.kernel == "emb_lookup":
                computeFunc = self.backendFuncs.emb_lookup

                emb_dim = commsParams.emb_dim
                num_embeddings = commsParams.num_embs
                avg_length = commsParams.avg_len
                batch_size = commsParams.batch_size
                print(
                    f"emb_dim {emb_dim} num_embs {num_embeddings} avg_len {avg_length} bs {batch_size}"
                )
                self.collectiveArgs.EmbWeights = self.backendFuncs.alloc_empty(
                    [num_embeddings, emb_dim], torch.double, curDevice
                )
                self.collectiveArgs.TableOffsets = torch.LongTensor(
                    [0, num_embeddings]
                ).to(curDevice)
                self.collectiveArgs.Indices = torch.LongTensor(
                    np.random.randint(0, num_embeddings - 1, avg_length * batch_size)
                ).to(curDevice)
                lengths = np.ones((1, batch_size)) * avg_length
                flat_lengths = lengths.flatten()
                self.collectiveArgs.Offsets = torch.LongTensor(
                    [0] + np.cumsum(flat_lengths).tolist()
                ).to(curDevice)
                self.collectiveArgs.LookupOut = self.backendFuncs.alloc_empty(
                    [batch_size, emb_dim], torch.double, curDevice
                )
                self.collectiveArgs.AvgLengths = avg_length
                self.collectiveArgs.numComputePerColl = commsParams.num_compute

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
            timeUsElapsedList, device=self.backendFuncs.get_device()
        )
        collectiveArgs.opTensor = None
        if commsParams.backend != "xla":
            timeList = [
                torch.ones_like(timeElapsedTensor) for _ in range(self.comm_size)
            ]
            collectiveArgs.opTensor = timeList

        collectiveArgs.ipTensor = timeElapsedTensor
        collectiveArgs.asyncOp = False
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
            header += "{:>15}{:>20}{:>10}{:>10}{:>25}{:>10}{:>10}{:>15}{:>15}{:>18}{:>18}".format(
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
            )
        else:
            if commsParams.bitwidth < 32:
                header += "-QUANT\t{:>15}{:>18}{:>25}{:>15}{:>15}{:>15}".format(
                    "size (B)",
                    "nElementsPerRank",
                    "P95 Latency(us): Quant",
                    "Comms",
                    "De-Quant",
                    "Overall",
                )
            elif not self.collectiveArgs.pair:
                header += (
                    "{:>15}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}".format(
                        "size (B)",
                        "nElementsPerRank",
                        "Latency(us):p50",
                        "p75",
                        "p95",
                        "Min",
                        "Max",
                        "AlgBW(GB/s)",
                        "BusBW(GB/s)",
                    )
                )
            else:
                header += "{:>15}{:>18}{:>22}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}".format(
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
        if commsParams.backend == "xla":
            latencyAcrossRanks = torch.transpose(tensorList.view(-1, 1), 0, 1)[0]
            latencyAcrossRanks = latencyAcrossRanks.cpu().detach().numpy()
            # quant tensor
            quantLatencyAcrossRanks = torch.transpose(
                quantTimeTensorList.view(-1, 1), 0, 1
            )[0]
            quantLatencyAcrossRanks = quantLatencyAcrossRanks.cpu().detach().numpy()
            # dequant tensor
            dequantLatencyAcrossRanks = torch.transpose(
                dequantTimeTensorList.view(-1, 1), 0, 1
            )[0]
            dequantLatencyAcrossRanks = dequantLatencyAcrossRanks.cpu().detach().numpy()
        else:
            latencyAcrossRanks = np.array(tensorList)
            # quant tensor
            quantLatencyAcrossRanks = np.array(quantTimeTensorList)
            # dequant tensor
            dequantLatencyAcrossRanks = np.array(dequantTimeTensorList)

        p95 = np.percentile(latencyAcrossRanks, 95)

        quant_p95 = np.percentile(quantLatencyAcrossRanks, 95)
        dequant_p95 = np.percentile(dequantLatencyAcrossRanks, 95)

        print(
            "\tCOMMS-RES-QUANT\t{:>15}{:>18}{:>25}{:>15}{:>15}{:>15}".format(
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

    def reportBenchTime(
        self,
        commsParams,
        results,
        tensorList,
        quantTimeTensorList,
        dequantTimeTensorList,
    ):
        # convernt num_elements to # of elements per rank
        if commsParams.collective in ("all_to_all", "all_to_allv"):
            results["numElements"] = int(
                results["numElements"] // commsParams.comms_world_info.world_size
            )

        if commsParams.collective == "pt2pt":
            self.reportBenchTimePt2Pt(commsParams, tensorList, results)
        elif commsParams.bitwidth < 32:
            self.reportBenchTimeCollWithQuant(
                commsParams,
                results,
                tensorList,
                quantTimeTensorList,
                dequantTimeTensorList,
            )
        else:
            self.reportBenchTimeColl(commsParams, results, tensorList)

    def reportBenchTimeColl(self, commsParams, results, tensorList):
        if commsParams.backend == "xla":
            latencyAcrossRanks = torch.transpose(tensorList.view(-1, 1), 0, 1)[0]
            latencyAcrossRanks = latencyAcrossRanks.cpu().detach().numpy()
        else:
            latencyAcrossRanks = np.array(tensorList)

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

        p50 = np.percentile(latencyAcrossCommRanks, 50)
        p75 = np.percentile(latencyAcrossCommRanks, 75)
        p95 = np.percentile(latencyAcrossCommRanks, 95)
        minlat = np.amin(latencyAcrossCommRanks)
        maxlat = np.amax(latencyAcrossCommRanks)

        # adjust busBW
        busBW = results["busBW"] * (commsParams.bitwidth / 32.0)

        if not self.collectiveArgs.pair:
            print(
                "\tCOMMS-RES{:>15}{:>18}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}".format(
                    results["memSize"],
                    str("%d" % (results["numElements"])),
                    str("%.1f" % (p50)),
                    str("%.1f" % (p75)),
                    str("%.1f" % (p95)),
                    str("%.1f" % (minlat)),
                    str("%.1f" % (maxlat)),
                    str("%.3f" % (results["algBW"])),
                    str("%.3f" % (busBW)),
                )
            )
        else:
            # convernt to # of elements per rank
            if commsParams.collective_pair in ("all_to_all", "all_to_allv"):
                results["numElements_pair"] = int(
                    results["numElements_pair"]
                    // commsParams.comms_world_info.world_size
                )
            print(
                "\tCOMMS-RES{:>15}{:>18}{:>22}{:>18}{:>12}{:>12}{:>12}{:>12}{:>15}{:>12}".format(
                    results["memSize"],
                    str("%d" % (results["numElements"])),
                    str("%d" % (results["numElements_pair"])),
                    str("%.1f" % (p50)),
                    str("%.1f" % (p75)),
                    str("%.1f" % (p95)),
                    str("%.1f" % (minlat)),
                    str("%.1f" % (maxlat)),
                    str("%.3f" % (results["algBW"])),
                    str("%.3f" % (busBW)),
                )
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
            "\tCOMMS-RES{:>15}{:>20}{:>10}{:>10}{:>25}{:>10}{:>10}{:>15}{:>15}{:>18}{:>18}".format(
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

    def benchTime(self, index, commsParams, backendFuncs):
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
        if global_rank == 0:
            self.printPreamble(commsParams)

        for curSize in allSizes:
            results = {}
            timeUsElapsedList = []
            quantTimeElapsedList = []
            dequantTimeElapsedList = []
            numElements = int(curSize // commsParams.element_size)
            collectiveFunc = self.backendFuncs.noop
            collectiveFunc_pair = self.backendFuncs.noop

            if (
                commsParams.mode != "compute"
            ):  # comms specific initializations if not in compute-only mode
                # set corresponding function pointers
                if commsParams.collective != "pt2pt":
                    collectiveFunc = backendFuncs.collectiveFunc[commsParams.collective]

                (
                    self.collectiveArgs.ipTensor,
                    self.collectiveArgs.opTensor,
                ) = self.prepComm(
                    curComm={
                        "in_msg_size": numElements,
                        "out_msg_size": numElements,
                        "world_size": world_size,
                    },
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
                (
                    self.collectiveArgs.ipTensor_pair,
                    self.collectiveArgs.opTensor_pair,
                ) = self.prepComm(
                    curComm={
                        "in_msg_size": self.collectiveArgs.numElements_pair,
                        "out_msg_size": self.collectiveArgs.numElements_pair,
                        "world_size": world_size,
                    },
                    commsParams=commsParams,
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
                    )
                )
                timeUsElapsedList = [results["timeUS"]]

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

            # gather and report performance to stdout
            tensorList = self.gatherBenchTime(
                self.collectiveArgs, commsParams, timeUsElapsedList
            )
            if global_rank == 0:
                self.reportBenchTime(
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

    def runBench(self, comms_world_info, commsParams):
        # Init the desired backend
        if commsParams.nw_stack == "pytorch-dist":
            from pytorch_dist_backend import PyTorchDistBackend

            backendObj = PyTorchDistBackend(comms_world_info, commsParams)
        elif commsParams.nw_stack == "pytorch-xla-tpu":
            from pytorch_tpu_backend import PyTorchTPUBackend

            backendObj = PyTorchTPUBackend(comms_world_info, commsParams)
        else:
            logger.error("Unsupported NW stack! ")
            comms_utils.gracefulExit()

        self.backendFuncs = backendObj
        try:
            backendObj.benchmark_comms()
        except ValueError as ve:
            if commsParams.backend == "ucc":
                logger.critical("PyTorch UCC not implemented? {}".format(repr(ve)))
            raise


def main():
    collBenchObj = commsCollBench()

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="PARAM-Comm Benchmark",
        formatter_class=MultilineFormatter,
    )
    args, leftovers = collBenchObj.readArgs(parser)

    collBenchObj.checkArgs(args)

    mpi_env_params = comms_utils.read_mpi_env_vars()
    if mpi_env_params["global_rank"] == 0:
        print("\t MPI environment: %s " % (str(mpi_env_params)))
        print(
            "\t backend: %s nw-stack: %s mode: %s args.b: %d args.e: %d args.f: %d args.z: %s args.master_ip: %s "
            % (
                args.backend,
                args.nw_stack,
                args.mode,
                args.b,
                args.e,
                args.f,
                args.z,
                args.master_ip,
            )
        )

    element_size = torch.ones([1], dtype=args.dtype).element_size()
    comms_world_info = comms_utils.comms_world_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, mpi_env_params
    )

    commsParams = comms_utils.commsParamsHolder(
        args, comms_world_info, element_size, collBenchObj.benchTime
    )

    if args.pair and args.overlap_pair_pgs:
        commsParams.num_pgs = 2
    collBenchObj.runBench(comms_world_info, commsParams)


if __name__ == "__main__":
    main()
