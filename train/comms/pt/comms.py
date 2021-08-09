#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import time
from itertools import cycle

import comms_utils
import numpy as np

# pytorch
import torch
from comms_utils import paramCommsBench

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
            "--mode", type=str, default="comms", help="benchmark mode"
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
            "--c",
            type=int,
            default=0,
            help="enable data validation check",
            choices=[0, 1],
        )  # validation check
        parser.add_argument(
            "--collective",
            type=str,
            default="all_reduce",
            help="Collective operation to be evaluated",
            choices=supportedCollectives,
        )  # collective op to benchmark
        # For comm-compute or compute mode
        parser.add_argument(
            "--kernel", type=str, default="gemm", help="compute kernel"
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
        )  # root process for reduce (and gather, scatter, bcast, etc., if support in the future)
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
            type=int,
            default=0,
            help="enable pair mode",
            choices=[0, 1],
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
            type=int,
            default=0,
            help="overlap collective pair with two pgs",
            choices=[0, 1],
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
                print(
                    "\t ERROR: Specified pt2pt pattern: %d is not one of the supported pt2pt patterns: %s"
                    % (args.pt2pt, str(pt2ptPatterns))
                )
                comms_utils.gracefulExit()

        args.b = comms_utils.parsesize(args.b)
        args.e = comms_utils.parsesize(args.e)
        args.dtype = self.dtypeMap[args.data_type]

        if args.b < 1:
            print("\t Starting size: %d should atleast be 1! " % (args.b))
            args.b = 1

        if args.e < args.b:
            print(
                "\t ERROR: In COMMS-mode, the begin-size: %d is larger than the end-size: %d "
                % (args.b, args.e)
            )

        if args.device == "cpu" and args.backend == "nccl":
            raise ValueError("NCCL is not supported for device type CPU")

        if args.c == 1 and args.z == 0:
            logging.warning(
                "Data validation is not supported for non-blocking mode...disable validation check and proceed..."
            )
            args.c = 0

        # run a few sanity checks
        if args.bitwidth < 32:
            if args.collective not in (
                "all_to_all",
                "all_to_allv",
                "reduce",
                "all_reduce",
            ):
                raise NotImplementedError(
                    f"quantized communication for {args.collective} is currently unsupported."
                )
            if args.collective in ("all_to_all", "all_to_allv"):
                if (args.beginSize // 4) % args.quant_a2a_embedding_dim != 0:
                    raise ValueError(
                        "begin size (elements) must be a multiple of --quant-a2a-embedding-dim for all_to_all operation"
                    )
                if args.blockingFlag != 1:
                    raise NotImplementedError(
                        "quantized All_to_all must be synchronous."
                    )
            if args.dtype != torch.float32:
                raise NotImplementedError(
                    f"quantization for {args.dtype} is not supported. Use float32 instead."
                )

    def runColl(self, comm_fn=None, compute_fn=None, comm_fn_pair=None):
        self.backendFuncs.complete_accel_ops(self.collectiveArgs, initOp=True)
        numElements = self.collectiveArgs.numElements
        if comm_fn_pair is not None:
            numElements_pair = self.collectiveArgs.numElements_pair
        # Initial warmup iters.
        for _ in range(self.collectiveArgs.numWarmupIters):
            if comm_fn is not None:
                if self.collectiveArgs.num_pgs > 1:
                    self.collectiveArgs.group = self.collectiveArgs.groups[0]
                comm_fn(self.collectiveArgs)
            if comm_fn_pair is not None:
                if self.collectiveArgs.num_pgs > 1:
                    self.collectiveArgs.group = self.collectiveArgs.groups[1]
                comm_fn_pair(self.collectiveArgs, pair=True)
            if compute_fn is not None:
                for _ in range(self.collectiveArgs.numComputePerColl):
                    compute_fn(self.collectiveArgs)
            if not self.collectiveArgs.asyncOp:  # should be sychronous, do wait.
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)

        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_begin")

        # Measuring time.
        elapsedTimeNS = 0.0
        self.collectiveArgs.quant_time.reset()
        self.collectiveArgs.dequant_time.reset()
        for _ in range(self.collectiveArgs.numIters):
            if (
                not self.collectiveArgs.asyncOp
            ):  # should be sychronous, do barrier and wait for collective
                self.setTensorVal(self.collectiveArgs.opTensor)  # reset tensor values
                if comm_fn_pair is not None:
                    self.setTensorVal(self.collectiveArgs.opTensor_pair)
                self.backendFuncs.sync_barrier(self.collectiveArgs)
            oldAsyncOp = self.collectiveArgs.asyncOp
            round_robin_group = cycle(self.collectiveArgs.groups)
            if comm_fn_pair is not None:
                self.collectiveArgs.asyncOp = True

            start = time.monotonic()  # available only in py3
            if comm_fn is not None:
                self.collectiveArgs.group = next(round_robin_group)
                comm_fn(self.collectiveArgs)
            if comm_fn_pair is not None:
                self.collectiveArgs.group = next(round_robin_group)
                comm_fn_pair(self.collectiveArgs, pair=True)
            if compute_fn is not None:
                for _ in range(self.collectiveArgs.numComputePerColl):
                    # TODO: investigate the cache effect
                    # Flush the cache
                    # _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
                    compute_fn(self.collectiveArgs)
            self.collectiveArgs.asyncOp = oldAsyncOp
            if (
                not self.collectiveArgs.asyncOp
            ):  # should be sychronous, wait for the collective
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            # Measuring time.
            elapsedTimeNS += (
                time.monotonic() - start
            ) * 1e9  # keeping time in NS, helps in divising data by nanosecond

        start = time.monotonic()  # available only in py3
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)
        end = time.monotonic()  # available only in py3
        if isinstance(self.collectiveArgs.opTensor, list):
            # allgather is a list of tensors
            x = self.collectiveArgs.opTensor[-1][
                -1
            ].item()  # to ensure collective won't be optimized away.
        else:
            x = self.collectiveArgs.opTensor[
                numElements - 1
            ].item()  # to ensure collective won't be optimized away.
        x_pair = None
        if comm_fn_pair is not None:
            if isinstance(self.collectiveArgs.opTensor_pair, list):
                # allgather is a list of tensors
                x_pair = self.collectiveArgs.opTensor_pair[-1][
                    -1
                ].item()  # to ensure collective won't be optimized away.
            else:
                x_pair = self.collectiveArgs.opTensor_pair[
                    numElements_pair - 1
                ].item()  # to ensure collective won't be optimized away.

        elapsedTimeNS += (
            end - start
        ) * 1e9  # keeping time in NS, helps in divising data by nanoseconds
        # calculate average (de-)quantization overhead (in nanoseconds)
        avgQuantNS = {
            "Quant": (
                self.collectiveArgs.quant_time.getTimeNS()
                / self.collectiveArgs.numIters
            ),
            "DeQuant": (
                self.collectiveArgs.dequant_time.getTimeNS()
                / self.collectiveArgs.numIters
            ),
        }

        memSize = self.backendFuncs.get_mem_size(self.collectiveArgs)

        avgIterNS, algBW = comms_utils.getAlgBW(
            elapsedTimeNS, memSize, self.collectiveArgs.numIters
        )
        busBW = self.backendFuncs.getBusBW(
            self.collectiveArgs.collective,
            algBW,
            self.collectiveArgs,
        )
        if comm_fn_pair is not None:
            memSize_pair = self.backendFuncs.get_mem_size(
                self.collectiveArgs, pair=True
            )
            memSize += memSize_pair

            _, algBW_pair = comms_utils.getAlgBW(
                elapsedTimeNS, memSize_pair, self.collectiveArgs.numIters
            )
            algBW += algBW_pair
            busBW_pair = self.backendFuncs.getBusBW(
                self.collectiveArgs.collective_pair,
                algBW_pair,
                self.collectiveArgs,
            )
            busBW += busBW_pair

        self.backendFuncs.sync_barrier(self.collectiveArgs, desc="runColl_end")
        return (avgIterNS, algBW, busBW, memSize, avgQuantNS, x_pair)

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
        return (pingPerIterNS, pingPongPerIterNS, avgUniBW, avgBiBW, memSize)

    def getPingLatency(self, numIters):
        logging.debug(
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
        logging.debug("STATUS: end ping test.")
        return pingLatencyNS

    def getPingPongLatency(self, numIters):
        logging.debug(
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
        logging.debug("STATUS: end ping-pong test.")
        return pingPongLatencyNS

    def getUniBW(self, numIters, memSize):
        logging.debug(
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
        logging.debug("STATUS: end UniBW test.")
        return avgUniBW

    def getBiBW(self, numIters, memSize):
        logging.debug(
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
        logging.debug("STATUS: end UniBW test.")
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
                    logging.error(
                        "One2one Pt2Pt requires only a single rank is specified in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()
        elif self.collectiveArgs.pt2pt == "pairwise":
            # pairwise pt2pt requires identical number of ranks in src_ranks and dst_ranks.
            if len(self.collectiveArgs.src_ranks) != len(self.collectiveArgs.dst_ranks):
                if self.global_rank == 0:
                    logging.error(
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
                    logging.error(
                        "Pairwise Pt2Pt requires distinct members in src_ranks and dst_ranks! "
                    )
                comms_utils.gracefulExit()

        if self.global_rank == 0:
            print(
                "\t collectiveArgs.collective=%s  %s, src_ranks=%s, dst_ranks=%s"
                % (
                    self.collectiveArgs.collective,
                    self.collectiveArgs.pt2pt,
                    self.collectiveArgs.src_ranks,
                    self.collectiveArgs.dst_ranks,
                )
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
                "\t collectiveArgs.collective=%s, src_ranks=%s, dst_ranks=%s"
                % (
                    self.collectiveArgs.collective,
                    self.collectiveArgs.src_ranks,
                    self.collectiveArgs.dst_ranks,
                )
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

        self.comm_size = world_size
        self.global_rank = global_rank

        comms_utils.fixBeginSize(
            commsParams, world_size
        )  # Ensuring that all-reduce and all-to-all has atleast one member per rank.
        self.backendFuncs.sayHello()  # Informs us where each process is running.
        allSizes = comms_utils.getSizes(
            commsParams.beginSize, commsParams.endSize, commsParams.stepFactor
        )  # Given the begin-size, end-size, step-factor what are the message sizes to iterate on.

        if global_rank == 0:
            print(
                "\t global_rank: %d allSizes: %s local_rank: %d element_size: %d "
                % (global_rank, allSizes, local_rank, commsParams.element_size)
            )
            print(
                "\t global_rank: %d commsParams: %s "
                % (global_rank, commsParams.__dict__)
            )

        # self.collectiveArgs = comms_utils.collectiveArgsHolder()
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

        if commsParams.bitwidth < 32:
            comms_utils.initQuantCommCtx(self.collectiveArgs, commsParams)

        if self.collectiveArgs.collective == "pt2pt":
            self.checkPt2PtRanks()
        else:
            self.checkCollectiveRanks()

        computeFunc = None
        if commsParams.mode != "comms":  # Compute mode related initialization.
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

    def gatherBenchTime(self, collectiveArgs, commsParams, timeElapsedList):
        # Push the list to device, then do an all-gather.
        timeElapsedTensor = torch.tensor(
            timeElapsedList, device=self.backendFuncs.get_device()
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

    def reportBenchTimeWithQuant(
        self,
        commsParams,
        allSizes,
        tensorList,
        quantTimeTensorList,
        dequantTimeTensorList,
        results,
    ):
        print(
            "\tCOMMS-QUANT-RES\t{:>15}\t{:>15}\t{:>25}\t{:>15}\t{:>15}\t{:>15}".format(
                "size (B)",
                "num-elements",
                "P95 Latency(us): Quant",
                "Comms",
                "De-Quant",
                "Overall",
            )
        )

        for idx, curSize in enumerate(allSizes):
            if commsParams.backend == "xla":
                latencyAcrossRanks = torch.transpose(
                    tensorList.view(-1, len(allSizes)), 0, 1
                )[idx]
                latencyAcrossRanks = latencyAcrossRanks.cpu().detach().numpy()
                # quant tensor
                quantLatencyAcrossRanks = torch.transpose(
                    quantTimeTensorList.view(-1, len(allSizes)), 0, 1
                )[idx]
                quantLatencyAcrossRanks = quantLatencyAcrossRanks.cpu().detach().numpy()
                # dequant tensor
                dequantLatencyAcrossRanks = torch.transpose(
                    dequantTimeTensorList.view(-1, len(allSizes)), 0, 1
                )[idx]
                dequantLatencyAcrossRanks = (
                    dequantLatencyAcrossRanks.cpu().detach().numpy()
                )
            else:
                latencyAcrossRanks = []
                for curRankTensor in tensorList:
                    rank_lat = curRankTensor[idx].item()
                    latencyAcrossRanks.append(rank_lat)

                latencyAcrossRanks = np.array(latencyAcrossRanks)
                # quant tensor
                quantLatencyAcrossRanks = []
                for curRankTensor in quantTimeTensorList:
                    rank_lat = curRankTensor[idx].item()
                    quantLatencyAcrossRanks.append(rank_lat)
                quantLatencyAcrossRanks = np.array(quantLatencyAcrossRanks)
                # dequant tensor
                dequantLatencyAcrossRanks = []
                for curRankTensor in dequantTimeTensorList:
                    rank_lat = curRankTensor[idx].item()
                    dequantLatencyAcrossRanks.append(rank_lat)
                dequantLatencyAcrossRanks = np.array(dequantLatencyAcrossRanks)

            p95 = np.percentile(latencyAcrossRanks, 95)

            quant_p95 = np.percentile(quantLatencyAcrossRanks, 95)
            dequant_p95 = np.percentile(dequantLatencyAcrossRanks, 95)

            self.collectiveArgs.dataSize = curSize

            print(
                "\tCOMMS-QUANT-RES\t{:>15}\t{:>15}\t{:>25}\t{:>15}\t{:>15}\t{:>15}".format(
                    results[curSize]["memSize"],
                    str("%d" % (results[curSize]["num_elements"])),
                    str("%.1f" % (quant_p95)),
                    str("%.1f" % (p95 - quant_p95 - dequant_p95)),
                    str("%.1f" % (dequant_p95)),
                    str("%.1f" % (p95)),
                    # str("%.3f" % (algBW)),
                    # str("%.3f" % (busBW)),
                )
            )

    def reportBenchTime(self, commsParams, allSizes, tensorList, results):
        self.collectiveArgs.collective = commsParams.collective
        self.collectiveArgs.numIters = 1  # commsParams.numIters
        self.collectiveArgs.pair = commsParams.pair
        self.collectiveArgs.collective_pair = commsParams.collective_pair

        if self.collectiveArgs.pair == 0:
            print(
                "\n\tCOMMS-RES\tsize (B)\t num-elements\t Latency(us):p50\tp75\t\tp95\tmin\tmax\t algBW(GB/s)\t busBW(GB/s)"
            )
        else:
            print(
                "\n\tCOMMS-RES\ttotal-pair-size (B)\t num-elements\t num-elements-pair\t Latency(us):p50\tp75\t\tp95\tmin\tmax\t algBW(GB/s)\t busBW(GB/s)"
            )
        for idx, curSize in enumerate(allSizes):
            if commsParams.backend == "xla":
                latencyAcrossRanks = torch.transpose(
                    tensorList.view(-1, len(allSizes)), 0, 1
                )[idx]
                latencyAcrossRanks = latencyAcrossRanks.cpu().detach().numpy()
            else:
                latencyAcrossRanks = []
                for curRankTensor in tensorList:
                    rank_lat = curRankTensor[idx].item()
                    latencyAcrossRanks.append(rank_lat)

                latencyAcrossRanks = np.array(latencyAcrossRanks)

            logging.debug("Latency across all ranks: %s" % (latencyAcrossRanks))

            # Include only communicating ranks
            if self.collectiveArgs.collective == "multicast":
                commRanks = [
                    self.collectiveArgs.srcOrDst
                ] + self.collectiveArgs.dst_ranks
            elif self.collectiveArgs.collective == "incast":
                commRanks = [
                    self.collectiveArgs.srcOrDst
                ] + self.collectiveArgs.src_ranks
            else:
                commRanks = range(self.collectiveArgs.world_size)

            latencyAcrossCommRanks = latencyAcrossRanks[commRanks]
            logging.debug(
                "Latency across communicating ranks (%s): %s"
                % (commRanks, latencyAcrossCommRanks)
            )

            p50 = np.percentile(latencyAcrossCommRanks, 50)
            p75 = np.percentile(latencyAcrossCommRanks, 75)
            p95 = np.percentile(latencyAcrossCommRanks, 95)
            minlat = np.amin(latencyAcrossCommRanks)
            maxlat = np.amax(latencyAcrossCommRanks)

            # adjust busBW
            busBW = results[curSize]["busBW"] * (commsParams.bitwidth / 32.0)

            if self.collectiveArgs.pair == 0:
                print(
                    "\tCOMMS-RES\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s"
                    % (
                        results[curSize]["memSize"],
                        str("%d" % (results[curSize]["num_elements"])),
                        str("%.1f" % (p50)),
                        str("%.1f" % (p75)),
                        str("%.1f" % (p95)),
                        str("%.1f" % (minlat)),
                        str("%.1f" % (maxlat)),
                        str("%.3f" % (results[curSize]["algBW"])),
                        str("%.3f" % (busBW)),
                    )
                )
            else:
                print(
                    "\tCOMMS-RES\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s"
                    % (
                        results[curSize]["memSize"],
                        str("%d" % (results[curSize]["num_elements"])),
                        str("%d" % (results[curSize]["num_elements_pair"])),
                        str("%.1f" % (p50)),
                        str("%.1f" % (p75)),
                        str("%.1f" % (p95)),
                        str("%.1f" % (minlat)),
                        str("%.1f" % (maxlat)),
                        str("%.3f" % (results[curSize]["algBW"])),
                        str("%.3f" % (busBW)),
                    )
                )

    def reportBenchTimePt2Pt(self, commsParams, allSizes, resultsAcrossRanks, results):
        print(
            "\n\tCOMMS-RES\tsize (B)\t pingLatency(us):p50\tp75\t\tp95\t pingPongLatency(us):p50\tp75\t\tp95\t avgUniBW(GB/s)\t avgBiBW(GB/s) totalUniBW(GB/s)\t totalBiBW(GB/s)"
        )
        for idx, curSize in enumerate(allSizes):
            pingLatencyAcrossRanks = []
            pingPongLatencyAcrossRanks = []
            uniBWAcrossRanks = []
            biBWAcrossRanks = []
            for curRankTensor in resultsAcrossRanks:
                pingLatencyAcrossRanks.append(curRankTensor[idx][0].item())
                pingPongLatencyAcrossRanks.append(curRankTensor[idx][1].item())
                uniBWAcrossRanks.append(curRankTensor[idx][2].item())
                biBWAcrossRanks.append(curRankTensor[idx][3].item())

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

            logging.debug(
                "Ping latency across communicating ranks (%s): %s"
                % (commRanks, pingLatencyAcrossCommRanks)
            )
            logging.debug(
                "PingPong latency across communicating ranks (%s): %s"
                % (commRanks, pingPongLatencyAcrossCommRanks)
            )
            logging.debug(
                "UniBW across all communicating ranks (%s): %s"
                % (commRanks, uniBWAcrossCommRanks)
            )
            logging.debug(
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
                "\tCOMMS-RES\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s"
                % (
                    results[curSize]["memSize"],
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

        results = {}
        timeElapsedList = []
        quantTimeElapsedList = []
        dequantTimeElapsedList = []
        for curSize in allSizes:
            # Allocating memory.
            numElements = int(curSize // commsParams.element_size)
            scaleFactor = numElements * numElements
            if commsParams.collective == "all_to_all":
                # numElements = int(numElements // world_size)  # assuming that world_size won't be zero!
                scaleFactor = 1

            if commsParams.dcheck == 1:
                # use all ones for easy data validation check
                ipTensor = backendFuncs.alloc_ones(
                    [numElements], curDevice, commsParams.dtype, self.initVal
                )
            else:
                ipTensor = backendFuncs.alloc_random(
                    [numElements], curDevice, commsParams.dtype, scaleFactor
                )

            opTensor = ipTensor
            asyncOp = True
            collectiveFunc = None

            if (
                commsParams.blockingFlag == 1
            ):  # if blockingFlag is 1, it means asyncOp should be false.
                asyncOp = False

            if commsParams.mode != "compute":  # comms specific initializations
                if commsParams.collective.startswith("all_to_all"):
                    # all_to_all(v) requires two tensors
                    opTensor = backendFuncs.alloc_random(
                        [numElements], curDevice, commsParams.dtype, scaleFactor
                    )
                    # all_to_allv requires tensors to specify split
                    if commsParams.collective == "all_to_allv":
                        self.collectiveArgs.ipTensor_split = [
                            int(numElements // world_size) for i in range(world_size)
                        ]
                        self.collectiveArgs.opTensor_split = [
                            int(numElements // world_size) for i in range(world_size)
                        ]
                elif commsParams.collective == "all_gather":
                    # allgather requires a tensor list, e.g., List[torch.Tensor]
                    opTensor = []
                    for _ in range(world_size):
                        opTensor.append(
                            backendFuncs.alloc_random(
                                [numElements], curDevice, commsParams.dtype, scaleFactor
                            )
                        )
                elif commsParams.collective == "incast":
                    # incast requires a tensor list with length of src_ranks, e.g., List[torch.Tensor]
                    opTensor = []
                    for _ in self.collectiveArgs.src_ranks:
                        opTensor.append(
                            backendFuncs.alloc_random(
                                [numElements], curDevice, commsParams.dtype, scaleFactor
                            )
                        )
                elif commsParams.collective == "reduce_scatter":
                    ipTensor = []
                    for _ in range(world_size):
                        ipTensor.append(
                            backendFuncs.alloc_random(
                                [numElements], curDevice, commsParams.dtype, scaleFactor
                            )
                        )
                elif commsParams.collective == "all_gather_base":
                    # this is a single all gather
                    opTensor = backendFuncs.alloc_random(
                        numElements * world_size,
                        curDevice,
                        commsParams.dtype,
                        scaleFactor,
                    )
                # set corresponding function pointers
                if commsParams.collective != "pt2pt":
                    collectiveFunc = backendFuncs.collectiveFunc[commsParams.collective]

            # Setup the arguments.
            self.collectiveArgs.ipTensor = ipTensor
            self.collectiveArgs.opTensor = opTensor
            self.collectiveArgs.asyncOp = asyncOp
            self.collectiveArgs.dataSize = curSize
            self.collectiveArgs.numElements = numElements
            self.collectiveArgs.waitObj = []

            collectiveFunc_pair = None
            if commsParams.pair != 0:
                curSize_pair = curSize
                # Allocating memory.
                numElements_pair = int(curSize_pair // commsParams.element_size)
                scaleFactor_pair = numElements_pair * numElements_pair
                if commsParams.collective_pair == "all_to_all":
                    # numElements = int(numElements // world_size)  # assuming that world_size won't be zero!
                    scaleFactor_pair = 1

                if commsParams.dcheck == 1:
                    # use all ones for easy data validation check
                    ipTensor_pair = backendFuncs.alloc_ones(
                        [numElements_pair], curDevice, commsParams.dtype, self.initVal
                    )
                else:
                    ipTensor_pair = backendFuncs.alloc_random(
                        [numElements_pair],
                        curDevice,
                        commsParams.dtype,
                        scaleFactor_pair,
                    )

                opTensor_pair = ipTensor_pair
                collectiveFunc_pair = None

                if commsParams.mode != "compute":  # comms specific initializations
                    if commsParams.collective_pair.startswith("all_to_all"):
                        # all_to_all(v) requires two tensors
                        opTensor_pair = backendFuncs.alloc_random(
                            [numElements_pair],
                            curDevice,
                            commsParams.dtype,
                            scaleFactor_pair,
                        )
                        # all_to_allv requires tensors to specify split
                        if commsParams.collective_pair == "all_to_allv":
                            self.collectiveArgs.ipTensor_split_pair = [
                                int(numElements_pair // world_size)
                                for i in range(world_size)
                            ]
                            self.collectiveArgs.opTensor_split_pair = [
                                int(numElements_pair // world_size)
                                for i in range(world_size)
                            ]
                    elif commsParams.collective_pair == "all_gather":
                        # allgather requires a tensor list, e.g., List[torch.Tensor]
                        opTensor_pair = []
                        for _ in range(world_size):
                            opTensor_pair.append(
                                backendFuncs.alloc_random(
                                    [numElements_pair],
                                    curDevice,
                                    commsParams.dtype,
                                    scaleFactor_pair,
                                )
                            )
                    # set corresponding function pointers
                    if commsParams.collective_pair != "pt2pt":
                        collectiveFunc_pair = backendFuncs.collectiveFunc[
                            commsParams.collective_pair
                        ]

                # Setup the arguments.
                self.collectiveArgs.ipTensor_pair = ipTensor_pair
                self.collectiveArgs.opTensor_pair = opTensor_pair
                self.collectiveArgs.dataSize_pair = curSize_pair
                self.collectiveArgs.numElements_pair = numElements_pair

            # self.collectiveArgs has all the information on the experiment.
            if commsParams.collective != "pt2pt":
                timeElapsedNS, algBW, busBW, memSize, avgQuantNS, x_pair = self.runColl(
                    comm_fn=collectiveFunc,
                    compute_fn=computeFunc,
                    comm_fn_pair=collectiveFunc_pair,
                )

                # perfom data validation check on the final opTensor
                if commsParams.dcheck == 1:
                    self.dcheck(commsParams, curSize, opTensor)

                results[curSize] = {}
                results[curSize]["timeUS"] = timeElapsedNS / 1e3
                timeElapsedList.append(
                    results[curSize]["timeUS"]
                )  # assuming that order is known at each rank, so it's OK to not identify it by message-size
                results[curSize]["algBW"] = algBW
                results[curSize]["busBW"] = busBW
                results[curSize]["memSize"] = memSize
                if (commsParams.collective == "all_to_all") or (
                    commsParams.collective == "all_to_allv"
                ):
                    results[curSize]["num_elements"] = int(numElements // world_size)
                else:
                    results[curSize]["num_elements"] = int(numElements)

                results[curSize]["quantTimeUS"] = avgQuantNS["Quant"] / 1e3
                results[curSize]["dequantTimeUS"] = avgQuantNS["DeQuant"] / 1e3
                quantTimeElapsedList.append(results[curSize]["quantTimeUS"])
                dequantTimeElapsedList.append(results[curSize]["dequantTimeUS"])

                if commsParams.pair:
                    results[curSize]["curSizePair"] = curSize_pair
                    if (commsParams.collective_pair == "all_to_all") or (
                        commsParams.collective_pair == "all_to_allv"
                    ):
                        results[curSize]["num_elements"] = int(
                            numElements // world_size
                        )
                    else:
                        results[curSize]["num_elements"] = int(numElements)
                        results[curSize]["x_pair"] = x_pair
            else:
                (
                    pingPerIterNS,
                    pingPongPerIterNS,
                    avgUniBW,
                    avgBiBW,
                    memSize,
                ) = self.runPt2Pt()
                results[curSize] = {}
                results[curSize]["pingPerIterNS"] = pingPerIterNS
                results[curSize]["pingPongPerIterNS"] = pingPongPerIterNS
                results[curSize]["avgUniBW"] = avgUniBW
                results[curSize]["avgBiBW"] = avgBiBW
                results[curSize]["memSize"] = memSize
                timeElapsedList.append(
                    [
                        np.mean(np.array(pingPerIterNS)) / 1e3,
                        np.mean(np.array(pingPongPerIterNS)) / 1e3,
                        avgUniBW,
                        avgBiBW,
                    ]  # time in US
                )

            del ipTensor
            del opTensor
            if commsParams.pair:
                del ipTensor_pair
                del opTensor_pair
            backendFuncs.clear_memory()
            self.backendFuncs.sync_barrier(
                self.collectiveArgs, desc=f"curSize_{curSize}"
            )

        comms_utils.clearQuantCommCtx(self.collectiveArgs)

        tensorList = self.gatherBenchTime(
            self.collectiveArgs, commsParams, timeElapsedList
        )

        if global_rank == 0:
            if self.collectiveArgs.collective != "pt2pt":
                self.reportBenchTime(commsParams, allSizes, tensorList, results)
                quantTimeTensorList = []
                if commsParams.bitwidth < 32:
                    logging.debug(quantTimeElapsedList)
                    quantTimeTensorList = self.gatherBenchTime(
                        self.collectiveArgs, commsParams, quantTimeElapsedList
                    )
                    dequantTimeTensorList = self.gatherBenchTime(
                        self.collectiveArgs, commsParams, dequantTimeElapsedList
                    )
                    if global_rank == 0:
                        self.reportBenchTimeWithQuant(
                            commsParams,
                            allSizes,
                            tensorList,
                            quantTimeTensorList,
                            dequantTimeTensorList,
                            results,
                        )
            else:
                self.reportBenchTimePt2Pt(commsParams, allSizes, tensorList, results)

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
            print("\t Error: Unsopported NW stack! ")
            comms_utils.gracefulExit()

        self.backendFuncs = backendObj
        try:
            backendObj.benchmark_comms()
        except ValueError as ve:
            if commsParams.backend == "ucc":
                logging.critical("PyTorch UCC not implemented? {}".format(repr(ve)))
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

    if args.pair == 1 and args.overlap_pair_pgs == 1:
        commsParams.num_pgs = 2
    collBenchObj.runBench(comms_world_info, commsParams)


if __name__ == "__main__":
    main()
