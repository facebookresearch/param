#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import gzip
import json
import logging
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from et_replay.comm import comms_utils, commsTraceParser, profiler_trace_analysis
from et_replay.comm.backend.base_backend import supportedC10dBackends, supportedP2pOps
from et_replay.comm.comms_utils import (
    bootstrap_info_holder,
    commsArgs,
    commsParamsHolderBase,
    paramCommsBench,
    paramStreamGuard,
    paramToCommName,
)
from et_replay.comm.param_profile import paramProfile, paramTimer
from torch.profiler import ProfilerActivity

try:
    from et_replay.vendor_internal.fb_internal import (
        get_fb_profiler_activities,
        get_fb_profiler_trace_handler,
    )

    has_fb_internal_libs = True
except ImportError:
    has_fb_internal_libs = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# sleep for 20ms to wait for next collective
LOOP_TIMER_S = 0.02

# index 0 is default value of trace type
VALID_TRACE_TYPES = ["et"]


def writeCommDetails(commsTracePerf: list, rank: int, folder: str = "./") -> None:
    """
    Writes the replayed comm details of the current rank.

    Args:
        commsTracePerf: List that contains the metrics of each replayed collective in the current rank.
        rank: The current rank that the comm details will be written for.
        folder: Directory path to where the comm details for all ranks will be written.
                                If none, no output will be written.
    Returns:
        None
    """
    if len(folder) == 0:
        # skip output if the path is explicitly set to ""
        return
    comms_file = folder + f"/replayedCommsPerf.rank-{rank}.json"
    logger.info(f"[Rank {rank:3}] Writing comms details to {comms_file}")

    saveToLocal = True
    if "://" in comms_file:  # assume that "://" in directory path means remote store
        saveToLocal = False
        try:
            from param_bench.et_replay.comm.vendor_internal.fb_internals import (
                writeRemoteTrace as writeFbRemoteTrace,
            )

        except ImportError:
            saveToLocal = True
            pass
        else:
            writeFbRemoteTrace(commsTracePerf, remotePath=comms_file)

    if saveToLocal:
        try:
            import pathlib

            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied to create directory {folder}")

        with open(comms_file, "w") as write_file:
            json.dump(commsTracePerf, write_file, indent=2)


# pyre-ignore[13]: lint complained about self.backendFuncs is never initlized.
#                  it is initialized in initBackend
class commsTraceReplayBench(paramCommsBench):
    """
    A class to replay and benchmark generated traces for collective communications.

    This class will read a provided trace and replay it based on runtime parameters specified in the command line.
    At the end of a replay, the benchmarks for the run will be recorded in different JSON files in the specified out_path, if provided.
    The goal of this class is to help scale AI training optimizations by studying the behaviours of AI backends.
    """

    def __init__(self):
        super().__init__(supportedNwstacks=["pytorch-dist", "pytorch-xla-tpu"])
        self.comms_trace = {}
        self.trace_file = ""
        self.trace_type = ""
        self.use_remote_trace = False
        self.use_one_trace = False
        self.disable_parallel_read = False
        self.is_dry_run = False
        self.shrink = False
        self.max_msg_cnt = 0  # 0 means no limit
        self.num_msg = 0
        self.is_blocking = False
        self.warmup_iter = 2
        self.reuse_tensors = False

        self.allowList = ""
        self.out_path = ""
        self.outputRanks = None
        self.colls_per_batch = -1
        self.coll_in_batch_num = 0
        self.replay_start_time = -1
        self.use_timestamp = False
        self.num_replays = 5
        self.profiler_num_replays_start = 0
        self.profiler_num_replays = 5

        self.collInMsgBytes: dict[str, list] = {}
        self.collInUniMsgBytes: dict[str, set] = {}
        self.collOutMsgBytes: dict[str, list] = {}
        self.collOutUniMsgBytes: dict[str, set] = {}

        self.batchLat = []
        self.collLat: dict[str, list] = {}
        self.compLat: dict[str, list] = {}

        self.comms_blocks: dict[str, list] = {}
        self.traceWithPerf = []
        self.blockStack = []
        self.replayIter = 0

        self.rebalance_policy = ""

        # for blocking collectives this is the sum of all the collective latencies
        # for nonblocking collectives this is the sum of how long each collective took to be sent to the device
        self.totalCommsLatency = 0.0
        # sum of all compute kernel latencies
        self.totalCompsLatency = 0.0
        # how long it took to finish all collectives in the trace
        self.totalTraceLatency = 0.0

        self.et_to_tensors = {}

        self.gemmTensor = None

        self.embLookupReuse = {}

    def readArgs(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        """
        Reads command line args to set runtime parameters for replay.

        Args:
            parser: ArgumentParser that will handle parsing command line arguments.
        Returns:
            Namespace containing a collection of parser arguments.
        """
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--trace-path",
            type=str,
            default="./",
            help="File path to read the trace. All rank read their own trace file unless `--use-one-trace` is used.",
        )
        parser.add_argument(
            "--trace-type",
            type=str,
            choices=VALID_TRACE_TYPES,
            default=VALID_TRACE_TYPES[0],
            help=f"Select trace type used for replay. Supported trace types: {VALID_TRACE_TYPES}. \
                   'et' represents Chakra host execution trace.",
        )
        parser.add_argument(
            "--use-one-trace",
            action="store_true",
            default=False,
            help="Toggle to use only one trace for all ranks",
        )
        parser.add_argument(
            "--disable-parallel-read",
            action="store_true",
            default=False,
            help="Disable parallel read from input trace path. Instead, rank 0 will read and broadcast to other ranks. "
            + "Valid only when `--use-one-trace` is used.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=self.is_dry_run,
            help="Toggle to only analyze trace without actually replaying collectives",
        )
        parser.add_argument(
            "--auto-shrink",
            action="store_true",
            default=self.shrink,
            help="Toggle to shrink message size when it does not match with the current scale (only for debug purpose)",
        )
        parser.add_argument(
            "--max-msg-cnt",
            type=int,
            default=self.max_msg_cnt,
            help="Only replay first N operations (0 means no limit)",
        )
        parser.add_argument(
            "--warmup-iter",
            type=int,
            default=self.warmup_iter,
            help="Number of warmup iterations",
        )
        parser.add_argument(
            "--reuse-tensors",
            action="store_true",
            default=self.reuse_tensors,
            help="Toggle to cache and reuse the same input/output for each compute kernel",
        )
        parser.add_argument(
            "--allow-ops",
            "--allow-list",
            type=str,
            default="all",
            help="List of desired collectives (separate by comma) to be replayed, e.g., `--allow-ops all_reduce,all_to_allv,wait`, typo or not supported collectives will be ignored.",
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=self.out_path,
            nargs="?",
            const="",
            help="Path to store generated results (e.g., replayed trace, profiler trace) for post performance analysis. (Default: %(default)s)",
        )

        parser.add_argument(
            "--output-ranks",
            type=str,
            default=None,
            help="List of ranks separated by comma (e.g. 1,2,3) OR a range specified by start:end (e.g., 1:3) to enable replayed trace dumping for post performance analysis. (Default: %(default)s)",
        )
        parser.add_argument(
            "--colls-per-batch",
            type=int,
            default=self.colls_per_batch,
            help="Toggle to set number of consecutive collectives in a batch. This also enables per batch latency stats.",
        )
        parser.add_argument(
            "--use-timestamp",
            action="store_true",
            default=self.use_timestamp,
            help="Toggle to use time-based replay.",
        )
        parser.add_argument(
            "--rebalance-policy",
            type=str,
            default="",
            help="Balancing policy for all_to_allv splits, this will occur during warm-up. Supported policies:['equal']. Unsupported policies will be ignored.",
        )
        parser.add_argument(
            "--num-replays",
            type=int,
            default=self.num_replays,
            help="Number of times to replay the given trace, used to get more accurate replay for small traces.",
        )

        parser.add_argument(
            "--profiler-num-replays-start",
            type=int,
            default=self.profiler_num_replays_start,
            help="Number of replay iteration to start collecting profiler trace after warmup in all ranks. (Default: %(default)s)",
        )
        parser.add_argument(
            "--profiler-num-replays",
            type=int,
            default=self.profiler_num_replays,
            help="Number of replay iterations to collect profiler trace in all ranks. (Default: %(default)s)",
        )

        args, _ = parser.parse_known_args()
        return args

    def checkArgs(self, args: argparse.Namespace) -> None:
        """
        Validates command line args, will raise an error and gracefully exit if an invalid command line arg is found.

        Args:
            args: Namespace containing collection of args to validate.
        Returns:
            None
        """
        super().checkArgs(args)

        if (
            not self.use_remote_trace
            and not os.path.isfile(self.trace_file)
            and not os.path.isdir(self.trace_file)
        ):
            raise ValueError(
                f"The specified trace path '{self.trace_file}' is neither a "
                "file nor a directory. Please provide a valid path."
            )

        if args.disable_parallel_read and not args.use_one_trace:
            raise ValueError(
                "--disable-parallel-read is valid only when --use-one-trace is used."
            )

        if args.trace_type not in VALID_TRACE_TYPES:
            raise ValueError(
                f"Trace type {self.trace_type} is not valid! Please specify one supported trace type from {str(VALID_TRACE_TYPES)} by using --trace-type."
            )

        if (
            args.output_ranks is not None
            and len(args.output_ranks) > 0
            and not len(args.output_path)
        ):
            raise ValueError('"--output-path" is not set for replay trace dumping')

        if (
            args.enable_profiler
            and not len(args.output_path)
            and not has_fb_internal_libs
        ):
            raise ValueError('"--output-path" is not set for profiler trace dumping')

    def reportBenchTime(self):
        """
        Prints replay benchmarks for current rank. This should only be called after setBench() and benchTime()
        to ensure that replay statistics are available to read.

        Args:
            None
        Returns:
            None
        """
        # TODO:
        #   1) dry run: output some statistics, e.g., # of msgs, distribtuion of sizes (max, min, avg, p50, p95...ect)
        #   2) normal run: output 1) as well as perf. breakdown (e.g., a2a latencies at different phase, some percentages...ect)
        # some basic stats
        print(
            f"\n+++++ {len(self.comms_trace)} msgs recorded in {self.trace_file} +++++\n"
        )

        for curBlock, blockComms in self.comms_blocks.items():
            lat_list = []
            if not self.is_dry_run:
                lat_list = [comm["latency_us"] for comm in blockComms]
            Lats = np.array(lat_list)

            logger.info(
                f"+ {len(blockComms)} comms in block {curBlock}: {Lats.sum():.2f} us in total"
            )

        logger.info("\n{} Message size Statistcs {}".format("=" * 20, "=" * 20))

        for name, collMsgs in self.collInMsgBytes.items():
            # input tensor
            msgSizes = np.array(collMsgs)
            print("-" * 50)
            print(f"+ {len(msgSizes)} {name}")
            print("-" * 50)
            print(
                f"Size of Input tensors (bytes)\n {'Total (MB)':>10} {'Max.':>15} {'Min.':>10} {'Average':>13} {'p50':>13} {'p95':>13}"
            )
            print(
                "{:>10.2f} {:15.2f} {:10.2f} {:15.2f} {:15.2f} {:15.2f}".format(
                    msgSizes.sum() / 1024.0 / 1024.0,
                    msgSizes.max(),
                    msgSizes.min(),
                    np.average(msgSizes),
                    np.percentile(msgSizes, 50),
                    np.percentile(msgSizes, 95),
                )
            )
            logger.debug(
                f"  - Used sizes (bytes): {sorted(self.collInUniMsgBytes[name])}"
            )

            # output tensor
            msgSizes = np.array(self.collOutMsgBytes[name])
            print(
                f"Size of Output tensors (bytes)\n {'Total (MB)':>10} {'Max.':>15} {'Min.':>10} {'Average':>13} {'p50':>13} {'p95':>13}"
            )
            print(
                "{:>10.2f} {:15.2f} {:10.2f} {:15.2f} {:15.2f} {:15.2f}".format(
                    msgSizes.sum() / 1024.0 / 1024.0,
                    msgSizes.max(),
                    msgSizes.min(),
                    np.average(msgSizes),
                    np.percentile(msgSizes, 50),
                    np.percentile(msgSizes, 95),
                )
            )
            logger.debug(
                f"  - Used sizes (bytes): {sorted(self.collOutUniMsgBytes[name])}"
            )

        if not self.is_dry_run:
            print("\n{} Performance of replayed comms {}".format("=" * 20, "=" * 20))
            print(
                "{}\nE2E latency (us): {} for {} iters, {:10.2f} per iter in avg\n{}".format(
                    "-" * 50,
                    self.totalTraceLatency,
                    self.num_replays,
                    self.totalTraceLatency / self.num_replays,
                    "-" * 50,
                )
            )
            for coll, lats in self.collLat.items():
                if len(lats) == 0:
                    continue

                Lat = np.array(lats)
                print(
                    "{}\n Replayed {} {} ({:.2f}%): \n{}".format(
                        "-" * 50,
                        len(lats),
                        coll,
                        (Lat.sum() / self.totalCommsLatency) * 100,
                        "-" * 50,
                    )
                )

                print(
                    f"Latency (us)\n {'Total':>10} {'Max.':>10} {'Min.':>10} {'Average':>10} {'p50':>10} {'p95':>10}"
                )
                print(
                    " {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(
                        Lat.sum(),
                        Lat.max(),
                        Lat.min(),
                        np.average(Lat),
                        np.percentile(Lat, 50),
                        np.percentile(Lat, 95),
                    )
                )
                msgSizeAndLatency = (
                    tuple(
                        zip(lats, self.collInMsgBytes[coll], self.collOutMsgBytes[coll])
                    )
                    if coll in self.collInMsgBytes
                    else lats
                )
                logger.debug(
                    f"Latency and size (bytes) of First ten: {msgSizeAndLatency[:10]}"
                )

            if self.colls_per_batch > 0:
                print("\n{} Batch Latency Performance {}".format("=" * 20, "=" * 20))
                BatchLat = np.array(self.batchLat)
                print(
                    f"Batch Latency (ms)\n {'Total':>10} {'Max.':>10} {'Min.':>10} {'Average':>10} {'p50':>10} {'p95':>10}"
                )
                print(
                    " {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(
                        BatchLat.sum(),
                        BatchLat.max(),
                        BatchLat.min(),
                        np.average(BatchLat),
                        np.percentile(BatchLat, 50),
                        np.percentile(BatchLat, 95),
                    )
                )

    def initTraceStat(self):
        """
        Do a first pass on the trace to gather statistics on msg count, msg sizes,
        and record how many collectives each block has.

        Args:
            None
        Returns:
            None
        """
        self.num_msg = len(self.comms_trace)
        self.max_msg_cnt = self.num_msg if self.max_msg_cnt == 0 else self.max_msg_cnt
        # first pass to know the statistics and get required info.
        for curComm in self.comms_trace[: self.max_msg_cnt]:
            # record the current comp
            if curComm.compute is not None:
                compName = curComm.compute
                if compName not in self.compLat.keys():
                    self.compLat[compName] = []
                continue

            # record the current comm
            collName = paramToCommName(curComm.comms)
            curBlocks = curComm.markerStack if curComm.markerStack is not None else []
            if collName not in self.collLat.keys():
                self.collLat[collName] = []
                # some ops don't have sizes
                if curComm.inMsgSize is not None:
                    self.collInMsgBytes[collName] = []
                    self.collInUniMsgBytes[collName] = set()
                    self.collOutMsgBytes[collName] = []
                    self.collOutUniMsgBytes[collName] = set()
            if curComm.inMsgSize is not None:
                dtypeSize = self.dtypeSizeMap[curComm.dtype]
                self.collInMsgBytes[collName].append(curComm.inMsgSize * dtypeSize)
                self.collInUniMsgBytes[collName].add(curComm.inMsgSize * dtypeSize)
                self.collOutMsgBytes[collName].append(curComm.outMsgSize * dtypeSize)
                self.collOutUniMsgBytes[collName].add(curComm.outMsgSize * dtypeSize)
            # get info sorted by code block
            for curBlock in curBlocks:
                if curBlock not in self.comms_blocks:
                    self.comms_blocks[curBlock] = []
                # only add entries if on dry run, otherwise, we'll deal with later during replay w/ more info
                if self.is_dry_run:
                    if collName not in ("wait", "barrier"):
                        self.comms_blocks[curBlock].append(
                            {
                                "comms": collName,
                                "in_msg_size": curComm.inMsgSize,
                                "out_msg_size": curComm.outMsgSize,
                            }
                        )
                    else:
                        self.comms_blocks[curBlock].append(
                            {
                                "comms": collName,
                            }
                        )

    def rebalanceSplit(self, curComm: commsArgs) -> None:
        """
        Policy-based rebalancing function for all_to_allv splits.

        Args:
           curComm: Contains info for collective. Will be modified in place.
        Returns:
            None
        """
        if self.rebalance_policy == "equal":
            # Equally split sizes across ranks.

            self.collectiveArgs.ipTensor = torch.tensor(
                [curComm.inMsgSize], dtype=torch.int, device=self.collectiveArgs.device
            )
            self.backendFuncs.collectiveFunc["all_reduce"](self.collectiveArgs)
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            # in and out sizes are the same for equal splits.
            newInSize = self.collectiveArgs.ipTensor[0].item()
            newInSize = (
                self.collectiveArgs.world_size * self.collectiveArgs.world_size
            ) * round(
                newInSize
                / (self.collectiveArgs.world_size * self.collectiveArgs.world_size)
            )
            curComm.inMsgSize = newInSize // self.collectiveArgs.world_size
            curComm.outMsgSize = curComm.inMsgSize
            curComm.inSplit = [
                (curComm.inMsgSize // self.collectiveArgs.world_size)
                for _ in range(self.collectiveArgs.world_size)
            ]
            curComm.outSplit = curComm.inSplit
        else:
            logger.error("Unsupported balancing policy. Ignoring.")

    def resetComms(self):
        """
        Reset collective group to default PG
        """
        self.collectiveArgs.group = self.backendFuncs.get_default_group()
        self.world_size = self.backendFuncs.get_world_size()

    def getCommGroupInfo(
        self, curComm: commsArgs, commsParams: commsParamsHolderBase
    ) -> tuple[int, str]:
        """
        Return the group infomation of the current process group
        including group rank of the local process, and a description string for logging purpose.
        A -1 group rank indicates an invalid process group on the local process.
        """

        # If a PG is associated, the process needs to be included in the PG (group_rank != -1);
        # otherwise invalid communication to the local process.
        if curComm.pgId is not None and not self.shrink:
            group = self.collectiveArgs.groups[curComm.pgId]
            groupDesc = f"PG: id={curComm.pgId}, world_ranks={commsParams.groupRanks[curComm.pgId]}"
        else:
            group = self.backendFuncs.get_default_group()
            groupDesc = "PG: default group"

        return (self.backendFuncs.get_group_rank(group), groupDesc)

    def hashEtCommsOp(self, commsOp: commsArgs) -> int:
        """
        Hash the current collective communication into a unique integer for tensors reuse

        """
        op = None
        if commsOp.comms in supportedP2pOps:
            op = (
                commsOp.comms,
                commsOp.src_rank,
                commsOp.dst_rank,
                commsOp.inMsgSize,
                commsOp.outMsgSize,
            )
        elif commsOp.inSplit or commsOp.outSplit:
            op = (
                commsOp.comms,
                commsOp.pgId,
                commsOp.inMsgSize,
                commsOp.outMsgSize,
                # inSplit and outSplit are list type, need to be converted for hash
                tuple(commsOp.inSplit),
                tuple(commsOp.outSplit),
            )
        else:
            op = (
                commsOp.comms,
                commsOp.pgId,
                commsOp.inMsgSize,
                commsOp.outMsgSize,
            )

        return hash(op)

    def generate_io_tensors(
        self,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        regenerateTensors: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | torch.Tensor]:
        # Use exactly specified inMsgSize/outMsgSize if call from trace replay
        # This avoid regenerating sizes such as in _prep_all_gather_base
        commsParams.size_from_trace = True
        commsParams.dtype = self.dtypeMap[curComm.dtype]
        if not curComm.id or regenerateTensors:
            return super().prepComm(curComm, commsParams)
        else:
            commsOpHash = self.hashEtCommsOp(curComm)
            if commsOpHash in self.et_to_tensors:
                # Allocate input/output tensors if first time replay, otherwise reuse the previous ones.
                super().prepComm(curComm, commsParams, False)
                (ipTensor, opTensor) = self.et_to_tensors[commsOpHash]
            else:
                (ipTensor, opTensor) = super().prepComm(curComm, commsParams, True)
                self.et_to_tensors[commsOpHash] = (ipTensor, opTensor)
        return (ipTensor, opTensor)

    def prepComms(
        self,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        regenerateTensors: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | torch.Tensor]:
        """
        Update process group and prepare the appropriate tensors for the current collective communication.

        Args:
            curComm: The current communication that we are preparing the correct tensor for.
            commsParams: Holds the comms param arguments that will determine tensor attributes.
            regenerateTensors: when an id is being replayed multiple times, setting this to false will use tensors from previous runs
        Returns:
            (ipTensor, opTensor) if the current communication requires tensors, None otherwise.
        """
        # prep process group for hard-coded traces
        if curComm.pgId is not None and not self.shrink:
            self.collectiveArgs.group = self.collectiveArgs.groups[curComm.pgId]
            self.collectiveArgs.world_size = (
                curComm.worldSize
            )  # match world size to the size of the current PG
        else:  # use default process group if no pg_id is provided or shrink is enabled
            self.collectiveArgs.group = self.backendFuncs.get_default_group()
            self.world_size = self.backendFuncs.get_world_size()

        commOp = paramToCommName(curComm.comms)
        if commOp in ("wait", "barrier", "batch_isend_irecv"):
            return (torch.Tensor(), torch.Tensor())

        # for all_to_allv, we can shrink the size if running on smaller scale
        # this is for sanity test or debug purpose only since we don't always get to run very large scale
        if self.shrink:
            cur_world_size = self.collectiveArgs.world_size
            real_world_size = cur_world_size

            if curComm.worldSize is not None:
                real_world_size = curComm.worldSize
            else:
                # if the trace does not record world size, we may use a2av splits to infer it
                if commOp == "all_to_allv":
                    in_split_len = len(curComm.inSplit)
                    out_split_len = len(curComm.outSplit)
                    if in_split_len > 0:
                        real_world_size = in_split_len
                    elif out_split_len > 0:
                        real_world_size = out_split_len

            newNumElemsIn = (curComm.inMsgSize // real_world_size) * cur_world_size
            newNumElemsOut = (curComm.outMsgSize // real_world_size) * cur_world_size

            if commOp == "all_to_allv":
                curComm.outSplit = (
                    curComm.outSplit[:cur_world_size]
                    if (curComm.outSplit is not None)
                    else []
                )
                curComm.inSplit = (
                    curComm.inSplit[:cur_world_size]
                    if (curComm.inSplit is not None)
                    else []
                )
                if len(curComm.inSplit) > 0:
                    newNumElemsIn = sum(curComm.inSplit)
                if len(curComm.outSplit) > 0:
                    newNumElemsOut = sum(curComm.outSplit)
            elif commOp == "all_gather":
                newNumElemsOut = newNumElemsIn * cur_world_size

            curComm.inMsgSize = newNumElemsIn
            curComm.outMsgSize = newNumElemsOut

            logger.debug(
                f"shrink message sizes to curInNumElem {curComm.inMsgSize}, curOutNumElem {curComm.outMsgSize}"
            )

        return self.generate_io_tensors(curComm, commsParams, regenerateTensors)

    def commRebalance(self, curComm: commsArgs) -> None:
        """
        Optionally rebalance data size for a collective.

        Args:
            curComm: The current communication that we are preparing the correct tensor for.
        Returns:
            None
        """
        commName = paramToCommName(curComm.comms)
        # Rebalance all_to_allv if a policy is specified.
        if (
            commName in self.backendFuncs.collectiveFunc.keys()
            and commName == "all_to_allv"
            and len(self.rebalance_policy) > 0
        ):
            # We need to set world_size correctly for rebalancing.
            self.collectiveArgs.world_size = (
                self.backendFuncs.get_world_size()
                if curComm.pgId is None or self.shrink
                else curComm.worldSize
            )
            # Pass in curComm to modify it in the trace
            self.rebalanceSplit(curComm)

    def runCompute(self, func, curBlockStack: str) -> tuple[float, float]:
        """
        Replays a specified compute operation and records metrics for benchmarking.

        Args:
            func: function pointer of the compute kernel
            curBlockStack: str containg the marker_stack(s) that this collective is a part of
        Returns:
            (latency, global_latency), returns the timings of how long the replay or posting (if nonblocking) of the collective took.
        """
        computeTimer = paramTimer()

        # replay the compute and measuring latency
        with paramProfile(
            timer=computeTimer,
            description=f"# PARAM replay {self.replayIter}: " + curBlockStack,
        ):
            # switch to compute stream and post compute kernel
            with paramStreamGuard(
                stream=self.collectiveArgs.compute_stream,
                curDevice=self.collectiveArgs.device,
                backendFuncs=self.backendFuncs,
                is_blocking=self.is_blocking,  # for blocking case, stream synchornization will be performed
            ):
                # Post the kernel *computeCount* times
                for _ in range(self.collectiveArgs.computeCount):
                    func(self.collectiveArgs)

        # For compute, latency and global_latency are the same
        global_latency = latency = computeTimer.getTimeUS()

        return (latency, global_latency)

    def runComms(
        self, collName: str, curComm: commsArgs, curBlockStack: str
    ) -> tuple[float, float]:
        """
        Replays collective communication operation and records metrics for benchmarking.

        Args:
            collName: Name of collective that is going to be replayed.
            curComm: Object containing information on the current collective.
            curBlockStack: str containg the marker_stack(s) that this collective is a part of
        Returns:
            (latency, global_latency), returns the timings of how long the replay or posting (if nonblocking) of the collective took.
        """
        self.collectiveArgs.quant_time.reset()
        self.collectiveArgs.dequant_time.reset()
        collTimer = paramTimer()

        if self.is_blocking:
            with paramProfile(
                description=f"# PARAM replay {self.replayIter} pre-comm barrier # "
                + curBlockStack
            ):
                self.backendFuncs.sync_barrier(self.collectiveArgs)

        # replay the collective
        with paramProfile(
            timer=collTimer,
            description=f"# PARAM replay {self.replayIter}:" + curBlockStack,
        ):
            if collName in self.backendFuncs.collectiveFunc.keys():
                # record wait_obj_key for wait ops
                if curComm.req is not None and curComm.pgId is not None:
                    if isinstance(curComm.req, list):
                        seq_id = curComm.req[0]
                        is_p2p_op = curComm.req[1]
                    else:
                        seq_id = curComm.req
                        is_p2p_op = False
                    self.collectiveArgs.wait_obj_key = (
                        curComm.pgId,
                        seq_id,
                        is_p2p_op,
                    )
                else:
                    self.collectiveArgs.wait_obj_key = None

                # handle point-to-point separately
                if collName in supportedP2pOps:
                    self.collectiveArgs.src_rank = curComm.src_rank
                    self.collectiveArgs.dst_rank = curComm.dst_rank

                    if curComm.batch_p2p:
                        self.collectiveArgs.collective = collName
                        self.backendFuncs.P2POp(self.collectiveArgs, retFlag=True)

                if collName in ["reduce", "broadcast", "gather", "scatter"]:
                    self.collectiveArgs.srcOrDst = curComm.root

                retObj = self.backendFuncs.collectiveFunc[collName](
                    self.collectiveArgs, retFlag=True
                )
            else:
                # skip not supported ops
                logger.warn(
                    f"Unsupported collective name: {collName}. Skipping replaying the collective"
                )
                retObj = None

            # if blocking, post outstanding ops and wait for them to complete. if nonblocking, just post op
            if self.is_blocking:
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)

            # if nonblocking, then store the pair {(pg_id, reqID, isP2P), future} so that we can wait on it later
            # check if req id is recorded in trace for backwards compatibility
            if (
                not self.is_blocking
                and collName != "wait"
                and self.collectiveArgs.wait_obj_key is not None
            ):
                self.collectiveArgs.waitObjIds[self.collectiveArgs.wait_obj_key] = (
                    retObj
                )

        # For non-blocking, latency and global_latency are the same
        global_latency = latency = collTimer.getTimeUS()

        if self.is_blocking:
            with paramProfile(
                description=f"# PARAM replay {self.replayIter} post-comm barrier # "
                + curBlockStack
            ) as bt:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            # We sync the global_latency for blocking
            global_latency = latency + (bt.intervalNS / 1e3)

        return (latency, global_latency)

    def waitForTimestamp(self, curComm: commsArgs, startTime: float) -> None:
        """
        Sleep until enough time has passed to match the collective's timestamp, based on the start time.

        Args:
            curComm: Current collective to sleep/wait for.
            startTime: Start time when replay began.
        Returns:
            None
        """
        # sleep for until it is time for the next collective to run
        # if the collective is less than LOOP_TIMER_S (.02s) away, continue looping for the duration. This is because of time.sleep()'s accuracy.
        if curComm.startTimeNs is not None:  # for backwards compatibility
            while time.monotonic_ns() - startTime <= curComm.startTimeNs:
                timeDiff = curComm.startTimeNs - (time.monotonic_ns() - startTime)
                if timeDiff / 1e9 >= LOOP_TIMER_S:  # make it seconds
                    time.sleep(LOOP_TIMER_S)

    def prepComputeReplay(self, commsParams: commsParamsHolderBase, curComm):
        computeFunc = None

        # Set the computeCount, which is the number of time to run the compute kernel
        self.collectiveArgs.computeCount = curComm.count

        # Set reuseTensors, which is whether we reuse the tensors between kernels
        self.collectiveArgs.reuseTensors = self.reuse_tensors

        # Prep to run GEMM kernel
        if curComm.compute == "gemm":
            if self.gemmTensor is None and self.reuse_tensors:
                self.gemmTensor = self.backendFuncs.alloc_random(
                    [1073741824],
                    self.collectiveArgs.device,
                    self.dtypeMap[curComm.dtype],
                )

            (
                self.collectiveArgs.MMout,
                self.collectiveArgs.MMin1,
                self.collectiveArgs.MMin2,
            ) = self.prepGemmNotSquare(
                curComm.mm0_dim0,
                curComm.mm0_dim1,
                curComm.mm1_dim0,
                curComm.mm1_dim1,
                self.dtypeMap[curComm.dtype],
                self.collectiveArgs.device,
                self.gemmTensor if self.reuse_tensors else None,
            )

            computeFunc = self.backendFuncs.gemm

        # Prep to run TBE/embedding lookup kernel
        elif curComm.compute == "emb_lookup":
            # Check if we are to reuse tensors and emb lookup call has been done before -- shortcut init if so
            if (
                curComm.toEmbLookupTuple() in self.embLookupReuse.keys()
                and self.reuse_tensors
            ):
                if curComm.direction == "forward":
                    (
                        self.collectiveArgs.embRequests,
                        self.collectiveArgs.emb,
                    ) = self.embLookupReuse[curComm.toEmbLookupTuple()]
                else:
                    (
                        self.collectiveArgs.embRequests,
                        self.collectiveArgs.LookupOut,
                        self.collectiveArgs.grad_output,
                    ) = self.embLookupReuse[curComm.toEmbLookupTuple()]

            # Otherwise, do init, then add to dictionary if reuse tensors is enabled
            else:
                curComm.device = commsParams.device
                comms_utils.init_emb_lookup(
                    self.collectiveArgs, curComm, self.backendFuncs
                )
                if self.reuse_tensors:
                    if curComm.direction == "forward":
                        self.embLookupReuse[curComm.toEmbLookupTuple()] = (
                            self.collectiveArgs.embRequests,
                            self.collectiveArgs.emb,
                        )
                    else:
                        self.embLookupReuse[curComm.toEmbLookupTuple()] = (
                            self.collectiveArgs.embRequests,
                            self.collectiveArgs.LookupOut,
                            self.collectiveArgs.grad_output,
                        )

            # Set embedded lookup as function to run
            computeFunc = self.backendFuncs.emb_lookup

        # Spawn the compute stream if it was not already created
        if self.collectiveArgs.compute_stream is None:
            self.collectiveArgs.compute_stream = self.backendFuncs.get_new_stream()

        return computeFunc

    def recordCommReplay(
        self,
        commsParams: commsParamsHolderBase,
        curComm,
        collName,
        latency,
        curBlockStack,
        global_latency,
        curBlocks,
    ):
        recordComm = curComm.toDict()

        recordComm["dtype_size"] = self.dtypeSizeMap.get(curComm.dtype, 0)
        recordComm["marker_stack"] = curBlockStack
        recordComm["quant_us"] = self.collectiveArgs.quant_time.getTimeUS()
        recordComm["dequant_us"] = self.collectiveArgs.dequant_time.getTimeUS()
        recordComm["latency_us"] = latency
        recordComm["global_latency_us"] = global_latency

        # record compute metrics
        if curComm.compute is not None:
            self.compLat[collName].append(latency)
            self.totalCompsLatency += latency
        else:
            # record comm metrics
            self.collLat[collName].append(latency)
            self.totalCommsLatency += latency

            # record comm block metrics
            # categorized by the marker
            for curBlock in curBlocks:
                # elem_size = self.collectiveArgs.ipTensor.element_size()
                self.comms_blocks[curBlock].append(recordComm)
        # Keep a copy of trace with performance (latency) and id
        self.traceWithPerf.append(recordComm)

    def replayTrace(
        self,
        commsParams: commsParamsHolderBase,
        warmup: bool = False,
    ) -> None:
        """
        Replay comms trace.

        Args:
            commsParams: Run-time parameters for replay.
            warmup: Indicating whether this round is for warmup.
        Returns:
            None
        """
        self.coll_in_batch_num = 0
        self.replay_start_time = time.monotonic_ns()
        for cnt, curComm in enumerate(self.comms_trace[: self.max_msg_cnt]):
            self.replaySingle(commsParams, curComm, cnt, warmup)

    def replaySingle(
        self,
        commsParams: commsParamsHolderBase,
        curComm: commsArgs,
        cnt: int,
        warmup: bool = False,
    ):
        if warmup:
            logLable = "[Warm-up]"
        else:
            logLable = f"[Replay {self.replayIter}]"

        curBlocks = curComm.markerStack if curComm.markerStack is not None else []
        curBlockStack = " ".join(curBlocks) if len(curBlocks) > 0 else "Unamed/Unknown"

        # Replay compute
        if curComm.compute is not None:
            # Prepare to run the compute function
            computeFunc = self.prepComputeReplay(commsParams, curComm)

            # Running the kernel
            logger.info(
                f"{logLable}[Rank {self.collectiveArgs.global_rank:3}] [{cnt+1} / {self.max_msg_cnt}] Replaying {curComm.compute}"
            )

            # Run the kernel and report the total time
            (latency, global_latency) = self.runCompute(
                func=computeFunc, curBlockStack=curBlockStack
            )
            recordName = curComm.compute

        # Replay comm
        else:
            if warmup:
                self.commRebalance(curComm)

            # Get the name of the collective from the comm object
            collName = paramToCommName(curComm.comms)
            (groupRank, groupDesc) = self.getCommGroupInfo(curComm, commsParams)
            # Skip comm if the local process doesn't belong to the PG or encounter an unexpected collective
            if (
                collName not in self.allowList
                or groupRank == -1
                or (
                    collName in ("send", "isend")
                    and curComm.src_rank != self.backendFuncs.get_global_rank()
                )
                or (
                    collName in ("recv", "irecv")
                    and curComm.dst_rank != self.backendFuncs.get_global_rank()
                )
            ):
                logger.warn(f"Skip collective {collName} id = {curComm.id}")
                return

            if groupRank >= 0:
                commDesc = f"{str(curComm.comms)}: NumElemsIn={curComm.inMsgSize}, NumElemsOut={curComm.outMsgSize}, Dtype={curComm.dtype}"
                if curComm.comms in ("all_to_all", "all_to_allv"):
                    commDesc += (
                        f", InSplit={curComm.inSplit}, OutSplit={curComm.outSplit}"
                    )
                if curComm.comms in supportedP2pOps:
                    commDesc += (
                        f", Src_Rank={curComm.src_rank}, Dst_Rank={curComm.dst_rank}"
                    )

                logger.info(
                    f"{logLable}[Rank {self.collectiveArgs.global_rank:3}] [{cnt+1} / {self.max_msg_cnt}] Replaying {commDesc} with {groupDesc} id = {curComm.id}"
                )

            # read fields and prepare the tensors
            (
                self.collectiveArgs.ipTensor,
                self.collectiveArgs.opTensor,
            ) = self.prepComms(curComm, commsParams, not self.reuse_tensors)

            if not warmup and self.colls_per_batch > 0 and self.coll_in_batch_num == 0:
                batch_begin = time.monotonic()

            # wait for collective timestamp if enabled.
            if not warmup and self.use_timestamp:
                self.waitForTimestamp(curComm, self.replay_start_time)

            # send comm request to pytorch backend
            (latency, global_latency) = self.runComms(collName, curComm, curBlockStack)

            # perform data validation check on the final opTensor
            if (
                self.is_blocking
                and commsParams.dcheck == 1
                and collName not in ("wait", "barrier")
            ):
                commsParams.collective = collName
                commsParams.srcOrDst = curComm.root if curComm.root is not None else 0

                self.dcheck(
                    commsParams, curComm.outMsgSize, self.collectiveArgs.opTensor
                )

            # calculating batch latency (batch defined by --colls-per-batch)
            if not warmup and collName == "wait" and self.colls_per_batch > 0:
                self.coll_in_batch_num += 1
                if self.coll_in_batch_num == self.colls_per_batch:
                    batch_latency = (
                        time.monotonic() - batch_begin
                    ) * 1e3  # make it millisecond
                    self.coll_in_batch_num = 0
                    self.batchLat.append(batch_latency)

            recordName = collName

        if not warmup:
            # record performance metrics
            self.recordCommReplay(
                commsParams,
                curComm,
                recordName,
                latency,
                curBlockStack,
                global_latency,
                curBlocks,
            )

        if self.backendFuncs.get_global_rank() == 0:
            logger.info(
                f"{logLable}[{cnt+1} / {self.max_msg_cnt}] Replayed {recordName} with id={curComm.id} in block [{curBlockStack}]... {global_latency:.2f} us"
            )

    def benchTime(self, commsParams: commsParamsHolderBase) -> None:
        """
        Run all collectives in current rank and record timing metrics for benchmarkng.

        The comms trace format is expecting to be either:
        all_to_allv
        {
            "startTimeNs": 0
            "markerStack": ["## all2all ##"]
            "comms": "all_to_allv",
            "req": 0
            "inMsgSize": 10357149,
            "outMsgSize": 23093760,
            "inSplit": [],
            "outSplit": [],
            "dtype": Int
            "worldSize": 16
        },
        or w/o in/out_split
        {
            "startTimeNs": 0
            "markerStack": ["## all2all ##"]
            "comms": "all_reduce",
            "req": 0
            "inMsgSize": 1048576,
            "outMsgSize": 1048576,
            "dtype": Int,
            "worldSize": 16
        }
        or wait/barrier
        {
            "startTimeNs": 0
            "markerStack": ["## all2all ##"]
            "req": 0
            "comms": "wait",
            "worldSize": 16
        }
        NOTE:
            - this format is subject to be changed anytime
            - the unit of all size fields is # of elements (not bytes)

        Args:
            commsParams: Holds comms params to pass into prepComms() to aqcuire appropriate tensors
                                                 and perform data validation in blocking runs.
        Returns:
            None
        """
        if commsParams.enable_profiler:
            # num of iterations to profile, at most num_replays iterations
            numProfileIters = (
                self.profiler_num_replays
                if self.profiler_num_replays_start + self.profiler_num_replays
                < self.num_replays
                else self.num_replays - self.profiler_num_replays_start
            )

            if has_fb_internal_libs:
                activities = get_fb_profiler_activities(self.collectiveArgs.device)
                trace_handler = get_fb_profiler_trace_handler(
                    self.backendFuncs.get_global_rank()
                )
            else:
                activities = {ProfilerActivity.CPU, ProfilerActivity.CUDA}

                def trace_handler(p):
                    import pathlib

                    folder_path = os.path.join(self.out_path, "profiler_trace")
                    try:
                        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
                    except PermissionError:
                        logger.error(
                            f"Permission denied to create directory {folder_path}"
                        )
                    p.export_chrome_trace(
                        os.path.join(
                            folder_path,
                            f"rank-{self.backendFuncs.get_global_rank()}.pt.json",
                        )
                    )

            logger.debug("GPU Trace Collection: Enabled")
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=self.profiler_num_replays_start,
                    active=numProfileIters,
                    repeat=1,
                ),
                on_trace_ready=trace_handler,
                activities=activities,
            )
            self.collectiveArgs.enable_profiler = True
        else:
            profiler = nullcontext()

        # sync everything before starting real runs
        with paramProfile(description="# PARAM replay warmup post-replay global sync"):
            self.backendFuncs.sync_barrier(self.collectiveArgs)

        if self.backendFuncs.get_global_rank() == 0:
            logger.info(
                f"{self.max_msg_cnt} messages in the trace...replaying (if present) {list(self.allowList)}"
            )
            for coll, sizes in self.collInMsgBytes.items():
                logger.info(f"\t{coll}: {len(sizes)}")

        # warmup runs
        for i in range(self.warmup_iter):
            # replay comms trace
            self.replayIter = i
            self.replayTrace(commsParams=commsParams, warmup=True)
            self.resetComms()

            # make sure all ops are completed
            with paramProfile(
                description=f"# PARAM replay {self.replayIter} post-replay global sync"
            ):
                self.backendFuncs.sync_barrier(self.collectiveArgs)

        traceStartTime = time.monotonic_ns()

        with profiler as prof:
            for i in range(self.num_replays):
                # replay comms trace
                self.replayIter = i
                self.replayTrace(commsParams=commsParams, warmup=False)
                self.resetComms()

                # make sure all ops are completed
                with paramProfile(
                    description=f"# PARAM replay {self.replayIter} post-replay global sync"
                ):
                    self.backendFuncs.sync_barrier(self.collectiveArgs)

                if prof:
                    prof.step()

        # record how long it took for trace-replay to complete
        traceEndTime = time.monotonic_ns()
        self.totalTraceLatency = (traceEndTime - traceStartTime) / 1e3  # make it us

        # stop profiler if used
        if self.collectiveArgs.enable_profiler:
            self.collectiveArgs.enable_profiler = False

        # cleanup any memory left in use
        self.backendFuncs.clear_memory(self.collectiveArgs)

        self.backendFuncs.barrier_all_ranks()

    def runBench(
        self,
        commsParams: commsParamsHolderBase,
    ) -> None:
        """
        Run the comms-replay benchmark:
        1) Each rank reads its trace
        2) First pass of the trace to ensure the format is valid and get basic stats
        3) Execute communication replay [Skip if on dry-run mode]
        4) report stats and performance (if not dry-run)

        Args:
            commsParams: Holds comms params to pass into inner functions.
        Returns:
            None
        """

        global_rank = self.backendFuncs.get_global_rank()
        logger.info(
            f"[Rank-{global_rank}] reading {self.trace_type} trace from {self.trace_file}"
        )
        self.report = (
            True
            if global_rank == 0
            or (
                commsParams.enable_local_report
                and self.backendFuncs.get_local_rank() == 0
            )
            else False
        )
        self.readTrace(remotePath=self.trace_file, rank=global_rank)

        self.initTraceStat()
        # only setup and perform collectives if not dry run mode
        if not self.is_dry_run:
            self.setBench(commsParams)
            # start benchmark
            self.benchTime(commsParams)
        elif self.report:
            logger.info(
                "+ Dry run mode...No replaying, Only Rank 0 read and analyze the trace..."
            )

        # global/local rank 0 reports statistics
        if self.report:
            self.reportBenchTime()
            # writeCommDetails(self.comms_blocks, rank=global_rank)

        if not self.is_dry_run:
            if self.backendFuncs.get_global_rank() in self.outputRanks:
                writeCommDetails(
                    self.traceWithPerf,
                    folder=os.path.join(self.out_path, "replayed_trace"),
                    rank=global_rank,
                )
            # TODO: collect perf. from all ranks to rank 0 and detect any imbalanced perf?

            if (
                commsParams.enable_profiler
                and self.backendFuncs.get_global_rank() == 0
                and not has_fb_internal_libs
            ):
                profiler_trace_analysis.analyze_profiler_trace(
                    os.path.join(self.out_path, "profiler_trace"), self.out_path
                )

            self.backendFuncs.barrier_all_ranks()

    def replayInit(
        self,
        commsParams: commsParamsHolderBase,
    ) -> None:
        """
        Init the comms-replay benchmark:
        1) Each rank reads its trace
        2) First pass of the trace to ensure the format is valid and get basic stats

        Args:
            commsParams: Holds comms params to pass into inner functions.
        Returns:
            None
        """
        global_rank = self.backendFuncs.get_global_rank()
        logger.info(f"[Rank-{global_rank}] reading trace from {self.trace_file}")
        self.readTrace(remotePath=self.trace_file, rank=global_rank)

        self.initTraceStat()
        # only setup and perform collectives if not dry run mode
        if not self.is_dry_run:
            self.setBench(commsParams)

    def initBackend(
        self,
        bootstrap_info: bootstrap_info_holder,
        commsParams: commsParamsHolderBase,
    ) -> None:
        """
        Initializes backend.

        Args:
            bootstrap_info: Holds current environment information.
            commsParams: Holds comms params to pass into backend for initialization.
        Returns:
            None
        """
        # init backend and corresponding function pointers
        if (
            commsParams.nw_stack == "pytorch-dist"
            and commsParams.backend in supportedC10dBackends
        ):
            from et_replay.comm.backend.pytorch_dist_backend import PyTorchDistBackend

            self.backendFuncs = PyTorchDistBackend(bootstrap_info, commsParams)
        elif commsParams.nw_stack == "pytorch-xla-tpu":
            from et_replay.comm.backend.pytorch_tpu_backend import PyTorchTPUBackend

            self.backendFuncs = PyTorchTPUBackend(bootstrap_info, commsParams)
        else:
            # check for customized backend
            try:
                logging.warning(
                    f"Attempt loading customized backend {commsParams.backend} if registered. Note that this is not officially supported. Use it with caution and at your own risk."
                )
                from et_replay.comm.backend.base_backend import customized_backend

                self.backendFuncs = customized_backend[commsParams.backend](
                    bootstrap_info, commsParams
                )
            except KeyError as e:
                logger.error(
                    f"Unsupported NW stack for backend {commsParams.backend}: {e}"
                )
                comms_utils.gracefulExit()

        self.backendFuncs.initialize_backend(
            bootstrap_info.master_ip,
            bootstrap_info.master_port,
            backend=commsParams.backend,
        )
        self.backendFuncs.sayHello()

    def setBench(
        self,
        commsParams: commsParamsHolderBase,
    ) -> None:
        """
        Initializes replay basic collective info.

        Args:
            commsParams: Holds comms params to pass into backend for initialization.
        Returns:
            None
        """
        # init process groups
        for curComm in self.comms_trace[: self.max_msg_cnt]:
            # record process group info
            if curComm.comms == "init":
                commsParams.groupRanks[curComm.pgId] = curComm.groupRanks
                commsParams.pgsDesc[curComm.pgId] = curComm.pgDesc
        self.backendFuncs.initialize_groups(commsParams.backend)

        # set basic collective info
        (
            local_rank,
            global_rank,
            world_size,
            group,
            curDevice,
            curHwDevice,
        ) = comms_utils.get_rank_details(
            self.backendFuncs
        )  # Getting ranks from backendFuncs object, since we cannot use MPI (e.g.: TPU) to launch all the processes

        self.collectiveArgs.group = group  # default group
        self.collectiveArgs.groups = self.backendFuncs.get_groups()
        self.collectiveArgs.device = curDevice
        self.collectiveArgs.world_size = world_size
        self.collectiveArgs.global_rank = global_rank
        self.collectiveArgs.backendFuncs = self.backendFuncs
        # FIXME:  0 is a common case, need this info from trace for more accurate replay
        self.collectiveArgs.srcOrDst = 0
        # FIXME: assuming it's always sum for reduce/allreduce operations
        self.collectiveArgs.op = self.backendFuncs.get_reduce_op("sum")
        self.collectiveArgs.asyncOp = not self.is_blocking
        self.collectiveArgs.ipTensor = None
        self.collectiveArgs.opTensor = None
        self.collectiveArgs.quant_threshold = commsParams.quant_threshold
        self.collectiveArgs.enable_profiler = commsParams.enable_profiler

        # set of collectives to be replayed
        if self.allowList in ("all", "default", "*"):
            self.allowList = self.backendFuncs.collectiveFunc.keys()
        else:
            self.allowList = [paramToCommName(op) for op in self.allowList.split(",")]

    def initBench(
        self, commsParams: commsParamsHolderBase, args: argparse.Namespace
    ) -> None:
        """
        Initializes replay parameters.

        Args:
            commsParams: Holds bitwidth information to initialize quantized communication context.
            args: Namespace containing command line args that we will set our parameters with.
        Returns:
            None
        """
        self.is_dry_run = args.dry_run
        self.shrink = args.auto_shrink
        self.max_msg_cnt = args.max_msg_cnt
        self.is_blocking = args.blocking
        self.warmup_iter = args.warmup_iter
        self.reuse_tensors = args.reuse_tensors
        self.allowList = args.allow_ops
        if args.output_ranks == "all":
            self.outputRanks = [*range(self.backendFuncs.get_world_size())]
        else:
            self.outputRanks = comms_utils.parseRankList(args.output_ranks)
        self.out_path = args.output_path
        self.colls_per_batch = args.colls_per_batch
        self.use_timestamp = args.use_timestamp
        self.rebalance_policy = args.rebalance_policy.lower()
        self.num_replays = args.num_replays
        self.profiler_num_replays_start = args.profiler_num_replays_start
        self.profiler_num_replays = args.profiler_num_replays
        self.disable_parallel_read = args.disable_parallel_read
        self.use_one_trace = args.use_one_trace

        if commsParams.bitwidth < 32:
            comms_utils.initQuantCommCtx(self.collectiveArgs, commsParams)

    def setTraceFile(self, args, comms_env_params):
        # TODO: file name may get changed later
        self.trace_file = args.trace_path
        self.trace_type = args.trace_type
        # assume the prefix is always "xxx://" when reading remote trace, e.g., http://xxx
        if "://" in args.trace_path:
            self.use_remote_trace = True

    def readRawTrace(self, remotePath: str, rank: int) -> None:
        """
        Read trace file from remote server or local disk, supporting both
        directory (with rank-specific files) and single file modes.

        Args:
            remotePath: Path to read from remotely if use_remote_trace is enabled.
            rank: The rank of the current process, used to select the correct
                 trace file in directory mode.
        Returns:
            None
        """
        if self.use_remote_trace:
            # format "<protocol prefix>://<url or path>"
            protocol = remotePath.split("://", 2)[0]
            raw_comms_trace = []
            if protocol in ("http", "https", "ftp"):
                raw_comms_trace = comms_utils.commonUrlRead(remotePath=remotePath)
            else:
                try:
                    from param_bench.et_replay.comm.vendor_internal.fb_internals import (
                        readRemoteTrace as readFbRemoteTrace,
                    )

                except ImportError:
                    logger.error(
                        f"Not supported protocol for the URL provided {remotePath}"
                    )
                else:
                    raw_comms_trace = readFbRemoteTrace(
                        remotePath=remotePath,
                        rank=rank,
                        full_trace_path=self.use_one_trace,
                        trace_type=self.trace_type,
                    )
            self.comms_trace = json.load(raw_comms_trace)
        else:
            # Check if self.trace_file is a directory or a single file
            if os.path.isdir(self.trace_file):
                # Directory mode: construct the path to the rank-specific file
                trace_file_path = f"{self.trace_file}/rank-{rank}.json"
            else:
                # Single file mode: use self.trace_file as is
                trace_file_path = self.trace_file

            # Read the json file from local disk
            # with open(trace_file_path) as f:
            with (
                gzip.open(trace_file_path, "rb")
                if trace_file_path.endswith("gz")
                else open(trace_file_path)
            ) as execution_data:
                self.comms_trace = json.load(execution_data)

    def readTrace(self, remotePath: str, rank: int) -> None:
        """
        Read trace file and convert/parse traces files.

        Args:
            remotePath: Path to read from remotely if use_remote_trace is enabled.
            globalRead: Whether to read trace on all ranks
        Returns:
            None
        """

        if self.disable_parallel_read and not self.is_dry_run:
            # checkArgs already checks whether disable_parallel_read is used together with use_one_trace. Sanity check here.
            assert self.use_one_trace
            # Rank 0 loads trace and broadcast
            if rank == 0:
                logger.info(f"[Rank-{rank}] reading trace from {remotePath}")
                self.readRawTrace(remotePath=remotePath, rank=rank)

                comms_trace_str = json.dumps(self.comms_trace)
                logger.info(f"[Rank-{rank}] broadcasting comms_trace")
                self.backendFuncs.store_set(remotePath, comms_trace_str)

            logger.info(f"[Rank-{rank}] receiving comms_trace with key {remotePath}")
            comms_trace_str = self.backendFuncs.store_get(remotePath)
            self.comms_trace = json.loads(comms_trace_str.decode())
            logger.info(f"[Rank-{rank}] received trace")
        else:
            # By default everyone loads trace in parallel
            self.readRawTrace(remotePath=remotePath, rank=rank)

        # Convert trace to comms trace.
        self.comms_trace = commsTraceParser.parseTrace(
            self.comms_trace,
            self.trace_type,
            (
                self.trace_file
                if not os.path.isdir(self.trace_file)
                else f"{self.trace_file}/rank-{rank}.json"
            ),
            rank,
            self.backendFuncs.get_world_size(),
        )


def main() -> None:
    """
    1) Read environment variables.
    2) Parse commmand line arguments.
    3) Read and analyze trace file.
    4) Run replay.
    """
    comms_env_params = comms_utils.read_comms_env_vars()

    traceBench = commsTraceReplayBench()
    parser = argparse.ArgumentParser(
        description="PARAM-Comms Trace Replay Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    args = traceBench.readArgs(parser)
    traceBench.setTraceFile(args, comms_env_params)
    traceBench.checkArgs(args)

    bootstrap_info = bootstrap_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, comms_env_params
    )
    commsParams = commsParamsHolderBase(args)
    # always initialize backend
    traceBench.initBackend(bootstrap_info, commsParams)

    traceBench.initBench(commsParams, args)
    traceBench.runBench(commsParams)


if __name__ == "__main__":
    main()
