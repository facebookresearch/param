#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging
import time
from os import path
from typing import Dict, List, Set

import comms_utils
import numpy as np
import torch
from comms_utils import (
    comms_world_info_holder,
    commsParamsHolderBase,
    paramCommsBench,
    paramProfile,
    paramTimer,
    paramToCommName,
)

logger = logging.getLogger(__name__)

# sleep for 20ms to wait for next collective
LOOP_TIMER_S = .02

def writeCommDetails(commsTracePerf: List, rank: int, folder: str ="./") -> None:
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
    comms_file = folder + f"/replayedCommsPerf.rank{rank}.json"
    logger.info(f"[Rank {rank:3}] Writing comms details to {comms_file}")

    saveToLocal = True
    if "://" in comms_file: # assume that "://" in directory path means remote store
        saveToLocal = False
        try:
            from internals import writeRemoteTrace as writeFbRemoteTrace
        except ImportError:
            saveToLocal = True
            pass
        else:
            writeFbRemoteTrace(commsTracePerf, remotePath=comms_file)

    if saveToLocal:
        try:
            import subprocess

            subprocess.check_output(
                ["mkdir", "-p", str(folder)], universal_newlines=True
            )
        except Exception as err:
            logger.error("\t Error: %s while creating directory: %s " % (err, folder))
            pass
        with open(comms_file, "w") as write_file:
            json.dump(commsTracePerf, write_file, indent=2)


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
        self.use_remote_trace = False
        self.is_dry_run = False
        self.shrink = False
        self.max_msg_cnt = 0  # 0 means no limit
        self.num_msg = 0
        self.is_blocking = True
        self.do_warm_up = True
        self.allowList = ""
        self.out_path = "/tmp/paramReplayedTrace"
        self.colls_per_batch = -1
        self.use_timestamp = False

        self.collInMsgSizes: Dict[str, List] = {}
        self.collInUniMsgSizes: Dict[str, Set] = {}
        self.collOutMsgSizes: Dict[str, List] = {}
        self.collOutUniMsgSizes: Dict[str, Set] = {}

        self.batchLat = []
        self.collLat: Dict[str, List] = {}

        self.comms_blocks: Dict[str, List] = {}
        self.traceWithPerf = []
        self.blockStack = []

        # for blocking collectives this is the sum of all the collective latencies
        # for nonblocking collectives this is the sum of how long each collective took to be sent to the device
        self.totalCommsLatency = 0.0
        # how long it took to finish all collectives in the trace
        self.totalTraceLatency = 0.0

    def readArgs(self, parser: argparse.ArgumentParser) -> None:
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
            "--use-one-trace",
            action="store_true",
            default=False,
            help="Toggle to use only one trace for all ranks",
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
            "--no-warm-up",
            action="store_true",
            default=False,
            help="Toggle to disable performing extra replaying for warm-up",
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
            help='Output path to write the replayed trace for post performance analysis. Set as empty string, i.e., "", to skip output',
        )
        parser.add_argument(
            "--colls-per-batch",
            type=int,
            default=self.colls_per_batch,
            help="Toggle to set number of consecutive collectives in a batch. This also enables per batch latency stats.",
        )
        parser.add_argument(
           "--use_timestamp",
            action="store_true",
            default=self.use_timestamp,
            help="Toggle to use time-based replay.",
        )
        return parser.parse_args()

    def checkArgs(self, args: argparse.Namespace) -> None:
        """
        Validates command line args, will raise an error and gracefully exit if an invalid command line arg is found.

        Args:
            args: Namespace containing collection of args to validate.
        Returns:
            None
        """
        super().checkArgs(args)

        if (not self.use_remote_trace) and (
            path.exists(self.trace_file) is False
            or path.isfile(self.trace_file) is False
        ):
            raise ValueError(
                f"Trace file {self.trace_file} does not exist or is not a file! Please specify the correct path by using --trace-path."
            )
            comms_utils.gracefulExit()

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

        for (name, collMsgs) in self.collInMsgSizes.items():
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
                    msgSizes.sum() / 1024 / 1024,
                    msgSizes.max(),
                    msgSizes.min(),
                    np.average(msgSizes),
                    np.percentile(msgSizes, 50),
                    np.percentile(msgSizes, 95),
                )
            )
            logger.debug(f"  - Used sizes: {sorted(self.collInUniMsgSizes[name])}")

            # output tensor
            msgSizes = np.array(self.collOutMsgSizes[name])
            print(
                f"Size of Output tensors (bytes)\n {'Total (MB)':>10} {'Max.':>15} {'Min.':>10} {'Average':>13} {'p50':>13} {'p95':>13}"
            )
            print(
                "{:>10.2f} {:15.2f} {:10.2f} {:15.2f} {:15.2f} {:15.2f}".format(
                    msgSizes.sum() / 1024 / 1024,
                    msgSizes.max(),
                    msgSizes.min(),
                    np.average(msgSizes),
                    np.percentile(msgSizes, 50),
                    np.percentile(msgSizes, 95),
                )
            )
            logger.debug(f"  - Used sizes: {sorted(self.collOutUniMsgSizes[name])}")

        if not self.is_dry_run:
            print("\n{} Performance of replayed comms {}".format("=" * 20, "=" * 20))
            print(
                "{}\n Total latency (us) of comms in trace {}: \n{}".format(
                    "-" * 50,
                    self.totalTraceLatency,
                    "-" * 50,
                )
            )
            for (coll, lats) in self.collLat.items():
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
                        zip(lats, self.collInMsgSizes[coll], self.collOutMsgSizes[coll])
                    )
                    if coll in self.collInMsgSizes
                    else lats
                )
                logger.debug(f"Latency and size of First ten: {msgSizeAndLatency[:10]}")

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
        maxInMsgsize = 0
        maxOutMsgsize = 0
        self.num_msg = len(self.comms_trace)
        self.max_msg_cnt = self.num_msg if self.max_msg_cnt == 0 else self.max_msg_cnt
        # first pass to know the statistics and get required info.
        for curComm in self.comms_trace[: self.max_msg_cnt]:
            # record the current comm
            collName = paramToCommName(curComm["comms"])
            curBlocks = curComm["marker_stack"] if "marker_stack" in curComm else []
            if collName not in self.collLat.keys():
                self.collLat[collName] = []
                # some ops don't have sizes
                if "in_msg_size" in curComm:
                    self.collInMsgSizes[collName] = []
                    self.collInUniMsgSizes[collName] = set()
                    self.collOutMsgSizes[collName] = []
                    self.collOutUniMsgSizes[collName] = set()
            if "in_msg_size" in curComm:
                self.collInMsgSizes[collName].append(curComm["in_msg_size"])
                self.collInUniMsgSizes[collName].add(curComm["in_msg_size"])
                self.collOutMsgSizes[collName].append(curComm["out_msg_size"])
                self.collOutUniMsgSizes[collName].add(curComm["out_msg_size"])
                maxInMsgsize = max(curComm["in_msg_size"], maxInMsgsize)
                maxOutMsgsize = max(curComm["out_msg_size"], maxOutMsgsize)
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
                                "in_msg_size": curComm["in_msg_size"],
                                "out_msg_size": curComm["out_msg_size"],
                            }
                        )
                    else:
                        self.comms_blocks[curBlock].append(
                            {
                                "comms": collName,
                            }
                        )

    def prepComms(self, curComm: Dict, commsParams: commsParamsHolderBase) -> (torch.Tensor, torch.Tensor):
        """
        Prepares the appropriate tensors for the current collective communication.

        Args:
            curComm: The current communication that we are preparing the correct tensor for.
            commsParams: Holds the comms param arguments that will determine tensor attributes.
        Returns:
            (ipTensor, opTensor) if the current communication requires tensors, None otherwise.
        """
        commOp = paramToCommName(curComm["comms"])
        if commOp in ("wait", "barrier"):
            return ([], [])

        # prep process group for hard-coded traces
        if "pg_id" in curComm and not self.shrink:
            self.collectiveArgs.group = self.collectiveArgs.groups[curComm["pg_id"]]
            self.collectiveArgs.world_size = curComm["world_size"] # match world size to the size of the current PG
        else: # use default process group if no pg_id is provided or shrink is enabled
            self.collectiveArgs.group = self.backendFuncs.get_default_group()

        # for all_to_allv, we can shrink the size if running on smaller scale
        # this is for sanity test or debug purpose only since we don't always get to run very large scale
        if self.shrink:

            cur_world_size = self.collectiveArgs.world_size
            real_world_size = cur_world_size

            if "world_size" in curComm.keys():
                real_world_size = curComm["world_size"]
            else:
                # if the trace does not record world size, we may use a2av splits to infer it
                if commOp == "all_to_allv":
                    in_split_len = len(curComm["in_split"])
                    out_split_len = len(curComm["out_split"])
                    if in_split_len > 0:
                        real_world_size = in_split_len
                    elif out_split_len > 0:
                        real_world_size = out_split_len

            newNumElemsIn = (curComm["in_msg_size"] // real_world_size) * cur_world_size
            newNumElemsOut = (
                curComm["out_msg_size"] // real_world_size
            ) * cur_world_size

            if commOp == "all_to_allv":
                curComm["out_split"] = (
                    curComm["out_split"][:cur_world_size]
                    if ("out_split" in curComm.keys())
                    else []
                )
                curComm["in_split"] = (
                    curComm["in_split"][:cur_world_size]
                    if ("in_split" in curComm.keys())
                    else []
                )
                if len(curComm["in_split"]) > 0:
                    newNumElemsIn = sum(curComm["in_split"])
                if len(curComm["out_split"]) > 0:
                    newNumElemsOut = sum(curComm["out_split"])
            elif commOp == "all_gather":
                newNumElemsOut = newNumElemsIn * cur_world_size

            curComm["in_msg_size"] = newNumElemsIn
            curComm["out_msg_size"] = newNumElemsOut

            logger.debug(
                f"shrink message sizes to curInNumElem {curComm['in_msg_size']}, curOutNumElem {curComm['out_msg_size']}"
            )

        commsParams.dtype = self.dtypeMap[curComm["dtype"]]
        # allocate and return tensors
        return super().prepComm(curComm, commsParams)

    def warmUpBench(self, commsParams: commsParamsHolderBase) -> None:
        """
        Replays collectives without recording statistics to warm up devices.

        Args:
            commsParams: Holds comms params to be passed into prepComms() for appropriate tensor allocation.
        Returns:
            None
        """
        for cnt, curComm in enumerate(self.comms_trace[: self.max_msg_cnt]):
            commEntry = curComm.copy()
            commName = paramToCommName(commEntry["comms"])
            if commName not in self.allowList:
                continue
            if self.backendFuncs.get_global_rank() == 0:
                logger.debug(
                    f"[Rank {self.collectiveArgs.global_rank:3}] Replaying \n{str(commEntry)}\n"
                )
                print(
                    f"[Warm-up][{cnt} / {self.max_msg_cnt}] Replaying {commName:>10}...",
                    end="\r",
                )

            # read fields and prepare the tensors
            (
                self.collectiveArgs.ipTensor,
                self.collectiveArgs.opTensor,
            ) = self.prepComms(commEntry, commsParams)

            if commName in self.backendFuncs.collectiveFunc.keys():
                self.backendFuncs.collectiveFunc[commName](self.collectiveArgs)
            # skip not supported ops

            self.backendFuncs.complete_accel_ops(self.collectiveArgs)

    def runComms(self, collName: str, curComm: Dict, curBlockStack: str) -> (float, float):
        """
        Replays collective communication operation and records metrics for benchmarking.

        Args:
            collName: Name of collective that is going to be replayed.
            curComm: dict containing information on the current collective.
            curBlockStack: str containg the marker_stack(s) that this collective is a part of
        Returns:
            (latency, global_latency), returns the timings of how long the replay or posting (if nonblocking) of the collective took.
        """
        self.collectiveArgs.quant_time.reset()
        self.collectiveArgs.dequant_time.reset()
        collTimer = paramTimer()

        if self.is_blocking:
            self.backendFuncs.sync_barrier(self.collectiveArgs)

        # replay the collective
        with paramProfile(
            timer=collTimer, description="# PARAM replay: " + curBlockStack
        ):
            if collName in self.backendFuncs.collectiveFunc.keys():
                # record collectiveID for wait ops
                if "req" in curComm:
                    self.collectiveArgs.collectiveId = curComm["req"]

                retObj = self.backendFuncs.collectiveFunc[collName](
                    self.collectiveArgs, retFlag=True
                )
            # skip not supported ops

            # if blocking, post outstanding ops and wait for them to complete. if nonblocking, just post op
            self.backendFuncs.complete_accel_ops(self.collectiveArgs, devSync=self.is_blocking)

            # if nonblocking, then store the pair {reqID, future} so that we can wait on it later
            # check if req id is recorded in trace for backwards compatibility
            if "req" in curComm and not self.is_blocking and collName != "wait":
                self.collectiveArgs.waitObjIds[curComm["req"]] = retObj


        # For non-blocking, latency and global_latency are the same
        global_latency = latency = collTimer.getTimeUS()

        if self.is_blocking:
            with paramProfile(
                description="# PARAM replay barrier # " + curBlockStack
            ) as bt:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

            # We sync the global_latency for blocking
            global_latency = latency + (bt.intervalNS / 1e3)

        return (latency, global_latency)


    def benchTime(self, commsParams: commsParamsHolderBase) -> None:
        """
        Run all collectives in current rank and record timing metrics for benchmarkng.

        The json format is expecting to be either
        {
            "startTime_ns": 0
            "timestamp": 12345
            "marker_stack": ["## all2all ##"]
            "comms": "all_to_allv",
            "seqnum": 0
            "req": 0
            "in_msg_size": 10357149,
            "out_msg_size": 23093760,
            "in_split": [],
            "out_split": [],
            "dtype": Int
            "world_size": 16
        },
        or w/o in/out_split
        {
            "startTime_ns": 0
            "timestamp": 12345
            "marker_stack": ["## all2all ##"]
            "comms": "all_reduce",
            "seqnum": 0
            "req": 0
            "in_msg_size": 1048576,
            "out_msg_size": 1048576,
            "dtype": Int,
            "world_size": 16
        }
        or wait/barrier
        {
            "startTime_ns": 0
            "timestamp": 12345
            "marker_stack": ["## all2all ##"]
            "seqnum": 0
            "req": 0
            "comms": "wait",
            "world_size": 16
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
        # warm-up
        if self.do_warm_up:
            self.warmUpBench(commsParams)

        # sync everything before starting real runs
        self.backendFuncs.sync_barrier(self.collectiveArgs)

        if self.backendFuncs.get_global_rank() == 0:
            print(
                f"\n+ {self.max_msg_cnt} messages in the trace...replaying (if present) {list(self.allowList)}"
            )
            for coll, sizes in self.collInMsgSizes.items():
                logger.info(f"\t{coll}: {len(sizes)}")

        coll_in_batch_num = 0
        startTime = time.monotonic_ns()
        for cnt, curComm in enumerate(self.comms_trace[: self.max_msg_cnt]):
            collName = paramToCommName(curComm["comms"])
            if collName not in self.allowList:
                continue

            curBlocks = curComm["marker_stack"] if "marker_stack" in curComm else []
            curBlockStack = (
                " ".join(curBlocks) if len(curBlocks) > 0 else "Unamed/Unknown"
            )

            if self.backendFuncs.get_global_rank() == 0:
                logger.debug(
                    f"[Rank {self.collectiveArgs.global_rank:3}] Replaying \n{str(curComm)}\n"
                )
                print(f"[{cnt} / {self.max_msg_cnt}]", end="\r")

            # read fields and prepare the tensors
            (
                self.collectiveArgs.ipTensor,
                self.collectiveArgs.opTensor,
            ) = self.prepComms(curComm, commsParams)

            if self.colls_per_batch > 0 and coll_in_batch_num == 0:
                batch_begin = time.monotonic()

            # sleep for until it is time for the next collective to run
            # if the collective is less than LOOP_TIMER_S (.02s) away, continue looping for the duration. This is because of time.sleep()'s accuracy.
            if self.use_timestamp:
                if "startTime_ns" in curComm: # for backwards compatibility
                    while(time.monotonic_ns() - startTime <= curComm["startTime_ns"]):
                        timeDiff = curComm["startTime_ns"] - (time.monotonic_ns() - startTime)
                        if(timeDiff/1e9 >= LOOP_TIMER_S): # make it seconds
                            time.sleep(LOOP_TIMER_S)

            # send comm request to pytorch backend
            (latency, global_latency) = self.runComms(collName, curComm, curBlockStack)

            # perform data validation check on the final opTensor
            if self.is_blocking and commsParams.dcheck == 1 and collName not in ("wait","barrier"):
                commsParams.collective = collName
                commsParams.srcOrDst = curComm["root"] if "root" in curComm else 0
                self.dcheck(commsParams, curComm["out_msg_size"], self.collectiveArgs.opTensor)

            # calculating batch latency (batch defined by --colls-per-batch)
            if collName == "wait" and self.colls_per_batch > 0:
                coll_in_batch_num += 1
                if coll_in_batch_num == self.colls_per_batch:
                    batch_latency = (
                        time.monotonic() - batch_begin
                    ) * 1e3  # make it millisecond
                    coll_in_batch_num = 0
                    self.batchLat.append(batch_latency)

            # record comm metrics
            self.collLat[collName].append(latency)
            self.totalCommsLatency += latency
            curComm["seqnum"] = cnt
            curComm["quant_us"] = self.collectiveArgs.quant_time.getTimeUS()
            curComm["dequant_us"] = self.collectiveArgs.dequant_time.getTimeUS()
            curComm["latency_us"] = latency
            curComm["global_latency_us"] = global_latency

            # record comm block metrics
            # categorized by the marker
            for curBlock in curBlocks:
                # elem_size = self.collectiveArgs.ipTensor.element_size()
                self.comms_blocks[curBlock].append(curComm)

            # Keep a copy of trace with performance (latency) and seqnum
            self.traceWithPerf.append(curComm)

            if self.backendFuncs.get_global_rank() == 0:
                logger.info(
                    f"[{cnt} / {self.max_msg_cnt}] Replayed {collName} in block [{curBlockStack}]... {global_latency:.2f} us"
                )

        # make sure all ops are completed, in the case of nonblocking, this will enqueue all remaining operations that did not have a wait op
        self.backendFuncs.sync_barrier(self.collectiveArgs)

        # record how long it took for trace-replay to complete
        endTime = time.monotonic_ns()
        self.totalTraceLatency = (endTime-startTime) / 1e3 # make it us

        # cleanup any memory left in use
        self.backendFuncs.clear_memory(self.collectiveArgs)

    def runBench(self, comms_world_info: comms_world_info_holder, commsParams: commsParamsHolderBase) -> None:
        """
        Run the comms-replay benchmark:
        1) Each rank reads its trace
        2) First pass of the trace to ensure the format is valid and get basic stats
        3) Execute communication replay [Skip if on dry-run mode]
        4) report stats and performance (if not dry-run)

        Args:
            comms_world_info: Holds information on the current environment.
            commsParams: Holds comms params to pass into inner functions.
        Returns:
            None
        """
        logger.info(
            f"[Rank-{comms_world_info.global_rank}] reading trace from {self.trace_file}"
        )
        self.comm_size = comms_world_info.world_size
        self.global_rank = comms_world_info.global_rank

        self.readTrace(remotePath=self.trace_file)

        self.initTraceStat()
        # only setup and perform collectives if not dry run mode
        if not self.is_dry_run:
            self.setBench(comms_world_info, commsParams)
            # start benchmark
            self.benchTime(commsParams)
        elif comms_world_info.global_rank == 0:
            print(
                "+ Dry run mode...No replaying, Only Rank 0 read and analyze the trace..."
            )

        # rank 0 reports statistics
        if comms_world_info.global_rank == 0:
            self.reportBenchTime()
            # writeCommDetails(self.comms_blocks, rank=comms_world_info.global_rank)

        if not self.is_dry_run:
            writeCommDetails(
                self.traceWithPerf,
                folder=self.out_path,
                rank=comms_world_info.global_rank,
            )
            # TODO: collect perf. from all ranks to rank 0 and detect any imbalanced perf?
            self.backendFuncs.barrier(self.collectiveArgs)
            self.backendFuncs.complete_accel_ops(self.collectiveArgs)

    def setBench(self, comms_world_info: comms_world_info_holder, commsParams: commsParamsHolderBase) -> None:
        """
        Initializes the replay backend.

        Args:
            comms_world_info: Holds current environment information.
            commsParams: Holds comms params to pass into backend for initialization.
        Returns:
            None
        """
        # init process groups
        for curComm in self.comms_trace[: self.max_msg_cnt]:
            # record process group info
            if curComm["comms"] == "init":
                commsParams.groupRanks[curComm["pg_id"]] = curComm["global_ranks"]

        # init backend and corresponding function pointers
        if commsParams.nw_stack == "pytorch-dist":
            from pytorch_dist_backend import PyTorchDistBackend

            self.backendFuncs = PyTorchDistBackend(comms_world_info, commsParams)
        elif commsParams.nw_stack == "pytorch-xla-tpu":
            from pytorch_tpu_backend import PyTorchTPUBackend

            self.backendFuncs = PyTorchTPUBackend(comms_world_info, commsParams)
        else:
            logger.error("Unsopported NW stack! ")
            comms_utils.gracefulExit()

        self.backendFuncs.initialize_backend(
            comms_world_info.master_ip,
            comms_world_info.master_port,
            backend=commsParams.backend,
        )
        self.backendFuncs.sayHello()

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
        )  # Getting ranks from backednFuncs object, since we cannot use MPI (e.g.: TPU) to launch all the processes

        self.collectiveArgs.group = group # default group
        self.collectiveArgs.groups = self.backendFuncs.get_groups()
        self.collectiveArgs.num_pgs = self.backendFuncs.get_num_pgs()
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

        # set of collectives to be replayed
        if self.allowList in ("all", "default", "*"):
            self.allowList = self.backendFuncs.collectiveFunc.keys()
        else:
            self.allowList = [paramToCommName(op) for op in self.allowList.split(",")]


    def initBench(self, commsParams: commsParamsHolderBase, args: argparse.Namespace) -> None:
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
        self.is_blocking = args.z
        self.do_warm_up = not args.no_warm_up
        self.allowList = args.allow_ops
        self.out_path = args.output_path
        self.colls_per_batch = args.colls_per_batch
        self.use_timestamp = args.use_timestamp

        if commsParams.bitwidth < 32:
            comms_utils.initQuantCommCtx(self.collectiveArgs, commsParams)

    def setTraceFile(self, args, comms_env_params):
        # TODO: file name may get changed later
        if args.use_one_trace:
            self.trace_file = args.trace_path
        else:
            self.trace_file = (
                f"{args.trace_path}/rank{comms_env_params['global_rank']}.json"
            )
        # assume the prefix is always "xxx://" when reading remote trace, e.g., http://xxx
        if "://" in args.trace_path:
            self.use_remote_trace = True

    def readTrace(self, remotePath: str) -> None:
        """
        Read trace file from remote server or local disk. This will also convert/parse traces files if needed.
        Supports conversions from KinetoTrace and PyTorch EG trace conversion is coming soon.

        Args:
            remotePath: Path to read from remotely if use_remote_trace is enabled.
        Returns:
            None
        """
        if self.use_remote_trace:
            protocol = remotePath.split("://", 2)[
                0
            ]  # format "<protocol prefix>://<url or path>"
            raw_comms_trace = []
            if protocol in ("http", "https", "ftp"):
                raw_comms_trace = comms_utils.commonUrlRead(remotePath=remotePath)
            else:
                try:
                    from internals import readRemoteTrace as readFbRemoteTrace
                except ImportError:
                    logger.error(
                        f"Not supported protocol for the URL provided {remotePath}"
                    )
                else:
                    raw_comms_trace = readFbRemoteTrace(remotePath=remotePath)

            self.comms_trace = json.load(raw_comms_trace)
        else:
            # read the json file from local disk
            with open(self.trace_file) as f:
                self.comms_trace = json.load(f)

        # additional check the trace format and convert it if needed
        try:
            from internals import fbTraceParser
        except ImportError:
            logger.info("FB internals not present, skipping Kineto fbTraceParser")
        else:
            self.comms_trace = fbTraceParser(
                self.comms_trace, target_rank=self.global_rank
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
    )

    args = traceBench.readArgs(parser)
    traceBench.setTraceFile(args, comms_env_params)
    traceBench.checkArgs(args)

    time.sleep(1)
    comms_world_info = comms_utils.comms_world_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, comms_env_params
    )
    commsParams = comms_utils.commsParamsHolderBase(args)
    traceBench.initBench(commsParams, args)
    traceBench.runBench(comms_world_info, commsParams)


if __name__ == "__main__":
    main()
