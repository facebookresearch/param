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

import comms_utils as comms_utils
import numpy as np
from comms_utils import paramCommsBench
from torch.autograd.profiler import record_function

logger = logging.getLogger(__name__)


def writeCommDetails(commsTracePerf, folder="./"):
    try:
        import subprocess
        subprocess.check_output(["mkdir", "-p", str(folder)], universal_newlines=True)
    except Exception as err:
        print("\t Error: %s while creating directory: %s " % (err, folder))
        pass
    comms_file = folder + "/replayedCommsPerf.json"
    logger.info(f"Writing comms details to {comms_file}")
    with open(comms_file, "w") as write_file:
        json.dump(commsTracePerf, write_file, indent=2)


class commsTraceReplayBench(paramCommsBench):
    def __init__(self):
        super().__init__(supportedNwstacks=["pytorch-nccl", "pytorch-xla-tpu"])
        self.comms_trace = {}
        self.trace_file = ""
        self.is_dry_run = False
        self.shrink = False
        self.max_msg_cnt = 0  # 0 means no limit
        self.num_msg = 0
        self.force_async = False
        self.do_warm_up = True

        self.collInMsgSizes: Dict[str, List] = {}
        self.collInUniMsgSizes: Dict[str, Set] = {}
        self.collOutMsgSizes: Dict[str, List] = {}
        self.collOutUniMsgSizes: Dict[str, Set] = {}

        self.collLat: Dict[str, List] = {}
        self.collTraceStat: Dict[str, List] = {}

        self.comms_blocks: Dict[str, List] = {}
        self.blockStack = []
        self.totalCommsLatency = 0.0

        import torch

        self.strToTorchDtype = {
            "Byte": torch.uint8,
            "Float": torch.float32,
            "Int": torch.int32,
            "Long": torch.long,
            "Double": torch.double,
        }

    def readArgs(self, parser, defaultModel="dlrm"):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--trace-path",
            type=str,
            default="traces",
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
            help="Only replay first N messages (0 means no limit)",
        )
        parser.add_argument(
            "--force-async",
            action="store_true",
            default=self.force_async,
            help="Toggle to make all communication asynchronous unless replaying barrier",
        )
        parser.add_argument(
            "--no-warm-up",
            action="store_true",
            default=False,
            help="Toggle to disable performing extra replaying for warm-up",
        )

        return parser.parse_args()

    def checkArgs(self, args):
        super().checkArgs(args)

        if (
            path.exists(self.trace_file) is False
            or path.isfile(self.trace_file) is False
        ):
            raise ValueError(
                f"Trace file {self.trace_file} not exist or not a file! Please specifiy the correct path using --trace-path"
            )
            comms_utils.gracefulExit()

    def reportBenchTime(self, commsParams):
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

            print(
                f"+ {len(blockComms)} comms in block {curBlock}: {Lats.sum():.2f} us in total"
            )

        print("\n{} Message size Statistcs {}".format("=" * 20, "=" * 20))

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
            for (coll, lats) in self.collLat.items():
                if len(lats) == 0:
                    continue

                Lat = np.array(lats)
                print("{}\n Replayed {} {} ({:.2f}%): \n{}".format("-" * 50, len(lats), coll, (Lat.sum()/self.totalCommsLatency)*100, "-" * 50))

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
                msgSizeAndLatency = tuple(
                    zip(lats, self.collInMsgSizes[coll], self.collOutMsgSizes[coll])
                )
                logger.debug(f"Latency and size of First ten: {msgSizeAndLatency[:10]}")

    def initTraceStat(self):
        maxInMsgsize = 0
        maxOutMsgsize = 0
        self.num_msg = len(self.comms_trace)
        self.max_msg_cnt = self.num_msg if self.max_msg_cnt == 0 else self.max_msg_cnt
        # FIXME: ideally, need to know actually elemement size of datatype indicated in the trace
        elem_size = 4
        # first pass to know the statistics and get required info.
        for curComm in self.comms_trace[:self.max_msg_cnt]:
            # record the current comm
            collName = curComm["comms"]
            if collName not in self.collTraceStat.keys():
                self.collTraceStat[collName] = []
                self.collLat[collName] = []
                # if collName not in self.collInMsgSizes.keys():
                self.collInMsgSizes[collName] = []
                self.collInUniMsgSizes[collName] = set()
                # if collName not in self.collOutMsgSizes.keys():
                self.collOutMsgSizes[collName] = []
                self.collOutUniMsgSizes[collName] = set()
            self.collInMsgSizes[collName].append(curComm["in_msg_size"]*elem_size)
            self.collInUniMsgSizes[collName].add(curComm["in_msg_size"]*elem_size)
            self.collOutMsgSizes[collName].append(curComm["out_msg_size"]*elem_size)
            self.collOutUniMsgSizes[collName].add(curComm["out_msg_size"]*elem_size)
            maxInMsgsize = max(curComm["in_msg_size"]*elem_size, maxInMsgsize)
            maxOutMsgsize = max(curComm["out_msg_size"]*elem_size, maxOutMsgsize)
            # get info sorted by code block
            # TODO: should we care about the comms without any markers?
            for curBlock in curComm["marker_stack"]:
                if curBlock not in self.comms_blocks:
                    self.comms_blocks[curBlock] = []
                # only add entries if on dry run, otherwise, we'll deal with later during replay w/ more info
                if self.is_dry_run:
                    self.comms_blocks[curBlock].append(
                        {
                            "comms": collName,
                            "in_msg_size_bytes": curComm["in_msg_size"]*elem_size,
                            "out_msg_size_bytes": curComm["out_msg_size"]*elem_size,
                        }
                    )

    # TODO: this function could be generalized for both this and comms.py
    def prepComms(self, curComm):
        # either in_split or out_split should be specified for alltoallv; meaningless for other collectives
        self.collectiveArgs.opTensor_split = (
            curComm["out_split"] if ("out_split" in curComm.keys()) else []
        )
        self.collectiveArgs.ipTensor_split = (
            curComm["in_split"] if ("in_split" in curComm.keys()) else []
        )

        curInNumElem = curComm["in_msg_size"]
        curOutNumElem = curComm["out_msg_size"]

        # FIXME: this is for debug purpose only since we don't always get to run very large scale; this is useful to debug in small scale, but the same trace
        if self.shrink:
            self.collectiveArgs.opTensor_split = (
                curComm["out_split"][: self.collectiveArgs.world_size]
                if ("out_split" in curComm.keys())
                else []
            )
            self.collectiveArgs.ipTensor_split = (
                curComm["in_split"][: self.collectiveArgs.world_size]
                if ("in_split" in curComm.keys())
                else []
            )
            curInNumElem = (
                sum(self.collectiveArgs.ipTensor_split)
                if len(self.collectiveArgs.ipTensor_split) > 0
                else curInNumElem
            )
            curOutNumElem = (
                sum(self.collectiveArgs.opTensor_split)
                if len(self.collectiveArgs.opTensor_split) > 0
                else curOutNumElem
            )
            logger.debug(
                f"shrink message sizes to curInNumElem {curInNumElem}, curOutNumElem {curOutNumElem}"
            )

        # allocate tensors
        # FIXME: could we re-use the tensors?
        # FIXME: cannot use alloc_rand for torch.int32, does it matter for perf?
        self.collectiveArgs.ipTensor = self.backendFuncs.alloc_empty(
            [curInNumElem],
            curRankDevice=self.collectiveArgs.device,
            dtype=self.strToTorchDtype[curComm["dtype"]],
        )
        if curComm["comms"] in ("all_to_all", "all_to_allv"):
            # alltoall requires two tensors
            self.collectiveArgs.opTensor = self.backendFuncs.alloc_empty(
                [curOutNumElem],
                curRankDevice=self.collectiveArgs.device,
                dtype=self.strToTorchDtype[curComm["dtype"]],
            )
        elif curComm["comms"] in ("allgather", "all_gather"):
            # allgather requires a tensor list, e.g., List[torch.Tensor]
            self.collectiveArgs.opTensor = []
            for _ in range(self.collectiveArgs.world_size):
                self.collectiveArgs.opTensor.append(
                    self.backendFuncs.alloc_empty(
                        [curInNumElem],
                        curRankDevice=self.collectiveArgs.device,
                        dtype=self.strToTorchDtype[curComm["dtype"]],
                    )
                )
        else:
            # only one tensor required for allreduce, reduce and broadcast
            self.collectiveArgs.opTensor = self.collectiveArgs.ipTensor

    # TODO: recursive call to get the proper marker stack for prettier trace in Kineto
    def commStack(self, blockStack, blockname, curComm):
        if self.blockStack[-1] != curComm["marker_stack"][-1]:
            # a new stack is encouter
            pass
        with record_function(blockname):
            # do comms
            if len(blockStack) > 0:
                nextBlock = blockStack.pop()
                nextComm = next(self.comms_trace)
                self.commStack(blockStack, nextBlock, nextComm)
            pass

    def benchTime(self, commsParams):
        """
        The json format is expecting to be either
        {
            "marker_stack": ["## all2all_tw_data:init ##"]
            "comms": "all_to_allv",
            "in_msg_size": 10357149,
            "out_msg_size": 23093760,
            "in_split": [],
            "out_split": [],
            "dtype": "Int"
        },
        or w/o in/out_split
        {
            "marker_stack": ["## all2all_tw_data:init ##"]
            "comms": "all_reduce",
            "in_msg_size": 1048576,
            "out_msg_size": 1048576,
            "dtype": "Int"
        }
        NOTE:
            - this format is subject to be changed/defined later
            - the unit of all size fields is # of elements (not bytes)
        """
        # FIXME: ideally, need to know actually elemement size of datatype indicated in the trace
        elem_size = 4

        # warm-up
        if self.do_warm_up:
            for cnt, curComm in enumerate(self.comms_trace[:self.max_msg_cnt]):

                if self.backendFuncs.get_global_rank() == 0:
                    print(
                        f"[Warm-up][{cnt} / {self.max_msg_cnt}] Replaying {curComm['comms']}...", end="\r"
                    )
                # read fields and prepare the tensors
                self.prepComms(curComm)

                self.collectiveArgs.waitObj.append(
                    self.backendFuncs.collectiveFunc[curComm["comms"]](
                        self.collectiveArgs, retFlag=self.collectiveArgs.asyncOp
                    )
                )
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)

        # sync everything before starting real runs
        self.backendFuncs.complete_accel_ops(self.collectiveArgs, initOp=True)
        self.backendFuncs.barrier(self.collectiveArgs)

        if self.backendFuncs.get_global_rank() == 0:
            print(
                f"\n+ {self.max_msg_cnt} messages to be replayed..."
            )
            for coll, sizes in self.collInMsgSizes.items():
                print(f"\t{coll}: {len(sizes)}")

        # second pass to perform collectives
        for cnt, curComm in enumerate(self.comms_trace[:self.max_msg_cnt]):
            collName = curComm["comms"]
            curBlocks = curComm["marker_stack"]
            curBlockStack = ' '.join(curBlocks) if len(curBlocks) > 0 else "Unamed/Unknown"

            logger.debug(
                f"[Rank {self.collectiveArgs.global_rank:3}] Got {collName} in block {curBlocks}, in_size {curComm['in_msg_size']}, out_size {curComm['out_msg_size']}, dtype {curComm['dtype']}"
            )

            if self.backendFuncs.get_global_rank() == 0:
                print(
                    f"[{cnt} / {self.max_msg_cnt}] Replaying {collName} in block [{curBlockStack}]...",
                    end="",
                )

            # read fields and prepare the tensors
            self.prepComms(curComm)

            # perform the collective and wait for it
            begin = time.monotonic()
            with record_function(curBlockStack):
                self.collectiveArgs.waitObj.append(
                    self.backendFuncs.collectiveFunc[collName](
                        self.collectiveArgs, retFlag=self.collectiveArgs.asyncOp
                    )
                )
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)
            end = time.monotonic()
            latency = (end - begin) * 1e6  # make it microsecond

            self.collLat[collName].append(latency)
            self.collTraceStat[collName].append(
                (curComm["in_msg_size"], curComm["out_msg_size"], latency)
            )

            curComm["latency(us)"] = latency
            self.totalCommsLatency += latency

            for curBlock in curComm["marker_stack"]:
                self.comms_blocks[curBlock].append(
                    {
                        "comms": collName,
                        "in_msg_size_bytes": curComm["in_msg_size"] * elem_size,
                        "out_msg_size_bytes": curComm["out_msg_size"] * elem_size,
                        "latency_us": latency,
                    }
                )

            if self.backendFuncs.get_global_rank() == 0:
                print(f"{latency:.2f} us")

            if not self.force_async:
                self.backendFuncs.barrier(self.collectiveArgs)

        self.backendFuncs.clear_memory()

    def runBench(self, comms_world_info, commsParams):
        self.setBench(comms_world_info, commsParams)

        self.initTraceStat()
        # only perform collectives if not dry run mode
        if not self.is_dry_run:
            self.backendFuncs.initialize_backend(
                comms_world_info.master_ip,
                comms_world_info.master_port,
                backend=commsParams.backend,
            )
            self.backendFuncs.sayHello()

            # start benchmark
            self.benchTime(commsParams)
        elif comms_world_info.global_rank == 0:
            print(
                "+ Dry run mode...No replaying, Only Rank 0 read and analyze the trace..."
            )

        # rank 0 reports statistics
        if comms_world_info.global_rank == 0:
            self.reportBenchTime(commsParams)
            # dump trace sorted with block and with latency if not dry run
            writeCommDetails(self.comms_blocks)
            # TODO: collect perf. from all ranks to rank 0 and detect any imbalanced perf?

        if not self.is_dry_run:
            self.backendFuncs.barrier(self.collectiveArgs)

    def setBench(self, comms_world_info, commsParams):
        # init backend and corresponding function pointers
        if commsParams.nw_stack == "pytorch-nccl":
            from pytorch_dist_backend import PyTorchDistBackend

            self.backendFuncs = PyTorchDistBackend(comms_world_info, commsParams)
        elif commsParams.nw_stack == "pytorch-xla-tpu":
            from pytorch_tpu_backend import PyTorchTPUBackend

            self.backendFuncs = PyTorchTPUBackend(comms_world_info, commsParams)
        else:
            print("\t Error: Unsopported NW stack! ")
            comms_utils.gracefulExit()

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

        self.collectiveArgs.group = group
        self.collectiveArgs.device = curDevice
        self.collectiveArgs.world_size = world_size
        self.collectiveArgs.global_rank = global_rank
        self.collectiveArgs.backendFuncs = self.backendFuncs
        # FIXME:  0 is a common case, need this info from trace for more accurate replay
        self.collectiveArgs.srcOrDst = 0
        # FIXME: assuming it's always sum for reduce/allreduce operations
        self.collectiveArgs.op = self.backendFuncs.get_reduce_op("sum")
        # FIXME: alwasy perfom blocking comms; may study non-blocking in the future
        self.collectiveArgs.asyncOp = False
        self.collectiveArgs.ipTensor = None
        self.collectiveArgs.opTensor = None

        # read the json file
        with open(self.trace_file) as f:
            logger.info(f"rank-{global_rank} reading {self.trace_file}")
            self.comms_trace = json.load(f)

    def initBench(self, comms_world_info, commsParams, args):
        self.is_dry_run = args.dry_run
        self.shrink = args.auto_shrink
        self.max_msg_cnt = args.max_msg_cnt
        self.force_async = args.force_async
        self.do_warm_up = not args.no_warm_up

    def setTraceFile(self, args, mpi_env_params):
        # TODO: file name may get changed later
        if args.use_one_trace:
            self.trace_file = args.trace_path
        else:
            self.trace_file = (
                f"{args.trace_path}/rank{mpi_env_params['global_rank']}.json"
            )

def main():

    mpi_env_params = comms_utils.read_mpi_env_vars()

    traceBench = commsTraceReplayBench()
    parser = argparse.ArgumentParser(
        description="PARAM-Comms Trace Replay Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    args = traceBench.readArgs(parser)
    traceBench.setTraceFile(args, mpi_env_params)
    traceBench.checkArgs(args)

    time.sleep(1)
    comms_world_info = comms_utils.comms_world_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, mpi_env_params
    )
    commsParams = comms_utils.commsParamsHolderBase(args)
    traceBench.initBench(comms_world_info, commsParams, args)
    traceBench.runBench(comms_world_info, commsParams)


if __name__ == "__main__":
    main()
