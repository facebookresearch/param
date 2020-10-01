#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

# import bisect # import shutil
import time
import numpy as np

import comms_utils as comms_utils

# pytorch
import torch

CUDA_MAX_THREADS = 1024


def runColl(collectiveArgs, comm_fn=None, compute_fn=None):
    collectiveArgs.backendFuncs.complete_accel_ops(collectiveArgs, initOp=True)
    numElements = collectiveArgs.numElements
    # Initial warmup iters.
    for _ in range(collectiveArgs.numWarmupIters):
        if comm_fn is not None:
            comm_fn(collectiveArgs)
        if compute_fn is not None:
            for _ in range(collectiveArgs.numComputePerColl):
                compute_fn(collectiveArgs)
        if not collectiveArgs.asyncOp:  # should be sychronous, do wait.
            collectiveArgs.backendFuncs.complete_accel_ops(collectiveArgs)
    collectiveArgs.backendFuncs.complete_accel_ops(
        collectiveArgs
    )  # should be done regardless of blocking or non-blocking.

    # Measuring time.
    start = time.monotonic()  # available only in py3
    for _ in range(collectiveArgs.numIters):
        if comm_fn is not None:
            comm_fn(collectiveArgs)
        if compute_fn is not None:
            for _ in range(collectiveArgs.numComputePerColl):
                # TODO: investigate the cache effect
                # Flush the cache
                # _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache
                compute_fn(collectiveArgs)
        if not collectiveArgs.asyncOp:  # should be sychronous, do wait.
            collectiveArgs.backendFuncs.complete_accel_ops(collectiveArgs)

    collectiveArgs.backendFuncs.complete_accel_ops(collectiveArgs)
    end = time.monotonic()  # available only in py3
    x = collectiveArgs.opTensor[
        numElements - 1
    ].item()  # to ensure collective won't be optimized away.

    elapsedTimeNS = (
        end - start
    ) * 1e9  # keeping time in NS, helps in divising data by nanoseconds
    avgIterNS, algBW = comms_utils.getAlgBW(
        elapsedTimeNS, collectiveArgs.dataSize, collectiveArgs.numIters
    )
    busBW = collectiveArgs.backendFuncs.getBusBW(
        collectiveArgs.collective, algBW, collectiveArgs.world_size
    )
    memSize = collectiveArgs.backendFuncs.get_mem_size(collectiveArgs)

    # dist.barrier(group=collectiveArgs.group)  # TODO: Make it generic! only works on GPU
    return (avgIterNS, algBW, busBW, memSize, x)


def initializeCollectiveArgs(commsParams, backendFuncs):
    # lint was complaining that benchTime was too complex!
    (
        local_rank,
        global_rank,
        world_size,
        group,
        curDevice,
    ) = comms_utils.get_rank_details(backendFuncs)  # Getting ranks from backednFuncs object, since we cannot use MPI (e.g.: TPU) to launch all the processes.

    comms_utils.fixBeginSize(commsParams, world_size)  # Ensuring that all-reduce and all-to-all has atleast one member per rank.
    backendFuncs.sayHello()  # Informs us where each process is running.
    allSizes = comms_utils.getSizes(
        commsParams.beginSize, commsParams.endSize, commsParams.stepFactor
    )  # Given the begin-size, end-size, step-factor what are the message sizes to iterate on.

    if global_rank == 0:
        print(
            "\t global_rank: %d allSizes: %s local_rank: %d element_size: %d "
            % (global_rank, allSizes, local_rank, commsParams.element_size)
        )
        print("\t global_rank: %d commsParams: %s " % (global_rank, commsParams))

    collectiveArgs = comms_utils.collectiveArgsHolder()
    collectiveArgs.group = group
    collectiveArgs.device = curDevice
    collectiveArgs.world_size = world_size
    collectiveArgs.numIters = commsParams.numIters
    collectiveArgs.numWarmupIters = commsParams.numWarmupIters
    collectiveArgs.global_rank = global_rank
    collectiveArgs.backendFuncs = backendFuncs
    collectiveArgs.srcOrDst = ""
    collectiveArgs.collective = commsParams.collective
    op = backendFuncs.get_reduce_op("sum")
    collectiveArgs.op = op
    collectiveArgs.dst = commsParams.dst

    computeFunc = None
    if commsParams.mode != "comms":  # Compute mode related initialization.
        if commsParams.kernel == "gemm":
            computeFunc = backendFuncs.gemm

            mm_dim = commsParams.mm_dim
            in1 = np.random.rand(mm_dim, mm_dim)
            MMin1 = torch.FloatTensor(in1).to(curDevice)
            in2 = np.random.rand(mm_dim, mm_dim)
            MMin2 = torch.FloatTensor(in2).to(curDevice)
            in3 = np.random.rand(mm_dim, mm_dim)
            MMin3 = torch.FloatTensor(in3).to(curDevice)
            MMout = backendFuncs.alloc_empty(
                [mm_dim, mm_dim], commsParams.dtype, curDevice
            )
            collectiveArgs.MMout = MMout
            collectiveArgs.MMin1 = MMin1
            collectiveArgs.MMin2 = MMin2
            collectiveArgs.MMin3 = MMin3
            collectiveArgs.numComputePerColl = commsParams.num_compute
        else:
            print("Compute kernel " + commsParams.kernel + " not supported...Abort!")
            comms_utils.gracefulExit()

    return (
        collectiveArgs,
        local_rank,
        global_rank,
        world_size,
        group,
        curDevice,
        allSizes,
        computeFunc,
    )

def reportBenchTime(collectiveArgs, commsParams, allSizes, tensorList, results):
    collectiveArgs.collective = commsParams.collective
    collectiveArgs.numIters = 1  # commsParams.numIters

    print(
        "\n\tCOMMS-RES\tsize (B)\t num-elements\t Latency(us):p50\tp75\t\tp95\t algBW(GB/s)\t busBW(GB/s)"
    )
    for idx, curSize in enumerate(allSizes):
        latencyAcrossRanks = []
        for curRankTensor in tensorList:
            rank_lat = curRankTensor[idx].item()
            latencyAcrossRanks.append(rank_lat)

        latencyAcrossRanks = np.array(latencyAcrossRanks)
        p50 = np.percentile(latencyAcrossRanks, 50)
        p75 = np.percentile(latencyAcrossRanks, 75)
        p95 = np.percentile(latencyAcrossRanks, 95)

        collectiveArgs.dataSize = curSize
        avgIterNS, algBW = comms_utils.getAlgBW(
            p50 * 1e3, collectiveArgs.dataSize, collectiveArgs.numIters
        )
        busBW = collectiveArgs.backendFuncs.getBusBW(
            collectiveArgs.collective, algBW, collectiveArgs.world_size
        )

        print(
            "\tCOMMS-RES\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s\t%12s"
            % (
                results[curSize]["memSize"],
                str("%d" % (results[curSize]["num_elements"])),
                str("%.1f" % (p50)),
                str("%.1f" % (p75)),
                str("%.1f" % (p95)),
                str("%.3f" % (algBW)),
                str("%.3f" % (busBW)),
            )
        )

def benchTime(index, commsParams, backendFuncs):
    # Get NW stack specific parameters
    (
        collectiveArgs,
        local_rank,
        global_rank,
        world_size,
        group,
        curDevice,
        allSizes,
        computeFunc,
    ) = initializeCollectiveArgs(commsParams, backendFuncs)

    results = {}
    timeElapsedList = []
    for curSize in allSizes:
        # Allocating memory.
        numElements = int(curSize // commsParams.element_size)
        scaleFactor = numElements * numElements
        if commsParams.collective == "all_to_all":
            # numElements = int(numElements // world_size)  # assuming that world_size won't be zero!
            scaleFactor = 1
        ipTensor = backendFuncs.alloc_random(
            [numElements], curDevice, commsParams.dtype, scaleFactor
        )

        opTensor = ipTensor
        # ignoring all_gather, scatter-gather, for now # FUTURE-TODO- make interface accept scatter and gather list.
        asyncOp = True
        collectiveFunc = None

        if (
            commsParams.blockingFlag == 1
        ):  # if blockingFlag is 1, it means asyncOp should be false.
            asyncOp = False

        if commsParams.mode != "compute":  # comms specific initializations
            if commsParams.collective == "all_reduce":
                collectiveFunc = backendFuncs.all_reduce

            elif commsParams.collective == "all_to_all":
                opTensor = backendFuncs.alloc_empty(
                    [numElements], commsParams.dtype, curDevice
                )
                collectiveFunc = backendFuncs.all_to_all

            elif commsParams.collective == "all_to_allv":
                opTensor = backendFuncs.alloc_empty(
                    [numElements], commsParams.dtype, curDevice
                )
                collectiveArgs.ipTensor_split = [
                    int(numElements // world_size) for i in range(world_size)
                ]
                collectiveArgs.opTensor_split = [
                    int(numElements // world_size) for i in range(world_size)
                ]
                collectiveFunc = backendFuncs.all_to_allv

            elif commsParams.collective == "reduce":
                collectiveFunc = backendFuncs.reduce

        # Setup the arguments.
        collectiveArgs.ipTensor = ipTensor
        collectiveArgs.opTensor = opTensor
        collectiveArgs.asyncOp = asyncOp
        collectiveArgs.dataSize = curSize
        collectiveArgs.numElements = numElements
        collectiveArgs.waitObj = None

        # collectiveArgs has all the information on the experiment.
        timeElapsedNS, algBW, busBW, memSize, x = runColl(
            collectiveArgs, comm_fn=collectiveFunc, compute_fn=computeFunc
        )

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
        results[curSize]["x"] = x

        del ipTensor
        del opTensor
        backendFuncs.clear_memory()

    # Push the list to device, then do an all-gather.
    timeElapsedTensor = torch.tensor(timeElapsedList, device=curDevice)
    tensorList = [torch.ones_like(timeElapsedTensor) for _ in range(world_size)]

    collectiveArgs.ipTensor = timeElapsedTensor
    collectiveArgs.tensorList = tensorList
    collectiveArgs.asyncOp = False
    collectiveArgs.dataSize = (
        timeElapsedTensor.nelement() * timeElapsedTensor.element_size()
    )
    collectiveArgs.numElements = timeElapsedTensor.nelement()
    collectiveArgs.waitObj = backendFuncs.all_gather(collectiveArgs, retFlag=True)
    backendFuncs.complete_accel_ops(collectiveArgs)

    if global_rank == 0:
        reportBenchTime(collectiveArgs, commsParams, allSizes, tensorList, results)


def runComms(comms_world_info, commsParams):
    # Run sanity checks.
    if commsParams.endSize < commsParams.beginSize:
        print(
            "\t ERROR: In COMMS-mode, the begin-size: %d is larger than the end-size: %d "
            % (commsParams.beginSize, commsParams.endSize)
        )

    # Run-loop
    if commsParams.nw_stack == "pytorch-nccl":
        # from pytorch_nccl_backend import PyTorchNCCLBackend
        from pytorch_nccl_backend import PyTorchNCCLBackend

        backendObj = PyTorchNCCLBackend(comms_world_info, commsParams)
    elif commsParams.nw_stack == "pytorch-xla-tpu":
        from tpu_backend import PyTorchTPUBackend

        backendObj = PyTorchTPUBackend(comms_world_info, commsParams)
    else:
        print("\t Error: Unsopported NW stack! ")
        comms_utils.gracefulExit()

    backendObj.benchmark_comms()
    return


def main():
    ### import packages ###
    import sys
    import argparse

    supportedCommStyle = [0, 1]  # 0 : non-blocking, 1 : blocking.
    supportedCollectives = [
        "reduce",
        "all_reduce",
        "all_to_all",
        "all_to_allv",
    ]  # , "scatter", "gather", "all_gather", "broadcast", "all_to_all"]
    supportedNwstacks = ["pytorch-nccl", "pytorch-xla-tpu"]
    supported_tpu_core_valuses = [1, 8]
    dtypeMap = {
        "float32": torch.float32,
        "int32": torch.int32,
        "float16": torch.half,
        "float64": torch.double,
    }
    supportedDtype = list(dtypeMap.keys())

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="PARAM-Comm Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # experiment related parameters
    parser.add_argument(
        "--backend", type=str, default="nccl",
        help="The backend to be used in PyTorch distributed process group"
    )  # alternative is DLRM mode.
    parser.add_argument(
        "--mode", type=str, default="comms",
        help="benchmark mode"
    )  # alternative is DLRM mode or comm-compute mode
    parser.add_argument("--b", type=str, default="8",
        help="minimum size, in bytes, to start with"
    )  # COMMS mode, begin the sweep at.
    parser.add_argument("--e", type=str, default="64",
        help="maximum size, in bytes, to end at"
    )  # COMMS mode, end the sweep at.
    parser.add_argument(
        "--f", type=int, default=2,
        help="multiplication factor between sizes"
    )  # COMMS mode, multiplication factor.
    parser.add_argument(
        "--z", type=int, default=1,
        help="use blocking mode for collectives"
    )  # 'sync/blocking' : 1 , 'async/non-blocking' : 0

    parser.add_argument("--w", type=int, default=5,
        help="number of warmup iterations"
    )  # number of warmup-iterations
    parser.add_argument("--n", type=int, default=5,
        help="number of iterations"
    )  # number of iterations
    parser.add_argument(
        "--collective", type=str, default="all_reduce",
        help='Collective to benchmark, supports ' + str(supportedCollectives)
    )  # collective op to benchmark
    parser.add_argument(
        "--master-ip", type=str, default="127.0.0.1",
        help="The master-IP to coordinate"
    )  # The master-IP to coordinate.
    parser.add_argument(
        "--master-port", type=str, default="29500",
        help="The master-port to coordinate"
    )  # The master-port to coordinate.
    parser.add_argument(
        "--nw-stack", type=str, default="pytorch-nccl",
        help="network stack to be used, supports " + str(supportedNwstacks)
    )  # The network stack to profile.
    parser.add_argument(
        "--dtype", type=torch.dtype, default=torch.float32
    )  # will be overwritten based on args.data_type and dtypeMap.
    parser.add_argument(
        "--data-type", type=str, default="float32",
        help="the base data type, supports " + str(supportedDtype)
    )  # The data type

    parser.add_argument(
        "--num-tpu-cores", type=int, default=1,
        help="number of TPU cores to be used"
    )  # number of TPU cores

    # For comm-compute or compute mode
    parser.add_argument(
        "--kernel", type=str, default="gemm",
        help="compute kernel"
    )  # Compute kernel: "gemm"
    parser.add_argument(
        "--num-compute", type=int, default=100,
        help="one collective for every NUM_COMPUTE compute kernels"
    )  # Launch one coll for every n compute kernels
    # For GEMM
    parser.add_argument(
        "--mm-dim", type=int, default=100,
        help="dimension size for GEMM compute kernel"
    )  # Matrix multiplication dim n, A[n,n] * B [n,n]
    # For emb lookup
    parser.add_argument("--emb-dim", type=int, default=128,
        help="dimension size for Embedding table compute kernel"
    )  # Embedding table dimension
    parser.add_argument(
        "--num-embs", type=int, default=100000,
        help="Embedding table hash size for Embedding table compute kernel"
    )  # Embedding table hash size
    parser.add_argument("--avg-len", type=int, default=28,
        help="Average lookup operations per sample"
    )  # Average #lookup per sample
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="number of samples reading the table concurrently"
    )  # #Samples reading the table concurrently
    parser.add_argument(
        "--root", type=int, default=0,
        help="root process for reduce benchmark"
    )  # root process for reduce (and gather, scatter, bcast, etc., if support in the future)
    # TODO: check the correctness of root, should be between 0 to [world_size -1]

    args, leftovers = parser.parse_known_args()
    args.b = comms_utils.parsesize(args.b)
    args.e = comms_utils.parsesize(args.e)

    if args.nw_stack not in supportedNwstacks:
        print(
            "\t ERROR: Specified backend: %s is not one of the supported backends: %s. Make sure the input is using the correct case."
            % (args.nw_stack, str(supportedNwstacks))
        )
        sys.exit()  # WARNING: Assuming sys is always used, should find a platform-independent way to gracefully exit.

    if args.collective not in supportedCollectives:
        print(
            "\t ERROR: Specified collective: %s is not one of the supported collectives: %s. Make sure the input is using the correct case."
            % (args.collective, str(supportedCollectives))
        )
        sys.exit()  # WARNING: Assuming sys is always used, should find a platform-independent way to gracefully exit.

    if args.z not in supportedCommStyle:
        print(
            "\t ERROR: Specified blocking: %d is not one of the supported commstyle: %s"
            % (args.z, str(supportedCommStyle))
        )
        comms_utils.gracefulExit()

    if args.data_type not in supportedDtype:
        print(
            "\t ERROR: Specified dtype: %d is not one of the supported commstyle: %s"
            % (args.data_type, str(supportedDtype))
        )
        comms_utils.gracefulExit()

    args.dtype = dtypeMap[args.data_type]

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

    if args.num_tpu_cores not in supported_tpu_core_valuses:
        print(
            "\t ERROR: TPU core value: %d is not one of the supported values: %s "
            % (args.num_tpu_cores, supported_tpu_core_valuses)
        )
        comms_utils.gracefulExit()

    if args.b < 1:
        print("\t Starting size: %d should atleast be 1! " % (args.b))
        args.b = 1

    element_size = torch.ones([1], dtype=args.dtype).element_size()
    comms_world_info = comms_utils.comms_world_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, mpi_env_params
    )
    commsParams = comms_utils.commsParamsHolder(args, element_size, benchTime)
    runComms(comms_world_info, commsParams)


if __name__ == "__main__":
    main()
