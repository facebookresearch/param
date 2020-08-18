#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

# import bisect # import shutil
import time

import numpy as np

# import param_bench.train.comms.comms_utils as comms_utils
# import comms_utils
import param_bench.train.comms.comms_utils as comms_utils

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

    start = time.monotonic()  # available only in py3
    for _ in range(collectiveArgs.numIters):
        if comm_fn is not None:
            comm_fn(collectiveArgs)
        if compute_fn is not None:
            for _ in range(collectiveArgs.numComputePerColl):
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
    ) = comms_utils.get_rank_details(backendFuncs)

    comms_utils.fixBeginSize(commsParams, world_size)
    backendFuncs.sayHello()
    allSizes = comms_utils.getSizes(
        commsParams.beginSize, commsParams.endSize, commsParams.stepFactor
    )

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

    computeFunc = None
    if commsParams.mode != "comms":
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
        elif commsParams.kernel == "emb_lookup":
            computeFunc = backendFuncs.emb_lookup

            emb_dim = commsParams.emb_dim
            num_embeddings = commsParams.num_embs
            avg_length = commsParams.avg_len
            batch_size = commsParams.batch_size
            print(
                f"emb_dim {emb_dim} num_embs {num_embeddings} avg_len {avg_length} bs {batch_size}"
            )
            collectiveArgs.EmbWeights = backendFuncs.alloc_empty(
                [num_embeddings, emb_dim], torch.double, curDevice
            )
            collectiveArgs.TableOffsets = torch.LongTensor([0, num_embeddings]).to(
                curDevice
            )
            collectiveArgs.Indices = torch.LongTensor(
                np.random.randint(0, num_embeddings - 1, avg_length * batch_size)
            ).to(curDevice)
            lengths = np.ones((1, batch_size)) * avg_length
            flat_lengths = lengths.flatten()
            collectiveArgs.Offsets = torch.LongTensor(
                [0] + np.cumsum(flat_lengths).tolist()
            ).to(curDevice)
            collectiveArgs.LookupOut = backendFuncs.alloc_empty(
                [batch_size, emb_dim], torch.double, curDevice
            )
            collectiveArgs.AvgLengths = avg_length
            collectiveArgs.numComputePerColl = commsParams.num_compute

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
    # Input/output for MM
    results = {}
    timeElapsedList = []
    for curSize in allSizes:
        # Allocating memory.
        numElements = int(curSize // commsParams.element_size)
        scaleFactor = numElements * numElements
        if commsParams.collective == "all_to_all":
            # numElements = int(numElements // world_size)  # assuming that world_size won't be zero!
            scaleFactor = 1
        # ipTensor = backendFuncs.alloc_random(
        #     [numElements], curDevice, commsParams.dtype, scaleFactor
        # )
        array = np.random.rand(numElements).tolist()
        ipTensor = torch.tensor(array).to(curDevice)

        # Looks like we should have superset of all arguments on all the collectives as done in NCCL
        # WARNING: Datatype is being changed, is that ok?
        opTensor = ipTensor

        # HAZARDOUS WARNING: What about C++-connections ?
        # ignoring all_gather, scatter-gather, for now # FUTURE-TODO- make interface accept scatter and gather list.
        asyncOp = True
        collectiveFunc = None

        if (
            commsParams.blockingFlag == 1
        ):  # if blockingFlag is 1, it means asyncOp should be false.
            asyncOp = False

        if commsParams.mode != "compute":
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
                ]  # torch.ones([world_size], dtype=int, device=curDevice) * int(numElements // world_size)
                collectiveArgs.opTensor_split = [
                    int(numElements // world_size) for i in range(world_size)
                ]  # torch.ones([world_size], dtype=int, device=curDevice) * int(numElements // world_size)
                collectiveFunc = backendFuncs.all_to_allv

            elif commsParams.collective == "reduce":
                collectiveFunc = backendFuncs.reduce

        # Setup the arguments.
        collectiveArgs.ipTensor = ipTensor  # assuming this'd be a pointer.
        collectiveArgs.opTensor = opTensor  # assuming this'd be a pointer.
        collectiveArgs.asyncOp = asyncOp
        collectiveArgs.dataSize = curSize
        collectiveArgs.numElements = numElements
        collectiveArgs.waitObj = None

        timeElapsedNS, algBW, busBW, memSize, x = runColl(
            collectiveArgs, comm_fn=collectiveFunc, compute_fn=computeFunc
        )

        # if(global_rank == 0):
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
        backendFuncs.clear_memory()  # torch.cuda.empty_cache()

    # Push the list to device, then do an all-reduce (doing reduce to rank-0 is sufficient!)
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
        print(
            "\n\tCOMMS-RES\tsize (B)\t num-elements\t Latency(us):p50\tp75\t\tp95\t algBW(GB/s)\tbusBW(GB/s)\top[0]"
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
            # print("\t latencyAcrossRanks: %s p50: %.3f p95: %.3f " % (latencyAcrossRanks, p50, p95))
            print(
                "\tCOMMS-RES\t%12s\t%12s\t%12s\t%12s\t%12s\t%6s\t%6s\t%20s"
                % (
                    results[curSize]["memSize"],
                    str("%d" % (results[curSize]["num_elements"])),
                    str("%.1f" % (p50)),
                    str("%.1f" % (p75)),
                    str("%.1f" % (p95)),
                    str("%.3f" % (results[curSize]["algBW"])),
                    str("%.3f" % (results[curSize]["busBW"])),
                    str("%.3f" % (results[curSize]["x"])),
                )
            )


def runComms(comms_world_info, commsParams):
    # Run sanity checks.
    if commsParams.endSize < commsParams.beginSize:
        print(
            "\t ERROR: In COMMS-mode, the begin-size: %d is larger than the end-size: %d "
            % (commsParams.beginSize, commsParams.endSize)
        )

    # Run-loop
    if commsParams.nw_stack == "pytorch-nccl":
        # from param_bench.train.comms.pytorch_nccl_backend import PyTorchNCCLBackend
        # from pytorch_nccl_backend import PyTorchNCCLBackend
        from param_bench.train.comms.pytorch_nccl_backend import PyTorchNCCLBackend

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
        description="Deep Learning Recommendation Model (DLRM)- communications benchmark"
    )
    # experiment related parameters
    parser.add_argument(
        "--backend", type=str, default="nccl"
    )  # alternative is DLRM mode.
    parser.add_argument(
        "--mode", type=str, default="comms"
    )  # alternative is DLRM mode or comm-compute mode
    parser.add_argument("--b", type=str, default="8")  # COMMS mode, begin the sweep at.
    parser.add_argument("--e", type=str, default="64")  # COMMS mode, end the sweep at.
    parser.add_argument(
        "--f", type=int, default=2
    )  # COMMS mode, multiplication factor.
    parser.add_argument(
        "--z", type=int, default=1
    )  # 'sync/blocking' : 1 , 'async/non-blocking' : 0

    parser.add_argument("--w", type=int, default=5)  # number of warmup-iterations
    parser.add_argument("--n", type=int, default=5)  # number of iterations
    parser.add_argument(
        "--collective", type=str, default="all_reduce"
    )  # number of iterations
    parser.add_argument(
        "--master-ip", type=str, default="127.0.0.1"
    )  # The master-IP to coordinate.
    parser.add_argument(
        "--master-port", type=str, default="29500"
    )  # The master-IP to coordinate.
    parser.add_argument(
        "--nw-stack", type=str, default="pytorch-nccl"
    )  # The network stack to profile.
    parser.add_argument(
        "--dtype", type=torch.dtype, default=torch.float32
    )  # will be overwritten based on args.data_type and dtypeMap.
    parser.add_argument(
        "--data-type", type=str, default="float32"
    )  # The network stack to profile.

    parser.add_argument(
        "--num-tpu-cores", type=int, default=1
    )  # The network stack to profile.

    # For comm-compute or compute mode
    parser.add_argument(
        "--kernel", type=str, default="gemm"
    )  # Compute kernel: "gemm" or "emb_lookup"
    parser.add_argument(
        "--num-compute", type=int, default=100
    )  # Launch one coll for every n compute kernels
    # For GEMM
    parser.add_argument(
        "--mm-dim", type=int, default=100
    )  # Matrix multiplication dim n, A[n,n] * B [n,n]
    # For emb lookup
    parser.add_argument("--emb-dim", type=int, default=128)  # Embedding table dimension
    parser.add_argument(
        "--num-embs", type=int, default=100000
    )  # Embedding table hash size
    parser.add_argument("--avg-len", type=int, default=28)  # Average #lookup per sample
    parser.add_argument(
        "--batch-size", type=int, default=512
    )  # #Samples reading the table concurrently

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
