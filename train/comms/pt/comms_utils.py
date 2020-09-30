# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
from abc import ABC, abstractmethod


def gracefulExit():
    # TODO: Is this the best way to exit?
    # WARNING: Assuming sys is always used, should find a platform-independent way to gracefully exit.
    sys.exit()


def parsesize(ipValue):
    """ nccl-tests compatible input-size parsing. """
    units = 0
    size = 0.0

    value = ""
    if ipValue.find("G") != -1:
        units = 1024 * 1024 * 1024
        unitIdx = ipValue.find("G")
        value = ipValue[0:unitIdx]

    elif ipValue.find("M") != -1:
        units = 1024 * 1024
        unitIdx = ipValue.find("M")
        value = ipValue[0:unitIdx]

    elif ipValue.find("K") != -1:
        units = 1024
        unitIdx = ipValue.find("K")
        value = ipValue[0:unitIdx]
    elif ipValue.isnumeric():
        units = 1
        value = ipValue
    else:
        print("\t ERROR: Could not parse input size %s " % (ipValue))
        gracefulExit()

    size = int(value) * units
    return int(size)


def getAlgBW(elapsedTimeNS, dataSize, numIters):
    # Similar to how algorithmic bandwidth is computed in nccl-tests.
    avgIterNS = 0.0
    if numIters != 0:
        avgIterNS = elapsedTimeNS / (numIters)

    algBW = 0.0
    if avgIterNS != 0:
        algBW = (dataSize) / (avgIterNS)  # dataSize dividied by ns gives us GBps
    return (avgIterNS, algBW)


def getSizes(beginSize, endSize, stepFactor):
    curSize = beginSize
    numIters = 0
    maxIters = 100
    allSizes = []
    while curSize <= endSize:
        allSizes.append(curSize)
        curSize = curSize * stepFactor
        numIters = numIters + 1
        if numIters > 100:
            print(
                "\t ERROR: For finding allSizes numIters: %d is greater than maxIters: %d "
                % (numIters, maxIters)
            )
            break
    return allSizes


def fixBeginSize(commsParams, world_size):
    # ensures we will have atleast one member/rank
    if (commsParams.collective == "all_to_all") or (
        commsParams.collective == "all_to_allv"
    ):
        if (commsParams.beginSize / commsParams.element_size) < world_size:
            commsParams.beginSize = world_size * commsParams.element_size
    elif (commsParams.collective == "all_reduce") or (
        commsParams.collective == "reduce"
    ):
        if commsParams.beginSize < commsParams.element_size:
            commsParams.beginSize = commsParams.element_size


def get_rank_details(backendFuncs):
    local_rank = backendFuncs.get_local_rank()
    global_rank = backendFuncs.get_global_rank()
    world_size = backendFuncs.get_world_size()
    group = backendFuncs.get_group(world_size)
    curDevice = backendFuncs.get_device()

    return (local_rank, global_rank, world_size, group, curDevice)


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def read_mpi_env_vars():
    world_size = env2int(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)

    local_size = env2int(
        ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"],
        1,
    )

    global_rank = env2int(
        ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0
    )

    local_rank = env2int(
        ["MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"],
        0,
    )

    mpi_env_params = {}
    mpi_env_params["world_size"] = world_size
    mpi_env_params["local_size"] = local_size
    mpi_env_params["global_rank"] = global_rank
    mpi_env_params["local_rank"] = local_rank
    return mpi_env_params


class backendFunctions(ABC):
    """ Abstract base class, provides common abstraction for all the backends. """
    def getBusBW(self, collective, algBW, world_size):
        busBW = algBW
        mulFactor = 1.0
        if collective == "all_reduce":
            if world_size != 0:
                mulFactor = 2 * (world_size - 1) / (world_size)
            busBW = algBW * mulFactor
        elif (collective == "all_to_all") or (collective == "all_to_allv"):
            if world_size != 0:
                mulFactor = (world_size - 1) / (world_size)
            busBW = algBW * mulFactor
        elif collective == "reduce":
            busBW = algBW
        else:
            print(
                "\t ERROR: collective: %s is not supported in computing bus BW! "
                % (collective)
            )
        return busBW

    @abstractmethod
    def sayHello(self, global_rank, local_rank, world_size, master_ip):
        pass

    # Collectives
    @abstractmethod
    def all_reduce(self, collectiveArgs, retFlag=False):
        pass

    @abstractmethod
    def reduce(self, collectiveArgs, retFlag=False):
        pass

    @abstractmethod
    def all_to_all(self, collectiveArgs, retFlag=False):
        pass

    @abstractmethod
    def all_to_allv(self, collectiveArgs, retFlag=False):
        pass

    @abstractmethod
    def complete_accel_ops(self, collectiveArgs, initOp=False):
        pass

    @abstractmethod
    def get_reduce_op(self, opName):
        pass

    # Compute functions
    @abstractmethod
    def gemm(self, collectiveArgs):
        pass

    # Memory related
    @abstractmethod
    def get_mem_size(self, collectiveArgs):
        pass

    @abstractmethod
    def alloc_random(self, sizeArr, curRankDevice, dtype, scaleFactor=1.0):
        pass

    @abstractmethod
    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        pass

    @abstractmethod
    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        pass

    @abstractmethod
    def clear_memory(self):
        pass

    # Getting world-size and other information.
    @abstractmethod
    def get_local_rank(self):
        pass

    @abstractmethod
    def get_global_rank(self):
        pass

    @abstractmethod
    def get_world_size(self):
        pass

    @abstractmethod
    def get_device(self):
        pass

    @abstractmethod
    def get_group(self, world_size):
        pass

    # Init functions
    @abstractmethod
    def initialize_backend(self, master_ip, master_port, backend="gloo"):
        pass

    @abstractmethod
    def benchmark_comms(self):
        pass


class comms_world_info_holder:
    def __init__(self, master_ip, master_port, num_tpu_cores, mpi_env_params):
        # Holding communication-world related parameters.
        self.global_rank = mpi_env_params["global_rank"]
        self.local_rank = mpi_env_params["local_rank"]
        self.world_size = mpi_env_params["world_size"]

        self.master_ip = master_ip
        self.master_port = master_port
        self.num_tpu_cores = num_tpu_cores


class commsParamsHolder:
    def __init__(self, args, element_size, benchTime):
        # A holding object for all the input parameters.
        self.nw_stack = args.nw_stack

        self.dtype = args.dtype
        self.element_size = element_size
        self.beginSize = args.b
        self.endSize = args.e
        self.maxSize = int(args.e // self.element_size)
        self.stepFactor = args.f
        self.blockingFlag = args.z
        self.dst = args.root

        self.backend = args.backend
        self.numWarmupIters = args.w
        self.numIters = args.n
        self.collective = args.collective
        self.mode = args.mode

        self.kernel = args.kernel
        self.num_compute = args.num_compute
        self.mm_dim = args.mm_dim
        self.emb_dim = args.emb_dim
        self.avg_len = args.avg_len
        self.num_embs = args.num_embs
        self.batch_size = args.batch_size
        self.benchTime = benchTime


class collectiveArgsHolder:
    def __init__(self):
        # A holding object for all the parameters related to a collective operation/experiment.
        self.group = []
        self.device = {}
        self.world_size = 0

        self.numIters = 0
        self.numWarmupIters = 0
        self.global_rank = -1
        self.backendFuncs = {}
        self.collective = ""

        self.MMout = {}
        self.MMin1 = {}
        self.MMin2 = {}
        self.MMin3 = {}
        self.numComputePerColl = 0

        self.EmbWeights = {}
        self.TableOffsets = {}
        self.Indices = {}
        self.Offsets = {}
        self.BTBlockSize = {}
        self.LookupOut = {}
        self.AvgLengths = {}

        self.ipTensor_split = []
        self.opTensor_split = []

        self.ipTensor = []
        self.opTensor = []
        self.srcOrDst = -1
        self.asyncOp = -1
        self.dataSize = 0
        self.numElements = 0
        self.dst = 0
