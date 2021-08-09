# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO

import torch
from torch.autograd.profiler import record_function

random.seed()

logger = logging.getLogger(__name__)


def gracefulExit():
    # TODO: Is this the best way to exit?
    # WARNING: Assuming sys is always used, should find a platform-independent way to gracefully exit.
    sys.exit()


def parsesize(ipValue):
    """nccl-tests compatible input-size parsing."""
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


def parseRankList(ipStr, ipName, comms_world_info):
    rankList = [] # default empty

    if ipStr:
        if ipStr.isnumeric():
            # single rank
            rankList = [int(ipStr)]
        elif ipStr.find(",") != -1:
            # list of unique ranks separated by comma
            rankList = list(map(int, [r.strip() for r in ipStr.split(",")]))
            rankList = list(OrderedDict.fromkeys(rankList))
        elif ipStr.find(":") != -1:
            # a range of ranks defined by [start:end]
            pos = list(map(int, [r.strip() for r in ipStr.split(":")]))
            rankList = [*range(pos[0], pos[1] + 1)]

        # Check if input is valid
        if len(rankList) == 0 or any(
            r < 0 or r >= comms_world_info.world_size for r in rankList
        ):
            if comms_world_info.global_rank == 0:
                print("\t ERROR: Could not parse %s %s" % (ipName, ipStr))
            gracefulExit()
    return rankList


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
    group = backendFuncs.get_default_group(world_size)
    curDevice = backendFuncs.get_device()
    curHwDevice = backendFuncs.get_hw_device()

    return (local_rank, global_rank, world_size, group, curDevice, curHwDevice)


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


def commonUrlRead(remotePath):
    import urllib.request

    # TODO: Error handle
    with urllib.request.urlopen(remotePath) as rf:
        contents = rf.read()
    return StringIO(contents.decode("utf-8"))


def initQuantCommCtx(collectiveArgs, commsParams):
    logger.info(f"communication bitwidth set to {commsParams.bitwidth}")
    try:
        from internals import initialize_collectiveArgs_internal

        initialize_collectiveArgs_internal(collectiveArgs, commsParams)
    except ImportError:
        # cannot do quantization, reset bitwidth
        logger.warning("quantization not supported, disabled and continue...")
        commsParams.bitwidth = 32
        pass


def clearQuantCommCtx(collectiveArgs):
    try:
        logger.debug("Removing installed quantization handlers.")
        from internals import remove_quantization_handlers

        remove_quantization_handlers(collectiveArgs)
    except ImportError:
        pass


@dataclass
class paramTimer:
    elapsedTimeNS: float = 0.0  # keeping time in NS

    def reset(self, newTime=0.0):
        self.elapsedTimeNS = newTime

    def incrTimeNS(self, timeNS):
        self.elapsedTimeNS += timeNS

    def getTimeUS(self) -> float:
        return self.elapsedTimeNS / 1e3

    def getTimeNS(self) -> float:
        return self.elapsedTimeNS


class paramProfile(record_function):
    """Inherit from PyTorch profiler to enable autoguard profiling while measuring the time interval in PARAM"""

    def __init__(self, timer=None, description=""):
        self.description = description
        self.timer = timer
        super().__init__(name=description)

    def __enter__(self):
        super().__enter__()
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.monotonic()
        self.intervalNS = (self.end - self.start) * 1e9  # keeping time in NS
        # if given a valid paramTimer object, directly update the measured time interval
        if isinstance(self.timer, paramTimer):
            self.timer.incrTimeNS(self.intervalNS)
        logger.debug(f"{self.description} took {self.intervalNS} ns")
        super().__exit__(exc_type, exc_value, traceback)


class backendFunctions(ABC):
    """Abstract base class, provides common abstraction for all the backends."""

    def __init__(self):
        self.collectiveFunc = {
            "all_to_all": self.all_to_all,
            "all_to_allv": self.all_to_allv,
            "all_reduce": self.all_reduce,
            "broadcast": self.broadcast,
            "all_gather": self.all_gather,
            "all_gather_base": self.all_gather_base,
            "reduce": self.reduce,
            "reduce_scatter": self.reduce_scatter,
            "reduce_scatter_base": self.reduce_scatter,
            "barrier": self.barrier,
            "incast": self.incast,
            "multicast": self.multicast,
        }

    def getBusBW(self, collective, algBW, collectiveArgs):
        busBW = algBW
        mulFactor = 1.0
        if collective == "all_reduce":
            if collectiveArgs.world_size != 0:
                mulFactor = (
                    2 * (collectiveArgs.world_size - 1) / (collectiveArgs.world_size)
                )
            busBW = algBW * mulFactor
        elif collective in (
            "all_to_all",
            "all_to_allv",
            "all_gather",
            "reduce_scatter",
            "all_gather_base",
        ):
            if collectiveArgs.world_size != 0:
                mulFactor = (collectiveArgs.world_size - 1) / (
                    collectiveArgs.world_size
                )
            busBW = algBW * mulFactor
        elif collective in ("reduce", "broadcast", "incast", "multicast"):
            busBW = algBW
        else:
            print(
                "\t ERROR: collective: %s is not supported in computing bus BW! "
                % (collective)
            )
        return busBW

    def alloc_ones(
        self, sizeArr, curRankDevice="cuda", dtype=torch.float32, scaleFactor=1.0
    ):
        ipTensor = torch.ones(sizeArr, device=curRankDevice, dtype=dtype)
        if scaleFactor != 1.0:
            ipTensor = ipTensor * scaleFactor
        return ipTensor

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
    def barrier(self, collectiveArgs, name="dummy"):
        pass

    def sync_barrier(self, collectiveArgs, desc="world"):
        self.barrier(collectiveArgs, name=desc)

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
    def get_hw_device(self):
        pass

    @abstractmethod
    def get_default_group(self, world_size):
        pass

    @abstractmethod
    def get_groups(self):
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


class commsParamsHolderBase:
    def __init__(self, args):
        # A holding object for common input parameters
        self.nw_stack = args.nw_stack
        self.dtype = args.dtype
        self.backend = args.backend
        self.device = args.device
        self.blockingFlag = args.z
        # quantization
        self.bitwidth = args.bitwidth
        self.quant_a2a_embedding_dim = args.quant_a2a_embedding_dim
        self.quant_threshold = args.quant_threshold

        self.num_pgs = 1


class commsParamsHolder(commsParamsHolderBase):
    def __init__(self, args, comms_world_info, element_size, benchTime):
        # A holding object for the input parameters from collective benchmark
        super().__init__(args)

        self.element_size = element_size
        self.beginSize = args.b
        self.endSize = args.e
        self.maxSize = int(args.e // self.element_size)
        self.stepFactor = args.f
        self.srcOrDst = args.root
        self.dcheck = args.c
        self.quant_threshold = max(
            self.endSize, self.quant_threshold
        )  # use quantization for all sizes in collective benchmark

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

        self.pair = args.pair
        self.collective_pair = args.collective_pair

        self.pt2pt = args.pt2pt
        self.window = args.window

        self.src_ranks = parseRankList(args.src_ranks, "src_ranks", comms_world_info)
        self.dst_ranks = parseRankList(args.dst_ranks, "dst_ranks", comms_world_info)


class collectiveArgsHolder:
    def __init__(self):
        # A holding object for all the parameters related to a collective operation/experiment.
        self.group = None
        self.groups = []
        self.num_pgs = 0
        self.device = {}
        self.world_size = 0

        self.numIters = 0
        self.numWarmupIters = 0
        self.global_rank = -1
        self.backendFuncs = {}
        self.collective = ""
        self.pt2pt = ""
        self.src_rank = -1
        self.dst_rank = -1

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
        self.waitObj = []

        self.ipTensor_split_pair = []
        self.opTensor_split_pair = []

        self.ipTensor_pair = []
        self.opTensor_pair = []
        self.dataSize_pair = 0
        self.numElements_pair = 0

        self.all2all_qcomm = None
        self.reducescatter_allgather_qcomm = None
        self.allreduce_qcomm = 32  # TODO: set it as the bitwidth for now until the quantization kernels be supported
        self.reduce_qcomm = 32
        self.quant_threshold = 0
        self.quant_time = paramTimer()
        self.dequant_time = paramTimer()


class paramCommsBench(ABC):
    def __init__(self, supportedNwstacks=None):
        self.supportedNwstacks = supportedNwstacks
        self.supported_tpu_core_valuses = [1, 8]
        self.dtypeMap = {
            "float32": torch.float32,
            "int32": torch.int32,
            "long": torch.long,
            "float16": torch.half,
            "float64": torch.double,
        }
        self.supportedDtype = list(self.dtypeMap.keys())
        self.backendFuncs = ""
        self.collectiveArgs = collectiveArgsHolder()
        self.comm_size = 1
        self.global_rank = -1
        # update initVal to test difffernt value
        self.initVal = 1.0

    def isCudaAvail(self):
        return torch.cuda.is_available()

    def dcheck(self, commsParams, curSize, tensor):
        expRes = self.initVal
        if (commsParams.collective == "all_reduce") or (
            self.backendFuncs.get_global_rank() == commsParams.srcOrDst
            and commsParams.collective == "reduce"
        ):
            # NOTE: this is for sum op. and the inital value is "self.initVal"
            expRes = self.collectiveArgs.world_size * self.initVal

        if (
            # Check results for incast only on root
            commsParams.collective == "incast"
            and self.backendFuncs.get_global_rank() != commsParams.srcOrDst
        ) or (
            # Check results of multicast only for dst_ranks
            commsParams.collective == "multicast"
            and self.backendFuncs.get_global_rank() not in commsParams.dst_ranks
        ):
            return

        if isinstance(tensor, list):
            # for allgather and incast, it's a list of tensors:
            for (rank, t) in enumerate(tensor):
                for (index, val) in enumerate(t):
                    if val != expRes:
                        raise ValueError(
                            f"[{curSize}-bytes {commsParams.collective}] Wrong value at [{rank}][{index}] = {tensor[index]}, expected {expRes}\n {tensor}"
                        )
        else:
            for (index, val) in enumerate(tensor):
                if val != expRes:
                    raise ValueError(
                        f"[{curSize}-bytes {commsParams.collective}] Wrong value at [{index}] = {tensor[index]}, expected {expRes}\n {tensor}"
                    )

    def setTensorVal(self, tensor, useRandVal=True):
        newVal = random.random() if useRandVal else self.initVal
        # reset values
        if self.collectiveArgs.collective in ("all_reduce", "reduce"):
            # all processes use initVal to have predictable results
            tensor[:] = self.initVal
        elif self.collectiveArgs.collective in ("broadcast", "multicast"):
            # root process uses initVal and others use random values
            tensor[:] = (
                self.initVal
                if (self.backendFuncs.get_global_rank() == self.collectiveArgs.srcOrDst)
                else newVal
            )
        elif isinstance(tensor, list):
            # could be a list of tensor, for all_gather/gather
            for t in tensor:
                t[:] = newVal
        else:
            tensor[:] = newVal

    @abstractmethod
    def runBench(self, *args, **kwargs):
        """Must override to start the desired benchmarking"""
        pass

    @abstractmethod
    def benchTime(self, *args, **kwargs):
        """Must override to run the desired benchmarking"""
        pass

    @abstractmethod
    def reportBenchTime(self, *args, **kwargs):
        """Must override to report/print the desired output"""
        pass

    @abstractmethod
    def readArgs(self, parser):
        """Basic/Common arguments for all PARAM-Comm benchmarks"""
        parser.add_argument(
            "--master-ip",
            type=str,
            default="127.0.0.1",
            help="The master-IP to coordinate",
        )  # The master-IP to coordinate.
        parser.add_argument(
            "--master-port",
            type=str,
            default="29500",
            help="The master-port to coordinate",
        )  # The master-port to coordinate.
        parser.add_argument(
            "--nw-stack",
            type=str,
            default="pytorch-dist",
            help="network stack to be used, supports " + str(self.supportedNwstacks),
        )  # The network stack to profile.
        parser.add_argument(
            "--dtype", type=torch.dtype, default=torch.float32
        )  # will be overwritten based on args.data_type and dtypeMap.
        parser.add_argument(
            "--data-type",
            type=str,
            default="float32",
            help="the base data type, supports " + str(self.supportedDtype),
        )  # The data type
        parser.add_argument(
            "--num-tpu-cores",
            type=int,
            default=1,
            help="number of TPU cores to be used",
        )  # number of TPU cores
        parser.add_argument(
            "--log",
            type=str,
            default="ERROR",
            help="Logging level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )  # logging level
        parser.add_argument(
            "--device",
            type=str,
            default=("cuda" if self.isCudaAvail() else "cpu"),
            choices=["cpu", "cuda", "tpu"],
            help="data placement",
        )  # device to place data for collective benchmarking
        parser.add_argument(
            "--backend",
            type=str,
            default=("nccl" if self.isCudaAvail() else "mpi"),
            help="The backend to be used in PyTorch distributed process group",
            choices=["nccl", "gloo", "mpi", "ucc", "xla"],
        )  #  backend used for the network stack
        parser.add_argument(
            "--z",
            type=int,
            default=1,
            help="use blocking mode for collectives",
            choices=[0, 1],
        )  # 'sync/blocking' : 1 , 'async/non-blocking' : 0
        parser.add_argument(
            "--bitwidth",
            type=int,
            default=32,
            help="Quantization bitwidth",
            choices=[2, 4, 8, 16, 32],
        )  # comms quantization
        parser.add_argument(
            "--quant-a2a-embedding-dim",
            type=int,
            default=32,
            help="Embedding dimension used by quantization alltoall if enabled",
            choices=[32, 64, 128, 256],
        )  # Row dimension for quantization
        parser.add_argument(
            "--quant-threshold",
            type=int,
            default=33554432,
            help="threshold of message sizes to perform quantization if enabled",
        )  # quantization threshold, default 32 MB
        pass

    @abstractmethod
    def checkArgs(self, args):
        """Validate some basic/common arguments for all PARAM-Comm benchmarks"""
        if args.nw_stack not in self.supportedNwstacks:
            print(
                "\t ERROR: Specified backend: %s is not one of the supported backends: %s. Make sure the input is using the correct case."
                % (args.nw_stack, str(self.supportedNwstacks))
            )
            gracefulExit()
        if args.data_type not in self.supportedDtype:
            print(
                "\t ERROR: Specified dtype: %d is not one of the supported commstyle: %s"
                % (args.data_type, str(self.supportedDtype))
            )
            gracefulExit()
        if args.num_tpu_cores not in self.supported_tpu_core_valuses:
            print(
                "\t ERROR: TPU core value: %d is not one of the supported values: %s "
                % (args.num_tpu_cores, self.supported_tpu_core_valuses)
            )
            gracefulExit()
        # check and set log level
        numeric_level = getattr(logging, args.log.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError("Invalid log level: %s" % args.log)
        logging.basicConfig(
            level=numeric_level,
            format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        )
