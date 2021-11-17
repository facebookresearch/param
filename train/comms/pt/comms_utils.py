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

default_master_ip = "127.0.0.1"
default_master_port = "29500"


def gracefulExit(args=0):
    # TODO: Is this the best way to exit?
    if args != 0:
        logger.error(args)
    # WARNING: Assuming sys is always used, should find a platform-independent way to gracefully exit.
    sys.exit(args)


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
        logger.error(f"Could not parse input size {ipValue}")
        gracefulExit()

    size = int(value) * units
    return int(size)


def parseRankList(ipStr, ipName, comms_world_info):
    rankList = []  # default empty

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
                logger.error(f"Could not parse {ipName}: {ipStr}")
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
            logger.error(
                f"For finding allSizes numIters: {numIters} is greater than maxIters: {maxIters}"
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

        if (
            commsParams.bitwidth < 32
            and (commsParams.beginSize / commsParams.element_size / world_size)
            < commsParams.quant_a2a_embedding_dim
        ):
            commsParams.beginSize = (
                commsParams.quant_a2a_embedding_dim
                * world_size
                * commsParams.element_size
            )
    elif (commsParams.collective == "all_reduce") or (
        commsParams.collective == "reduce"
    ):
        if commsParams.beginSize < commsParams.element_size:
            commsParams.beginSize = commsParams.element_size


def get_rank_details(backendFuncs):
    local_rank = backendFuncs.get_local_rank()
    global_rank = backendFuncs.get_global_rank()
    world_size = backendFuncs.get_world_size()
    group = backendFuncs.get_default_group()
    curDevice = backendFuncs.get_device()
    curHwDevice = backendFuncs.get_hw_device()

    return (local_rank, global_rank, world_size, group, curDevice, curHwDevice)


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def read_comms_env_vars():
    world_size = env2int(
        ["MV2_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"], -1
    )

    local_size = env2int(
        [
            "LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "MV2_COMM_WORLD_LOCAL_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
        ],
        -1,
    )

    global_rank = env2int(
        ["MV2_COMM_WORLD_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"], -1
    )

    local_rank = env2int(
        [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
        ],
        -1,
    )

    comms_env_params = {}
    comms_env_params["world_size"] = world_size
    comms_env_params["local_size"] = local_size
    comms_env_params["global_rank"] = global_rank
    comms_env_params["local_rank"] = local_rank
    return comms_env_params


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


def checkQuantArgs(collective, dtype, beginSize, quant_a2a_embedding_dim, blockingFlag):
    if collective not in (
        "all_to_all",
        "all_to_allv",
        "reduce",
        "all_reduce",
    ):
        raise NotImplementedError(
            f"quantized communication for {collective} is currently unsupported."
        )
    if collective in ("all_to_all", "all_to_allv"):
        if (beginSize // 4) % quant_a2a_embedding_dim != 0:
            logger.warning(
                f"begin size {beginSize} must be a multiple of --quant-a2a-embedding-dim {quant_a2a_embedding_dim} for all_to_all operation"
            )
        if blockingFlag != 1:
            raise NotImplementedError("quantized All_to_all must be synchronous.")
    if dtype != torch.float32:
        raise NotImplementedError(
            f"quantization for {dtype} is not supported. Use float32 instead."
        )


def clearQuantCommCtx(collectiveArgs):
    try:
        logger.debug("Removing installed quantization handlers.")
        from internals import remove_quantization_handlers

        remove_quantization_handlers(collectiveArgs)
    except ImportError:
        pass


def paramToCommName(name, supported_comms=None):
    """
    Map any possible creative collective names to the internal name
    Validate the `name` if `supported_comms` is providedd
    """
    name_aliases = {
        "alltoall": "all_to_all",
        "alltoallv": "all_to_allv",
        "alltoallbase": "all_to_allv",
        "allreduce": "all_reduce",
        "allgather": "all_gather",
        "allgatherbase": "all_gather_base",
        "reducescatter": "reduce_scatter",
        "recvanysource": "recv",
    }

    new_name = name.lower()

    new_name = "".join(x for x in new_name if x.isalpha())
    if new_name in name_aliases:
        new_name = name_aliases[new_name]
    else:
        new_name = name

    if supported_comms is not None and new_name not in supported_comms:
        gracefulExit(
            f"{name} is not a supported communication in PARAM! Supported comms: {supported_comms}"
        )

    return new_name


def ensureTensorFlush(tensors):
    x = None
    if isinstance(tensors, list) and len(tensors) > 0:
        # some collectives like allgather use a list of tensors
        x = tensors[-1][-1].item()  # to ensure collective won't be optimized away.
    elif isinstance(tensors, torch.Tensor) and tensors.nelement() > 0:
        x = tensors[-1].item()  # to ensure collective won't be optimized away.

    return x


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
            "reduce_scatter_base": self.reduce_scatter_base,
            "barrier": self.barrier,
            "incast": self.incast,
            "multicast": self.multicast,
            "noop": self.noop,
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
            "reduce_scatter_base",
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
            logger.error(
                f"collective: {collective} is not supported in computing bus BW! "
            )
        return busBW

    def alloc_ones(
        self, sizeArr, curRankDevice="cuda", dtype=torch.float32, scaleFactor=1.0
    ):
        ipTensor = torch.ones(sizeArr, device=curRankDevice, dtype=dtype)
        if scaleFactor != 1.0:
            ipTensor = ipTensor * scaleFactor
        return ipTensor

    def noop(self, collectiveArgs=None, retFlag=False, pair=False):
        """no-op for the case we want to skip comms/compute"""
        pass

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
    def clear_memory(self, collectiveArgs):
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
    def get_default_group(self):
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
    def __init__(self, master_ip, master_port, num_tpu_cores, comms_env_params):
        # Holding communication-world related parameters.
        self.global_rank = comms_env_params["global_rank"]
        self.local_rank = comms_env_params["local_rank"]
        self.world_size = comms_env_params["world_size"]

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
        self.dcheck = args.c
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
        self.comms_world_info = comms_world_info


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

        self.ipTensor_pair = None
        self.opTensor_pair = None
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
        if (
            commsParams.collective
            in ("all_reduce", "reduce_scatter", "reduce_scatter_base")
        ) or (
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
            commsParams.collective in ("multicast", "pt2pt")
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
            # could be a list of tensor, for all_gather, gather, reduce_scatter
            for t in tensor:
                t[:] = newVal
        else:
            tensor[:] = newVal

    def prepComm(self, curComm, commsParams):
        """Allocate the tensors for collective"""
        commOp = paramToCommName(
            curComm["comms"] if ("comms" in curComm.keys()) else commsParams.collective,
            supported_comms=self.backendFuncs.collectiveFunc.keys(),
        )

        if commOp in ("wait", "barrier"):
            return ([], [])

        numElementsIn = curComm["in_msg_size"]
        # numElementsOut is only meaningful for out-of-place collectives and pt2pt
        numElementsOut = curComm["out_msg_size"]
        world_size = self.collectiveArgs.world_size
        dtype = commsParams.dtype
        curDevice = commsParams.device
        # scaleFactor = 1 if commsParams.collective == "all_to_all" else numElements * numElements
        scaleFactor = numElementsOut * numElementsOut
        opTensor = []

        if commsParams.dcheck == 1:
            # use predictable values for data validation check
            ipTensor = self.backendFuncs.alloc_ones(
                [numElementsIn], curDevice, dtype, scaleFactor=self.initVal
            )
        else:
            ipTensor = self.backendFuncs.alloc_random(
                [numElementsIn], curDevice, dtype, scaleFactor
            )

        if commOp == "all_to_allv":
            # all_to_all(v) requires two tensors
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut], curDevice, dtype, scaleFactor
            )
            # all_to_allv requires tensors to specify split
            self.collectiveArgs.opTensor_split = (
                curComm["out_split"] if ("out_split" in curComm.keys()) else []
            )
            self.collectiveArgs.ipTensor_split = (
                curComm["in_split"] if ("in_split" in curComm.keys()) else []
            )
        elif commOp == "all_gather":
            # allgather requires a tensor list, e.g., List[torch.Tensor]
            for _ in range(world_size):
                opTensor.append(
                    self.backendFuncs.alloc_random(
                        [numElementsIn], curDevice, dtype, scaleFactor
                    )
                )
        elif commOp == "all_gather_base":
            # this is a single all gather with flat output tensor
            opTensor = self.backendFuncs.alloc_random(
                numElementsIn * world_size,
                curDevice,
                dtype,
                scaleFactor,
            )
        elif commOp == "incast":
            # incast requires a tensor list with length of src_ranks, e.g., List[torch.Tensor]
            for _ in self.collectiveArgs.src_ranks:
                opTensor.append(
                    self.backendFuncs.alloc_random(
                        [numElementsOut], curDevice, dtype, scaleFactor
                    )
                )
        elif commOp == "reduce_scatter":
            opTensor = ipTensor
            ipTensor = []
            if commsParams.dcheck == 1:
                for _ in range(world_size):
                    ipTensor.append(
                        self.backendFuncs.alloc_ones(
                            [numElementsOut], curDevice, commsParams.dtype, self.initVal
                        )
                    )
            else:
                for _ in range(world_size):
                    ipTensor.append(
                        self.backendFuncs.alloc_random(
                            [numElementsOut], curDevice, commsParams.dtype, scaleFactor
                        )
                    )
        elif commOp == "reduce_scatter_base":
            opTensor = ipTensor
            ipTensor = []
            if commsParams.dcheck == 1:
                ipTensor = self.backendFuncs.alloc_ones(
                    numElementsOut * world_size,
                    curDevice,
                    commsParams.dtype,
                    self.initVal,
                )
            else:
                ipTensor = self.backendFuncs.alloc_random(
                    numElementsOut * world_size,
                    curDevice,
                    commsParams.dtype,
                    scaleFactor,
                )
        elif commOp in ("all_to_all", "pt2pt"):
            # pt2pt or out-of-place collectives
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut],
                curDevice,
                dtype,
                scaleFactor,
            )
        else:
            # in-place case for other collectives such as allreduce, reduce, broadcast
            opTensor = ipTensor

        return (ipTensor, opTensor)

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
            default=default_master_ip,
            help="The master-IP to coordinate",
        )  # The master-IP to coordinate.
        parser.add_argument(
            "--master-port",
            type=str,
            default=default_master_port,
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
        parser.add_argument(
            "--c",
            type=int,
            default=0,
            help="enable data validation check",
            choices=[0, 1],
        )  # validation check
        pass

    @abstractmethod
    def checkArgs(self, args):
        """Validate some basic/common arguments for all PARAM-Comm benchmarks"""
        if args.nw_stack not in self.supportedNwstacks:
            logger.error(
                f"Specified backend: {args.nw_stack} is not one of the supported backends: {str(self.supportedNwstacks)}. Make sure the input is using the correct case."
            )
            gracefulExit()
        if args.data_type not in self.supportedDtype:
            logger.error(
                f"Specified dtype: {args.data_type} is not one of the supported commstyle: {str(self.supportedDtype)}"
            )
            gracefulExit()
        if args.num_tpu_cores not in self.supported_tpu_core_valuses:
            logger.error(
                f"TPU core value: {args.num_tpu_cores} is not one of the supported values: {self.supported_tpu_core_valuses}"
            )
            gracefulExit()
        # check and set log level
        numeric_level = getattr(logging, args.log.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {args.log}")
        comms_env_params = read_comms_env_vars()
        logging.basicConfig(
            level=numeric_level,
            format="[%(asctime)s][%(name)s][%(levelname)s][Rank{:3}] - %(message)s".format(
                comms_env_params["global_rank"]
            ),
        )
        # check master-ip and master-port with the following logic
        #   1) prefer the values passed to PARAM, i.e., through --master-ip and --master-port
        #   2) check and use the env. variable, i.e., MASTER_ADDR and MASTER_PORT
        #   3) if both #1 and #2 are not set, pre-defined default values will be used
        if "MASTER_ADDR" in os.environ:
            if args.master_ip not in (default_master_ip, os.environ["MASTER_ADDR"]):
                logger.warning(
                    f"--master-ip={args.master_ip} while MASTER_ADDR={os.environ['MASTER_ADDR']}, "
                    f"use --master-ip={args.master_ip} and continue..."
                )
                os.environ["MASTER_ADDR"] = args.master_ip
            else:
                logger.info(
                    "From environment variables, using MASTER_ADDR="
                    + os.environ["MASTER_ADDR"]
                )
        else:
            os.environ["MASTER_ADDR"] = args.master_ip

        if "MASTER_PORT" in os.environ:
            if args.master_port not in (default_master_port, os.environ["MASTER_PORT"]):
                logger.warning(
                    f"--master-port={args.master_port} while MASTER_PORT={os.environ['MASTER_PORT']}, "
                    f"use --master-port={args.master_port} and continue..."
                )
                os.environ["MASTER_PORT"] = args.master_port
            else:
                logger.info(
                    "From environment variables, using MASTER_PORT="
                    + os.environ["MASTER_PORT"]
                )
        else:
            os.environ["MASTER_PORT"] = args.master_port
