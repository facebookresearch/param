# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse

import logging
import os
import random
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from collections.abc import Callable
from contextlib import ContextDecorator
from io import StringIO
from typing import Any

try:
    from et_replay.vendor_internal.fb_internal import (
        initialize_collectiveArgs_internal,
        remove_quantization_handlers,
    )

    has_internal_libs = True
except ImportError:
    has_internal_libs = False

try:
    import et_replay.comm.backend.vendor_internal.fb_internals  # noqa: F401

    loaded_internals = True
except ImportError:
    loaded_internals = False


import numpy as np
import torch
from et_replay.comm.backend.base_backend import (
    BaseBackend,
    collectiveArgsHolder,
    customized_backend,
    supportedC10dBackends,
    supportedDevices,
)

from et_replay.comm.param_profile import paramTimer
from torch._C._distributed_c10d import ProcessGroup  # @manual

random.seed()

logger = logging.getLogger(__name__)

default_master_ip = "127.0.0.1"
default_master_port = "29500"


def gracefulExit(args: Any = 0) -> None:
    """
    Use this function to gracefully exit if any fatal errors are encountered.

    Args:
        args: Message you want to print out.
    Returns:
        None: Will cause program to terminate.
    """
    # TODO: Is this the best way to exit?
    if args != 0:
        logger.error(args)
    # WARNING: Assuming sys is always used, should find a platform-independent way to gracefully exit.
    sys.exit(args)


def parseRankList(ipStr: str) -> list[int]:
    """
    Parses a string into a rank list.

    Args:
        ipStr: String containing list of ranks or single rank.
    Returns:
        List: Returns list containing the ranks from ipStr
    """
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
    return rankList


def fixBeginSize(commsParams: commsParamsHolder, world_size: int) -> None:
    """
    Validate begin size to match other parameters.

    Args:
        commsParams: Holds beginSize and other parameters to perform validation.
        world_size: The total number of global ranks.
    Returns:
        None
    """
    # ensures we will have atleast one member/rank
    if commsParams.collective in (
        "all_to_all",
        "all_to_allv",
        "all_gather",
        "all_gather_base",
        "gather",
        "reduce_scatter_base",
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


def get_rank_details(
    backendFuncs: BaseBackend,
) -> tuple[int, int, int, ProcessGroup, str, str]:
    """
    Returns the details of the rank for the current backendFunction.

    Args:
        backendFuncs: Backend we are gathering information from.
    Returns:
        (local_rank, global_rank, world_size, group, curDevice, curHwDevice): Returns the values of these in the provided backendFunction.
    """
    local_rank = backendFuncs.get_local_rank()
    global_rank = backendFuncs.get_global_rank()
    world_size = backendFuncs.get_world_size()
    group = backendFuncs.get_default_group()
    curDevice = backendFuncs.get_device()
    curHwDevice = backendFuncs.get_hw_device()

    return (local_rank, global_rank, world_size, group, curDevice, curHwDevice)


def env2int(env_list: list[str], default: int = -1) -> int:
    """
    Takes environment variables list and returns the first value found.

    Args:
        env_list: List of environment variables.
        default: Default value to return if all environment variables are not set.
    Returns:
        val: Returns value located at one of the environment variables, or returns default value if none are set.
    """
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def read_comms_env_vars() -> dict[str, int]:
    """
    Reads environment variables and record them.

    Args:
        None
    Returns:
        comms_env_params: Dict containing env var name as key and int for that env var as value.
    """
    world_size = env2int(
        [
            "MV2_COMM_WORLD_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "WORLD_SIZE",
            "SLURM_NTASKS",
        ],
        1,
    )

    local_size = env2int(
        [
            "LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "MV2_COMM_WORLD_LOCAL_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "SLURM_NTASKS_PER_NODE",
        ],
        1,
    )

    global_rank = env2int(
        [
            "MV2_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "RANK",
            "SLURM_PROCID",
        ],
        0,
    )

    local_rank = env2int(
        [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "SLURM_LOCALID",
        ],
        0,
    )

    comms_env_params = {}
    comms_env_params["world_size"] = world_size
    comms_env_params["local_size"] = local_size
    comms_env_params["global_rank"] = global_rank
    comms_env_params["local_rank"] = local_rank
    return comms_env_params


def commonUrlRead(remotePath: str) -> StringIO:
    """
    Reads content at remotePath.

    Args:
        remotePath: URL of where to read from.
    Returns:
        StringIO: Return decoded StringIO for contents of url.
    """
    import urllib.request

    # TODO: Error handle
    with urllib.request.urlopen(remotePath) as rf:
        contents = rf.read()
    return StringIO(contents.decode("utf-8"))


def initQuantCommCtx(
    collectiveArgs: collectiveArgsHolder, commsParams: commsParamsHolderBase
) -> None:
    """
    Initialize quantization handlers.

    Args:
        collectiveArgs: This will be modified to support quantization.
        commsParams: Holds parameters used to setup quantization (bidwidth).
    Returns:
        None
    """
    logger.info(f"communication bitwidth set to {commsParams.bitwidth}")

    if has_internal_libs:
        initialize_collectiveArgs_internal(collectiveArgs, commsParams)
    else:
        # cannot do quantization, reset bitwidth
        logger.warning("quantization not supported, disabled and continue...")
        commsParams.bitwidth = 32


def checkQuantArgs(
    collective: str,
    dtype: torch.dtype,
    beginSize: int,
    quant_a2a_embedding_dim: int,
    blockingFlag: bool,
) -> None:
    """
    Checks quantized args passed in parameter list to make sure they are supported, will exit if not.

    Args:
        collective: Name of collective to be quantized.
        dtype: Torch datatype of collective.
        beginSize: Starting size.
        quant_a2a_embedding_dim: Quant embedding dimension for all_to_all.
        blockingFlag: Flag to specify whether the collective will be ran in blocking or non-blocking mode.
    Returns:
        None
    """
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
        if not blockingFlag:
            raise NotImplementedError("quantized All_to_all must be synchronous.")
    if dtype != torch.float32:
        raise NotImplementedError(
            f"quantization for {dtype} is not supported. Use float32 instead."
        )


def clearQuantCommCtx(collectiveArgs: collectiveArgsHolder) -> None:
    """
    Cleans up quantization handlers.

    Args:
        collectiveArgs: Contains the quantization handlers.
    Returns:
        None
    """
    if has_internal_libs:
        logger.debug("Removing installed quantization handlers.")
        remove_quantization_handlers(collectiveArgs)


def paramToCommName(name: str, supported_comms: list[str] | None = None) -> str:
    """
    Map any possible creative collective names to the internal name.
    Validate the `name` if `supported_comms` is provided.

    Args:
        name: Name of collective.
        supported_comms: List of supported comms to check in.
    Returns:
        new_name: Returns the formatted name if supported_comms is empty, or name is in supported_comms.
    """
    name_aliases = {
        "alltoall": "all_to_all",
        "alltoallv": "all_to_allv",
        "alltoallbase": "all_to_allv",
        "allreduce": "all_reduce",
        "allgather": "all_gather",
        "allgatherbase": "all_gather_base",
        "reducescatter": "reduce_scatter",
        "reducescatterbase": "reduce_scatter_base",
        "recvanysource": "recv",
    }

    new_name = name.lower()

    new_name = "".join(x for x in new_name if x.isalpha())
    if new_name in name_aliases:
        new_name = name_aliases[new_name]
    else:
        new_name = name

    if supported_comms is not None and new_name not in supported_comms:
        logger.error(
            f"{name} is not a supported communication in PARAM! Supported comms: {supported_comms}"
        )
        gracefulExit()

    return new_name


def ensureTensorFlush(tensors: list[torch.Tensor] | torch.Tensor) -> Any:
    """
    Use this to flush non-blocking ops to ensure they are really complete.

    Args:
        tensors: Retrieve item of last tensor to force flush.
    Returns:
        x: A standard python number, can be float or int.
    """
    x = None
    if isinstance(tensors, list) and len(tensors) > 0 and len(tensors[-1]) > 0:
        # some collectives like allgather use a list of tensors
        x = tensors[-1][-1].item()  # to ensure collective won't be optimized away.
    elif isinstance(tensors, torch.Tensor) and tensors.nelement() > 0:
        x = tensors[-1].item()  # to ensure collective won't be optimized away.

    return x


class commsArgs:
    """
    This class contains all of the args that we can use to perform a single collective.

    Public Attributes:
        Global/Comm Attributes:
            comms: Name of collective.
            compute: Name of compute kernel.
            id: Current trace object ID.
            req: Request ID of collective to map to wait operation.
            inMsgSize: Size of input tensor.
            outMsgSize: Size of output tensor.
            dtype: Data type of tensor values.
            inSplit: List of input split sizes for rank across current process group.
            outSplit: List of output split sizes for ranks across current process group.
            startTimeNs: Start time of current collective.
            pgId: Unique indentifier for the process group this collective will use.
            groupRanks: Global ranks of the process group, this is used with PG init.
            worldSize: World size of current process group.
            markerStack: Current markers that this collective is a part of.
            root: Used to determine if collective is src or dst.
            src_rank: Src rank of a send/recv op.
            dst_rank: Dst rank of a send/recv op.

        GEMM Attributes:
            mm0_dim0: dimension 0 of the first matrix for replaying GEMM kernels
            mm0_dim1: dimension 1 of the first matrix for replaying GEMM kernels
            mm1_dim0: dimension 0 of the second matrix for replaying GEMM kernels
            mm1_dim1: dimension 1 of the second matrix for replaying GEMM kernels

        Embedded Lookup Attributes:
            direction: direction of embedding lookup kernel (forward or backward)
            embDim: dimension size for Embedding table compute kernel
            numEmbs: Embedding table hash size for Embedding table compute kernel
            batchSize: number of samples reading the table concurrently
            numEmbTables: number of embedding tables (per device)
            numEmbTablesBatched: number of embedding tables batched together (-1 means no batching)
            bagSize: bag size
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize arguments used for comm replay.
        """
        self.comms = kwargs["comms"] if "comms" in kwargs else None
        self.compute = kwargs["compute"] if "compute" in kwargs else None
        self.id = kwargs["id"] if "id" in kwargs else None
        self.req = kwargs["req"] if "req" in kwargs else None
        self.inMsgSize = kwargs["inMsgSize"] if "inMsgSize" in kwargs else None
        self.outMsgSize = kwargs["outMsgSize"] if "outMsgSize" in kwargs else None
        self.dtype = kwargs["dtype"] if "dtype" in kwargs else None
        self.inSplit = kwargs["inSplit"] if "inSplit" in kwargs else None
        self.outSplit = kwargs["outSplit"] if "outSplit" in kwargs else None
        self.startTimeNs = kwargs["startTimeNs"] if "startTimeNs" in kwargs else None
        self.pgId = kwargs["pgId"] if "pgId" in kwargs else None
        self.pgDesc = kwargs["pgDesc"] if "pgDesc" in kwargs else None
        self.groupRanks = kwargs["groupRanks"] if "groupRanks" in kwargs else None
        self.worldSize = kwargs["worldSize"] if "worldSize" in kwargs else None
        self.markerStack = kwargs["markerStack"] if "markerStack" in kwargs else None
        self.root = kwargs["root"] if "root" in kwargs else None
        self.src_rank = kwargs["src_rank"] if "src_rank" in kwargs else None
        self.dst_rank = kwargs["dst_rank"] if "dst_rank" in kwargs else None
        self.batch_p2p = kwargs["use_batch"] if "use_batch" in kwargs else None

        self.mm0_dim0 = kwargs["mm0_dim0"] if "mm0_dim0" in kwargs else None
        self.mm0_dim1 = kwargs["mm0_dim0"] if "mm0_dim1" in kwargs else None
        self.mm1_dim0 = kwargs["mm1_dim1"] if "mm1_dim0" in kwargs else None
        self.mm1_dim1 = kwargs["mm1_dim1"] if "mm1_dim1" in kwargs else None

        self.direction = kwargs["direction"] if "direction" in kwargs else 0
        self.embDim = kwargs["embDim"] if "embDim" in kwargs else None
        self.numEmbs = kwargs["numEmbs"] if "numEmbs" in kwargs else None
        self.numEmbTables = kwargs["numEmbTables"] if "numEmbTable" in kwargs else None
        self.numEmbTablesBatched = (
            kwargs["numEmbTablesBatched"] if "numEmbTablesBatched" in kwargs else None
        )
        self.bagSize = kwargs["bagSize"] if "bagSize" in kwargs else None

    def toDict(self) -> dict:
        """
        Convert commsArgs to dictionary for storing in json.

        Args:
            None
        Returns:
            commData: Dictionary containing the comms metadata.
        """
        commData = {}
        if self.comms is not None:
            commData["comms"] = self.comms
        if self.compute is not None:
            commData["compute"] = self.compute
            if commData["compute"] == "gemm":
                if self.mm0_dim0 is not None:
                    commData["mm0_dim0"] = self.mm0_dim0
                if self.mm0_dim1 is not None:
                    commData["mm0_dim1"] = self.mm0_dim1
                if self.mm1_dim0 is not None:
                    commData["mm1_dim0"] = self.mm1_dim0
                if self.mm1_dim1 is not None:
                    commData["mm1_dim1"] = self.mm1_dim1
            elif commData["compute"] == "emb_lookup":
                if self.embDim is not None:
                    commData["embDim"] = self.embDim
                if self.numEmbs is not None:
                    commData["numEmbs"] = self.numEmbs
                if self.numEmbTables is not None:
                    commData["numEmbTables"] = self.numEmbTables
                if self.numEmbTablesBatched is not None:
                    commData["numEmbTablesBatched"] = self.numEmbTablesBatched
                if self.bagSize is not None:
                    commData["bagSize"] = self.bagSize

        if self.req is not None:
            commData["req"] = self.req
        if self.inMsgSize is not None:
            commData["in_msg_size"] = self.inMsgSize
            commData["out_msg_size"] = self.outMsgSize
            commData["dtype"] = self.dtype
        if self.inSplit is not None:
            commData["in_split"] = self.inSplit
        if self.outSplit is not None:
            commData["out_split"] = self.outSplit
        if self.startTimeNs is not None:
            commData["startTime_ns"] = self.startTimeNs
        if self.pgId is not None:
            commData["pg_id"] = self.pgId
        if self.worldSize is not None:
            commData["world_size"] = self.worldSize
        if self.root is not None:
            commData["root"] = self.root

        return commData

    def __eq__(self, other: commsArgs) -> bool:
        """
        Used for testing. Check if two comms are equal.
        """
        return self.__dict__ == other.__dict__

    def __repr__(self):
        """
        Print repr of commsArgs in human readable format.
        """
        return self.__dict__.__str__()

    def __str__(self) -> str:
        """
        Print out the commsArgs in human readable format.
        """
        return self.__dict__.__str__()

    def toEmbLookupTuple(self):
        """
        Return tuple containing all values relevant to embedding lookup replay.
        """
        return (
            self.direction,
            self.emb_dim,
            self.num_embs,
            self.batch_size,
            self.num_emb_tables_per_device,
            self.bagSize,
        )


class paramStreamGuard(ContextDecorator):
    """guard execution on a stream"""

    def __init__(
        self,
        stream: torch.cuda.Stream | None,
        curDevice: torch.device,
        backendFuncs: BaseBackend,
        is_blocking: bool = True,
        timer: paramDeviceTimer | None = None,
    ) -> None:
        self.cur_stream = None
        self.stream = stream
        self.curDevice = curDevice
        self.backendFuncs = backendFuncs
        self.is_blocking = is_blocking
        self.timer = timer

    def __enter__(self) -> paramStreamGuard:
        self.cur_stream = self.backendFuncs.switch_stream(self.stream, self.curDevice)
        if self.timer:
            self.timer.start(self.stream)
        return self

    def __exit__(self, *exc) -> None:
        if self.timer:
            self.timer.end(self.stream)
        if self.is_blocking:
            self.backendFuncs.sync_stream(self.cur_stream, self.curDevice)
        self.backendFuncs.switch_stream(self.cur_stream, self.curDevice)


class paramDeviceTimer(paramTimer):
    """
    Device timer.
    """

    def __init__(self, name: str, backendFuncs: BaseBackend) -> None:
        """
        Initialize start and end device events
        """
        super().__init__()
        self.name = name
        self.start_event = backendFuncs.get_new_event(enable_timing=True)
        self.end_event = backendFuncs.get_new_event(enable_timing=True)

    def start(self, stream=None) -> None:
        self.start_event.record(stream)

    def end(self, stream=None) -> None:
        self.end_event.record(stream)

    def elapsedTime(self) -> None:
        """
        Record elapsedTime between start and end events.
        Must be called after syncrhonization ensuring completion of the start and end recording
        """
        _elapsedTimeNS = self.start_event.elapsed_time(self.end_event) * 1e6
        self.elapsedTimeNS += _elapsedTimeNS  # torch elapsed_time is in MS


class bootstrap_info_holder:
    """Class holding communication-world related parameters."""

    def __init__(
        self,
        master_ip: str,
        master_port: str,
        num_tpu_cores: int,
        comms_env_params: dict[str, int],
    ) -> None:
        self.global_rank = comms_env_params["global_rank"]
        self.local_rank = comms_env_params["local_rank"]
        self.local_size = comms_env_params["local_size"]
        self.world_size = comms_env_params["world_size"]

        self.master_ip = master_ip
        self.master_port = master_port
        self.num_tpu_cores = num_tpu_cores


class commsParamsHolderBase:
    """Class holding object for common input parameters"""

    def __init__(self, args: Namespace) -> None:
        self.nw_stack = args.nw_stack
        self.dtype = args.dtype
        self.backend = args.backend
        self.device = args.device
        self.blockingFlag = args.blocking
        # quantization
        self.bitwidth = args.bitwidth
        self.quant_a2a_embedding_dim = args.quant_a2a_embedding_dim
        self.quant_threshold = args.quant_threshold
        self.dcheck = args.c
        self.groupRanks = {}  # record what ranks each process group will work on {pg_id, ranks}
        self.pgsDesc = {}  # {pg_id: pg_desc}
        self.use_ext_dist = args.use_ext_dist
        self.size_from_trace = False
        self.init_method = args.init_method
        self.enable_local_report = args.enable_local_report
        self.enable_profiler = args.enable_profiler
        self.use_perf_logger = args.use_perf_logger
        self.init_only = args.init_only


class commsParamsHolder(commsParamsHolderBase):
    """Class holding object for the input parameters from collective benchmark."""

    def __init__(
        self,
        args,
        bootstrap_info: bootstrap_info_holder,
        element_size: int,
        benchTime: Callable,
    ) -> None:
        super().__init__(args)

        self.element_size = element_size
        self.sizes = args.ss
        self.beginSize = args.b
        self.endSize = args.e
        self.maxSize = int(args.e // self.element_size)
        self.inSplit = args.i
        self.outSplit = args.o
        self.data_type = args.data_type
        self.stepFactor = args.f
        self.stepBytes = args.sb
        self.srcOrDst = args.root
        self.quant_threshold = max(
            self.endSize, self.quant_threshold
        )  # use quantization for all sizes in collective benchmark

        self.numWarmupIters = args.w
        self.numIters = args.n
        self.collective = args.collective
        self.collective_list = args.collective.split(",")
        self.mode = args.mode

        self.kernel = args.kernel
        self.num_compute = args.num_compute
        self.num_coll = args.num_coll
        self.mm_dim = args.mm_dim
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.num_embs = args.num_embs
        self.num_emb_tables_per_device = args.num_emb_tables_per_device
        self.num_emb_tables_batched = args.num_emb_tables_batched
        self.bag_size = args.bag_size

        self.pair = args.pair
        self.overlap_pair_pgs = args.overlap_pair_pgs
        self.collective_pair = args.collective_pair
        self.multi_comms = args.multi_comms

        self.pt2pt = args.pt2pt
        self.window = args.window

        self.src_ranks = args.src_ranks
        self.dst_ranks = args.dst_ranks
        self.bootstrap_info = bootstrap_info

        self.size_start_profiler = args.size_start_profiler


class paramCommsBench(ABC):
    """Abstract class for any param comms benchmark."""

    def __init__(self, supportedNwstacks: list[str]) -> None:
        self.supportedNwstacks = supportedNwstacks
        self.supported_tpu_core_valuses = [1, 8]
        self.dtypeMap = {
            "float": torch.float32,
            "float32": torch.float32,
            "float16": torch.half,
            "float64": torch.double,
            "double": torch.double,
            "int32": torch.int32,
            "int": torch.int32,
            "long": torch.long,
            "bfloat16": torch.bfloat16,
            "bool": torch.bool,
            "half": torch.half,
            "byte": torch.uint8,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "short": torch.short,
            "char": torch.int8,
            "signed char": torch.int8,
            "unsigned char": torch.uint8,
        }
        self.dtypeSizeMap = {
            k: torch.tensor([], dtype=v).element_size()
            for k, v in self.dtypeMap.items()
        }
        self.supportedDtype = list(self.dtypeMap.keys())
        self.backendFuncs: BaseBackend

        self.collectiveArgs = collectiveArgsHolder()
        self.comm_size = 1
        self.global_rank = -1
        # update initVal to test different value
        self.initVal = 1
        self.report = False

    def isCudaAvail(self) -> bool:
        return torch.cuda.is_available()

    def dcheck(
        self, commsParams: commsParamsHolderBase, curSize: int, tensor: torch.Tensor
    ) -> None:
        """ "
        Data validaton check for collectives, will raise an exception if invalid.

        Args:
            commsParams: Contains collective information.
            curSize: Current size in bytes.
            tensor: Tensor to validate.
        Returns:
            None
        """
        expRes = self.initVal
        if (
            commsParams.collective
            in (
                "all_reduce",
                "reduce_scatter",
                "reduce_scatter_base",
            )
        ) or (
            self.backendFuncs.get_global_rank() == commsParams.srcOrDst
            and commsParams.collective == "reduce"
        ):
            # NOTE: for sum op. and the inital value is "self.initVal", for boolean type, self.initVal is always True
            expRes = (
                self.initVal
                if tensor.dtype == torch.bool
                else self.collectiveArgs.world_size * self.initVal
            )

        if (
            commsParams.collective in ("reduce", "gather")
            and self.backendFuncs.get_global_rank() != commsParams.srcOrDst
        ) or (
            commsParams.collective in ("pt2pt",)
            and self.backendFuncs.get_global_rank() not in commsParams.dst_ranks
        ):
            return

        if isinstance(tensor, list):
            # for allgather, it's a list of tensors:
            for rank, t in enumerate(tensor):
                if not torch.all(torch.eq(t, expRes)):
                    for index, val in enumerate(t):
                        if val != expRes:
                            raise ValueError(
                                f"[{curSize}-bytes {commsParams.collective}] Wrong value at [{rank}][{index}] = {t[index]}, expected {expRes}\n {tensor}"
                            )
        else:
            if not torch.all(torch.eq(tensor, expRes)):
                for index, val in enumerate(tensor):
                    if val != expRes:
                        raise ValueError(
                            f"[{curSize}-bytes {commsParams.collective}] Wrong value at [{index}] = {tensor[index]}, expected {expRes}\n {tensor}"
                        )

    def setTensorVal(self, tensor: torch.Tensor, useRandVal: bool = True) -> None:
        """
        Set tensor value, use initVal if useRandVal is false.

        Args:
            tensor: Tensor to set value on.
            useRandVal: Determines whether to use predictable values or not.
        Returns:
            None
        """
        newVal = random.random() if useRandVal else self.initVal
        t = tensor[0] if isinstance(tensor, list) else tensor
        if t.type == torch.bool:
            newVal = newVal > 0.5
        # reset values
        if self.collectiveArgs.collective in ("all_reduce", "reduce"):
            # all processes use initVal to have predictable results
            newVal = self.initVal
        elif self.collectiveArgs.collective in ("broadcast",):
            # root process uses initVal and others use random values
            newVal = (
                self.initVal
                if (self.backendFuncs.get_global_rank() == self.collectiveArgs.srcOrDst)
                else newVal
            )

        # reset the tensor(s)
        if isinstance(tensor, list):
            # could be a list of tensor, for all_gather, gather, reduce_scatter
            for t in tensor:
                t.fill_(newVal)
        else:
            tensor.fill_(newVal)

    # Collection of prepComm private methods. These methods prepare tensors for the respective collective.

    def _prep_all_to_allv(
        self,
        ipTensor: torch.Tensor,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the all_to_allv mode"""

        opTensor = torch.Tensor()
        if allocate:
            # all_to_allv requires two tensors
            # ipTensor has been allocated outside of this function, just pass in
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut], curDevice, dtype, scaleFactor
            )
        # recorded splits in trace is only for dim 0, but tensor in replay has been flattened.
        # need to recalculate the splits for flattened 1D tensor
        # corner case: one rank sends zeor data out, but receives data from other ranks, and vice versa.
        self.collectiveArgs.opTensor_split = (
            [
                numElementsOut // max(sum(curComm.outSplit), 1) * i
                for i in curComm.outSplit
            ]
            if curComm.outSplit
            else None
        )
        self.collectiveArgs.ipTensor_split = (
            [numElementsIn // max(sum(curComm.inSplit), 1) * i for i in curComm.inSplit]
            if curComm.inSplit
            else None
        )
        return (ipTensor, opTensor)

    def _prep_all_to_all(
        self,
        ipTensor: list[torch.Tensor],
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        ipTensor = []
        opTensor = []
        if allocate:
            i_alloc_func = (
                self.backendFuncs.alloc_ones
                if commsParams.dcheck == 1
                else self.backendFuncs.alloc_random
            )
            i_scale_factor = self.initVal if commsParams.dcheck == 1 else scaleFactor
            ipTensor = [
                i_alloc_func([i], curDevice, commsParams.dtype, i_scale_factor)
                for i in curComm.inSplit
            ]

            opTensor = [
                self.backendFuncs.alloc_random(
                    [i], curDevice, commsParams.dtype, scaleFactor
                )
                for i in curComm.outSplit
            ]
        return (ipTensor, opTensor)

    def _prep_all_gather(
        self,
        ipTensor: torch.Tensor,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        opTensor = []

        if not commsParams.size_from_trace:
            numElementsIn = numElementsIn // world_size

        if allocate:
            if commsParams.dcheck == 1:
                ipTensor = self.backendFuncs.alloc_ones(
                    [numElementsIn],
                    curDevice,
                    dtype,
                    scaleFactor=self.initVal,
                )
            else:
                ipTensor = self.backendFuncs.alloc_random(
                    [numElementsIn], curDevice, dtype, scaleFactor
                )
            # allgather requires a tensor list, e.g., List[torch.Tensor]
            for _ in range(world_size):
                opTensor.append(
                    self.backendFuncs.alloc_random(
                        [numElementsIn], curDevice, dtype, scaleFactor
                    )
                )
        return (ipTensor, opTensor)

    def _prep_all_gather_base(
        self,
        ipTensor: torch.Tensor,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        opTensor = torch.Tensor()
        if not commsParams.size_from_trace:
            numElementsOut = numElementsIn
            numElementsIn = numElementsIn // world_size
        if allocate:
            if commsParams.dcheck == 1:
                ipTensor = self.backendFuncs.alloc_ones(
                    [numElementsIn],
                    curDevice,
                    dtype,
                    scaleFactor=self.initVal,
                )
            else:
                ipTensor = self.backendFuncs.alloc_random(
                    [numElementsIn], curDevice, dtype, scaleFactor
                )
            # this is a single all gather with flat output tensor
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut],
                curDevice,
                dtype,
                scaleFactor,
            )
        return (ipTensor, opTensor)

    def _prep_reduce_scatter(
        self,
        ipTensor: list[torch.Tensor],
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        ipTensor = []
        opTensor = torch.Tensor()
        if not commsParams.size_from_trace:
            numElementsIn = numElementsOut // world_size
            numElementsOut = numElementsOut // world_size
        else:
            numElementsIn = numElementsIn // world_size
        if allocate:
            if commsParams.dcheck == 1:
                for _ in range(world_size):
                    ipTensor.append(
                        self.backendFuncs.alloc_ones(
                            [numElementsIn],
                            curDevice,
                            commsParams.dtype,
                            self.initVal,
                        )
                    )
            else:
                for _ in range(world_size):
                    ipTensor.append(
                        self.backendFuncs.alloc_random(
                            [numElementsIn],
                            curDevice,
                            commsParams.dtype,
                            scaleFactor,
                        )
                    )
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut], curDevice, dtype, scaleFactor
            )
        return (ipTensor, opTensor)

    def _prep_reduce_scatter_base(
        self,
        ipTensor: torch.Tensor,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ipTensor = torch.Tensor()
        opTensor = torch.Tensor()
        if not commsParams.size_from_trace:
            numElementsIn = numElementsOut
            numElementsOut = numElementsOut // world_size
        if allocate:
            if commsParams.dcheck == 1:
                ipTensor = self.backendFuncs.alloc_ones(
                    [numElementsIn],
                    curDevice,
                    commsParams.dtype,
                    self.initVal,
                )
            else:
                ipTensor = self.backendFuncs.alloc_random(
                    [numElementsIn],
                    curDevice,
                    commsParams.dtype,
                    scaleFactor,
                )
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut], curDevice, dtype, scaleFactor
            )
        return (ipTensor, opTensor)

    def _prep_pt2pt(
        self,
        ipTensor: torch.Tensor,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        numElementsIn: int,
        numElementsOut: int,
        world_size: int,
        curDevice: str,
        dtype: torch.dtype,
        scaleFactor: float,
        allocate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pt2pt or out-of-place collectives
        opTensor = torch.Tensor()
        if allocate:
            opTensor = self.backendFuncs.alloc_random(
                [numElementsOut],
                curDevice,
                dtype,
                scaleFactor,
            )
        return (ipTensor, opTensor)

    def prepGemmNotSquare(
        self,
        mm0_dim0: int,
        mm0_dim1: int,
        mm1_dim0: int,
        mm1_dim1: int,
        dtype: torch.dtype,
        curDevice: str,
        gemmTensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gemmTensor is None:
            in1 = np.random.rand(mm0_dim0, mm0_dim1)
            in2 = np.random.rand(mm1_dim0, mm1_dim1)

            MMin1 = torch.FloatTensor(in1).to(curDevice)
            MMin2 = torch.FloatTensor(in2).to(curDevice)
            MMout = self.backendFuncs.alloc_empty(
                [mm0_dim0, mm1_dim1], dtype, curDevice
            )
        else:
            mm_size0 = mm0_dim0 * mm0_dim1
            mm_size1 = mm1_dim0 * mm1_dim1
            out_size = mm0_dim0 * mm1_dim1
            MMin1 = gemmTensor[0:mm_size0].view((mm0_dim0, mm0_dim1))
            MMin2 = gemmTensor[mm_size0 : mm_size0 + mm_size1].view(
                (mm1_dim0, mm1_dim1)
            )
            MMout = gemmTensor[
                mm_size0 + mm_size1 : mm_size0 + mm_size1 + out_size
            ].view((mm0_dim0, mm1_dim1))

        return MMout, MMin1, MMin2

    # Prepare generic compute operations that uses 1 or 2 input tensors, and 1 output tensor
    def prepComp(
        self, mm_dim: int, dtype: torch.dtype, curDevice: str, kernel: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        compIn1 = self.backendFuncs.alloc_random([mm_dim, mm_dim], curDevice, dtype)
        compOut = self.backendFuncs.alloc_empty([mm_dim, mm_dim], dtype, curDevice)
        compIn2 = None
        if kernel in ["add", "sub"]:
            compIn2 = self.backendFuncs.alloc_random([mm_dim, mm_dim], curDevice, dtype)
        return (compOut, compIn1, compIn2)

    def prepGemm(
        self, mm_dim: int, dtype: torch.dtype, curDevice: str, gemmTensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.prepGemmNotSquare(
            mm_dim, mm_dim, mm_dim, mm_dim, dtype, curDevice, gemmTensor
        )

    def prepComm(
        self,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        allocate: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | torch.Tensor]:
        """
        Allocate the tensors for collective.

        Args:
            curComm: Current collective communication.
            commsParams: Holds parameters that affect tensor allocation.
        Returns:
            (iptensor, optensor): Appropriate input and output tensors for collective.
        """
        commOp = paramToCommName(
            curComm.comms if (curComm.comms is not None) else commsParams.collective,
            supported_comms=self.backendFuncs.collectiveFunc.keys(),
        )

        if commOp in ("wait", "barrier"):
            return (torch.Tensor(), torch.Tensor())

        numElementsIn = curComm.inMsgSize
        # numElementsOut is only meaningful for out-of-place collectives and pt2pt
        numElementsOut = curComm.outMsgSize
        world_size = self.collectiveArgs.world_size
        dtype = commsParams.dtype
        curDevice = commsParams.device
        # seed to generate random value; let's use a small value to avoid potential "overflow when unpacking long"
        scaleFactor = world_size
        opTensor = torch.Tensor()

        if allocate:
            if commsParams.dcheck == 1:
                # use predictable values for data validation check
                ipTensor = self.backendFuncs.alloc_ones(
                    [numElementsIn], curDevice, dtype, scaleFactor=self.initVal
                )
            else:
                ipTensor = self.backendFuncs.alloc_random(
                    [numElementsIn], curDevice, dtype, scaleFactor
                )
        else:
            ipTensor = torch.Tensor()
        # TODO: consider using this dictionary to check valid keywords rather than silently defaulting

        dispatchDict = {
            "all_to_allv": self._prep_all_to_allv,
            "all_to_all": self._prep_all_to_all,
            "all_gather": self._prep_all_gather,
            "gather": self._prep_all_gather,
            "all_gather_base": self._prep_all_gather_base,
            "reduce_scatter": self._prep_reduce_scatter,
            "reduce_scatter_base": self._prep_reduce_scatter_base,
            "scatter": self._prep_reduce_scatter,
            "pt2pt": self._prep_pt2pt,
        }

        function_to_call = dispatchDict.get(commOp)
        if function_to_call is not None:
            ipTensor, opTensor = function_to_call(
                ipTensor,
                curComm,
                commsParams,
                numElementsIn,
                numElementsOut,
                world_size,
                curDevice,
                dtype,
                scaleFactor,
                allocate,
            )
        else:
            # in-place case for other collectives such as allreduce, reduce, broadcast
            opTensor = ipTensor

        return (ipTensor, opTensor)

    @abstractmethod
    def runBench(self, commsParams: commsParamsHolderBase) -> None:
        """Must override to start the desired benchmarking"""
        pass

    @abstractmethod
    def benchTime(self, commsParams: commsParamsHolderBase) -> None:
        """Must override to run the desired benchmarking"""
        pass

    @abstractmethod
    def reportBenchTime(self, *args, **kwargs) -> None:
        """Must override to report/print the desired output"""
        pass

    @abstractmethod
    def readArgs(self, parser: ArgumentParser) -> argparse.Namespace:
        """Basic/Common arguments for all PARAM-Comm benchmarks"""
        parser.add_argument(
            "--master-ip",
            type=str,
            default=(
                default_master_ip
                if "MASTER_ADDR" not in os.environ
                else os.environ["MASTER_ADDR"]
            ),
            help="The master-IP to coordinate for Pytorch distributed stack",
        )  # The master-IP to coordinate.
        parser.add_argument(
            "--master-port",
            type=str,
            default=(
                default_master_port
                if "MASTER_PORT" not in os.environ
                else os.environ["MASTER_PORT"]
            ),
            help="The master-port to coordinate for Pytorch distributed stack",
        )  # The master-port to coordinate.
        parser.add_argument(
            "--nw-stack",
            type=str,
            default="pytorch-dist",
            help="network stack to be used, supports " + str(self.supportedNwstacks),
        )  # The network stack to profile.
        parser.add_argument(
            "--dtype", type=torch.dtype, default=torch.float32
        )  # will be overwritten based on args.data_types and dtypeMap.
        parser.add_argument(
            "--num-tpu-cores",
            type=int,
            default=1,
            help="number of TPU cores to be used",
        )  # number of TPU cores
        parser.add_argument(
            "--log",
            "--log-level",
            type=str,
            default="ERROR",
            help="Logging level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        )  # logging level
        parser.add_argument(
            "--device",
            type=str,
            default=("cuda" if self.isCudaAvail() else "cpu"),
            choices=supportedDevices,
            help="data placement",
        )  # device to place data for collective benchmarking
        parser.add_argument(
            "--backend",
            type=str,
            default=("nccl" if self.isCudaAvail() else "gloo"),
            choices=supportedC10dBackends + list(customized_backend.keys()),
            help="The backend to be used in PyTorch distributed process group",
        )  #  backend used for the network stack
        parser.add_argument(
            "-b",
            "--blocking",
            action="store_true",
            help="use blocking/non-blocking mode for collectives",
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
        parser.add_argument(
            "--use-ext-dist",
            "--use-ext-pg",
            action="store_true",
            default=False,
            help="use extend_distributed wrapper",
        )  # use extend_distributed wrapper to init and create PGs
        parser.add_argument(
            "--init-method",
            "--pg-init-method",
            type=str,
            default=None,
            help="URL specifying how to initialize the process group. See https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group",
        )  # URL specifying how to initialize the process group. See https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
        parser.add_argument(
            "--enable-local-report",
            action="store_true",
            default=False,
            help="Toggle to enable all nodes' local rank report the output",
        )  # let all localRank-0 report the output
        parser.add_argument(
            "--enable-profiler",
            action="store_true",
            default=False,
            help="toggle to enable pytorch profiler",
        )  # enable pytorch profiler
        parser.add_argument(
            "--use-perf-logger",
            "--use-custom-perf-logger",
            nargs="+",
            type=str,
            default=None,
            help="add name of custom performer loggers to use them in additional to text output, user is responsible to implement and register the custom performance logger",
        )  # use custom performer logger
        parser.add_argument(
            "--init-only",
            action="store_true",
            default=False,
            help="Toggle to skip running collectives and only do initalization",
        )
        pass

    @abstractmethod
    def checkArgs(self, args: Namespace) -> None:
        """Validate some basic/common arguments for all PARAM-Comm benchmarks"""
        if args.nw_stack not in self.supportedNwstacks:
            logger.error(
                f"Specified backend: {args.nw_stack} is not one of the supported backends: {str(self.supportedNwstacks)}. Make sure the input is using the correct case."
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
        # overwrite existing logging config. Require Python 3.8+
        logging.basicConfig(
            level=numeric_level,
            format="[%(asctime)s][%(name)s][%(levelname)s][Rank{:3}] - %(message)s".format(
                comms_env_params["global_rank"]
            ),
            force=True,
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


def init_emb_lookup(collectiveArgs, commsParams, backendFuncs):
    """
    Initialize embedding table op

    Args:
        collectiveArgs: collective arguments.
        commsParams: Holds parameters that affect tensor allocation.
        backendFuncs: backend function
    Returns:
        None
    """
    try:
        # fbgemm_gpu can be downloaded from https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu
        from fbgemm_gpu.split_embedding_utils import generate_requests
        from fbgemm_gpu.split_table_batched_embeddings_ops import (
            ComputeDevice,
            EmbeddingLocation,
            OptimType,
            SplitTableBatchedEmbeddingBagsCodegen,
        )
    except ImportError:
        logger.error("benchmarking with emb_lookup kernels requires fbgemm_gpu library")
        return
    collectiveArgs.direction = commsParams.direction
    collectiveArgs.emb_dim = commsParams.emb_dim
    num_embeddings = commsParams.num_embs
    collectiveArgs.batch_size = commsParams.batch_size
    num_tables_per_device = commsParams.num_emb_tables_per_device
    collectiveArgs.num_emb_tables_batched = commsParams.num_emb_tables_batched
    bag_size = commsParams.bag_size

    num_emb_tables_batched = (
        num_tables_per_device
        if collectiveArgs.num_emb_tables_batched == -1
        else collectiveArgs.num_emb_tables_batched
    )
    collectiveArgs.num_emb_ops = num_tables_per_device // num_emb_tables_batched

    collectiveArgs.emb = [
        SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    num_embeddings,
                    collectiveArgs.emb_dim,
                    (
                        EmbeddingLocation.DEVICE
                        if commsParams.device == "cuda"
                        else EmbeddingLocation.HOST
                    ),
                    (
                        ComputeDevice.CUDA
                        if commsParams.device == "cuda"
                        else ComputeDevice.CPU
                    ),
                )
                for _ in range(num_emb_tables_batched)
            ],
            device=backendFuncs.get_device(),
            optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        )
        for _ in range(collectiveArgs.num_emb_ops)
    ]

    collectiveArgs.embRequests = generate_requests(
        iters=collectiveArgs.num_emb_ops,
        B=collectiveArgs.batch_size,
        T=num_emb_tables_batched,
        L=bag_size,
        E=num_embeddings,
    )

    # If we are doing backward pass, then we need to initialize Lookup tensor using forward pass and grad output
    if collectiveArgs.direction == "backward":
        for i in range(len(collectiveArgs.embRequests)):
            (indices, offsets, weights) = collectiveArgs.embRequests[i].unpack_3()
            collectiveArgs.LookupOut = collectiveArgs.emb[i].forward(
                indices,
                offsets,
                weights,
            )

        collectiveArgs.grad_output = torch.rand_like(collectiveArgs.LookupOut).to(
            collectiveArgs.device
        )
