# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import json
from typing import List, Tuple

from et_replay import ExecutionTrace
from et_replay.comm import comms_utils
from et_replay.comm.comms_utils import commsArgs
from et_replay.comm.pytorch_backend_utils import supportedP2pOps

tensorDtypeMap = {
    "Tensor(int)": "int",
    "Tensor(float)": "float",
    "Tensor(bool)": "bool",
    "Tensor(long)": "long",
    "Tensor(long int)": "long",
    "Tensor(double)": "double",
    "Tensor(half)": "half",
    "Tensor(byte)": "byte",
    "Tensor(c10::Half)": "half",
    "Tensor(c10::BFloat16)": "bfloat16",
    "Tensor(unsigned char)": "char",
    "Tensor(signed char)": "char",
}


def parseTrace(
    in_trace: List, trace_type: str, trace_file_path: str, target_rank: int, total_ranks: int
) -> List:
    """
    Parse trace files to be compatible with PARAM replay-mode.
    Currently supports: Chakra host execution trace.

    Args:
        in_trace: Trace file to be parsed.
        trace_type: Trace type to be parsed with
        trace_file_path: Path of input trace file being loaded.
        target_rank: The current rank of the device.
        total_ranks: Total number of ranks.
    Returns:
        parsed_trace: Parsed trace that is compatible with PARAM replay-mode.
    """

    if trace_type == "et":  # Execution Trace (e.g. Chakra host execution trace)
        parsed_trace = _parseExecutionTrace(
            ExecutionTrace(in_trace), target_rank, total_ranks
        )
    else:
        raise ValueError(f"Specified trace type {trace_type} to {trace_file_path} is not supported. \
Please check supported types with '--help'")

    return parsed_trace


def _getTensorInfoFromPyTorchETEntry(
    tensor_container: List, container_type: str
) -> Tuple[int, int, str]:
    """
    Extract message size, tensor count, type from PyTorch ET entry inputs/outputs field.
    NOTE: This format can be changed at anytime. TODO: When an extract/parsing tool is available in ATC, switch to it.
    """
    list_count = container_type.count("GenericList")
    tensors = []
    if list_count == 2:
        # GenericList[GenericList[Tensor(), Tensor()]]
        tensors = tensor_container[0][0]
        dtype = container_type.replace("GenericList[", "").split(",", 1)[0]
    elif list_count == 1:
        # GenericList[Tensor()]
        tensors = tensor_container[0]
        dtype = container_type.replace("GenericList[", "").replace("]", "")
    else:
        tensors.append(tensor_container[0])
        dtype = container_type

    msg_size = 0
    for tensor in tensors:
        msg_size += tensor[3]

    return msg_size, dtype


def _parseExecutionTrace(
    in_trace: ExecutionTrace, target_rank: int, total_ranks: int
) -> List:
    """
    Convert the Execution Trace comms metadata to the common trace format for replay.
    """
    # Execution Trace PG_ID types availability
    ET_PG_NAME_TUPLE = in_trace.schema_pytorch() >= (1, 0, 3)
    ET_BACKENDID = in_trace.schema_pytorch() < (1, 0, 3)

    initOps = []
    newCommsTrace = []
    backendIdToPgid = {}
    pgRanksMap = {}
    groupCnt = -1

    # Parse PG info from ET
    for node in in_trace.nodes.values():
        if "process_group:init" in node.name:
            pgJson = node.inputs[0]
            try:
                pgObj = json.loads(pgJson)
            except json.decoder.JSONDecodeError:  # skip if pg_config_info is truncated
                break

            for pg in pgObj:
                if not pg["pg_name"].isdecimal():
                    # TODO support local synchronization pg
                    continue
                pgId = int(pg["pg_name"])
                ranks = pg["ranks"]
                groupCnt = pg["group_count"]
                pgRanksMap[pgId] = (
                    ranks
                    if len(ranks) > 0
                    else list(range(pg["group_size"]))
                    # rank list is empty when all ranks are in a pg
                )
                if ET_BACKENDID:
                    backendId = pg["uid"] if "uid" in pg else pg["backend_id"]
                    backendIdToPgid[backendId] = pgId
            break  # only one process_group init node per trace

    # Parse comms nodes
    for node in in_trace.nodes.values():
        if node.name == "record_param_comms":
            # wait/barrier ops do not have an input tensor which is the 1st element in other ops
            # 1 - shift: req
            # 2 - shift: pgId
            # 3 - shift: local_src/dst_rank
            # 4 - shift: comm
            shift = 1 if any(k in node.inputs[0:5] for k in ["wait", "barrier"]) else 0
            newComm = commsArgs()
            newComm.id = node.id
            newComm.comms = comms_utils.paramToCommName(
                node.inputs[4 - shift].lower()
            )  # 5th value of inputs is colName
            if newComm.comms == "init":
                continue
            newComm.req = node.inputs[
                1 - shift
            ]  # 2nd value of inputs is the req id of the collective

            pgIdentifier = node.inputs[
                2 - shift
            ]  # 3rd value of inputs is the pg identifier of the collective
            # Assign pg_id info for PGs that were created.
            if ET_BACKENDID and pgIdentifier in backendIdToPgid:
                newComm.pgId = backendIdToPgid[pgIdentifier]
                newComm.groupRanks = pgRanksMap[newComm.pgId]
                newComm.worldSize = len(newComm.groupRanks)
            elif ET_PG_NAME_TUPLE and pgIdentifier[0].isdecimal():
                newComm.pgId = int(pgIdentifier[0])
                newComm.groupRanks = pgRanksMap[newComm.pgId]
                newComm.worldSize = len(newComm.groupRanks)

            if newComm.comms not in ("wait", "barrier"):
                (
                    newComm.inMsgSize,
                    inMsgType,
                ) = _getTensorInfoFromPyTorchETEntry(node.inputs, node.input_types[0])
                (
                    newComm.outMsgSize,
                    _,
                ) = _getTensorInfoFromPyTorchETEntry(node.outputs, node.output_types[0])
                newComm.dtype = tensorDtypeMap[
                    inMsgType
                ]  # 1st value of input_types is the data type for the tensors

            if newComm.comms in supportedP2pOps:
                if "send" in newComm.comms:
                    newComm.src_rank = target_rank
                    local_dst_rank = node.inputs[3 - shift]
                    newComm.dst_rank = newComm.groupRanks[local_dst_rank]
                if "recv" in newComm.comms:
                    local_src_rank = node.inputs[3 - shift]
                    newComm.src_rank = newComm.groupRanks[local_src_rank]
                    newComm.dst_rank = target_rank

            if newComm.comms == "broadcast":
                newComm.root = newComm.groupRanks[0]
                newComm.srcOrDst = newComm.groupRanks[0]

            if newComm.comms == "all_to_allv":
                # 6th value of inputs is in_split, split evenly if not provided
                if not newComm.worldSize:
                    # if no pg info provided, use total ranks as world size
                    newComm.worldSize = total_ranks
                newComm.inSplit = (
                    node.inputs[5]
                    if node.inputs[5]
                    else [int(newComm.inMsgSize / newComm.worldSize)]
                    * newComm.worldSize
                )
                # 7th value of inputs is out_split, split evenly if not provided
                newComm.outSplit = (
                    node.inputs[6]
                    if node.inputs[6]
                    else [int(newComm.outMsgSize / newComm.worldSize)]
                    * newComm.worldSize
                )
            newCommsTrace.append(newComm)

    # Build init node
    initOps = []
    if groupCnt < 0:
        # old format: To be removed
        for pgId, ranks in pgRanksMap.items():
            newComm = create_pg_init_node(pgId, ranks, len(ranks))
            initOps.append(newComm)
    else:
        for pgId in range(groupCnt):
            if pgId in pgRanksMap:
                ranks = pgRanksMap[pgId]
            else:
                # create a dummy pg that the current rank is not part of
                ranks = [0] if target_rank != 0 else [1]

            newComm = create_pg_init_node(pgId, ranks, len(ranks))
            initOps.append(newComm)

    return initOps + newCommsTrace


def create_pg_init_node(pg_id: int, ranks: List[int], world_size: int):
    newComm = commsArgs()
    newComm.comms = "init"
    newComm.pgId = pg_id
    newComm.req = -1
    newComm.groupRanks = ranks
    newComm.worldSize = world_size
    return newComm