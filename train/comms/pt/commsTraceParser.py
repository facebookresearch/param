# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import json

from typing import List, Tuple

import comms_utils
from comms_utils import commsArgs

tensorDtypeMap = {
    "Tensor(int)": "Int",
    "Tensor(float)": "Float",
    "Tensor(bool)": "Bool",
    "Tensor(long)": "Long",
    "Tensor(long int)": "Long",
    "Tensor(double)": "Double",
    "Tensor(half)": "Half",
    "Tensor(byte)": "Byte",
    "Tensor(c10::Half)": "Half",
    "Tensor(c10::BFloat16)": "bfloat16",
}


def parseTrace(in_trace: List, target_rank: int) -> List:
    """
    Parse trace files to be compatible with PARAM replay-mode.
    Currently supports: Kineto Unitrace, UCC Trace, and PyTorch ET trace.

    Args:
        in_trace: Trace file to be parsed.
        target_rank: The current rank of the device.
    Returns:
        parsed_trace: Parsed trace that is compatible with PARAM replay-mode.
    """

    if type(in_trace) is dict and "schema" in in_trace:  # PyTorch ET trace
        in_trace = _parsePyTorchET(in_trace["nodes"])
    elif "comms" not in in_trace[0] and "ph" in in_trace[0]:  # Kineto Unitrace
        in_trace = _parseKinetoUnitrace(in_trace, target_rank)
    elif "comms" in in_trace[0]:  # Basic Trace
        in_trace = _parseBasicTrace(in_trace)
    else:
        raise ValueError("Unrecognized trace format.")

    return in_trace


def _parseBasicTrace(in_trace: List):
    """
    Convert Basic Trace to comms trace format.
    """
    newCommsTrace = []
    for cnt, curComm in enumerate(in_trace):

        newComm = commsArgs()
        newComm.id = cnt
        newComm.markerStack = curComm.get("markers")
        if "comms" in curComm:
            _parseBasicTraceComms(curComm, newComm)

        elif "compute" in curComm:
            _parseBasicTraceCompute(curComm, newComm)

        if newComm.comms is not None or newComm.compute is not None:
            newCommsTrace.append(newComm)
        else:
            raise ValueError(
                "Trace file contains an element that is not a supported in PARAM! Please format all elements as comms or compute for replay."
            )

    return newCommsTrace


def _parseBasicTraceComms(curComm, newComm: commsArgs) -> None:

    newComm.comms = comms_utils.paramToCommName(curComm["comms"].lower())
    if newComm.markerStack is None:
        newComm.markerStack = [newComm.comms]
    newComm.req = curComm.get("req")
    newComm.startTimeNs = curComm.get("startTime_ns")
    newComm.worldSize = curComm.get("world_size")
    newComm.root = curComm.get("root")
    newComm.pgId = curComm.get("pg_id")
    newComm.groupRanks = curComm.get("global_ranks")

    if newComm.comms not in ("wait", "barrier", "init"):
        newComm.inMsgSize = curComm["in_msg_size"]
        newComm.outMsgSize = curComm["out_msg_size"]
        newComm.dtype = curComm["dtype"]

    if newComm.comms == "all_to_allv":
        newComm.inSplit = curComm["in_split"]
        newComm.outSplit = curComm["out_split"]


def _parseBasicTraceCompute(curComm, newComm: commsArgs) -> None:
    newComm.compute = curComm["compute"].lower()
    if newComm.markerStack is None:
        newComm.markerStack = [newComm.compute]
    # count = number of times to call the compute kernel
    if "count" in curComm:
        newComm.count = curComm["count"]
    # if no count is specified, assume 1
    else:
        newComm.count = 1
    if newComm.compute == "gemm":
        if "mm_dim" in curComm:
            newComm.mm0_dim0 = curComm.get("mm_dim")
            newComm.mm0_dim1 = curComm.get("mm_dim")
            newComm.mm1_dim0 = curComm.get("mm_dim")
            newComm.mm1_dim1 = curComm.get("mm_dim")
        else:
            newComm.mm0_dim0 = curComm.get("mm0_dim0")
            newComm.mm0_dim1 = curComm.get("mm0_dim1")
            newComm.mm1_dim0 = curComm.get("mm1_dim0")
            newComm.mm1_dim1 = curComm.get("mm1_dim1")
        newComm.dtype = curComm.get("dtype")
    elif newComm.compute == "emb_lookup":
        if "direction" in curComm:
            newComm.direction = curComm["direction"]
        else:
            newComm.direction = "forward"
        newComm.emb_dim = curComm.get("emb_dim")
        newComm.num_embs = curComm.get("num_embs")
        newComm.batch_size = curComm.get("batch_size")
        newComm.num_emb_tables_per_device = curComm.get("num_emb_tables")
        newComm.num_emb_tables_batched = -1
        newComm.bag_size = curComm.get("bag_size")
    else:
        raise ValueError(
            f"Trace file contains {str(newComm.compute)} compute element that is not supported in PARAM!"
        )


def _parseKinetoUnitrace(in_trace: List, target_rank: int) -> List:
    """
    Convert the Kineto unitrace w/ comms metadata to the clean common trace format for replay.
    """
    newCommsTrace = []
    commsCnt = 0
    for entry in in_trace:
        # TODO: figure the current marker stack if present
        marker = "unknown"
        pass

        if (
            "name" in entry
            and entry["name"] == "record_param_comms"
            and entry["args"]["rank"] == target_rank
        ):

            newComm = commsArgs()
            newComm.comms = comms_utils.paramToCommName(entry["args"]["comms"].lower())
            newComm.id = commsCnt
            newComm.inMsgSize = entry["args"]["in_msg_size"]
            newComm.outMsgSize = entry["args"]["out_msg_size"]
            newComm.dtype = entry["args"]["dtype"]
            newComm.inSplit = entry["args"]["in_split"]
            newComm.outSplit = entry["args"]["out_split"]
            newComm.markerStack = marker

            newCommsTrace.append(newComm)
            commsCnt += 1

    return newCommsTrace


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

    return msg_size, len(tensors), dtype


def _parsePyTorchET(in_trace: List) -> List:
    """
    Convert the PyTorch ET w/ comms metadata to the clean common trace format for replay.

    NOTE: This format can be changed at anytime. When an extract/parsing tool is available in ATC, switch to it.
    """

    # first pass through to extract process group info.
    etIdToRanks = {}
    pgIdToEtId = {}

    for entry in in_trace:
        if "name" in entry and "process_group:init" in entry["name"]:
            pgInfo = entry["name"]
            pgInfo = pgInfo[
                pgInfo.find("[") : pgInfo.rfind("]") + 1
            ]  # extract [0,1,...]
            ranks = json.loads(pgInfo)  # convert str of ranks to list
            etIdToRanks[entry["id"]] = ranks

        elif "name" in entry and entry["name"] == "record_param_comms":
            if (
                len(entry["inputs"]) == 6
            ):  # this is wait, barrier, or init op since there is no input tensor
                opName = entry["inputs"][3]
                if opName == "init":
                    pgIdToEtId[entry["inputs"][1]] = entry[
                        "parent"
                    ]  # map pgId to parent function where pg was created

    newCommsTrace = []

    # Create process groups
    pgIdToRanks = {}
    for pgId, etId in pgIdToEtId.items():
        if etId in etIdToRanks:
            newComm = commsArgs()
            newComm.comms = "init"
            newComm.pgId = pgId
            newComm.groupRanks = etIdToRanks[etId]
            pgIdToRanks[pgId] = newComm.groupRanks
            newCommsTrace.append(newComm)

    commsCnt = 0
    for entry in in_trace:
        if "name" in entry and entry["name"] == "record_param_comms":

            shift = (
                0 if len(entry["inputs"]) == 7 else 1
            )  # wait and barrier ops do not have an input tensor, shift index one over

            newComm = commsArgs()
            if entry.get("id") is not None:
                newComm.id = entry["id"]
            if entry.get("eg_id") is not None:
                newComm.id = entry["eg_id"]
            if entry.get("et_id") is not None:
                newComm.id = entry["et_id"]
            newComm.comms = comms_utils.paramToCommName(
                entry["inputs"][4 - shift].lower()
            )  # 5th value of inputs is colName

            if newComm.comms == "init":
                continue  # We extracted pgs in earlier loop.

            newComm.req = entry["inputs"][
                1 - shift
            ]  # 2nd value of inputs is the req id of the collective

            if newComm.comms not in ("wait"):  # wait doesn't need pg info
                pgId = entry["inputs"][
                    2 - shift
                ]  # 3rd value of inputs is the pg id of the collective

                # Assign pgId info for PGs that were created.
                if pgId in pgIdToRanks:
                    newComm.pgId = pgId
                    newComm.worldSize = len(pgIdToRanks[pgId])

            if newComm.comms not in ("wait", "barrier"):
                (
                    newComm.inMsgSize,
                    inTensorCnt,
                    inMsgType,
                ) = _getTensorInfoFromPyTorchETEntry(
                    entry["inputs"], entry["input_types"][0]
                )
                (
                    newComm.outMsgSize,
                    outTensorCnt,
                    outMsgType,
                ) = _getTensorInfoFromPyTorchETEntry(
                    entry["outputs"], entry["output_types"][0]
                )
                newComm.dtype = tensorDtypeMap[
                    inMsgType
                ]  # 1st value of input_types is the data type for the tensors
                if not newComm.worldSize:
                    newComm.worldSize = max(inTensorCnt, outTensorCnt)

                if newComm.comms in ("all_gather", "reduce_scatter"):
                    newComm.inMsgSize = inTensorCnt
                    newComm.outMsgSize = outTensorCnt

            if newComm.comms in ("all_to_allv"):
                newComm.inSplit = entry["inputs"][5]  # 6th value of inputs is in_split
                newComm.outSplit = entry["inputs"][
                    6
                ]  # 7th value of inputs is out_split
                if not newComm.inSplit or not newComm.outSplit:
                    continue

            newCommsTrace.append(newComm)
            commsCnt += 1

    return newCommsTrace
