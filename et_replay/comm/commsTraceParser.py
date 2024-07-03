# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import json
from typing import List, Tuple

from et_replay import ExecutionTrace
from et_replay.comm import comms_utils
from et_replay.comm.comms_utils import commsArgs
from et_replay.comm.backend.base_backend import supportedP2pOps

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
    ET_PG_NAME_TUPLE = in_trace.schema_pytorch() >= (1, 0, 3)
    if (in_trace.schema_pytorch() < (1, 0, 3)):
        raise ValueError(f"Only support trace version >1.0.3, but current trace version is {in_trace.schema.split('-')[0]}")

    pg_ranks_map = _parse_proc_group_info(in_trace) # key is pg id, value is global ranks in this pg
    comms_op_list = _parse_comms_op_node(in_trace, pg_ranks_map, target_rank, total_ranks)

    return comms_op_list

def _parse_proc_group_info(in_trace: ExecutionTrace):
    pg_ranks_map = {}
    for node in in_trace.nodes.values():
        if "process_group:init" in node.name:
            # info of this node is dumped using torch.distributed.distributed_c10d._world.pg_config_info
            # at the start of profiling, but not not callback to torch.distributed.init_process_group()
            # Pre-Assumption: all process groups has been created before profiling start.
            try:
                pg_objs = json.loads(node.inputs[0])
            except json.decoder.JSONDecodeError:  # skip if pg_config_info is truncated
                break

            for pg in pg_objs:
                if not pg["pg_name"].isdecimal():
                    # TODO support local synchronization pg
                    raise ValueError(f"Process group name is {pg['pg_name']} in node {node['id']}, which is not supported.")
                (pg_id, ranks, group_size, group_count) = [pg[k] for k in ["pg_name", "ranks", "group_size", "group_count"]]
                pg_id = int(pg_id)
                pg_ranks_map[pg_id] = (
                    ranks
                    if len(ranks) > 0
                    else list(range(group_size))
                    # rank list is empty when all ranks are in a pg
                )
            break  # only one process_group init node per trace
    return pg_ranks_map

def _parse_comms_op_node(in_trace: ExecutionTrace, pg_ranks_map: dict, target_rank: int, total_ranks: int):
    comms_op_list = []

    for pg_id, ranks in pg_ranks_map.items():
        comm_args = _create_pg_init_node(pg_id, ranks, len(ranks))
        comms_op_list.append(comm_args)

    for node in in_trace.nodes.values():
        if node.name == "record_param_comms":
            # according to macro RECORD_PARAM_COMMS and RECORD_PARAM_COMMS_DATA in torch/csrc/distributed/c10d/ParamCommsUtils.hpp
            # ["wait", "barrier", "init"] record 1st element as seq, others record starting from input tensor
            index_base = 0 if isinstance(node.inputs[0], int) else 1
            (req_id, pg_id_pair, recorded_rank, comm_name) = [node.inputs[index_base + i] for i in range(4)]
            comm_args = commsArgs()
            comm_args.id = node.id
            comm_args.comms = comms_utils.paramToCommName(comm_name.lower())
            if comm_args.comms == "init":
                # init node has been built
                continue
            comm_args.req = req_id

            if pg_id_pair[0].isdecimal():
                comm_args.pgId = int(pg_id_pair[0])
                comm_args.groupRanks = pg_ranks_map[comm_args.pgId]
                comm_args.worldSize = len(comm_args.groupRanks)

            if comm_args.comms not in ("wait", "barrier"):
                (comm_args.inMsgSize, in_msg_type) = _getTensorInfoFromPyTorchETEntry(node.inputs, node.input_types[0])
                (comm_args.outMsgSize, _) = _getTensorInfoFromPyTorchETEntry(node.outputs, node.output_types[0])
                comm_args.dtype = tensorDtypeMap[in_msg_type]  # 1st value of input_types is the data type for the tensors

            if comm_args.comms in supportedP2pOps:
                if "send" in comm_args.comms:
                    (comm_args.src_rank, comm_args.dst_rank) = (target_rank, recorded_rank)
                elif "recv" in comm_args.comms:
                    (comm_args.src_rank, comm_args.dst_rank) = (recorded_rank, target_rank)

            if comm_args.comms in ["reduce", "broadcast", "gather", "scatter"]:
                comm_args.root = recorded_rank
                comm_args.groupRanks = comm_args.groupRanks

            if comm_args.comms == "all_to_allv":
                # 6th value of inputs is in_split, split evenly if not provided
                if not comm_args.worldSize:
                    # if no pg info provided, use total ranks as world size
                    comm_args.worldSize = total_ranks
                comm_args.inSplit = (
                    node.inputs[5]
                    if node.inputs[5]
                    else [int(comm_args.inMsgSize / comm_args.worldSize)]
                    * comm_args.worldSize
                )
                # 7th value of inputs is out_split, split evenly if not provided
                comm_args.outSplit = (
                    node.inputs[6]
                    if node.inputs[6]
                    else [int(comm_args.outMsgSize / comm_args.worldSize)]
                    * comm_args.worldSize
                )
            comms_op_list.append(comm_args)

    return comms_op_list

def _create_pg_init_node(pg_id: int, ranks: List[int], world_size: int):
    comm_args = commsArgs()
    comm_args.comms = "init"
    comm_args.pgId = pg_id
    comm_args.req = -1
    comm_args.groupRanks = ranks
    comm_args.worldSize = world_size
    return comm_args
