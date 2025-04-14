# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import json

import logging

import math

from et_replay.comm import comms_utils
from et_replay.comm.backend.base_backend import supportedP2pOps
from et_replay.comm.comms_utils import commsArgs

from et_replay.execution_trace import ExecutionTrace

logger = logging.getLogger(__name__)


def parseTrace(
    in_trace: list,
    trace_type: str,
    trace_file_path: str,
    target_rank: int,
    total_ranks: int,
) -> list:
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
        raise ValueError(
            f"Specified trace type {trace_type} to {trace_file_path} is not supported. \
Please check supported types with '--help'"
        )

    return parsed_trace


def _parseExecutionTrace(
    in_trace: ExecutionTrace, target_rank: int, total_ranks: int
) -> list:
    """
    Convert the Execution Trace comms metadata to the common trace format for replay.
    """
    if in_trace.schema_pytorch() < (1, 0, 3):
        raise ValueError(
            f"Only support trace version >1.0.3, but current trace version is {in_trace.schema.split('-')[0]}"
        )

    # pg_ranks_map: key is pg id, value is global ranks in this pg
    # pg_desc_map: key is pg id, value is pg desc
    pg_ranks_map, pg_desc_map = _parse_proc_group_info(in_trace)
    comms_op_list = _parse_comms_op_node(
        in_trace, pg_ranks_map, pg_desc_map, target_rank, total_ranks
    )

    return comms_op_list


def _parse_proc_group_info(in_trace: ExecutionTrace):
    pg_ranks_map = {}  # {node_id : {process_group_id : [ranks] } }
    pg_desc_map = {}  # {node_id : {process_group_id : pg_desc }
    pg_init_nodes = (
        node for node in in_trace.nodes.values() if "process_group:init" in node.name
    )
    for node in pg_init_nodes:
        # info of this node is dumped using torch.distributed.distributed_c10d._world.pg_config_info
        # at the start of profiling, but not callback to torch.distributed.init_process_group()
        # Pre-Assumption: all process groups has been created before profiling start.
        try:
            pg_objs = json.loads(node.inputs[0])
        except json.decoder.JSONDecodeError:  # skip if pg_config_info is truncated
            break

        pg_ranks_map[node.id] = {}
        pg_desc_map[node.id] = {}
        for pg in pg_objs:
            if not pg["pg_name"].isdecimal():
                # TODO support local synchronization pg
                logger.warning(
                    f"Process group name is {pg['pg_name']} in node {node.id}, which is not supported. Skip."
                )
                continue
            (pg_id, pg_desc, ranks, group_size, group_count) = (
                pg[k]
                for k in ["pg_name", "pg_desc", "ranks", "group_size", "group_count"]
            )
            pg_id = int(pg_id)
            pg_ranks_map[node.id][pg_id] = (
                ranks if len(ranks) > 0 else list(range(group_size))
                # rank list is empty when all ranks are in a pg
            )
            pg_desc_map[node.id][pg_id] = pg_desc
        break  # only one process_group init node per trace
    return pg_ranks_map, pg_desc_map


def _parse_comms_op_node(  # noqa: C901
    in_trace: ExecutionTrace,
    pg_ranks_map: dict,
    pg_desc_map: dict,
    target_rank: int,
    total_ranks: int,
):
    comms_op_list = []

    for node_id in pg_ranks_map:
        for pg_id, ranks in pg_ranks_map[node_id].items():
            comm_args = _create_pg_init_node(
                node_id, pg_id, ranks, pg_desc_map[node_id][pg_id], len(ranks)
            )
            comms_op_list.append(comm_args)

    pg_ranks_map_flatten = {}
    for _, v in pg_ranks_map.items():
        pg_ranks_map_flatten.update(v)

    comm_nodes = (
        node for node in in_trace.nodes.values() if node.name == "record_param_comms"
    )
    is_seq_id = (
        lambda x: isinstance(x, list)
        and len(x) == 2
        and isinstance(x[0], int)
        and isinstance(x[1], bool)
    )
    for node in comm_nodes:
        # for ["wait", "barrier", "init"] ops, before having different seq_id for p2p op and non p2p op, seq_id is an integer for the first input
        # After having different seq_id for p2p op and non p2p op, seq_id is a list of [seq_id, isP2P] for the first input
        # Need to handle both cases, in the future this kind of change should have different version of schema, and we can use version to decide how to parse the trace
        if is_seq_id(node.inputs[0]) or isinstance(node.inputs[0], int):
            index_base = 0
        else:
            index_base = 1
        req_id = node.inputs[index_base]
        recorded_rank = node.inputs[index_base + 2]

        comm_args = commsArgs()
        comm_args.id = node.id
        comm_args.comms = comms_utils.paramToCommName(
            node.commArgs.collective_name.lower()
        )
        if comm_args.comms == "init":
            # init node has been built
            continue

        if isinstance(req_id, int):
            # this is the format before having different seq_id for p2p op and non p2p op
            comm_args.req = (req_id, False)
        else:
            comm_args.req = req_id

        if node.commArgs.pg_name and node.commArgs.pg_name.isdecimal():
            comm_args.pgId = int(node.commArgs.pg_name)
            comm_args.groupRanks = pg_ranks_map_flatten[comm_args.pgId]
            comm_args.worldSize = len(comm_args.groupRanks)

        if comm_args.comms not in ("wait", "barrier"):
            comm_args.inMsgSize = node.commArgs.in_msg_nelems
            comm_args.outMsgSize = node.commArgs.out_msg_nelems
            comm_args.dtype = node.commArgs.dtype.lower()

        # the recorded rank id in execution trace is local rank id in the process group
        # we need to convert it to global rank for replay, check the function broadcast() of pytorch below:
        # https://github.com/pytorch/pytorch/blob/6c4efd4e959017fc758fcc5dc32d8cc6a4b9164d/torch/distributed/distributed_c10d.py#L2404
        if comm_args.comms in supportedP2pOps:
            if "send" in comm_args.comms:
                (comm_args.src_rank, comm_args.dst_rank) = (
                    target_rank,
                    comm_args.groupRanks[recorded_rank],
                )
            elif "recv" in comm_args.comms:
                (comm_args.src_rank, comm_args.dst_rank) = (
                    comm_args.groupRanks[recorded_rank],
                    target_rank,
                )
        elif comm_args.comms in ["reduce", "broadcast", "gather", "scatter"]:
            comm_args.root = comm_args.groupRanks[recorded_rank]
            comm_args.groupRanks = comm_args.groupRanks

        if comm_args.comms == "all_to_all":
            # flatten each tensor and store the # of elements into split field
            comm_args.inSplit = [math.prod(i) for i in node.input_shapes[0]]
            comm_args.outSplit = [math.prod(i) for i in node.output_shapes[0]]
        elif comm_args.comms == "all_to_allv":
            if not comm_args.worldSize:
                # if no pg info provided, use total ranks as world size
                comm_args.worldSize = total_ranks
            comm_args.inSplit = json.loads(node.commArgs.in_split_size)
            comm_args.outSplit = json.loads(node.commArgs.out_split_size)

        comms_op_list.append(comm_args)

    return comms_op_list


def _create_pg_init_node(
    node_id: int, pg_id: int, ranks: list[int], pg_desc: str, world_size: int
):
    comm_args = commsArgs()
    comm_args.id = node_id
    comm_args.comms = "init"
    comm_args.pgId = pg_id
    comm_args.pgDesc = pg_desc
    comm_args.req = -1
    comm_args.groupRanks = ranks
    comm_args.worldSize = world_size
    return comm_args
