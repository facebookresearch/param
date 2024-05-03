# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from __future__ import annotations

import json
from typing import Dict, List, Tuple

from param.comm.backend.pytorch_utils import supportedP2pOps
from param.comm.comm_utils import commsArgs, param_to_comm_name
from param.execution_trace import ExecutionTrace

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


class ChakraTraceParser:
    """
    A class for parsing trace files to make them compatible with PARAM replay-mode.
    Supports various trace formats like Basic Trace, Kineto Unitrace, and PyTorch ET trace.
    """

    def parse_trace(self, in_trace: List, trace_type: str, target_rank: int, total_ranks: int) -> List:
        """
        Parses trace files based on the type specified and formats them for PARAM replay-mode.

        Args:
            in_trace (List): The trace file to be parsed.
            trace_type (str): The type of trace to parse.
            target_rank (int): The rank of the device that the trace pertains to.
            total_ranks (int): The total number of ranks involved in the trace.

        Returns:
            List: A list of parsed trace elements formatted for PARAM.
        """
        if trace_type == "basic":
            return self._parse_basic_trace(in_trace)
        elif trace_type == "et":
            return self._parse_execution_trace(ExecutionTrace(in_trace), target_rank, total_ranks)
        elif trace_type == "kineto":
            return self._parse_kineto_unitrace(in_trace, target_rank)
        else:
            raise ValueError("Unrecognized trace format")

    def _parse_basic_trace(self, in_trace: List) -> List:
        new_comms_trace = []
        for cnt, cur_comm in enumerate(in_trace):
            new_comm = commsArgs()
            new_comm.id = cnt
            new_comm.marker_stack = cur_comm.get("markers", [])
            if "comms" in cur_comm:
                self._parse_basic_trace_comms(cur_comm, new_comm)
            elif "compute" in cur_comm:
                self._parse_basic_trace_compute(cur_comm, new_comm)
            if new_comm.comms or new_comm.compute:
                new_comms_trace.append(new_comm)
            else:
                raise ValueError("Unsupported element in trace file. Please format all elements as comms or compute.")
        return new_comms_trace

    def _parse_basic_trace_comms(self, cur_comm: Dict, new_comm: commsArgs) -> None:
        new_comm.comms = param_to_comm_name(cur_comm["comms"].lower())
        new_comm.marker_stack = [new_comm.comms] if new_comm.marker_stack is None else new_comm.marker_stack
        new_comm.req = cur_comm.get("req")
        new_comm.start_time_ns = cur_comm.get("start_time_ns")
        new_comm.world_size = cur_comm.get("world_size")
        new_comm.root = cur_comm.get("root")
        new_comm.pg_id = cur_comm.get("pg_id")
        new_comm.group_ranks = cur_comm.get("global_ranks")

        if new_comm.comms not in ("wait", "barrier", "init", "batch_isend_irecv"):
            new_comm.in_msg_size = cur_comm["in_msg_size"]
            new_comm.out_msg_size = cur_comm["out_msg_size"]
            new_comm.dtype = tensorDtypeMap[cur_comm["dtype"].lower()]

        if new_comm.comms == "all_to_allv":
            new_comm.in_split = cur_comm["in_split"]
            new_comm.out_split = cur_comm["out_split"]

        if new_comm.comms in supportedP2pOps:
            new_comm.src_rank = cur_comm["src_rank"]
            new_comm.dst_rank = cur_comm["dst_rank"]
            new_comm.batch_p2p = cur_comm["use_batch"]

    def _parse_basic_trace_compute(self, cur_comm: Dict, new_comm: commsArgs) -> None:
        new_comm.compute = cur_comm["compute"].lower()
        new_comm.marker_stack = [new_comm.compute] if new_comm.marker_stack is None else new_comm.marker_stack
        new_comm.count = cur_comm.get("count", 1)

        if new_comm.compute == "gemm":
            if "mm_dim" in cur_comm:
                new_comm.mm0_dim0 = cur_comm["mm_dim"]
                new_comm.mm0_dim1 = cur_comm["mm_dim"]
                new_comm.mm1_dim0 = cur_comm["mm_dim"]
                new_comm.mm1_dim1 = cur_comm["mm_dim"]
            else:
                new_comm.mm0_dim0 = cur_comm.get("mm0_dim0")
                new_comm.mm0_dim1 = cur_comm.get("mm0_dim1")
                new_comm.mm1_dim0 = cur_comm.get("mm1_dim0")
                new_comm.mm1_dim1 = cur_comm.get("mm1_dim1")
            new_comm.dtype = cur_comm.get("dtype").lower()
        elif new_comm.compute == "emb_lookup":
            new_comm.direction = cur_comm.get("direction", "forward")
            new_comm.emb_dim = cur_comm.get("emb_dim")
            new_comm.num_embs = cur_comm.get("num_embs")
            new_comm.batch_size = cur_comm.get("batch_size")
            new_comm.num_emb_tables_per_device = cur_comm.get("num_emb_tables")
            new_comm.num_emb_tables_batched = cur_comm.get("num_emb_tables_batched", -1)
            new_comm.bag_size = cur_comm.get("bag_size")
        else:
            raise ValueError(f"Unsupported compute element '{new_comm.compute}' in trace file.")

    def _parse_kineto_unitrace(self, in_trace: List, target_rank: int) -> List:
        new_comms_trace = []
        comms_cnt = 0
        for entry in in_trace:
            # Placeholder logic to determine marker stack, to be implemented.
            marker = "unknown"  # Not fully implemented

            if entry["name"] == "record_param.comm" and entry["args"]["rank"] == target_rank:
                new_comm = commsArgs()
                new_comm.comms = param_to_comm_name(entry["args"]["comms"].lower())
                new_comm.id = comms_cnt
                new_comm.in_msg_size = entry["args"]["in_msg_size"]
                new_comm.out_msg_size = entry["args"]["out_msg_size"]
                new_comm.dtype = tensorDtypeMap[entry["args"]["dtype"].lower()]
                new_comm.in_split = entry["args"]["in_split"]
                new_comm.out_split = entry["args"]["out_split"]
                new_comm.marker_stack = marker
                new_comms_trace.append(new_comm)
                comms_cnt += 1
        return new_comms_trace

    def _parse_execution_trace(self, in_trace: ExecutionTrace, target_rank: int, total_ranks: int) -> List:
        new_comms_trace = []
        backend_id_to_pgid = {}
        pg_ranks_map = {}

        for node in in_trace.nodes.values():
            if "process_group:init" in node.name:
                pg_json = node.inputs[0]
                try:
                    pg_obj = json.loads(pg_json)
                except json.decoder.JSONDecodeError:
                    continue

                for pg in pg_obj:
                    if not pg["pg_name"].isdecimal():
                        continue
                    pg_id = int(pg["pg_name"])
                    ranks = pg["ranks"]
                    pg_ranks_map[pg_id] = ranks if ranks else list(range(pg["group_size"]))
                    backend_id_to_pgid[pg["backend_id"]] = pg_id

        for node in in_trace.nodes.values():
            if node.name == "record_param.comm":
                new_comm = commsArgs()
                new_comm.id = node.id
                new_comm.comms = param_to_comm_name(node.inputs[4].lower())
                new_comm.req = node.inputs[1]
                pg_identifier = node.inputs[2]
                if pg_identifier in backend_id_to_pgid:
                    new_comm.pg_id = backend_id_to_pgid[pg_identifier]
                    new_comm.group_ranks = pg_ranks_map[new_comm.pg_id]
                    new_comm.world_size = len(new_comm.group_ranks)

                if new_comm.comms not in ("wait", "barrier"):
                    new_comm.in_msg_size, in_msg_type = self._get_tensor_info_from_pytorch_et_entry(
                        node.inputs, node.input_types[0]
                    )
                    new_comm.out_msg_size, _ = self._get_tensor_info_from_pytorch_et_entry(
                        node.outputs, node.output_types[0]
                    )
                    new_comm.dtype = tensorDtypeMap[in_msg_type]

                if new_comm.comms in supportedP2pOps:
                    if "send" in new_comm.comms:
                        new_comm.src_rank = target_rank
                        local_dst_rank = node.inputs[3]
                        new_comm.dst_rank = new_comm.group_ranks[local_dst_rank]
                    elif "recv" in new_comm.comms:
                        local_src_rank = node.inputs[3]
                        new_comm.src_rank = new_comm.group_ranks[local_src_rank]
                        new_comm.dst_rank = target_rank

                if new_comm.comms == "broadcast":
                    new_comm.root = new_comm.group_ranks[0]
                    new_comm.src_or_dst = new_comm.group_ranks[0]

                if new_comm.comms == "all_to_allv":
                    new_comm.in_split = (
                        node.inputs[5]
                        if node.inputs[5]
                        else [int(new_comm.in_msg_size / new_comm.world_size)] * new_comm.world_size
                    )
                    new_comm.out_split = (
                        node.inputs[6]
                        if node.inputs[6]
                        else [int(new_comm.out_msg_size / new_comm.world_size)] * new_comm.world_size
                    )
                new_comms_trace.append(new_comm)

        return new_comms_trace

    def _get_tensor_info_from_pytorch_et_entry(
        self, tensor_container: List, container_type: str
    ) -> Tuple[int, int, str]:
        list_count = container_type.count("GenericList")
        tensors = []
        if list_count == 2:
            tensors = tensor_container[0][0]
            dtype = container_type.replace("GenericList[", "").split(",", 1)[0]
        elif list_count == 1:
            tensors = tensor_container[0]
            dtype = container_type.replace("GenericList[", "").replace("]", "")
        else:
            tensors.append(tensor_container[0])
            dtype = container_type

        msg_size = 0
        for tensor in tensors:
            msg_size += tensor[3]

        return msg_size, len(tensors), dtype

    def create_pg_init_node(self, pg_id: int, ranks: List[int], world_size: int) -> commsArgs:
        """
        Creates a process group initialization node for communication setup in distributed systems.

        Args:
            pg_id (int): Process group ID.
            ranks (List[int]): List of ranks in the process group.
            world_size (int): Total number of ranks in the process group.

        Returns:
            commsArgs: Initialized process group communication arguments.
        """
        new_comm = commsArgs()
        new_comm.comms = "init"
        new_comm.pg_id = pg_id
        new_comm.req = -1  # No specific request ID for initialization
        new_comm.group_ranks = ranks
        new_comm.world_size = world_size
        return new_comm
