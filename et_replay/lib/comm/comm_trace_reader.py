import json
import logging
import os
from typing import List, Optional, Tuple

from ..execution_trace import ExecutionTrace
from .backend.base_backend import SupportedP2pOps
from .comm_op_args import CommOpArgs

tensor_dtype_map = {
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


class CommTraceReader:
    """
    Handles the reading and parsing of communication traces from either remote
    URLs or local files.
    """

    def __init__(self, world_size) -> None:
        self.world_size = world_size
        self.logger = logging.getLogger(__name__)

    def read_trace(
        self,
        file_path: str,
        rank: int,
        trace_type: str,
    ) -> dict:
        """
        Main method to read and parse the trace file.

        Args:
            file_path (str): Path to read the trace from.
            rank (int): Rank of the current process.
            trace_type (str): Type of the trace to be parsed.

        Returns:
            dict: Parsed communication trace.
        """
        trace_content = self.read_raw_trace(file_path, rank, trace_type)
        trace = json.loads(trace_content) if trace_content else {}

        return self._parse_execution_trace(ExecutionTrace(trace), rank, self.world_size)

    def read_raw_trace(
        self, file_path: str, rank: int, trace_type: str
    ) -> Optional[str]:
        """
        Reads the trace file from remote or local storage based on the configuration.

        Args:
            file_path (str): Path to read the trace from.
            rank (int): Rank of the current process to handle rank-specific files.
            trace_type (str): Type of the trace to understand file handling.

        Returns:
            Optional[str]: JSON string of the trace content.
        """
        trace_file_path = (
            os.path.join(file_path, f"{rank}.json")
            if os.path.isdir(file_path)
            else file_path
        )
        with open(trace_file_path, "r") as file:
            return file.read()

    def _parse_execution_trace(
        self, in_trace: ExecutionTrace, target_rank: int, total_ranks: int
    ) -> List[CommOpArgs]:
        """
        Convert the Execution Trace comms metadata to the common trace format for
        replay.

        Args:
            in_trace (ExecutionTrace): Execution trace to be parsed.
            target_rank (int): The current rank of the device.
            total_ranks (int): Total number of ranks.

        Returns:
            List[CommOpArgs]: Parsed communication trace.
        """
        ET_PG_NAME_TUPLE = in_trace.schema_pytorch() >= (1, 0, 3)
        ET_BACKENDID = in_trace.schema_pytorch() < (1, 0, 3)

        init_ops = []
        new_comm_trace = []
        backend_id_to_pgid = {}
        pg_ranks_map = {}
        group_cnt = -1

        for node in in_trace.nodes.values():
            if "process_group:init" in node.name:
                pg_json = node.inputs[0]
                try:
                    pg_obj = json.loads(pg_json)
                except json.decoder.JSONDecodeError:
                    break

                for pg in pg_obj:
                    if not pg["pg_name"].isdecimal():
                        continue
                    pg_id = int(pg["pg_name"])
                    ranks = pg["ranks"]
                    group_cnt = pg["group_count"]
                    pg_ranks_map[pg_id] = (
                        ranks if len(ranks) > 0 else list(range(pg["group_size"]))
                    )
                    if ET_BACKENDID:
                        backend_id = pg.get("uid", pg.get("backend_id"))
                        backend_id_to_pgid[backend_id] = pg_id
                break

        for node in in_trace.nodes.values():
            if node.name == "record_param_comms":
                shift = 0 if len(node.inputs) in (8, 10) else 1
                new_comm = CommOpArgs()
                new_comm.id = node.id
                new_comm.comms = comm_utils.standardize_comm_name(
                    node.inputs[4 - shift].lower()
                )
                if new_comm.comms == "init":
                    continue
                new_comm.req = node.inputs[1 - shift]

                pg_identifier = node.inputs[2 - shift]
                if ET_BACKENDID and pg_identifier in backend_id_to_pgid:
                    new_comm.pg_id = backend_id_to_pgid[pg_identifier]
                    new_comm.group_ranks = pg_ranks_map[new_comm.pg_id]
                    new_comm.world_size = len(new_comm.group_ranks)
                elif ET_PG_NAME_TUPLE and pg_identifier[0].isdecimal():
                    new_comm.pg_id = int(pg_identifier[0])
                    new_comm.group_ranks = pg_ranks_map[new_comm.pg_id]
                    new_comm.world_size = len(new_comm.group_ranks)

                if new_comm.comms not in ("wait", "barrier"):
                    (
                        new_comm.in_msg_size,
                        in_msg_type,
                    ) = self._get_tensor_info_from_pytorch_et_entry(
                        node.inputs, node.input_types[0]
                    )
                    (
                        new_comm.out_msg_size,
                        _,
                    ) = self._get_tensor_info_from_pytorch_et_entry(
                        node.outputs, node.output_types[0]
                    )
                    new_comm.dtype = tensor_dtype_map[in_msg_type]

                if new_comm.comms in SupportedP2pOps:
                    if "send" in new_comm.comms:
                        new_comm.src_rank = target_rank
                        local_dst_rank = node.inputs[3 - shift]
                        new_comm.dst_rank = new_comm.group_ranks[local_dst_rank]
                    if "recv" in new_comm.comms:
                        local_src_rank = node.inputs[3 - shift]
                        new_comm.src_rank = new_comm.group_ranks[local_src_rank]
                        new_comm.dst_rank = target_rank

                if new_comm.comms == "broadcast":
                    new_comm.root = new_comm.group_ranks[0]
                    new_comm.src_or_dst = new_comm.group_ranks[0]

                if new_comm.comms == "all_to_allv":
                    if not new_comm.world_size:
                        new_comm.world_size = total_ranks
                    new_comm.in_split = (
                        node.inputs[5]
                        if node.inputs[5]
                        else [int(new_comm.in_msg_size / new_comm.world_size)]
                        * new_comm.world_size
                    )
                    new_comm.out_split = (
                        node.inputs[6]
                        if node.inputs[6]
                        else [int(new_comm.out_msg_size / new_comm.world_size)]
                        * new_comm.world_size
                    )
                new_comm_trace.append(new_comm)

        if group_cnt < 0:
            for pg_id, ranks in pg_ranks_map.items():
                new_comm = self._create_pg_init_node(pg_id, ranks, len(ranks))
                init_ops.append(new_comm)
        else:
            for pg_id in range(group_cnt):
                if pg_id in pg_ranks_map:
                    ranks = pg_ranks_map[pg_id]
                else:
                    ranks = [0] if target_rank != 0 else [1]
                new_comm = self._create_pg_init_node(pg_id, ranks, len(ranks))
                init_ops.append(new_comm)

        return init_ops + new_comm_trace

    def _get_tensor_info_from_pytorch_et_entry(
        self, tensor_container: List, container_type: str
    ) -> Tuple[int, str]:
        """
        Extract message size, tensor count, type from PyTorch ET entry inputs/outputs
        field.

        Args:
            tensor_container (List): Container holding tensor information.
            container_type (str): Type of the container.

        Returns:
            Tuple[int, str]: Message size and tensor data type.
        """
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

        return msg_size, dtype

    def _create_pg_init_node(
        self, pg_id: int, ranks: List[int], world_size: int
    ) -> CommOpArgs:
        """
        Create a process group initialization node.

        Args:
            pg_id (int): Process group ID.
            ranks (List[int]): List of ranks in the process group.
            world_size (int): Size of the world (number of ranks).

        Returns:
            CommOpArgs: Communication arguments for the process group initialization.
        """
        new_comm = CommOpArgs()
        new_comm.comms = "init"
        new_comm.pg_id = pg_id
        new_comm.req = -1
        new_comm.group_ranks = ranks
        new_comm.world_size = world_size
        return new_comm
