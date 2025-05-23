import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum

import numpy as np
import torch

from et_replay.comm import comms_utils, param_profile
from et_replay.comm.comms_utils import (
    bootstrap_info_holder,
    commsArgs,
    commsParamsHolderBase,
    paramCommsBench,
    paramToCommName,
)
from et_replay.et_replay_utils import (
    build_torchscript_func,
    build_triton_func,
    fbgemm_input_args_indices,
    generate_fbgemm_tensors,
    generate_prefix,
    generate_suffix,
    get_input_tensors,
    get_output_tensors,
    is_fbgemm_forward,
    is_fbgemm_forward_unweighted,
    is_tensor,
    is_tensor_list,
    TORCH_DTYPES_RNG,
    TORCH_DTYPES_RNG_str,
)
from et_replay.execution_trace import ExecutionTrace, NodeType
from et_replay.tools.comm_replay import commsTraceReplayBench
from et_replay.utils import trace_handler
from param_bench.train.compute.python.lib import pytorch as lib_pytorch
from param_bench.train.compute.python.lib.init_helper import load_modules
from param_bench.train.compute.python.workloads import pytorch as workloads_pytorch
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.async_compile import AsyncCompile

# grid and split_scan_grid are dynamically loaded
from torch.profiler import ExecutionTraceObserver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CommsReplayManager(commsTraceReplayBench):  # pyre-ignore[13]:
    def __init__(self):
        super().__init__()

        self.comp_replay_manager = None

    def generate_io_tensors(
        self,
        curComm: commsArgs,
        commsParams: commsParamsHolderBase,
        regenerateTensors: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | torch.Tensor]:
        paramCommsBench.prepComm(self, curComm, commsParams, False)

        node = self.comp_replay_manager.et.nodes[curComm.id]

        ip_tensor, _ = self.comp_replay_manager.get_data(node, True, True)
        op_tensor, _ = self.comp_replay_manager.get_data(node, False, True)

        # "record_param_comms" node is recorded in NCCL C++ API (for example,
        # torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp). However, et_replay need to
        # call its Python API. Apparently C++ API and Python API is not compatable. For some
        # collectives, their Python API has tensor as their input while its C++ API has a vector
        #  of tensor as their input. The following code is to work around this issue.
        need_extract_ops = [
            "send",
            "recv",
            "broadcast",
            "reduce",
            "all_reduce",
            "all_gather",
            "gather",
            "scatter",
            "reduce_scatter",
        ]

        if len(ip_tensor) > 0:
            if paramToCommName(curComm.comms) not in need_extract_ops:
                ip_tensor = ip_tensor[0]
            else:
                ip_tensor = ip_tensor[0][0]

        if len(op_tensor) > 0:
            if paramToCommName(curComm.comms) not in need_extract_ops:
                op_tensor = op_tensor[0]
            else:
                op_tensor = op_tensor[0][0]

        return (ip_tensor, op_tensor)


class TensorAllcationMode(Enum):
    """
    Enum to represent the tensor allocation mode
    """

    # Allocate input tensors that can not be generated when replaying the trace
    # at the beginning and reuse them for all iterations.
    PRE_ALLOCATE = 1

    # Allocate tensors on the fly and free them after they are out of scope
    LAZY_ALLOCATE = 2


class ExgrReplayManager:
    def __init__(self):
        self.numWarmupIters = 1
        self.numIters = 1
        self.profile_replay = False
        self.profile_memory = False
        self.et = None
        self.et_profile = False
        self.cuda_id = 0
        self.debug = False
        self.compute_only = False
        self.generator = False
        self.trace_file = ""
        self.dump = False
        self.dump_path = ""
        self.args = None
        # Comms env.
        self.comms_env_params = comms_utils.read_comms_env_vars()
        self.commsBench = None
        self.comms_world_info = None
        self.commsParams = None

        self.cuda = "cuda"
        self.device = torch.device(self.cuda)

        # Permanent registry of the tensors that need to be initialized.
        self.tensor_registry_permanent = {}
        # Registry of all input tensors.
        self.input_tensor_ids = set()
        # Runtime registry of all tensors.
        self.tensor_registry = {}
        # Nodes/Ops to replay after preprocessing.
        self.sorted_nodes = []
        # Reconstructed function registry for each node/op.
        self.funcs = {}
        # Mark some intermediate tensors (output of operators) as unchangeable.
        self.unchangeable_intermediate_tensors = set()
        # Unique tensors in execution trace identified by (tensor_id, storage_id, offset, num_elem, elem_bytes).
        self.original_unique_tensors = set()
        # Number of unique tensors in replay since tensors may have multiple shapes and to accommodate that
        # we treat tensors with same identifier tuple but different shapes as different tensors.
        self.replay_unique_tensor_num = 0
        # Map (tensor_id, node,id) in et to unique tensor_id in replay.
        # We assume in only input or only output of an op, the shape of tensors with same id keeps the same.
        # Same tensor in input and output may be different (e.g., aten::unsqueeze()).
        self.tensors_mapping = {}
        # Dict that stores the shape of each unique tensor in replay.
        self.replay_tensors_shapes = {}
        # Dict that stores the shapes of a tensor, for the convenience of quickly determining whether
        # to create a unique tensor in replay if the id is same but shape is different.
        self.tensor_shapes = defaultdict(set)
        # Dict that maps tensor storage id to its size, and a map for {device, torch.Tensor}.
        # The tensor with the same storage id may located on different devices.
        self.tensor_storage_map: dict[int, []] = defaultdict(list)
        # Mark those tensors that occur first as an input in the original et as needing to be instantiated in replay
        # at the very beginning.
        self.instantiate = set()
        # Tensors that should be instantiated on cpu, e.g., input of aten::pin_memory and aten::to.
        self.cpu_tensor = set()

        self.parallel_nodes_parents = []
        # Ids of nodes that need to run in parallel.
        self.parallel_nodes_ids = []

        # This is used to pick out a single iteration when trace contains multiple iterations.
        # Basically this label should be captured at the beginning of each iteration so that one iteration
        # is between two consecutive label nodes.
        self.profile_step_label = "ProfilerStep#"

        try:
            from et_replay.vendor_internal.fb_internal import (
                add_internal_parallel_nodes_parents,
            )
        except ImportError:
            logger.info("FB internals not present")
        else:
            self.parallel_nodes_parents = add_internal_parallel_nodes_parents(
                self.parallel_nodes_parents
            )

        self.profile_step_node_ids = []

        # Only use for memory profile.
        self.current_allocated_mem = 0
        self.current_reserved_mem = 0
        self.op_allocated_mem = {}
        self.op_reserved_mem = {}

        # actual_skip_nodes is a dictinary which records the skipped node name
        # and the reason to skip it.
        # Dict {name => skip reason}
        self.actual_skip_nodes: dict[str, str] = {}
        self.initial_skip_node_count = 0
        self.n_skipped_nodes = 0

        self.tensor_with_device = True
        # A tensor may appear on multiple devices but here we only store the first device for initialization
        # since device change should be captured in operator execution and be naturally recovered by replaying
        # the operators.
        self.tensor_device = {}

        # This is the map between replay_tensor_id to the last node id that uses it.
        # After an op runs, we check the input/output tensors that op used, if the last_node_id
        # of the replay_tensor_id is the same as or greater than the current node id, that tensor is deleted
        self.replay_tensor_id_to_last_node_id_map = {}

        self.tensor_storage_id_to_last_node_id_map: dict[int, int] = defaultdict(int)

        self.tensor_allocate_mode: TensorAllcationMode = (
            TensorAllcationMode.PRE_ALLOCATE
        )

        # Unrecognized nodes that are neither operators nor predefined label nodes.
        self.exceptional_nodes = set()

        # Debug use for execution time breakdown.
        self.lookup_cnt = 0
        self.input_total_time = 0
        self.output_total_time = 0
        self.exec_time = []
        self.setup_time = []

        self.tf32 = False

        # Tensors that related to aten::to, for those tensors we still need to override its value after
        # the first iteration, otherwise the tensors may be recycled.
        self.special_tensors = set()

        # Replay on CPU.
        self.cpu = False

        self.available_memory = 0

    def initBench(self):
        self.numWarmupIters = self.args.warmup_iter
        self.numIters = self.args.iter
        self.profile_replay = self.args.profile_replay
        self.profile_memory = self.args.profile_memory
        self.et_profile = self.args.et
        self.cuda_id = self.args.cuda
        self.debug = self.args.debug
        self.compute_only = self.args.compute
        self.generator = self.args.generator
        self.dump = self.args.dump
        self.dump_path = self.args.dump_path
        self.wait_delay = self.args.delay
        self.cpu = self.args.cpu
        self.tf32 = self.args.tf32
        if self.args.enable_lazy_tensor_allocation:
            self.tensor_allocate_mode = TensorAllcationMode.LAZY_ALLOCATE

        # Single trace.
        if not self.args.trace_path:
            # Input et trace should be explicitly specified after --input.
            if "://" in self.args.input:
                try:
                    from et_replay.vendor_internal.fb_internal import read_remote_trace
                except ImportError:
                    logger.info("FB internals not present")
                    exit(1)
                else:
                    et, self.trace_file = read_remote_trace(self.args.input)
                    self.et = ExecutionTrace(json.load(et))
            else:
                self.trace_file = self.args.input
                with open(self.trace_file) as f:
                    self.et = ExecutionTrace(json.load(f))

            if self.cuda_id == -1:
                self.cuda = "cuda"
            else:
                self.cuda = f"cuda:{self.cuda_id}"
            self.dump_path += "benchmark.py"
        # Multiple traces.
        else:
            logger.info(
                f"process {os.getpid()} is rank {self.comms_env_params['global_rank']}"
            )
            self.cuda_id = self.comms_env_params["local_rank"]
            self.cuda = f"cuda:{self.comms_env_params['local_rank']}"
            # Different processes should read different traces based on global_rank_id.
            if "://" in self.args.trace_path:
                try:
                    from et_replay.vendor_internal.fb_internal import (
                        read_remote_skip_node_file,
                        read_remote_trace,
                    )
                except ImportError:
                    logger.info("FB internals not present")
                    exit(1)
                else:
                    et, self.trace_file = read_remote_trace(
                        f"{self.args.trace_path}/rank-{self.comms_env_params['global_rank']}.json"
                    )
                    self.et = ExecutionTrace(json.load(et))

                    # check if the remote path has skip-node.json
                    skip_node_json = read_remote_skip_node_file(
                        f"{self.args.trace_path}"
                    )
                    if skip_node_json:
                        self.actual_skip_nodes = json.load(skip_node_json)
                        self.initial_skip_node_count = len(self.actual_skip_nodes)
            else:
                self.trace_file = f"{self.args.trace_path}/rank-{self.comms_env_params['global_rank']}.json"
                with open(self.trace_file) as f:
                    self.et = ExecutionTrace(json.load(f))

            self.dump_path += f"benchmark_{self.comms_env_params['global_rank']}.py"

        # base_path is used to find the generated kernel files in the same directory of the trace file.
        base_path, file_name = os.path.split(self.trace_file)
        self.resource_dir = os.path.join(
            base_path, os.path.splitext(file_name)[-2] + "_resources"
        )

        if self.args.skip_node_file:
            try:
                with open(self.args.skip_node_file) as json_file:
                    self.actual_skip_nodes = json.load(json_file)
                    self.initial_skip_node_count = len(self.actual_skip_nodes)
            except OSError:
                logger.info(
                    f"Failed to load skip node file {self.args.skip_node_file}."
                )

        self.kernel_map = {}
        self.async_compile = AsyncCompile()

        if self.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.cuda)

        # Total memory of the device
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        # Memory currently allocated and reserved by PyTorch
        reserved_memory = torch.cuda.memory_reserved(self.device)
        # Available memory (approximate) by subtracting allocated and reserved from total
        self.available_memory = (total_memory - reserved_memory) / (1024**3)
        logger.info(f"Available memory: {self.available_memory} GB")

    def detect_tensor_device(self, root):
        # Automatically detect whether the captured tensor information includes device.
        # Just a temporary utility to accommodate old and new versions et and should be removed later.
        def traverse(root):
            for child in root.children:
                for _, t_id, _ in get_input_tensors(child):
                    if len(list(t_id)) == 5:
                        self.tensor_with_device = False
                    return
                for _, t_id, _ in get_output_tensors(child):
                    if len(list(t_id)) == 5:
                        self.tensor_with_device = False
                    return
                traverse(child)

        traverse(root)

    def reset_registry(self):
        if self.tensor_with_device:
            self.tensor_registry = {
                k: (
                    None
                    if v is None
                    else (
                        v.cpu()
                        if self.tensor_device[k] == "cpu" or self.cpu
                        else v.cuda(self.device)
                    )
                )
                for k, v in self.tensor_registry_permanent.items()
            }
        else:
            self.tensor_registry = {
                k: (
                    None
                    if v is None
                    else (
                        v.cpu()
                        if k in self.cpu_tensor or self.cpu
                        else v.cuda(self.device)
                    )
                )
                for k, v in self.tensor_registry_permanent.items()
            }
        gc.collect()
        torch.cuda.empty_cache()

    def add_skipped_nodes(self, node, reason: str) -> None:
        if node.name not in self.actual_skip_nodes:
            self.actual_skip_nodes[node.name] = reason

    def is_skipped(self, node) -> bool:
        if node.name in self.actual_skip_nodes:
            self.n_skipped_nodes += 1
            return True
        else:
            return False

    def extract_subgraph(self, root):
        """
        return: all nodes in the subgraph, in the order of node ID (also execution)
        """

        def analayze_node(node):
            for _, t_id, _ in get_input_tensors(node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if node.name == "record_param_comms" and self.compute_only:
                    continue
                self.input_tensor_ids.add(t_id)

            # The output tensors from the comm nodes also need to be allocated. Since the
            # output tensors from the comm nodes have similar behavior like the input tensors,
            # we take them as the input tensors, and put them in self.input_tensor_ids.
            # So in full_replay, self.input_tensor_ids contains both input tensor ids and
            # the comm nodes' output tensor ids
            if not self.compute_only:
                for _, t_id, _ in get_output_tensors(node):
                    if self.tensor_with_device:
                        t_id = tuple(list(t_id)[:5])
                    if node.name != "record_param_comms":
                        continue
                    self.input_tensor_ids.add(t_id)

            if node.name == "record_param_comms":
                # Node "record_param_comms" is not able to have a func created by self.build_func
                # but we still want to return success to keep it in self.sorted_nodes
                return True, ""

            func, output_count = self.build_func(node)
            if func is None:
                return False, "Failed to build function"
            else:
                self.funcs[node.id] = (func, output_count)
                return True, ""

        def dfs_traverse(node):
            if self.profile_step_label in node.name:
                self.profile_step_node_ids.append(node.id)
            if node.type == NodeType.OPERATOR:
                if not self.is_skipped(node):
                    self.sorted_nodes.append(node)
                return

            for child in node.children:
                dfs_traverse(child)

        self.n_skipped_nodes = 0
        dfs_traverse(root)
        self.sorted_nodes = sorted(self.sorted_nodes, key=lambda x: x.id)
        self.profile_step_node_ids = sorted(self.profile_step_node_ids)

        if len(self.profile_step_node_ids) > 1:
            # Only execute the ops in the first step
            start_id = self.profile_step_node_ids[0]
            end_id = self.profile_step_node_ids[1]
            self.sorted_nodes = [
                x for x in self.sorted_nodes if x.id > start_id and x.id < end_id
            ]

        logger.info(f"#Operators to execute: {len(self.sorted_nodes)}")
        picked_nodes = []
        for node in self.sorted_nodes:
            success, msg = analayze_node(node)
            if success:
                picked_nodes.append(node)
            else:
                self.add_skipped_nodes(node, msg)
        self.sorted_nodes = picked_nodes

        # triton kernels are compiled in parallel, need to wait until
        # all kernels are compiled.
        self.async_compile.wait(globals())
        del self.async_compile

        self.select_parallel_nodes()

        if self.args.skip_node_file is not None and self.args.update_skip_node_file:
            actual_skip_node_count = len(self.actual_skip_nodes)
            if actual_skip_node_count > self.initial_skip_node_count:
                with open(self.args.skip_node_file, "w") as outfile:
                    json.dump(self.actual_skip_nodes, outfile, indent=4)

    def select_parallel_nodes(self):
        def is_parallel_parent(node):
            return node.name in self.parallel_nodes_parents

        def has_parallel_parent(node):
            if not node.parent or node.parent.id == node.id:
                return False
            if is_parallel_parent(node):
                return True
            return has_parallel_parent(node.parent)

        for node in self.sorted_nodes:
            if has_parallel_parent(node):
                self.parallel_nodes_ids.append(node.id)

        assert len(self.parallel_nodes_ids) == len(set(self.parallel_nodes_ids))

    def analyze_tensors(self):
        def add_storage_tensor(node_id, t_id, device):
            # t_id is a tupe of (tensor_id, storage_id, offset, number of element,
            # number of bytes for each element, device)

            # ET does not save the size of the tensor storage, so we iterate over all the
            # tensors to find the maximum size of the storage.
            storage_id = t_id[1]
            self.tensor_storage_id_to_last_node_id_map[storage_id] = max(
                node_id, self.tensor_storage_id_to_last_node_id_map[storage_id]
            )
            if storage_id not in self.tensor_storage_map:
                # the storage size for this tensor is the sum of the storage offset and
                # number of elements * number of bytes per element.
                self.tensor_storage_map[storage_id] = [
                    t_id[2] + t_id[3] * t_id[4],
                    {},
                ]
            else:
                self.tensor_storage_map[storage_id][0] = max(
                    self.tensor_storage_map[storage_id][0], t_id[2] + t_id[3] * t_id[4]
                )

        def add_unique_tensor(node_name, node_id, t_id, shape, input, device=-1):
            add_storage_tensor(node_id, t_id, device)
            # If we did not see this tensor before, add it as a unique tensor.
            if t_id not in self.original_unique_tensors:
                self.original_unique_tensors.add(t_id)
                self.replay_unique_tensor_num += 1
                replay_t_id = self.replay_unique_tensor_num
                self.tensors_mapping[(node_id, t_id, input)] = replay_t_id
                self.replay_tensors_shapes[replay_t_id] = shape
                self.tensor_shapes[t_id].add((replay_t_id, tuple(shape)))
                self.replay_tensor_id_to_last_node_id_map[replay_t_id] = node_id
                if self.tensor_with_device:
                    self.tensor_device[replay_t_id] = device
                if node_name == "aten::to":
                    self.special_tensors.add(replay_t_id)
                return

            # If we saw this tensor before but with a different shape, add it as a unique tensor.
            for replay_t_id, pre_shape in self.tensor_shapes[t_id]:
                if tuple(shape) == pre_shape:
                    self.tensors_mapping[(node_id, t_id, input)] = replay_t_id
                    self.replay_tensor_id_to_last_node_id_map[replay_t_id] = max(
                        self.replay_tensor_id_to_last_node_id_map[replay_t_id], node_id
                    )
                    if node_name == "aten::to":
                        self.special_tensors.add(replay_t_id)
                    return

            self.replay_unique_tensor_num += 1
            self.tensors_mapping[(node_id, t_id, input)] = self.replay_unique_tensor_num
            replay_t_id = self.replay_unique_tensor_num
            self.replay_tensors_shapes[replay_t_id] = shape
            self.tensor_shapes[t_id].add((replay_t_id, tuple(shape)))
            self.replay_tensor_id_to_last_node_id_map[replay_t_id] = node_id
            if self.tensor_with_device:
                self.tensor_device[replay_t_id] = device
            if node_name == "aten::to":
                self.special_tensors.add(replay_t_id)

        for node in self.sorted_nodes:
            if node.name == "record_param_comms" and self.compute_only:
                continue
            for _, t_id, shape in get_input_tensors(node):
                if self.tensor_with_device:
                    device = list(t_id)[5]
                    t_id = tuple(list(t_id)[:5])
                    if t_id in self.input_tensor_ids:
                        add_unique_tensor(
                            node.name, node.id, t_id, shape, input=True, device=device
                        )
                else:
                    if t_id in self.input_tensor_ids:
                        add_unique_tensor(node.name, node.id, t_id, shape, input=True)

            for _, t_id, shape in get_output_tensors(node):
                if self.tensor_with_device:
                    device = list(t_id)[5]
                    t_id = tuple(list(t_id)[:5])
                    if t_id in self.input_tensor_ids:
                        # fake comm nodes output tensor as input
                        add_unique_tensor(
                            node.name,
                            node.id,
                            t_id,
                            shape,
                            input=(node.name == "record_param_comms"),
                            device=device,
                        )
                else:
                    if t_id in self.input_tensor_ids:
                        add_unique_tensor(
                            node.name,
                            node.id,
                            t_id,
                            shape,
                            input=(node.name == "record_param_comms"),
                        )

        # Simulate the execution progress and record the output tensors we have seen so far.
        output_set = set()
        for node in self.sorted_nodes:
            if node.name == "record_param_comms" and self.compute_only:
                continue
            for _, t_id, _ in get_input_tensors(node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if (
                    t_id in self.input_tensor_ids
                    and self.tensors_mapping[(node.id, t_id, True)] not in output_set
                ):
                    self.instantiate.add(self.tensors_mapping[(node.id, t_id, True)])

            for _, t_id, _ in get_output_tensors(node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if t_id in self.input_tensor_ids:
                    if (
                        node.name == "record_param_comms"
                    ):  # comm op needs alloc output tensors
                        self.instantiate.add(
                            self.tensors_mapping[(node.id, t_id, True)]
                        )
                    else:
                        output_set.add(self.tensors_mapping[(node.id, t_id, False)])

    def allocate_tensors(self):
        start_ns = time.time_ns()

        if not self.compute_only:
            for node in self.sorted_nodes:
                if node.name == "record_param_comms":
                    self.allocate_node_tensors(node, True, True)
                    self.allocate_node_tensors(node, False, True)
                else:
                    self.allocate_node_tensors(node, True, False)
        else:
            for node in self.sorted_nodes:
                if node.name != "record_param_comms":
                    self.allocate_node_tensors(node, True, False)

        logger.info(
            f"Tensor allocation time: {(time.time_ns() - start_ns) / 1000000.0} ms"
        )

    def allocate_node_tensors(self, node, is_input, is_comm_node):  # noqa: C901
        if is_input:
            tensor_strides = node.get_input_tensor_strides()
            tensors = get_input_tensors(node)
            node.pre_load_tensors = [None] * len(tensors)
        else:
            tensor_strides = node.get_output_tensor_strides()
            tensors = get_output_tensors(node)

        for idx, (data_type, t_id, shape) in enumerate(tensors):
            device = self.device
            if self.tensor_with_device:
                device = t_id[5]
                t_id = tuple(list(t_id)[:5])

            if is_comm_node:
                # Both input/output tensors of the comm nodes are treated as input tensors
                replay_t_id = self.tensors_mapping[(node.id, t_id, True)]
            else:
                replay_t_id = self.tensors_mapping[(node.id, t_id, is_input)]

            if t_id not in self.input_tensor_ids:
                continue
            found_tensor = False
            if self.tensor_allocate_mode == TensorAllcationMode.PRE_ALLOCATE:
                if replay_t_id in self.tensor_registry_permanent.keys():
                    found_tensor = True
            else:
                if replay_t_id in self.tensor_registry.keys():
                    found_tensor = True
            if not found_tensor:
                try:
                    dtype, _ = TORCH_DTYPES_RNG[
                        data_type.removeprefix("Tensor(").rstrip(")")
                    ]

                    strides = None
                    if node.input_strides is not None:
                        strides = tensor_strides[idx]

                    tensor = None
                    if (
                        dtype
                        in (
                            torch.int8,
                            torch.uint8,
                            torch.int16,
                            torch.uint16,
                            torch.int32,
                            torch.uint32,
                            torch.int64,
                            torch.uint64,
                            torch.long,
                        )
                        and is_input
                    ):
                        tensor = self.get_tensor_from_file(
                            node.id, idx, device, shape, dtype, strides
                        )
                        if tensor is not None:
                            node.pre_load_tensors[idx] = tensor

                    if tensor is None:
                        tensor = self.get_tensor_from_storage(
                            t_id[1],  # storage_id
                            t_id[2],  # offset
                            t_id[4],  # number of bytes per element
                            device,
                            shape,
                            dtype,
                            strides,
                            node.get_input_tensor_range(idx),
                        )

                        if (
                            self.tensor_allocate_mode
                            == TensorAllcationMode.PRE_ALLOCATE
                        ):
                            self.tensor_registry_permanent[replay_t_id] = tensor
                        else:
                            self.tensor_registry[replay_t_id] = tensor

                except KeyError:
                    if data_type != "Tensor(nullptr (uninitialized))":
                        logger.info(f"KeyError: {node.id}, {t_id}, {data_type}")
                    if self.tensor_allocate_mode == TensorAllcationMode.PRE_ALLOCATE:
                        self.tensor_registry_permanent[replay_t_id] = None
                    else:
                        self.tensor_registry[replay_t_id] = None

    def build_func(self, node):
        if node.kernel_backend == "triton":
            if node.kernel_file in self.kernel_map:
                func = self.kernel_map[node.kernel_file]
                # For a triton kernel, it is the caller's responsibility to allocate
                # the output tensors, and pass them in as the input arguments.
                # So the number of the output tensors is always 0
                output_count = 0
            else:
                func, output_count = build_triton_func(
                    node, self.resource_dir, self.async_compile, self.device
                )
                self.kernel_map[node.kernel_file] = func
        else:
            func, output_count = build_torchscript_func(node)

        return func, output_count

    def generate_code(self):
        def _generate_tensor_allocation_str():
            tensor_allocation_str = ""
            unallocated_tensor_allocation_str = ""
            unallocated_tensors = set()
            tensor_allocate_template = """{tensor} = {rng}({shape}).to({dtype}){cuda}"""
            for node in self.sorted_nodes:
                if node.name == "record_param_comms" and self.compute_only:
                    continue
                if is_fbgemm_forward(node):
                    if self.cpu:
                        tensor_allocation_str += f'input_args, _ = generate_fbgemm_tensors(nodes[{node.id}], "cpu", {self.args.rows}, {self.args.pooling_factor}, {self.args.alpha})\n'
                        input_args, _ = generate_fbgemm_tensors(
                            node,
                            "cpu",
                            self.args.rows,
                            self.args.pooling_factor,
                            self.args.alpha,
                        )
                    else:
                        tensor_allocation_str += f'input_args, _ = generate_fbgemm_tensors(nodes[{node.id}], "{self.cuda}", {self.args.rows}, {self.args.pooling_factor}, {self.args.alpha})\n'
                        input_args, _ = generate_fbgemm_tensors(
                            node,
                            self.cuda,
                            self.args.rows,
                            self.args.pooling_factor,
                            self.args.alpha,
                        )

                for idx, (dtype, t_id, shape) in enumerate(get_input_tensors(node)):
                    if self.tensor_with_device:
                        t_id = tuple(list(t_id)[:5])
                    replay_t_id = self.tensors_mapping[(node.id, t_id, True)]
                    if (
                        t_id in self.input_tensor_ids
                        and replay_t_id not in self.tensor_registry_permanent.keys()
                        and (
                            node.name == "aten::embedding_bag"
                            or "fbgemm::split_embedding_codegen_lookup" in node.name
                            or replay_t_id in self.instantiate
                        )
                    ):
                        try:
                            if is_fbgemm_forward(node):
                                tensor_allocation_str += (
                                    f"global tensor_{replay_t_id}\n"
                                )
                                tensor_allocation_str += (
                                    f"tensor_{replay_t_id} = input_args[{idx}]\n"
                                )
                                if (
                                    "fbgemm::split_embedding_codegen_lookup"
                                    in node.name
                                ):
                                    self.unchangeable_intermediate_tensors.add(
                                        replay_t_id
                                    )
                            else:
                                if node.name == "aten::embedding_bag":
                                    self.unchangeable_intermediate_tensors.add(
                                        replay_t_id
                                    )
                                if node.name == "aten::pin_memory" and idx == 0:
                                    self.cpu_tensor.add(replay_t_id)

                                if dtype == "Tensor(signed char)":
                                    dtype_str, rng_str = TORCH_DTYPES_RNG_str[
                                        "signed char"
                                    ]
                                else:
                                    dtype_str, rng_str = TORCH_DTYPES_RNG_str[
                                        dtype.removeprefix("Tensor(").rstrip(")")
                                    ]
                                tensor_str = f"tensor_{replay_t_id}"
                                shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
                                cuda_str = ""
                                if not self.cpu:
                                    if self.tensor_with_device:
                                        if self.tensor_device[replay_t_id] != "cpu":
                                            cuda_str = f'.cuda("{self.cuda}")'
                                    elif replay_t_id not in self.cpu_tensor:
                                        cuda_str = f'.cuda("{self.cuda}")'

                                tensor_allocation_str += f"global {tensor_str}\n"
                                tensor_allocation_str += (
                                    tensor_allocate_template.format(
                                        tensor=tensor_str,
                                        rng=rng_str,
                                        shape=shape_str,
                                        dtype=dtype_str,
                                        cuda=cuda_str,
                                    )
                                    + "\n"
                                )
                            self.tensor_registry_permanent[replay_t_id] = 1
                        except KeyError:
                            if dtype != "Tensor(nullptr (uninitialized))":
                                logger.info(f"KeyError: {node.id}, {t_id}, {dtype}")
                            tensor_allocation_str += f"global tensor_{replay_t_id}\n"
                            tensor_allocation_str += f"tensor_{replay_t_id} = None\n"
                            self.tensor_registry_permanent[replay_t_id] = 1
                    elif (
                        t_id in self.input_tensor_ids
                        and replay_t_id not in self.tensor_registry_permanent.keys()
                        and replay_t_id not in self.instantiate
                    ):
                        if replay_t_id not in unallocated_tensors:
                            tensor_allocation_str += f"global tensor_{replay_t_id}\n"
                            unallocated_tensor_allocation_str += (
                                f"    global tensor_{replay_t_id}\n"
                            )
                            unallocated_tensors.add(replay_t_id)

            return tensor_allocation_str, unallocated_tensor_allocation_str

        def _generate_inputs_str(node):
            inputs = ""
            if is_fbgemm_forward(node):
                idx_list = fbgemm_input_args_indices(node)
                for idx in idx_list:
                    if self.tensor_with_device:
                        inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(node.inputs[idx][:5]), True)]}, "
                    else:
                        inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(node.inputs[idx]), True)]}, "
                if is_fbgemm_forward_unweighted(node):
                    inputs += "None" + ", "
            else:
                for idx, item in enumerate(node.inputs):
                    if (
                        node.name == "aten::convolution_backward"
                        and idx == len(node.inputs) - 1
                    ):
                        inputs += "[True, True, True], "
                        continue
                    if is_tensor(node, idx, True):
                        if self.tensor_with_device:
                            item = tuple(item[:5])
                        # Workaround to handle tensor with same id but different data types.
                        if node.name == "aten::index_add_" and idx == 3:
                            inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(item), True)]}.to(torch.float64), "
                        elif node.name == "aten::index_add_" and idx == 2:
                            inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(item), True)]}.to(torch.int), "
                        elif node.name == "aten::index_select" and idx == 2:
                            inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(item), True)]}.to(torch.int), "
                        elif (
                            node.name == "aten::index_copy_"
                            and idx == 3
                            and node.input_types[idx] == "Tensor(double)"
                        ):
                            inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(item), True)]}.to(torch.float64), "
                        elif (
                            node.name == "aten::index_copy_"
                            and idx == 2
                            and node.input_types[idx] == "Tensor(long)"
                        ):
                            inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(item), True)]}.to(torch.int64), "
                        else:
                            inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(item), True)]}, "
                    elif is_tensor_list(node, idx):
                        inputs += "["
                        if self.tensor_with_device:
                            for t_id in item:
                                inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(t_id[:5]), True)]}, "
                        else:
                            for t_id in item:
                                inputs += f"tensor_{self.tensors_mapping[(node.id, tuple(t_id), True)]}, "
                        inputs = inputs[:-2] + "], "
                    elif item == "<None>" or item == "<Generator>":
                        inputs += "None" + ", "
                    elif item == "inf" or item == "-inf":
                        inputs += f'float("{item}"), '
                    elif node.input_types[idx] == "Device" and "cuda" in item:
                        if self.cpu:
                            inputs += '"cpu", '
                        else:
                            inputs += f'"{self.cuda}", '
                    elif isinstance(item, str):
                        inputs += f'"{item}", '
                    else:
                        inputs += str(item) + ", "
            return inputs[:-2]

        def _generate_outputs_str(node, override=True):
            def _generate_output_tensor_str(node, output_tensors, override):
                (_, t_id, _) = output_tensors.pop(0)
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if t_id in self.input_tensor_ids:
                    replay_t_id = self.tensors_mapping[(node.id, t_id, False)]
                    if (
                        replay_t_id not in self.unchangeable_intermediate_tensors
                        and replay_t_id not in self.instantiate
                        and (override or replay_t_id in self.special_tensors)
                    ):
                        self.tensor_registry[replay_t_id] = f"tensor_{replay_t_id}"
                        return f"tensor_{replay_t_id}"
                return "_"

            def _parse_element_type(node, output_type, output_tensors, override):
                outputs = ""
                if output_type.startswith("Tensor"):
                    outputs += (
                        _generate_output_tensor_str(node, output_tensors, override)
                        + ", "
                    )
                elif output_type.startswith("GenericList"):
                    outputs += "["
                    elements_type = output_type[len("GenericList[") : -1].split(",")
                    for element_type in elements_type:
                        outputs += _parse_element_type(
                            node, element_type, output_tensors, override
                        )
                    outputs = outputs[:-2] + "], "
                else:
                    outputs += "_, "
                return outputs

            try:
                outputs = ""
                output_tensors = get_output_tensors(node)
                if len(output_tensors) == 0:
                    return "_"

                for output_type in node.output_types:
                    outputs += _parse_element_type(
                        node, output_type, output_tensors, override
                    )

                assert len(output_tensors) == 0
                outputs = outputs[:-2]
                return outputs
            except Exception as e:
                logger.info(f"Generate outputs error: {e}, {node.id}")
                exit(1)

        def _generate_run_ops_str(override):
            code_str = ""
            exec_template = """        {outputs} = {func}[0]({inputs})"""
            start_special = False
            for node in self.sorted_nodes:
                if node.name == "record_param_comms" and not self.compute_only:
                    if node.id in self.parallel_nodes_ids:
                        if not start_special:
                            code_str += "        with torch.cuda.stream(s1):\n"
                        start_special = True
                        code_str += f"            # node id: {node.id}\n"
                        code_str += f"            _ = traceBench.replaySingle(commsParams, {node.id})\n"
                        if "wait" in node.inputs or "barrier" in node.inputs:
                            if self.wait_delay != 0:
                                code_str += f"            time.sleep({self.wait_delay / 1000.0})\n"
                    else:
                        start_special = False
                        code_str += f"        # node id: {node.id}\n"
                        code_str += f"        _ = traceBench.replaySingle(commsParams, {node.id})\n"
                        if "wait" in node.inputs or "barrier" in node.inputs:
                            if self.wait_delay != 0:
                                code_str += (
                                    f"        time.sleep({self.wait_delay / 1000.0})\n"
                                )
                    continue

                func, output_count = self.funcs[node.id]
                if not func:
                    continue

                func_str = f"funcs[{node.id}]"
                inputs_str = _generate_inputs_str(node)
                outputs_str = _generate_outputs_str(node, override=override)

                if node.id in self.parallel_nodes_ids:
                    if not start_special:
                        code_str += "        with torch.cuda.stream(s1):\n"
                    start_special = True
                    code_str += f"            # node id: {node.id}\n"
                    code_str += (
                        exec_template.format(
                            outputs="    " + outputs_str,
                            func=func_str,
                            inputs=inputs_str,
                        )
                        + "\n"
                    )
                else:
                    start_special = False
                    code_str += f"        # node id: {node.id}\n"
                    code_str += (
                        exec_template.format(
                            outputs=outputs_str, func=func_str, inputs=inputs_str
                        )
                        + "\n"
                    )
                    if override and node.name == "aten::repeat_interleave":
                        current_len = node.input_shapes[0][0]
                        target_len = node.output_shapes[0][0]
                        if current_len < target_len:
                            dtype_str, _ = TORCH_DTYPES_RNG_str[
                                node.output_types[0].removeprefix("Tensor(").rstrip(")")
                            ]
                            code_str += f'        tmp = torch.zeros({target_len - current_len}).to({dtype_str}).cuda("{self.cuda}")\n'
                            t_id = node.outputs[0]
                            if self.tensor_with_device:
                                t_id = tuple(list(t_id)[:5])
                            replay_t_id = self.tensors_mapping[(node.id, t_id, False)]
                            code_str += f"        tensor_{replay_t_id} = torch.cat((tmp, tensor_{replay_t_id}))\n"

            return code_str

        code_str = ""
        skip_nodes_str = ""
        for node in self.skip_node_names:
            skip_nodes_str += f'"{node}", '
        skip_nodes_str = skip_nodes_str[:-2]

        if self.cpu:
            code_str += generate_prefix(
                self.profile_step_label,
                skip_nodes_str,
                self.trace_file,
                "cpu",
                self.compute_only,
                self.tf32,
                self.args.rows,
            )
        else:
            code_str += generate_prefix(
                self.profile_step_label,
                skip_nodes_str,
                self.trace_file,
                self.cuda,
                self.compute_only,
                self.tf32,
                self.args.rows,
            )

        (
            tensor_allocation_str,
            unallocated_tensor_allocation_str,
        ) = _generate_tensor_allocation_str()

        code_str += tensor_allocation_str
        code_str += "\n"

        self.tensor_registry = {k: v for k, v in self.tensor_registry_permanent.items()}

        code_str += "\ndef run_ops(iter, s1):\n"
        code_str += unallocated_tensor_allocation_str
        code_str += "    if iter == 0:\n"
        code_str += _generate_run_ops_str(True)
        code_str += "    else:\n"
        code_str += _generate_run_ops_str(False)

        code_str += generate_suffix(
            self.numWarmupIters, self.numIters, self.cuda_id, str(self.profile_replay)
        )

        if self.dump:
            logger.info(f"Intermediate benchmark file dumped to {self.dump_path}")
            with open(self.dump_path, "w") as f:
                print(code_str, file=f)
        exec(code_str)

    def get_tensor_from_file(
        self,
        node_id,
        tensor_index,
        device,
        shape,
        data_type,
        strides,
    ):
        def to_numpy_data_type(data_type):
            if data_type == torch.int8:
                return np.int8
            elif data_type == torch.uint8:
                return np.uint8
            elif data_type == torch.int16:
                return np.int16
            elif data_type == torch.uint16:
                return np.uint16
            elif data_type == torch.int32:
                return np.int32
            elif data_type == torch.uint32:
                return np.uint32
            elif data_type == torch.int64 or data_type == torch.long:
                return np.int64
            elif data_type == torch.uint64:
                return np.uint64
            else:
                raise ValueError(f"Unsupported data type: {data_type}")

        device = torch.device(device)

        # check if the tensor data file exists
        storage_fn = (
            self.resource_dir
            + "/nid_"
            + str(node_id)
            + "_tid_"
            + str(tensor_index)
            + ".dat"
        )
        if os.path.isfile(storage_fn):
            np_x = np.fromfile(storage_fn, dtype=to_numpy_data_type(data_type))
            if len(shape) == 0:
                np_x = np_x.item()
                x = torch.tensor(np_x)
            else:
                np_x = np.reshape(np_x, shape)
                x = torch.from_numpy(np_x)
                if device != torch.device("cpu"):
                    x = x.cuda(device)
            return x
        else:
            return None

    def get_tensor_from_storage(
        self,
        storage_id,
        data_offset,
        elem_bytes,
        device,
        shape,
        data_type,
        strides,
        tensor_range,
    ):
        if storage_id not in self.tensor_storage_map:
            return None
        tensor_data = self.tensor_storage_map[storage_id]
        if device not in tensor_data[1]:
            if (
                data_type
                in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                    torch.uint16,
                    torch.uint32,
                    torch.uint64,
                    torch.int,
                    torch.long,
                ]
                and tensor_range is not None
            ):
                storage_tensor = torch.randint(
                    tensor_range[0],
                    tensor_range[1] + 1,
                    (tensor_data[0] // elem_bytes,),
                    dtype=data_type,
                    device=device,
                )
            elif data_type in [
                torch.half,
                torch.float32,
                torch.float64,
                torch.bfloat16,
            ]:
                storage_tensor = torch.rand(
                    (tensor_data[0] // elem_bytes), dtype=data_type, device=device
                )
            else:
                storage_tensor = torch.ones(
                    (tensor_data[0] // elem_bytes), dtype=data_type, device=device
                )

            tensor_data[1][device] = storage_tensor
        else:
            storage_tensor = tensor_data[1][device]

        x = torch.empty(0, dtype=data_type)
        device = torch.device(device)
        if device != torch.device("cpu"):
            x = x.cuda(device)
        if strides is None:
            x = x.set_(
                storage_tensor.untyped_storage(),
                storage_offset=data_offset,
                size=shape,
            )
        else:
            x = x.set_(
                storage_tensor.untyped_storage(),
                storage_offset=data_offset,
                size=shape,
                stride=strides,
            )

        return x

    def free_tensor_in_storage(self, storage_id, node_id):
        if (
            self.tensor_allocate_mode == TensorAllcationMode.LAZY_ALLOCATE
            and storage_id in self.tensor_storage_id_to_last_node_id_map
            and node_id >= self.tensor_storage_id_to_last_node_id_map[storage_id]
        ):
            self.tensor_storage_map[storage_id][1] = {}

    def get_data(self, node, is_input, is_comm_node):
        try:
            if self.tensor_allocate_mode == TensorAllcationMode.LAZY_ALLOCATE:
                self.allocate_node_tensors(node, is_input, is_comm_node)
            if is_input:
                data_in = node.inputs
            else:
                data_in = node.outputs
            data_out = []
            tensor_index = 0
            for idx, item in enumerate(data_in):
                if is_tensor(node, idx, is_input):
                    if (
                        is_input
                        and hasattr(node, "pre_load_tensors")
                        and node.pre_load_tensors[tensor_index] is not None
                    ):
                        data_out.append(node.pre_load_tensors[tensor_index])
                    else:
                        self.lookup_cnt += 1
                        if self.tensor_with_device:
                            item = tuple(item[:5])
                        data_out.append(
                            self.tensor_registry[
                                self.tensors_mapping[(node.id, tuple(item), True)]
                            ]
                        )

                    tensor_index += 1

                elif is_tensor_list(node, idx, is_input):
                    self.lookup_cnt += len(item)
                    tensor_list = []
                    for t_id in item:
                        if (
                            is_input
                            and hasattr(node, "pre_load_tensors")
                            and node.pre_load_tensors[tensor_index] is not None
                        ):
                            tensor_list.append(node.pre_load_tensors[tensor_index])
                        else:
                            if self.tensor_with_device:
                                t_id = tuple(t_id[:5])
                            else:
                                t_id = tuple(t_id)
                            tensor_list.append(
                                self.tensor_registry[
                                    self.tensors_mapping[(node.id, t_id, True)]
                                ]
                            )

                        tensor_index += 1
                    data_out.append(tensor_list)

                elif item == "<None>" or item == "<Generator>":
                    data_out.append(None)
                elif item == "inf" or item == "-inf":
                    data_out.append(float(item))
                elif node.input_types[idx] == "Device" and "cuda" in item:
                    if self.cpu:
                        data_out.append("cpu")
                    else:
                        data_out.append(self.cuda)
                else:
                    data_out.append(item)
            return data_out, ""
        except Exception as e:
            return None, f"Inputs error: {e}"

    def get_comm_outputs(self, node):
        try:
            outputs = []
            for idx, item in enumerate(node.outputs):
                if is_tensor(node, idx):
                    self.lookup_cnt += 1
                    if self.tensor_with_device:
                        item = tuple(item[:5])
                    outputs.append(
                        self.tensor_registry[
                            self.tensors_mapping[(node.id, tuple(item), True)]
                        ]
                    )
                elif is_tensor_list(node, idx):
                    self.lookup_cnt += len(item)
                    if self.tensor_with_device:
                        outputs.append(
                            [
                                self.tensor_registry[
                                    self.tensors_mapping[
                                        (node.id, tuple(t_id[:5]), True)
                                    ]
                                ]
                                for t_id in item
                            ]
                        )
                    else:
                        outputs.append(
                            [
                                self.tensor_registry[
                                    self.tensors_mapping[(node.id, tuple(t_id), True)]
                                ]
                                for t_id in item
                            ]
                        )
                elif item == "<None>" or item == "<Generator>":
                    outputs.append(None)
                elif item == "inf" or item == "-inf":
                    outputs.append(float(item))
                elif node.input_types[idx] == "Device" and "cuda" in item:
                    if self.cpu:
                        outputs.append("cpu")
                    else:
                        outputs.append(self.cuda)
                else:
                    outputs.append(item)
            return outputs
        except Exception as e:
            logger.info(f"Outputs error: {e} at node: {node.id}")

    def free_device_memory(self, force: bool = False):
        free_memory = force
        allocated_memory = torch.cuda.memory_allocated(self.device) / 1024 / 1024 / 1024
        if allocated_memory / self.available_memory > self.args.device_memory_threshold:
            free_memory = True

        if free_memory:
            for _, v in self.tensor_storage_map.items():
                if len(v) > 1:
                    v[1] = {}
            self.tensor_registry = {}

    def run_op(self, node, iter, cnt):  # noqa: C901
        if (
            self.tensor_allocate_mode == TensorAllcationMode.LAZY_ALLOCATE
            and self.args.device_memory_threshold != 1.0
        ):
            self.free_device_memory()
        if isinstance(node, commsArgs):
            if self.debug and iter >= self.numWarmupIters:
                start_ns = time.time_ns()
                before_execution = start_ns

            self.commsBench.replaySingle(self.commsParams, node, cnt)
            if self.debug and iter >= self.numWarmupIters:
                after_execution = time.time_ns()

            # Caution: some collectives are running asynchronously, not sure if freeing the tensors immediately
            # may cause any invalid tensor issue. The best solution is to free the tensors after the "wait"
            # op of this collective finishes
            et_node = self.et.nodes[node.id]
            for _, t_id, _ in get_input_tensors(et_node) + get_output_tensors(et_node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                replay_t_id = self.tensors_mapping[(node.id, t_id, True)]
                if (
                    node.id >= self.replay_tensor_id_to_last_node_id_map[replay_t_id]
                    and replay_t_id in self.tensor_registry
                ):
                    del self.tensor_registry[replay_t_id]
                self.free_tensor_in_storage(t_id[1], node.id)
            return True, ""
        else:
            # This is a comms node and it is handled by commsBench.replaySingle
            if node.name == "record_param_comms" or node.name.startswith("c10d::"):
                return True, ""

            if self.debug and iter >= self.numWarmupIters:
                start_ns = time.time_ns()

            func, output_count = self.funcs[node.id]
            inputs, msg = self.get_data(node, True, False)
            if msg != "":
                logger.info(f"Failed to get input data for node {node.id}: {msg}")
                return False, msg
            # TODO: why need this hack?
            # Workaround to eliminate the "strides() called on undefined Tensor" error.
            if node.name == "aten::convolution_backward":
                inputs[-1] = [True, True, True]

            if self.debug and iter >= self.numWarmupIters:
                before_execution = time.time_ns()

            try:
                outputs = []
                if output_count == 0:
                    if node.kernel_backend == "triton":
                        # The last entry in inputs is the stream id, however if we pass that back into func.run()
                        # ET will attemp to dereference the stream again. To fully support triton kernels we need
                        # to keep a mapping between stream address and stream id. This is not implemented yet so
                        # we will capture the raw stream from the device and pass that back to the replay.
                        # In reality all PT2 compute ops are in the same stream to begin with, only collectives
                        # are in different streams, so this is fine.
                        stream = get_raw_stream(self.cuda_id)
                        func.run(*inputs[:-1], stream=stream)
                    else:
                        func(*inputs)
                else:
                    if output_count == 1:
                        tmp = (func(*inputs),)
                    else:
                        tmp = func(*inputs)
                    # Flatten any tensor lists
                    # TODO: Simplify this
                    if not tmp:
                        logger.info(f"Not expect that {node.id} has no output.")
                        return
                    for x in tmp:
                        if isinstance(x, list) and isinstance(x[0], torch.Tensor):
                            outputs.extend(x)
                        elif isinstance(x, torch.Tensor):
                            outputs.append(x)

                if self.debug and iter >= self.numWarmupIters:
                    after_execution = time.time_ns()

                for _, t_id, _ in get_input_tensors(node):
                    if self.tensor_with_device:
                        t_id = tuple(list(t_id)[:5])
                    replay_t_id = self.tensors_mapping[(node.id, t_id, True)]
                    if (
                        node.id
                        >= self.replay_tensor_id_to_last_node_id_map[replay_t_id]
                        and replay_t_id in self.tensor_registry
                    ):
                        del self.tensor_registry[replay_t_id]
                    self.free_tensor_in_storage(t_id[1], node.id)

                for (_, t_id, _), output in zip(get_output_tensors(node), outputs):
                    if self.tensor_with_device:
                        t_id = tuple(list(t_id)[:5])

                    if t_id in self.input_tensor_ids:
                        replay_t_id = self.tensors_mapping[(node.id, t_id, False)]
                        if (
                            node.id
                            < self.replay_tensor_id_to_last_node_id_map[replay_t_id]
                            and replay_t_id not in self.tensor_registry
                        ):
                            self.tensor_registry[replay_t_id] = output
                        else:
                            del output
                    else:
                        del output
                    self.free_tensor_in_storage(t_id[1], node.id)

            except Exception as e:
                msg = f"Run op exception Error: {e}, node id: {node.id}, node name: {node.name}"
                logger.error(msg)
                return False, msg

        if self.profile_memory:
            self.op_allocated_mem[node] = (
                torch.cuda.memory_allocated(self.device) - self.current_allocated_mem
            )
            self.current_allocated_mem = torch.cuda.memory_allocated(self.device)
            self.op_reserved_mem[node] = (
                torch.cuda.memory_reserved(self.device) - self.current_reserved_mem
            )
            self.current_reserved_mem = torch.cuda.memory_reserved(self.device)

        if self.debug and iter >= self.numWarmupIters:
            self.setup_time.append(
                time.time_ns() - start_ns - (after_execution - before_execution)
            )
            self.exec_time.append(after_execution - before_execution)

        return True, ""

    def init_comms(self):
        comms_env_params = comms_utils.read_comms_env_vars()
        logger.info(f"{comms_env_params}, {self.cuda}")

        self.commsBench = CommsReplayManager()
        self.commsBench.comp_replay_manager = self
        self.commsBench.trace_file = self.args.trace_path
        if "://" in self.trace_file:
            self.commsBench.use_remote_trace = True

        parser = argparse.ArgumentParser(description="Execution Trace Comms Replay")
        # Meta internal testing tool seems automatically add --output to the args
        # The testing failed with the following error when commsBench readArgs:
        # "ambiguous option: --output could match --output-path, --output-ranks"
        # so we need to add this to suppress the error
        parser.add_argument("--output", help=argparse.SUPPRESS)
        comms_args = self.commsBench.readArgs(parser)

        self.commsBench.checkArgs(comms_args)

        bootstrap_info = bootstrap_info_holder(
            comms_args.master_ip,
            comms_args.master_port,
            comms_args.num_tpu_cores,
            comms_env_params,
        )
        self.commsParams = commsParamsHolderBase(comms_args)

        self.commsBench.trace_type = "et"
        self.commsBench.initBackend(bootstrap_info, self.commsParams)
        self.commsBench.initBench(self.commsParams, comms_args)
        self.commsBench.replayInit(self.commsParams)

    def remove_op_with_runtime_error(self):
        for cnt, node in enumerate(self.sorted_nodes):
            success, msg = self.run_op(node, 0, cnt)
            if success:
                continue
            if (
                msg.find("RuntimeError: CUDA error") != -1
                or msg.find("torch.OutOfMemoryError") != -1
            ):
                logger.info(f"Can not keep replaying due to {msg}")
                self.add_skipped_nodes(node, msg)
                break

            # The current node failed, if other nodes depend on the output of this
            # node, these nodes will also fail. Allocate the output tensors so other nodes
            # can keep playing.
            for data_type, t_id, shape in get_output_tensors(node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if t_id not in self.input_tensor_ids:
                    continue

                if data_type == "Tensor(nullptr (uninitialized))":
                    t = None
                else:
                    dtype, rng = TORCH_DTYPES_RNG[
                        data_type.removeprefix("Tensor(").rstrip(")")
                    ]
                    replay_t_id = self.tensors_mapping[(node.id, t_id, False)]
                    t = rng(shape).to(dtype)
                    if self.tensor_with_device:
                        if self.tensor_device[replay_t_id] != "cpu" and not self.cpu:
                            t.cuda(self.tensor_device[replay_t_id])
                    else:
                        if not self.cpu:
                            t.cuda(self.device)

                self.tensor_registry[replay_t_id] = t

            self.add_skipped_nodes(node, msg)

    def preprocess_graph(self):
        if not self.compute_only and not self.generator:
            self.init_comms()

        nodes = self.et.get_nodes(clean=True)
        assert isinstance(self.args.subgraph, str)
        if self.args.subgraph != "":
            find_subgraph = False
            for _, n in nodes.items():
                if self.args.subgraph in n.name:
                    root = n
                    find_subgraph = True
                    break
            if not find_subgraph:
                logger.info(f"Cannot find subgraph with name {self.args.subgraph}.")
                exit(1)
        else:
            root = nodes[1]  # 1-base

        self.detect_tensor_device(root)

        self.extract_subgraph(root)

        self.analyze_tensors()

        tensor_with_multiple_shape_count = 0
        for tensor in self.tensor_shapes:
            if len(self.tensor_shapes[tensor]) != 1:
                tensor_with_multiple_shape_count += len(self.tensor_shapes[tensor])
        logger.info(
            f"Tensor count with same identifier but different shapes:{tensor_with_multiple_shape_count}, total tensor: {len(self.tensor_shapes)}"
        )

        if self.generator:
            self.generate_code()
        elif self.tensor_allocate_mode == TensorAllcationMode.PRE_ALLOCATE:
            self.allocate_tensors()

    def benchTime(self):
        # A dictionary to save the benchmark result.
        benchmark_result = {"execution finished": False}

        start_time = datetime.now()
        self.preprocess_graph()
        if self.generator:
            return 0

        if self.args.update_skip_node_file:
            if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "1":
                logger.info(
                    "Please set CUDA_LAUNCH_BLOCKING=1 to get accurate skip node list."
                )
                benchmark_result["execution finished"] = False
                return benchmark_result

            self.remove_op_with_runtime_error()
            actual_skip_node_count = len(self.actual_skip_nodes)
            if actual_skip_node_count > self.initial_skip_node_count:
                with open(self.args.skip_node_file, "w") as outfile:
                    json.dump(self.actual_skip_nodes, outfile, indent=4)
                benchmark_result["execution finished"] = False
            else:
                benchmark_result["execution finished"] = True
            return benchmark_result

        logger.info("Start execution... ")

        total_time = 0.0
        event_1 = torch.cuda.Event(enable_timing=True)
        event_2 = torch.cuda.Event(enable_timing=True)

        def run_ops(event_1, event_2, iter):
            if not self.compute_only:
                self.commsBench.replayIter = iter
            event_1.record()
            for cnt, node in enumerate(self.sorted_nodes):
                success, _ = self.run_op(node, iter, cnt)
                if not success:
                    return False
            event_2.record()
            if not self.compute_only:
                self.commsBench.resetComms()
                # make sure all ops are completed
                with param_profile.paramProfile(
                    description=f"# PARAM replay {self.commsBench.replayIter} post-replay global sync"
                ):
                    self.commsBench.backendFuncs.sync_barrier(
                        self.commsBench.collectiveArgs
                    )
            torch.cuda.synchronize(self.device)
            if not self.compute_only:
                self.commsBench.backendFuncs.clear_memory(
                    self.commsBench.collectiveArgs
                )
            gc.collect()
            torch.cuda.empty_cache()

            return True

        # log real time qps every # iterations.
        qps_print_interval = 10

        prev_iter = self.numWarmupIters

        def run_iter(iter):
            nonlocal prev_iter
            nonlocal qps_print_interval
            nonlocal total_time

            logger.info(f"iteration = {iter}")
            if self.et_profile:
                if iter == self.numWarmupIters:
                    et.start()
                if iter == self.numWarmupIters + 1:
                    et.stop()
                    et.unregister_callback()
            if iter == prev_iter:
                start_ns = time.time_ns()
            if iter == prev_iter + qps_print_interval:
                logger.info(
                    f"Current QPS: {int(qps_print_interval / ((time.time_ns() - start_ns) / 1000000000))}"
                )
                logger.info(
                    f"Replay {qps_print_interval} iterations time: {(time.time_ns() - start_ns) / 1000000.0} ms"
                )
                prev_iter = iter
                start_ns = time.time_ns()

            if self.tensor_allocate_mode == TensorAllcationMode.LAZY_ALLOCATE:
                self.free_device_memory(force=True)
            else:
                self.reset_registry()
            ret = run_ops(event_1, event_2, iter)
            if iter >= self.numWarmupIters:
                total_time += event_1.elapsed_time(event_2)

            return ret

        if self.et_profile:
            et_file = "/tmp/replay_et.json"
            et = ExecutionTraceObserver()
            et.register_callback(et_file)

        if not self.compute_only:
            # since the comp replay will pick the 2nd iteration nodes, comm replay also needs
            if len(self.profile_step_node_ids) > 1:
                commNodes = []
                for node in self.commsBench.comms_trace[: self.commsBench.max_msg_cnt]:
                    if (
                        node.id > self.profile_step_node_ids[0]
                        and node.id < self.profile_step_node_ids[1]
                    ):
                        commNodes.append(node)
            else:
                commNodes = self.commsBench.comms_trace[: self.commsBench.max_msg_cnt]

            # commNodes is a list of commsArgs, since commsArgs also contains node id, it
            # can be mixed with sorted_nodes and re-sort them by id, this is an example after sort:
            # (Node 1000(comp op)) -> (Node 1001(name == "record_param_comms")) -> (commsArgs 1001) -> (Node 1002(comp op))
            # Function run_ops(self, node, iter, cnt) will check the type of the input node, if it is a "Node"
            # and its name is "record_param_comms", skip it; if it is a "commsArgs", use comm_replay to replay it
            # TODO: replace the "record_param_comms" node with commsArgs.
            self.sorted_nodes = self.sorted_nodes + commNodes
            self.sorted_nodes.sort(key=lambda x: x.id)
            self.commsBench.replay_start_time = time.monotonic_ns()

        torch.cuda.set_device(self.cuda_id)

        if self.profile_replay:
            try:
                from aiplatform.monitoring.unitrace.upload_manifold import (
                    export_trace_func,
                )

                rank = self.comms_env_params["local_rank"]
                on_trace_ready = export_trace_func(
                    "/tmp",
                    worker_name=f"rank-{rank}",
                    bucket_name="hpc_traces",
                    zoomer_request_callsite="hpc",
                )
            except ImportError:
                on_trace_ready = trace_handler
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=self.numWarmupIters, active=self.numIters
                ),
                record_shapes=True,
                on_trace_ready=on_trace_ready,
            ) as prof:
                success = True
                for iter in range(self.numWarmupIters + self.numIters):
                    if not run_iter(iter):
                        success = False
                        break
                    prof.step()
                benchmark_result["execution finished"] = success
        else:
            success = True
            for iter in range(self.numWarmupIters + self.numIters):
                if not run_iter(iter):
                    success = False
                    break
            benchmark_result["execution finished"] = success

        if not benchmark_result["execution finished"]:
            return benchmark_result

        if self.profile_memory:
            logger.info("Allocated GPU memory(B):")
            for node in dict(
                sorted(
                    self.op_allocated_mem.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:100]
            ):
                logger.info(f"{node.id}, {self.op_allocated_mem[node]}")
            logger.info("Reserved GPU memory(B):")
            for node in dict(
                sorted(
                    self.op_reserved_mem.items(), key=lambda item: item[1], reverse=True
                )[:100]
            ):
                logger.info(f"{node.id}, {self.op_reserved_mem[node]}")
        logger.info("Replay finished")
        logger.info(f"Replay time per iteration: {total_time / self.numIters} ms")
        logger.info(
            f"Operator coverage: {len(self.sorted_nodes)} / {len(self.sorted_nodes) + self.n_skipped_nodes} = {len(self.sorted_nodes) / (len(self.sorted_nodes) + self.n_skipped_nodes)}"
        )
        end_time = datetime.now()

        try:
            from et_replay.vendor_internal.fb_internal import generate_query_url
        except ImportError:
            logger.info("FB internals not present")
        else:
            generate_query_url(start_time, end_time, self.cuda_id)

        if self.debug:
            logger.info(f"Setup time: {sum(self.setup_time) / 1000000.0}")
            logger.info(f"Execution time: {sum(self.exec_time) / 1000000.0}")

            logger.info(f"Input time: {self.input_total_time / 1000000.0}")
            logger.info(f"Output time: {self.output_total_time / 1000000.0}")
            logger.info(f"Lookup count: {self.lookup_cnt}")
            logger.info(f"Remap tensor list size: {len(self.tensors_mapping)}")

            logger.info(
                f"Execution time: 50th:{np.percentile(self.exec_time, 50) / 1000.0}ms\t90th:{np.percentile(self.exec_time, 90) / 1000.0}ms\t95th:{np.percentile(self.exec_time, 95) / 1000.0}ms"
            )

        if not self.compute_only:
            self.commsBench.reportBenchTime()

        return benchmark_result

    def readComputeArgs(self, check_args: bool = True):
        parser = argparse.ArgumentParser(description="Execution Trace Compute Replay")
        parser.add_argument(
            "--warmup-iter", type=int, default=5, help="Number of warm up iterations."
        )
        parser.add_argument(
            "--iter", type=int, default=10, help="Number of replay iterations."
        )
        parser.add_argument(
            "--input",
            type=str,
            required=False,
            help="Input execution trace json file for single process use.",
        )
        parser.add_argument(
            "--profile-replay",
            default=False,
            action="store_true",
            help="Profile replay.",
        )
        parser.add_argument(
            "--profile-memory",
            default=False,
            action="store_true",
            help="Profile memory usage in replay.",
        )
        parser.add_argument(
            "--et",
            action="store_true",
            default=False,
            help="Capture execution trace for replay.",
        )
        parser.add_argument(
            "--cuda",
            type=int,
            default=-1,
            help="cuda device id, if not specify, will use the default cuda device.",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
            help="Enable debug mode.",
        )
        parser.add_argument(
            "-g",
            "--generator",
            action="store_true",
            default=False,
            help="Enable code generator mode.",
        )
        parser.add_argument(
            "--dump",
            action="store_true",
            default=False,
            help="Dump generated benchmark source file.",
        )
        parser.add_argument(
            "--dump-path",
            type=str,
            required=False,
            default="./",
            help="Path to dump generated benchmark file.",
        )
        parser.add_argument(
            "--trace-path",
            type=str,
            required=False,
            help="File path to read the trace. All rank read their own trace file.",
        )
        parser.add_argument(
            "-c",
            "--compute",
            action="store_true",
            default=False,
            help="Replay compute only.",
        )
        parser.add_argument(
            "--subgraph",
            type=str,
            required=False,
            default="",
            help="Root node name of subgraph.",
        )
        parser.add_argument(
            "--delay",
            type=int,
            default=0,
            help="Delayed time in ms for wait communication operators.",
        )
        parser.add_argument(
            "--tf32",
            action="store_true",
            default=False,
            help="Enable tf32.",
        )
        parser.add_argument(
            "--cpu",
            action="store_true",
            default=False,
            help="Replay on cpu.",
        )
        parser.add_argument(
            "--skip-node-file",
            type=str,
            required=False,
            default="",
            help="Path to the file that contains the list of nodes to skip.",
        )
        parser.add_argument(
            "--update-skip-node-file",
            action="store_true",
            default=False,
            help="When true, the node skip list will be updated with the nodes that are skipped during replay.",
        )

        parser.add_argument(
            "--enable-lazy-tensor-allocation",
            action="store_true",
            default=False,
            help="When true, the tensors will be allocated lazily during replay.",
        )

        parser.add_argument(
            "--device-memory-threshold",
            type=float,
            default=1.0,
            help="When the device memory usage is above this threshold, the replay will release allocated tensors.",
        )
        self.args, _ = parser.parse_known_args()

        # Check if both 'input' and 'trace_path' are not provided
        if check_args and self.args.input is None and self.args.trace_path is None:
            parser.print_help(sys.stderr)
            sys.exit(1)


def main():
    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    replay_manager = ExgrReplayManager()
    replay_manager.readComputeArgs()
    replay_manager.initBench()
    benchmark_result = replay_manager.benchTime()
    if benchmark_result["execution finished"]:
        logger.info("Replay finished successfully.")
        sys.exit(0)
    else:
        logger.info("Replay failed.")
        sys.exit(-1)


if __name__ == "__main__":
    main()
