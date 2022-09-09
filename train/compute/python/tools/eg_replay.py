import argparse
from datetime import datetime
import gc
import json
import time
from collections import defaultdict
from functools import reduce

import torch

from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph, NodeType
from torch.profiler import ExecutionGraphObserver

from param_bench.train.compute.python.lib import pytorch as lib_pytorch
from param_bench.train.compute.python.lib.init_helper import load_modules
from param_bench.train.compute.python.workloads import pytorch as workloads_pytorch
from param_bench.train.compute.python.tools.eg_replay_utils import (
    build_fbgemm_func,
    build_torchscript_func,
    fbgemm_input_args_indices,
    generate_fbgemm_tensors,
    has_backward_parent,
    get_input_tensors,
    get_output_tensors,
    is_backward_aten,
    is_fbgemm_backward,
    is_fbgemm_forward,
    is_fbgemm_forward_unweighted,
    is_qualified,
    is_tensor,
    is_tensor_list,
    skip_op,
    TORCH_DTYPES_BYTES,
    TORCH_DTYPES_RNG,
)
from param_bench.train.compute.python.tools.utility import generate_query_url, trace_handler


class ExgrReplayManager:
    def __init__(self, exgr, args):
        with open(exgr, 'r') as f:
            self.exgr = ExecutionGraph(json.load(f))
        self.numWarmupIters = args.warmup
        self.numIters = args.iter
        self.profile_replay = args.profile_replay
        self.profile_memory = args.profile_memory
        self.eg = args.eg
        self.batch = args.batch
        self.cuda_id = args.cuda

        # Permanent
        self.tensor_registry_permanent = {}
        self.dependency_permanent = defaultdict(int)
        self.sorted_nodes = []
        self.funcs = {}
        # Mark some intermediate tensors (output of operators) as unchangeable.
        self.unchangeable_intermediate_tensors = set()
        # Unique tensors in execution graph identified by [tensor_id, storage_id, offset, num_elem, elem_bytes].
        self.original_unique_tensors = set()
        # Number Unique tensors in replay since unique tensors in eg may have multiple shapes and to accommodate that
        # in replay we treat tensors with same identifier but different shapes as different tensors.
        self.replay_unique_tensor_num = 0
        # Map unique tensor with the node id of its operator in eg to unique tensors in replay. We assume
        # in either input or output, the shape of tensors with same id for an operator keeps the same.
        # Same tensor in input and output may be different (e.g., aten::unsqueeze()).
        self.tensors_mapping = {}
        # Dict that stores the shape of each unique tensor in replay.
        self.replay_tensors_shapes = {}
        # Dict that stores the shapes of a tensor that has appeared, for the convenience of quickly determining whether
        # to create a unique tensor in replay if the identifier is same but shape is different.
        self.tensor_shapes = defaultdict(set)
        # Mark those tensors that occur first as an input in the original run as needing to be instantiated in replay
        # at the very beginning.
        self.instantiate = set()
        # Tensors that should be instantiated on cpu, e.g., input of aten::pin_memory and aten::to.
        self.cpu_tensor = set()
        # Tensors lookup dict at runtime.
        self.tensor_registry = {}
        # Skip the node if their names contain any of the following strings.
        self.skip_node_names = ["DataLoader", "aten::set_", "fb::", "c10d::allreduce_", "pyspeech::", "gans_torchscript_ops::"]

        if self.profile_memory:
            self.current_allocated_mem = 0
            self.current_reserved_mem = 0
            self.op_allocated_mem = {}
            self.op_reserved_mem = {}

        if self.cuda_id == -1:
            self.cuda = "cuda"
        else:
            self.cuda = f"cuda:{self.cuda_id}"

        self.device = torch.device(self.cuda)

        self.fbgemm_backward_ops = []

        self.additional_tensors = set()
        self.top_tensors = {}
        self.additional_tensors_size = 0

        self.actual_skip_nodes = []
        self.actual_skip_nodes_cnt = 0

        self.tensor_with_device = True
        # A tensor may appear on multiple devices but here we only store the first device for initialization
        # since device change should be captured in operator execution and be naturally recovered by replaying
        # the operators.
        self.tensor_device = {}

        self.exceptional_nodes = set()


    def detect_tensor_device(self, root):
        # Automatically detect whether the captured tensor information includes device.
        # Just a util to accommodate old and new versions eg and can be removed later.
        def _traverse(root):
            for child in root.children:
                for _, t_id, _ in get_input_tensors(child):
                    if len(list(t_id)) == 5:
                        self.tensor_with_device = False
                        return
                for _, t_id, _ in get_output_tensors(child):
                    if len(list(t_id)) == 5:
                        self.tensor_with_device = False
                        return
                _traverse(child)
        _traverse(root)


    def reset_registry(self):
        if self.tensor_with_device:
            self.tensor_registry = {k: (None if v is None else (v if self.tensor_device[k] == 'cpu' else v.cuda(self.device))) for k, v in self.tensor_registry_permanent.items()}
        else:
            self.tensor_registry = {k: (None if v is None else (v if k in self.cpu_tensor else v.cuda(self.device))) for k, v in self.tensor_registry_permanent.items()}
        gc.collect()
        torch.cuda.empty_cache()


    def extract_subgraph(self, root):
        """
            return: all nodes in the subgraph, in the order of node ID
        """
        def _dfs_traverse(root):
            for child in root.children:
                try:
                    if any(x in child.name for x in self.skip_node_names):
                        self.actual_skip_nodes.append(child.name)
                        self.actual_skip_nodes_cnt += 1
                        continue

                    if is_qualified(child):
                        self.sorted_nodes.append(child)

                        self.top_tensors[child] = set()
                        for _, t_id, _ in get_input_tensors(child):
                            if self.tensor_with_device:
                                t_id = tuple(list(t_id)[:5])
                            self.top_tensors[child].add(t_id)
                        for _, t_id, _ in get_output_tensors(child):
                            if self.tensor_with_device:
                                t_id = tuple(list(t_id)[:5])
                            self.top_tensors[child].add(t_id)

                        for _, t_id, _ in get_input_tensors(child):
                            if self.tensor_with_device:
                                t_id = tuple(list(t_id)[:5])
                            self.dependency_permanent[t_id] += 1

                        func, output_count = self.build_func(child)
                        self.funcs[child.id] = (func, output_count)
                    else:
                        if skip_op(child):
                            self.actual_skip_nodes.append(child.name)
                            self.actual_skip_nodes_cnt += 1
                        _dfs_traverse(child)
                except Exception as e:
                    print(f"Graph parse error: {e}, node id: {child.id}")
                    exit(1)

        _dfs_traverse(root)
        self.sorted_nodes = sorted(self.sorted_nodes, key=lambda x: x.id)
        print("#Operators to execute: ", len(self.sorted_nodes))


    def analyze_subgraph(self, root):
        def _bfs_traverse(node):
            for child in node.children:
                if any(x in child.name for x in self.skip_node_names):
                    continue

                if is_backward_aten(child) or has_backward_parent(child):
                    continue
                else:
                    if child not in self.sorted_nodes and child.type == NodeType.OPERATOR:
                        node = child.parent
                        while (node and node not in self.sorted_nodes):
                            node = node.parent
                        if not node:
                            self.exceptional_nodes.add(child)
                            continue
                        for data_type, t_id, shape in get_output_tensors(child):
                            if self.tensor_with_device:
                                t_id = tuple(list(t_id)[:5])
                            if t_id not in self.top_tensors[node] and \
                                t_id in self.dependency_permanent and t_id not in self.additional_tensors:
                                self.additional_tensors.add(t_id)
                                if shape:
                                    self.additional_tensors_size += reduce(lambda x,y:x*y, shape) * TORCH_DTYPES_BYTES[data_type.lstrip('Tensor(').rstrip(')')]
                    _bfs_traverse(child)

        _bfs_traverse(root)
        # print("Exceptional nodes: ")
        # for node in self.exceptional_nodes:
        #     print(node.id, node.name)
        print(f"Additional allocated {len(self.additional_tensors)} tensors with total size of {self.additional_tensors_size/1024/1024}MB")


    def analyze_tensors(self):
        def add_unique_tensor(node_id, t_id, shape, input, device=-1):
            # If we did not see this tensor before, add it as a unique tensor.
            if t_id not in self.original_unique_tensors:
                self.original_unique_tensors.add(t_id)
                self.replay_unique_tensor_num += 1
                self.tensors_mapping[(node_id, t_id, input)] = self.replay_unique_tensor_num
                self.replay_tensors_shapes[self.tensors_mapping[(node_id, t_id, input)]] = shape
                self.tensor_shapes[t_id].add((self.tensors_mapping[(node_id, t_id, input)], tuple(shape)))
                if self.tensor_with_device:
                    self.tensor_device[self.tensors_mapping[(node_id, t_id, input)]] = device
                return

            # If we saw this tensor before but with a different shape, add it as a unique tensor.
            for (relay_t_id, pre_shape) in self.tensor_shapes[t_id]:
                if tuple(shape) == pre_shape:
                    self.tensors_mapping[(node_id, t_id, input)] = relay_t_id
                    return

            self.replay_unique_tensor_num += 1
            self.tensors_mapping[(node_id, t_id, input)] = self.replay_unique_tensor_num
            self.replay_tensors_shapes[self.tensors_mapping[(node_id, t_id, input)]] = shape
            self.tensor_shapes[t_id].add((self.tensors_mapping[(node_id, t_id, input)], tuple(shape)))
            if self.tensor_with_device:
                self.tensor_device[self.tensors_mapping[(node_id, t_id, input)]] = device


        for node in self.sorted_nodes:
            for _, t_id, shape in get_input_tensors(node):
                if self.tensor_with_device:
                    device = list(t_id)[5]
                    t_id = tuple(list(t_id)[:5])
                    if t_id in self.dependency_permanent.keys():
                        add_unique_tensor(node.id, t_id, shape, input=True, device=device)
                else:
                    if t_id in self.dependency_permanent.keys():
                        add_unique_tensor(node.id, t_id, shape, input=True)

            for _, t_id, shape in get_output_tensors(node):
                if self.tensor_with_device:
                    device = list(t_id)[5]
                    t_id = tuple(list(t_id)[:5])
                    if t_id in self.dependency_permanent.keys():
                        add_unique_tensor(node.id, t_id, shape, input=False, device=device)
                else:
                    if t_id in self.dependency_permanent.keys():
                        add_unique_tensor(node.id, t_id, shape, input=False)

        # Simulate the execution progress and record the output tensors we have seen so far.
        output_set = set()
        for node in self.sorted_nodes:
            for _, t_id, _ in get_input_tensors(node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if t_id in self.dependency_permanent.keys() and self.tensors_mapping[(node.id, t_id, True)] not in output_set:
                    self.instantiate.add(self.tensors_mapping[(node.id, t_id, True)])

            for _, t_id, _ in get_output_tensors(node):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                if t_id in self.dependency_permanent.keys():
                    output_set.add(self.tensors_mapping[(node.id, t_id, False)])


    def allocate_tensors(self):
        for node in self.sorted_nodes:
            if is_fbgemm_forward(node):
                input_args, _ = generate_fbgemm_tensors(node, self.cuda)
            for idx, (data_type, t_id, shape) in enumerate(get_input_tensors(node)):
                if self.tensor_with_device:
                    t_id = tuple(list(t_id)[:5])
                replay_t_id = self.tensors_mapping[(node.id, t_id, True)]
                if t_id in self.dependency_permanent.keys() and \
                        replay_t_id not in self.tensor_registry_permanent.keys() and \
                        (node.name == "aten::embedding_bag" or node.name == "fbgemm::split_embedding_codegen_lookup_sgd_function" \
                        or replay_t_id in self.instantiate):
                    try:
                        if is_fbgemm_forward(node):
                            self.tensor_registry_permanent[replay_t_id] = input_args[idx]
                            if node.name == "fbgemm::split_embedding_codegen_lookup_sgd_function":
                                self.unchangeable_intermediate_tensors.add(replay_t_id)
                        else:
                            dtype, rng = TORCH_DTYPES_RNG[data_type.lstrip('Tensor(').rstrip(')')]
                            self.tensor_registry_permanent[replay_t_id] = rng(shape).to(dtype)
                            if node.name == "aten::embedding_bag":
                                self.unchangeable_intermediate_tensors.add(replay_t_id)
                            if node.name == "aten::pin_memory" and idx == 0:
                                self.cpu_tensor.add(replay_t_id)
                    except KeyError:
                        if data_type != 'Tensor(nullptr (uninitialized))':
                            print("KeyError: ", node.id, t_id, data_type)
                        self.tensor_registry_permanent[replay_t_id] = None

            ######
            # Workaround to match offsets for embedding table
            # Currently assume a uniform distribution.
            if node.name == "aten::embedding_bag":
                indices_tensor_shape = node.input_shapes[1][0]
                offsets_tensor_shape = node.input_shapes[2][0]
                nnz = indices_tensor_shape / offsets_tensor_shape
                for i in range(offsets_tensor_shape):
                   self.tensor_registry_permanent[self.tensors_mapping[(node.id, node.inputs[2], True)]][i] = i * nnz
            ######


    def build_func(self, node):
        if is_fbgemm_forward(node):
            func, output_count = build_fbgemm_func(node, self.cuda)
            self.fbgemm_backward_ops.append((func.backward, node.id))
            return func.forward, output_count
        elif is_fbgemm_backward(node):
            assert self.fbgemm_backward_ops
            backward_op, forward_id = self.fbgemm_backward_ops.pop(-1)
            return backward_op, len(node.output_types)
        func, output_count = build_torchscript_func(node)
        if not func:
            self.actual_skip_nodes.append(node.name)
            self.actual_skip_nodes_cnt += 1
        return func, output_count


    def preprocess_graph(self):
        nodes = self.exgr.get_nodes(clean=True)
        root = nodes[1] # 1-base

        self.detect_tensor_device(root)

        self.extract_subgraph(root)

        # self.analyze_subgraph(root)

        self.analyze_tensors()

        tensor_with_multiple_shape_count = 0
        for tensor in self.tensor_shapes:
            if len(self.tensor_shapes[tensor]) != 1:
                tensor_with_multiple_shape_count += len(self.tensor_shapes[tensor])
        print(f"Tensor count with same identifier but different shapes:{tensor_with_multiple_shape_count}, total tensor: {len(self.tensor_shapes)}")

        self.allocate_tensors()
        self.reset_registry()


    def get_inputs(self, node):
        try:
            if is_fbgemm_forward(node):
                idx_list = fbgemm_input_args_indices(node)
                inputs = [self.tensor_registry[self.tensors_mapping[(node.id, tuple(node.inputs[idx]), True)]] for idx in idx_list]
                if is_fbgemm_forward_unweighted(node):
                    inputs.append(None)
            else:
                inputs = []
                for idx, item in enumerate(node.inputs):
                    if is_tensor(node, idx):
                        if self.tensor_with_device:
                            item = tuple(item[:5])
                        inputs.append(self.tensor_registry[self.tensors_mapping[(node.id, tuple(item), True)]])
                    elif is_tensor_list(node, idx):
                        if self.tensor_with_device:
                            inputs.append([self.tensor_registry[self.tensors_mapping[(node.id, tuple(t_id[:5]), True)]] for t_id in item])
                        else:
                            inputs.append([self.tensor_registry[self.tensors_mapping[(node.id, tuple(t_id), True)]] for t_id in item])
                    elif item == '<None>' or item == '<Generator>':
                        inputs.append(None)
                    elif item == 'inf' or item == '-inf':
                        inputs.append(float(item))
                    elif node.input_types[idx] == 'Device' and 'cuda' in item:
                        inputs.append(self.cuda)
                    else:
                        inputs.append(item)
            return inputs
        except Exception as e:
            print("Inputs error: ", e, node.id)


    def run_op(self, node, iter):
        func, output_count = self.funcs[node.id]
        if not func:
            return
        inputs = self.get_inputs(node)

        ######
        # Workaround to eliminate the "strides() called on undefined Tensor" error.
        if node.name == "aten::convolution_backward":
            inputs[-1] = [True, True, True]
        ######

        # # Workaround to handle tensors are on different devices.
        # if node.name == "aten::mul":
        #     if is_tensor(node, 0) and is_tensor(node, 1):
        #         if inputs[0].is_cuda ^ inputs[1].is_cuda:
        #             if inputs[0].is_cuda:
        #                 inputs[1] = inputs[1].to(self.device)
        #             else:
        #                 inputs[1] = inputs[1].to('cpu')

        # Workaround to handle tensor with same id but different data types (ads_cmf10x_single_iter_512_newest_eg.json).
        if node.name == "aten::index_add_":
            inputs[3] = inputs[3].to(torch.float64)
        if node.name == "aten::index_copy_":
            if node.input_types[3] == "Tensor(double)":
                inputs[3] = inputs[3].to(torch.float64)

        try:
            outputs = []
            if output_count == 0:
                func(*inputs)
            else:
                if output_count == 1:
                    tmp = (func(*inputs),)
                else:
                    tmp = func(*inputs)
                # Flatten any tensor lists
                # TODO: Simplify this
                for x in tmp:
                    if isinstance(x, list) and isinstance(x[0], torch.Tensor):
                        outputs.extend(x)
                    elif isinstance(x, torch.Tensor):
                        outputs.append(x)
        except Exception as e:
            print(f"Run op exception Error: {e}, node id: {node.id}, func: {func}, inputs: {inputs}")
            exit(1)

        for (_, t_id, _), output in zip(get_output_tensors(node), outputs):
            if self.tensor_with_device:
                t_id = tuple(list(t_id)[:5])
            if t_id in self.dependency_permanent.keys() and self.tensors_mapping[(node.id, t_id, False)] not in self.unchangeable_intermediate_tensors:
                if self.tensors_mapping[(node.id, t_id, False)] not in self.instantiate:
                    self.tensor_registry[self.tensors_mapping[(node.id, t_id, False)]] = output

        if self.profile_memory:
            self.op_allocated_mem[node] = torch.cuda.memory_allocated(self.device) - self.current_allocated_mem
            self.current_allocated_mem = torch.cuda.memory_allocated(self.device)
            self.op_reserved_mem[node] = torch.cuda.memory_reserved(self.device) - self.current_reserved_mem
            self.current_reserved_mem = torch.cuda.memory_reserved(self.device)


    def analyze_ops(self):
        fused_cnt = 0
        aten_up_cnt = 0
        aten_cnt = 0
        custom_cnt = 0
        for op in self.actual_skip_nodes:
            if "fused" in op:
                fused_cnt += 1
            elif "aten::record_stream" in op or "aten::set_" in op:
                aten_up_cnt += 1
            elif "aten::" in op:
                aten_cnt += 1
            elif "fb::" in op or "fbgemm::" in op:
                custom_cnt += 1
            else:
                print(op)
        print("fused cnt: ", fused_cnt)
        print("aten unsupported cnt: ", aten_up_cnt)
        print("aten cnt: ", aten_cnt)
        print("custom cnt: ", custom_cnt)
        print("skipped ops: ", self.actual_skip_nodes)


    def benchTime(self):
        start_time = datetime.now()

        self.preprocess_graph()
        print("Start to execution: ")
        time.sleep(2)
        total_time = 0.0
        event_1 = torch.cuda.Event(enable_timing=True)
        event_2 = torch.cuda.Event(enable_timing=True)

        if self.eg:
            eg_file = "/tmp/replay_eg.json"
            eg = ExecutionGraphObserver()
            eg.register_callback(eg_file)

        # Print real time qps every # iterations.
        qps_print_interval = 10

        prev_iter = self.numWarmupIters
        if self.profile_replay:
            with torch.profiler.profile(
                activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                # schedule=torch.profiler.schedule(
                #     skip_first=10,
                #     wait=10,
                #     warmup=10,
                #     active=10,
                # ),
                on_trace_ready=trace_handler,
                # profile_memory=True,
            ) as prof:
                for iter in range(self.numWarmupIters + self.numIters):
                    if self.eg:
                        if iter == self.numWarmupIters:
                            eg.start()
                        if iter == self.numWarmupIters + 1:
                            eg.stop()
                            eg.unregister_callback()
                    if iter == prev_iter:
                        start_ns = time.time_ns()
                    if iter == prev_iter + qps_print_interval:
                        print("Current QPS: ", int(self.batch * qps_print_interval / ((time.time_ns() - start_ns) / 1000000000)))
                        print("Replay {} iterations time: {}ms".format(qps_print_interval, (time.time_ns() - start_ns) / 1000000.0))
                        prev_iter = iter
                        start_ns = time.time_ns()
                    event_1.record()
                    for node in self.sorted_nodes:
                        self.run_op(node, iter)
                    event_2.record()
                    torch.cuda.synchronize()
                    if iter >= self.numWarmupIters:
                        total_time += event_1.elapsed_time(event_2)
                    # Comment out this for now since it will introduce additional cudaMalloc.
                    # self.reset_registry()
                    prof.step()
                print("Execution finished!")
        else:
            for iter in range(self.numWarmupIters + self.numIters):
                if self.eg:
                    if iter == self.numWarmupIters:
                        eg.start()
                    if iter == self.numWarmupIters + 1:
                        eg.stop()
                        eg.unregister_callback()
                if iter == prev_iter:
                    start_ns = time.time_ns()
                if iter == prev_iter + qps_print_interval:
                    print("Current QPS: ", int(self.batch * qps_print_interval / ((time.time_ns() - start_ns) / 1000000000)))
                    print("Replay {} iterations time: {}ms".format(qps_print_interval, (time.time_ns() - start_ns) / 1000000.0))
                    prev_iter = iter
                    start_ns = time.time_ns()
                event_1.record()
                for node in self.sorted_nodes:
                    self.run_op(node, iter)
                event_2.record()
                torch.cuda.synchronize()
                if iter >= self.numWarmupIters:
                    total_time += event_1.elapsed_time(event_2)
                # self.reset_registry()
            print("Execution finished!")

        if self.profile_memory:
            print("Allocated GPU memory(B):")
            for node in dict(sorted(self.op_allocated_mem.items(), key=lambda item: item[1], reverse=True)[:100]):
                print(node.id, self.op_allocated_mem[node])
            print("Reserved GPU memory(B):")
            for node in dict(sorted(self.op_reserved_mem.items(), key=lambda item: item[1], reverse=True)[:100]):
                print(node.id, self.op_reserved_mem[node])

        print("Replay time per iteration: {:.2f} ms".format(total_time / self.numIters))

        print("Operator coverage: {} / {} = {}".format(len(self.sorted_nodes), len(self.sorted_nodes) + self.actual_skip_nodes_cnt,
              len(self.sorted_nodes) / (len(self.sorted_nodes) + self.actual_skip_nodes_cnt)))
        end_time = datetime.now()
        generate_query_url(start_time, end_time, self.cuda_id)


def main():
    parser = argparse.ArgumentParser(description="Execution Graph Replay")
    parser.add_argument(
        "-w", "--warmup", type=int, default=5, help="Number of warm up iterations."
    )
    parser.add_argument(
        "--iter", type=int, default=30, help="Number of replay iterations."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input execution graph json file."
    )
    parser.add_argument(
        "-p", "--profile-replay", action="store_true", help="Profile replay and get trace."
    )
    parser.add_argument(
        "-m", "--profile-memory", action="store_true", help="Profile memory usage in replay."
    )
    parser.add_argument(
        "--eg", action="store_true", default=False, help="Capture execution graph."
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Batch size (number of queries) in one replay iteration, used to calculate QPS."
    )
    parser.add_argument(
        "--cuda", type=int, default=-1, help="cuda device id, if not specify, will use the default"
    )

    args = parser.parse_args()

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    exgr = args.input
    replay_manager = ExgrReplayManager(exgr, args)
    replay_manager.benchTime()

if __name__ == "__main__":
    main()
