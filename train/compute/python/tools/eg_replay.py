import argparse
import gc
import json
from collections import defaultdict
import time
import torch

from ..lib import pytorch as lib_pytorch
from ..lib.init_helper import load_modules
from ..workloads import pytorch as workloads_pytorch
from .eg_replay_utils import (
    build_torchscript_func,
    is_backward_aten,
    is_backward_parent,
    is_op,
    is_output_tensor,
    is_tensor,
    TORCH_DTYPES_RNG,
    trace_handler,
)
from .execution_graph import ExecutionGraph


class ExgrReplayManager:
    def __init__(self, exgr, args):
        with open(exgr, 'r') as f:
            self.exgr = ExecutionGraph(json.load(f))
        self.numWarmupIters = args.warmup
        self.numIters = args.iter
        self.profile_replay = args.profile_replay
        self.profile_memory = args.profile_memory

        # Permanent
        self.tensor_registry_permanent = {}
        self.dependency_permanent = defaultdict(int)
        self.sorted_nodes = []
        self.funcs = {}
        # Mark some intermediate tensors (output of operators) as unchangeable
        self.unchangeable_intermediate_tensors = set()
        # Unique tensors in execution graph identified by [tensor_id, storage_id, offset, num_elem, elem_bytes]
        self.original_unique_tensors = set()
        # Number Unique tensors in replay since unique tensors in eg may have multiple shapes and to accommodate that
        # in replay we treat tensors with same identifier but different shapes as different tensors
        self.replay_unique_tensor_num = 0
        # Map unique tensor with the node id of its operation in eg to unique tensors in replay. We assume
        # the shape of a tensor for an operation keeps the same (e.g., a tensor can be both input and output)
        self.tensors_mapping = {}
        # Dict that stores the shape of each unique tensor in replay
        self.replay_tensors_shapes = {}
        # Dict that stores the shapes of a tensor that has appeared, for the convenience of quickly determining whether
        # to create a unique tensor in replay if the identifier is same but shape is different
        self.tensor_shapes = defaultdict(set)
        # Mark those tensors that occur first as an input in the original run as needing to be instantiated in replay
        # at the very beginning
        self.instantiate = set()
        # Tensors that should be instantiated on cpu, e.g., input of aten::pin_memory and aten::to
        self.cpu_tensor = set()
        # Temporary
        self.tensor_registry = {}
        # Skip the node if their names contain any of the following strings.
        self.skip_node_names = ["DataLoader", "aten::set_"]

        if self.profile_memory:
            self.current_allocated_mem = 0
            self.current_reserved_mem = 0
            self.op_allocated_mem = {}
            self.op_reserved_mem = {}

        self.cuda = torch.device('cuda:0')


    def reset_registry(self):
        self.tensor_registry = {k: (None if v is None else (v if k in self.cpu_tensor else v.cuda(self.cuda))) for k, v in self.tensor_registry_permanent.items()}
        gc.collect()
        torch.cuda.empty_cache()


    def extract_subgraph(self, root):
        """
            return: all nodes in the subgraph, in the order of node ID
        """
        def _dfs_traverse(root):
            for child in root.children:
                if any(x in child.name for x in self.skip_node_names):
                    continue

                # if "DataLoader" in child.name or "aten::set_" in child.name:
                #     continue

                if (is_backward_aten(child)) or (is_op(child, strict=True) and not is_backward_parent(child)):
                    self.sorted_nodes.append(child)

                    # Tensors dependency
                    for idxi, ip in enumerate(child.inputs):
                        if is_tensor(child, idxi):
                            if 'GenericList' in child.input_types[idxi]:
                                for t in ip:
                                    self.dependency_permanent[tuple(t)] += 1
                            else:
                                self.dependency_permanent[ip] += 1

                    # Build aten funcs
                    func, output_count = build_torchscript_func(child)
                    self.funcs[child.id] = (func, output_count)
                else:
                    _dfs_traverse(child)

        _dfs_traverse(root)
        self.sorted_nodes = sorted(self.sorted_nodes, key=lambda x: x.id)
        print("#Operations to execute: ", len(self.sorted_nodes))


    def analyze_tensors(self):

        def add_unique_tensor(n, ip, shape):
            # If we did not see this tensor before, add it as a unique tensor
            if ip not in self.original_unique_tensors:
                self.original_unique_tensors.add(ip)
                self.replay_unique_tensor_num += 1
                self.tensors_mapping[(n.id, ip)] = self.replay_unique_tensor_num
                self.replay_tensors_shapes[self.tensors_mapping[(n.id, ip)]] = shape
                self.tensor_shapes[ip].add((self.tensors_mapping[(n.id, ip)], tuple(shape)))
                return

            # If we saw this tensor before but with a different shape, add it as a unique tensor
            for (relay_id, pre_shape) in self.tensor_shapes[ip]:
                if tuple(shape) == pre_shape:
                    self.tensors_mapping[(n.id, ip)] = relay_id
                    return

            self.replay_unique_tensor_num += 1
            self.tensors_mapping[(n.id, ip)] = self.replay_unique_tensor_num
            self.replay_tensors_shapes[self.tensors_mapping[(n.id, ip)]] = shape
            self.tensor_shapes[ip].add((self.tensors_mapping[(n.id, ip)], tuple(shape)))


        for n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if not is_tensor(n, idx):
                    continue
                if 'GenericList' in n.input_types[idx]:
                    for idxi, t in enumerate(ip):
                        if tuple(t) not in self.dependency_permanent.keys():
                            continue
                        else:
                            add_unique_tensor(n, tuple(t), n.input_shapes[idx][idxi])
                else:
                    if ip in self.dependency_permanent.keys():
                        add_unique_tensor(n, ip, n.input_shapes[idx])
            for idx, ip in enumerate(n.outputs):
                if not is_output_tensor(n, idx):
                    continue
                if 'GenericList' in n.output_types[idx]:
                    for idxi, t in enumerate(ip):
                        if tuple(t) not in self.dependency_permanent.keys():
                            continue
                        else:
                            add_unique_tensor(n, tuple(t), n.output_shapes[idx][idxi])
                else:
                    if ip in self.dependency_permanent.keys():
                        add_unique_tensor(n, ip, n.output_shapes[idx])

        # Simulate the execution progress and record the output tensors we have seen so far
        output_set = set()
        for n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if not is_tensor(n, idx):
                    continue
                if 'GenericList' in n.input_types[idx]:
                    for t in ip:
                        if tuple(t) in self.dependency_permanent.keys() and self.tensors_mapping[(n.id, tuple(t))] not in output_set:
                            self.instantiate.add(self.tensors_mapping[(n.id, tuple(t))])
                else:
                    if ip in self.dependency_permanent.keys() and self.tensors_mapping[(n.id, ip)] not in output_set:
                        self.instantiate.add(self.tensors_mapping[(n.id, ip)])
            for idx, ip in enumerate(n.outputs):
                if not is_output_tensor(n, idx):
                    continue
                if 'GenericList' in n.output_types[idx]:
                    for t in ip:
                        if tuple(t) in self.dependency_permanent.keys():
                            output_set.add(self.tensors_mapping[(n.id, tuple(t))])
                else:
                    if ip in self.dependency_permanent.keys():
                        output_set.add(self.tensors_mapping[(n.id, ip)])


    def allocate_tensors(self):
        # Instantiation of tensors:
        for n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if is_tensor(n, idx):
                    if 'GenericList' in n.input_types[idx]:
                        for idxi, t in enumerate(ip):
                            if tuple(t) in self.dependency_permanent.keys() and \
                                    self.tensors_mapping[(n.id, tuple(t))] not in self.tensor_registry_permanent.keys() and \
                                    (n.name == "aten::embedding_bag" or self.tensors_mapping[(n.id, tuple(t))] in self.instantiate):
                                try:
                                    dtype, rng = TORCH_DTYPES_RNG[n.input_types[idx][idxi].lstrip('Tensor(').rstrip(')')]
                                    self.tensor_registry_permanent[self.tensors_mapping[(n.id, tuple(t))]] = rng(n.input_shapes[idx][idxi]).to(dtype)
                                    # Mark offsets tensor for retrieving embedding table as unchangeable
                                    # since we want to explicitly specify its distribution
                                    if n.name == "aten::embedding_bag":
                                        self.unchangeable_intermediate_tensors.add(self.tensors_mapping[(n.id, tuple(t))])
                                except KeyError:
                                    if n.input_types[idx][idxi] != 'Tensor(nullptr (uninitialized))':
                                        print("KeyError in list: ", n.id, ip, n.input_types[idx][idxi])
                                    self.tensor_registry_permanent[self.tensors_mapping[(n.id, tuple(t))]] = None
                    else:
                        if ip in self.dependency_permanent.keys() and \
                                self.tensors_mapping[(n.id, ip)] not in self.tensor_registry_permanent.keys() and \
                                (n.name == "aten::embedding_bag" or self.tensors_mapping[(n.id, ip)] in self.instantiate):
                            try:
                                dtype, rng = TORCH_DTYPES_RNG[n.input_types[idx].lstrip('Tensor(').rstrip(')')]
                                self.tensor_registry_permanent[self.tensors_mapping[(n.id, ip)]] = rng(n.input_shapes[idx]).to(dtype)
                                if n.name == "aten::embedding_bag":
                                    self.unchangeable_intermediate_tensors.add(self.tensors_mapping[(n.id, ip)])
                                if (n.name == "aten::pin_memory" or n.name == "aten::to") and idx == 0:
                                    self.cpu_tensor.add(self.tensors_mapping[(n.id, ip)])
                            except KeyError:
                                if n.input_types[idx] != 'Tensor(nullptr (uninitialized))':
                                    print("KeyError: ", n.id, ip, n.input_types[idx])
                                self.tensor_registry_permanent[self.tensors_mapping[(n.id, ip)]] = None
            ######
            # Workaround to match offsets for embedding table
            # Currently assume a uniform distribution
            if n.name == "aten::embedding_bag":
                indices_tensor_shape = n.input_shapes[1][0]
                offsets_tensor_shape = n.input_shapes[2][0]
                nnz = indices_tensor_shape / offsets_tensor_shape
                for i in range(offsets_tensor_shape):
                   self.tensor_registry_permanent[self.tensors_mapping[(n.id, n.inputs[2])]][i] = i * nnz
            ######
        # print(len(self.tensor_registry_permanent))


    def preprocess_graph(self):
        nodes = self.exgr.get_nodes(clean=True)
        root_node = nodes[1] # 1-base

        self.extract_subgraph(root_node)
        # self._dfs_traverse(root_node)
        self.analyze_tensors()

        tensor_with_multiple_shape_count = 0
        for tensor in self.tensor_shapes:
            if len(self.tensor_shapes[tensor]) != 1:
                tensor_with_multiple_shape_count += len(self.tensor_shapes[tensor])
        print(f"Tensor count with same identifier but different shapes:{tensor_with_multiple_shape_count}, total tensor: {len(self.tensor_shapes)}")

        self.allocate_tensors()
        self.reset_registry()


    def run_op(self, node):
        func, output_count = self.funcs[node.id]
        if not func:
            return

        inputs = []
        for idx, item in enumerate(node.inputs):
            if is_tensor(node, idx):
                if 'GenericList' in node.input_types[idx]:
                    ts = []
                    for t in item:
                        ts.append(self.tensor_registry[self.tensors_mapping[(node.id, tuple(t))]])
                    inputs.append(ts)
                else:
                    inputs.append(self.tensor_registry[self.tensors_mapping[(node.id, item)]])
            else:
                if item == '<None>' or item == '<Generator>':
                    inputs.append(None)
                elif item == 'inf':
                    inputs.append(float('inf'))
                elif item == '-inf':
                    inputs.append(float('-inf'))
                else:
                    inputs.append(item)

        ######
        # Workaround to eliminate the "strides() called on undefined Tensor" error
        if node.name == "aten::convolution_backward":
            inputs[-1] = [True, True, True]
        ######

        try:
            if output_count == 1:
                outputs = (func(*inputs),)
            else:
                outputs = func(*inputs)
        except Exception as e:
            print("Run op exception Error:", e, node.id, inputs)

        for idx, item in enumerate(node.outputs):
            if is_output_tensor(node, idx):
                if 'GenericList' in node.output_types[idx]:
                    for idxi, t in enumerate(item):
                        if tuple(t) in self.dependency_permanent.keys() and self.tensors_mapping[(node.id, tuple(t))] not in self.unchangeable_intermediate_tensors:
                            self.tensor_registry[self.tensors_mapping[(node.id, tuple(t))]] = outputs[idx][idxi]
                else:
                    if item in self.dependency_permanent.keys() and self.tensors_mapping[(node.id, item)] not in self.unchangeable_intermediate_tensors:
                        if self.tensors_mapping[(node.id, item)] not in self.instantiate:
                            self.tensor_registry[self.tensors_mapping[(node.id, item)]] = outputs[idx]

        if self.profile_memory:
            self.op_allocated_mem[node] = torch.cuda.memory_allocated(self.cuda) - self.current_allocated_mem
            self.current_allocated_mem = torch.cuda.memory_allocated(self.cuda)
            self.op_reserved_mem[node] = torch.cuda.memory_reserved(self.cuda) - self.current_reserved_mem
            self.current_reserved_mem = torch.cuda.memory_reserved(self.cuda)


    def benchTime(self):
        self.preprocess_graph()
        print("Start to execution: ")
        time.sleep(10)
        total_time = 0.0
        event_1 = torch.cuda.Event(enable_timing=True)
        event_2 = torch.cuda.Event(enable_timing=True)

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
                    event_1.record()
                    for node in self.sorted_nodes:
                        self.run_op(node)
                    event_2.record()
                    torch.cuda.synchronize()
                    if iter >= self.numWarmupIters:
                        total_time += event_1.elapsed_time(event_2)
                    # Comment out this for now since it will introduce additional cudaMalloc
                    # self.reset_registry()
                    prof.step()
                    # print(iter, torch.cuda.memory_allocated(self.cuda))
            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
        else:
            for iter in range(self.numWarmupIters + self.numIters):
                event_1.record()
                for node in self.sorted_nodes:
                    self.run_op(node)
                event_2.record()
                torch.cuda.synchronize()
                if iter >= self.numWarmupIters:
                    total_time += event_1.elapsed_time(event_2)
                # Comment out this for now since it will introduce additional cudaMalloc
                # self.reset_registry()

        if self.profile_memory:
            print("Allocated GPU memory(B):")
            for node in dict(sorted(self.op_allocated_mem.items(), key=lambda item: item[1], reverse=True)[:100]):
                print(node.id, self.op_allocated_mem[node])
            print("Reserved GPU memory(B):")
            for node in dict(sorted(self.op_reserved_mem.items(), key=lambda item: item[1], reverse=True)[:100]):
                print(node.id, self.op_reserved_mem[node])

        # print("Replay time{}: {:.2f} ms".format(
        #     " (profiled)" if self.profile_replay else "",
        #     total_time / self.numIters
        # ))


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
