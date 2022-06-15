import argparse
import json
import gc
import torch
from collections import defaultdict
from ..lib import pytorch as lib_pytorch
from ..lib.init_helper import load_modules
from ..workloads import pytorch as workloads_pytorch
from .execution_graph import ExecutionGraph
from .eg_replay_utils import is_tensor, is_output_tensor, is_qualified, trace_handler, TORCH_DTYPES_RNG, build_torchscript_func


class ExgrReplayManager:
    def __init__(self, exgr, args):
        with open(exgr, 'r') as f:
            self.exgr = ExecutionGraph(json.load(f))
        self.numWarmupIters = args.warmup
        self.numIters = args.iter
        self.profile_replay = args.profile_replay

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
        # Mark all intermediate tensors
        self.intermediate = set()

        # Temporary
        self.tensor_registry = {}
        self.dependency = {}


    def reset_registry(self):
        self.dependency = self.dependency_permanent.copy()
        self.tensor_registry = {k: (v.cuda() if v is not None else None) for k, v in self.tensor_registry_permanent.items()}
        gc.collect()
        torch.cuda.empty_cache()


    def _dfs_traverse(self, node):
        # moc: maximum output count
        # peculiar: has more outputs than its parents
        # processed: the whole subtree has been processed (ops taken or neglected)
        def get_moc_and_peculiar(node, last_moc):
            if len(node.outputs) > last_moc:
                return len(node.outputs), True
            return last_moc, False

        def dfs(node, depth, moc, peculiar, processed):
            # Update info of the current node
            moc, peculiar = get_moc_and_peculiar(node, moc)
            ret_peculiar, ret_processed = peculiar, processed
            c_peculiars, c_processeds = [], []

            # Search the subtree
            for child in node.children:
                c_peculiar, c_processed = dfs(child, depth+1, moc, peculiar, processed)
                c_peculiars.append(c_peculiar)
                c_processeds.append(c_processed)

            next_level_has_peculiar = any(c_peculiars)
            next_level_has_processed = any(c_processeds)

            # Either there's a peculiar op or there's a processed subtree in the next level
            # Example: aten::cross_entropy_loss

            # print(node.id, node.name, next_level_has_peculiar, next_level_has_processed)
            if next_level_has_peculiar or next_level_has_processed:
                for idxc, c in enumerate(node.children):
                    # Take this op if not processed
                    if not c_processeds[idxc] and is_qualified(c):
                        self.sorted_nodes.append((c.id, c))

                        # Tensors dependency
                        for idxi, ip in enumerate(c.inputs):
                            if is_tensor(c, idxi):
                                self.dependency_permanent[ip] += 1

                        # Build aten funcs
                        func, output_count = build_torchscript_func(c)
                        self.funcs[c.id] = (func, output_count)

                        # Mark as processed
                        c_processeds[idxc] = True

            # Mark an op and its subtree as processed if all branches are processed
            all_next_level_processed = len(c_processeds) != 0 and all(c_processeds)
            if all_next_level_processed:
                ret_processed = True

            return ret_peculiar, ret_processed

        dfs(node, 0, len(node.outputs), False, False)
        self.sorted_nodes = [(id, node) for id, node in sorted(self.sorted_nodes, key=lambda x: x[0])]


    def analyze_tensors(self):

        def add_unique_tensor(n, idx, ip, shape, add_to_intermediate=False):
            # If we did not see this tensor before, add it as a unique tensor
            if ip not in self.original_unique_tensors:
                self.original_unique_tensors.add(ip)
                self.replay_unique_tensor_num += 1
                self.tensors_mapping[(n.id, ip)] = self.replay_unique_tensor_num
                self.replay_tensors_shapes[self.tensors_mapping[(n.id, ip)]] = shape
                self.tensor_shapes[ip].add((self.tensors_mapping[(n.id, ip)], tuple(shape)))
                if add_to_intermediate:
                    self.intermediate.add(self.tensors_mapping[(n.id, ip)])
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
            if add_to_intermediate:
                self.intermediate.add(self.tensors_mapping[(n.id, ip)])

        for _, n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if not is_tensor(n, idx) or ip not in self.dependency_permanent.keys():
                    continue
                add_unique_tensor(n, idx, ip, n.input_shapes[idx], add_to_intermediate=False)
            for idx, ip in enumerate(n.outputs):
                if not is_output_tensor(n, idx) or ip not in self.dependency_permanent.keys():
                    continue
                add_unique_tensor(n, idx, ip, n.output_shapes[idx], add_to_intermediate=True)


    def allocate_tensors(self):
        # Instantiation of tensors:
        for _, n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if is_tensor(n, idx) and \
                        ip in self.dependency_permanent.keys() and \
                        self.tensors_mapping[(n.id, ip)] not in self.tensor_registry_permanent.keys() and \
                        (n.name == "aten::embedding_bag" or self.tensors_mapping[(n.id, ip)] not in self.intermediate):
                    try:
                        dtype, rng = TORCH_DTYPES_RNG[n.input_types[idx].lstrip('Tensor(').rstrip(')')]
                        self.tensor_registry_permanent[self.tensors_mapping[(n.id, ip)]] = rng(n.input_shapes[idx]).to(dtype)
                        # Mark offsets tensor for retrieving embedding table as unchangeable
                        # since we want to specify its distribution
                        if self.tensors_mapping[(n.id, ip)] in self.intermediate:
                            self.unchangeable_intermediate_tensors.add(self.tensors_mapping[(n.id, ip)])
                    except KeyError:
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
        # Get root node
        nodes = self.exgr.get_nodes(clean=True)
        root_node = nodes[1] # 1-base

        # Parse graph
        self._dfs_traverse(root_node)

        # Analyze tensors
        self.analyze_tensors()

        # tensor_with_multiple_shape_count = 0
        # for tensor in self.tensor_shapes:
        #     if len(self.tensor_shapes[tensor]) != 1:
        #         tensor_with_multiple_shape_count += len(self.tensor_shapes[tensor])
        # print("Tensor count with same identifier but different shapes: ", tensor_with_multiple_shape_count)

        # Allocate
        self.allocate_tensors()

        # Reset
        self.reset_registry()


    def run_op(self, node):
        # print("-----")
        # print(node.name, node.inputs, node.outputs)
        inputs = [
            self.tensor_registry[self.tensors_mapping[(node.id, item)]] if is_tensor(node, idx) else \
            (
                None if item == '<None>' else item
            ) for idx, item in enumerate(node.inputs)
        ]

        ######
        # Workaround to eliminate the "strides() called on undefined Tensor" error
        if node.name == "aten::convolution_backward":
            inputs[-1] = [True, True, True]
        ######

        # print(node.name, node.id, [(type(i), i.shape if torch.is_tensor(i) else i, i.dtype if torch.is_tensor(i) else i) for i in inputs], type(node.outputs[0]))
        func, output_count = self.funcs[node.id]
        if output_count == 1:
            outputs = (func(*inputs),)
        else:
            outputs = func(*inputs)

        for output_id, output in zip(node.outputs, outputs):
            if output_id in self.dependency_permanent.keys():
                if self.tensors_mapping[(node.id, output_id)] not in self.unchangeable_intermediate_tensors:
                    self.tensor_registry[self.tensors_mapping[(node.id, output_id)]] = output


    def benchTime(self):
        self.preprocess_graph()
        total_time = 0.0
        event_1 = torch.cuda.Event(enable_timing=True)
        event_2 = torch.cuda.Event(enable_timing=True)

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
        ) as prof:
            for iter in range(self.numWarmupIters + self.numIters):
                event_1.record()
                for _, node in self.sorted_nodes:
                    self.run_op(node)
                event_2.record()
                torch.cuda.synchronize()
                if iter >= self.numWarmupIters:
                    total_time += event_1.elapsed_time(event_2)
                # Comment out this for now since it will introduce additional cudaMalloc
                # self.reset_registry()
                prof.step()

        print("Replay time{}: {:.2f} ms".format(
            " (profiled)" if self.profile_replay else "",
            total_time / self.numIters
        ))


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
