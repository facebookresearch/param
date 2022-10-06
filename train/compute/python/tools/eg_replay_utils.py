import re

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import PoolingMode, WeightDecayMode

from param_bench.train.compute.python.lib.pytorch.config_util import create_op_args
from param_bench.train.compute.python.tools.execution_graph import NodeType

from param_bench.train.compute.python.workloads.pytorch.split_table_batched_embeddings_ops import (
    SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator,
    SplitTableBatchedEmbeddingBagsCodegenOp,
)


# TODO: Add all torch dtypes to here
TORCH_DTYPES_RNG = {
    "bool": (torch.bool, torch.ones),
    "int8": (torch.int8, torch.ones),
    "half": (torch.half, torch.ones),
    "int": (torch.int, torch.ones),
    "long": (torch.int64, torch.ones),
    "long int": (torch.int64, torch.ones),
    "float": (torch.float32, torch.randn),
    "double": (torch.float64, torch.randn),
    "unsigned char": (torch.int8, torch.ones),
    "c10::Half": (torch.half, torch.ones),
}

TORCH_DTYPES_RNG_str = {
    "bool": ("torch.bool", "torch.ones"),
    "int8": ("torch.int8", "torch.ones"),
    "half": ("torch.half", "torch.ones"),
    "int": ("torch.int", "torch.ones"),
    "long": ("torch.int64", "torch.ones"),
    "long int": ("torch.int64", "torch.ones"),
    "float": ("torch.float32", "torch.randn"),
    "double": ("torch.float64", "torch.randn"),
    "unsigned char": ("torch.int8", "torch.ones"),
    "c10::Half": ("torch.half", "torch.ones"),
}

TORCH_DTYPES_BYTES = {
    "bool": 1,
    "int8": 1,
    "half": 2,
    "int": 4,
    "long": 8,
    "long int": 8,
    "float": 4,
    "double": 8,
    "unsigned char": 1,
    "c10::Half": 2,
}


def is_tensor_list(n, idx):
    return isinstance(idx, int) and "GenericList[Tensor" in n.input_types[idx]


def is_tensor(n, idx):
    return (
        isinstance(idx, int)
        and "Tensor" in n.input_types[idx]
        and "GenericList" not in n.input_types[idx]
    )


def is_op(node, strict=False):
    if not strict:
        return node.type == NodeType.OPERATOR
    return node.type == NodeType.OPERATOR and (
        node.parent is not None and node.parent.type != NodeType.OPERATOR
    )


def has_backward_parent(op):
    if not op.parent or op.parent.id == op.id:  # Top op
        return False
    if is_backward_parent(op):
        return True
    return has_backward_parent(op.parent)


def is_backward_parent(op):
    return (
        "autograd::engine::evaluate_function: " in op.name
        or "Optimizer.step" in op.name
    )


def is_backward_aten(op):
    return op.name.startswith("aten::") and has_backward_parent(op)


def fbgemm_input_args_indices(n):
    idx_list = None
    if "sgd" in n.name or "adagrad" in n.name:
        # exact_sgd: 11: indices, 12: offsets, 14: indice_weights
        if n.inputs[14] == "<None>":
            idx_list = [11, 12]
        else:
            idx_list = [11, 12, 14]
    return idx_list


def is_fbgemm_forward(op):
    return "fbgemm::split_embedding_codegen_lookup_" in op.name


def is_fbgemm_forward_unweighted(op):
    return is_fbgemm_forward(op) and len(fbgemm_input_args_indices(op)) == 2


def is_fbgemm_backward(op):
    return "CppNode<SplitLookupFunction_" in op.name and not is_backward_parent(op)


def is_fbgemm(op):
    return is_fbgemm_forward(op) or is_fbgemm_backward(op)


# TODO: Hopefully merge is_fbgemm and skip_op, ignore tid == 2
def skip_op(op):
    # Workaround: skip bounds check indices and other ops under embedding lookup module
    return (
        not is_fbgemm_forward(op)
        and op.parent is not None
        and (
            "embedding_lookup" in op.parent.name
            or "param|SplitTableBatchedEmbeddingBagsCodegen" in op.parent.name
            # or "## forward:tw_global_sparse_arch ##" in op.parent.name or op.name == "fb::to_dense_representation" \
            or op.name == "fb::to_dense_representation"
            or (
                "fbgemm::" in op.name
                and "fbgemm::split_embedding_codegen_lookup_" not in op.name
            )
            or ("SymInt" in op.op_schema)
        )
        or ("fused" in op.name)
        or (
            op.name
            in [
                "aten::empty",
                "aten::to",
                "aten::lift",
                "aten::detach_",
                "aten::set_",
                "aten::pin_memory",
            ]
            and "thread" in op.parent.name
            and op.tid == 2
        )
    )


def is_qualified(op):
    return not skip_op(op) and (
        is_backward_aten(op)
        or is_fbgemm_backward(op)
        or (is_op(op, strict=True) and not is_backward_parent(op))
    )


def get_input_tensors(n):
    if is_fbgemm_forward(n):
        idx_list = fbgemm_input_args_indices(n)
        return zip(
            [n.input_types[x] for x in idx_list],
            [
                tuple(n.inputs[x]) if isinstance(n.inputs[x], list) else n.inputs[x]
                for x in idx_list
            ],
            [n.input_shapes[x] for x in idx_list],
        )
    return n.get_input_tensors()


def get_output_tensors(n):
    if is_fbgemm_forward(n):
        return list(zip(n.output_types, [tuple(x) for x in n.outputs], n.output_shapes))
    return n.get_output_tensors()


def c10_type_to_str(t):
    if "c10::Half" in t:
        return "fp16"
    return "fp32"
    # raise ValueError("c10 type not supported!")


def get_optimizer_from_fbgemm_function_name(s):
    # strip 'fbgemm::split_embedding_codegen_lookup_*_function'
    # opt = s[39:].split("_")[:-1] , # this one does not work with rowwise
    opt = s[39:-9]
    if "rowwise" in opt:
        opt = opt.replace("rowwise", "row_wise")
    return "exact_{}".format(opt)  # Workaround, should be more accurate


def get_fbgemm_info(n):
    num_tables = n.input_shapes[6][0] - 1
    rows = [1024] * num_tables
    batch_size = int((n.input_shapes[12][0] - 1) / num_tables)
    assert batch_size == n.output_shapes[0][0]

    # Assume most tables have same dim except the last fews, at the same time it is required that dim % 4 == 0
    # and dim <= 1024 (not necessary but would invoke error)
    avg_dim = int(n.inputs[7] / num_tables / 4) * 4
    dims = [avg_dim for _ in range(num_tables)]
    addition = n.inputs[7] - avg_dim * num_tables
    pos = len(dims) - 1
    while addition > 0 and pos >= 0:
        if addition >= 1024 - dims[pos]:
            addition -= 1024 - dims[pos]
            dims[pos] += 1024 - dims[pos]
        else:
            dims[pos] += addition
            addition = 0
        pos -= 1
    pooling_factor = [1] * num_tables

    weighted = "Float" not in n.input_types[1]  # e.g. c10:Half
    weights_precision = c10_type_to_str(n.input_types[1])
    optimizer = get_optimizer_from_fbgemm_function_name(n.name)

    if optimizer == "exact_sgd":
        lr = n.inputs[20]
    elif optimizer == "exact_row_wise_adagrad":
        lr = n.inputs[25]
    else:
        lr = 0.01
    if optimizer == "exact_row_wise_adagrad":
        eps = n.inputs[24]
        weight_decay = n.inputs[26]
        if n.inputs[27] == 0:
            weight_decay_mode = WeightDecayMode.NONE
        elif n.inputs[27] == 1:
            weight_decay_mode = WeightDecayMode.L2
        else:
            weight_decay_mode = WeightDecayMode.DECOUPLE
    else:
        eps = 1.0e-8
        weight_decay = 0.0
        weight_decay_mode = WeightDecayMode.NONE

    return (
        rows,
        num_tables,
        dims,
        batch_size,
        pooling_factor,
        weighted,
        weights_precision,
        optimizer,
        lr,
        eps,
        weight_decay,
        weight_decay_mode,
    )


def build_fbgemm_func(n, device):
    assert n.parent is not None
    (
        rows,
        num_tables,
        dims,
        _,
        _,
        weighted,
        weights_precision,
        optimizer,
        lr,
        eps,
        weight_decay,
        weight_decay_mode,
    ) = get_fbgemm_info(n)
    op = SplitTableBatchedEmbeddingBagsCodegenOp()
    op.device = device
    if num_tables == 1:
        op.build(
            num_tables,
            rows[0],
            dims[0],
            PoolingMode.SUM,
            weighted,
            weights_precision,
            optimizer,
            lr,
            eps,
            weight_decay,
            weight_decay_mode,
        )
    else:
        op.build(
            num_tables,
            rows,
            dims,
            PoolingMode.SUM,
            weighted,
            weights_precision,
            optimizer,
            lr,
            eps,
            weight_decay,
            weight_decay_mode,
        )
    return op, len(n.outputs)


def generate_fbgemm_tensors(n, device):
    assert n.parent is not None
    (
        rows,
        num_tables,
        dims,
        batch_size,
        pooling_factor,
        weighted,
        weights_precision,
        optimizer,
        _,
        _,
        _,
        _,
    ) = get_fbgemm_info(n)
    if num_tables == 1:
        rows = rows[0]
        dims = dims[0]
        pooling_factor = pooling_factor[0]

    data_generator_config = create_op_args(
        [
            {"type": "int", "name": "num_tables", "value": num_tables},
            {"type": "int", "name": "rows", "value": rows},
            {"type": "int", "name": "dim", "value": dims},
            {"type": "int", "name": "batch_size", "value": batch_size},
            {"type": "int", "name": "pooling_factor", "value": pooling_factor},
            {"type": "bool", "name": "weighted", "value": weighted},
            {"type": "str", "name": "weights_precision", "value": weights_precision},
        ],
        {"optimizer": {"type": "str", "value": optimizer}},
    )

    input_data_gen = SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator()

    (input_args, input_kwargs) = input_data_gen.get_data(data_generator_config, device)
    if is_fbgemm_forward_unweighted(n):
        input_args.pop(-1)

    return input_args, input_kwargs  # Discard weights if not needed


def build_torchscript_func(n):
    input_count = len(n.input_types)
    output_count = len(n.output_types)

    if (
        "pyspeech" in n.op_schema
        or n.op_schema == ""
        or n.name == "aten::record_stream"
    ):
        return None, None

    tmp = n.op_schema.split(") -> ")
    # items = [item for item in tmp[0].split(',') if item != ' *']
    types = [item for item in tmp[0].split(" ") if "," not in item][:-1]
    # print(n.name, n.id, types)
    types = [re.sub(r"\[[0-9]\]", "[]", t) for t in types]  # e.g. int[2] -> int[]
    # print(n.name, n.id, types)
    input_types = [
        "Tensor" if "Tensor(" in t else t
        for t in types
        if ("*)" not in t and "->" not in t)
    ]  # e.g. Tensor(float) -> Tensor; exception: aten::unbind(Tensor(a -> *) self, ...
    # print(n.name, n.id, input_types)
    input_types[0] = re.sub(
        r"^.*?\(", "", input_types[0]
    )  # Strip the op name, e.g. aten::zeros(int[] -> int[]
    # print(n.name, n.id, input_types)
    output_types = (
        tmp[-1].lstrip(" (").rstrip(")").split(", ")
    )  # e.g. (Tensor, Tensor) -> [Tensor, Tensor]
    # print(n.id, input_types, output_types)
    tmp = []
    for t in output_types:
        if t == "Tensor[]" or t == "Tensor(a)[]":
            tmp.append("Tensor[]")
        elif "Tensor" in t:
            tmp.append("Tensor")
        else:
            tmp.append(t)
    output_types = tmp
    # print(n.id, input_types, output_types)

    inputStr = """
        graph({}):
            {} = {}({})
            {}
            {}
    """.format(
        # Input arguments
        ", ".join(["%{}: {}".format(idx, t) for idx, t in enumerate(input_types)]),
        # Op
        "%output: {}".format(output_types[0] if output_count == 1 else "NoneType")
        if output_count <= 1
        else ", ".join(
            [
                "%{}: {}".format(idx + input_count, t)
                for idx, t in enumerate(output_types)
            ]
        ),
        n.name,
        ", ".join(["%{}".format(idx) for idx in range(input_count)]),
        # Tuple handling
        "%output : ({}) = prim::TupleConstruct({})".format(
            ", ".join(["Tensor" for _ in range(output_count)]),
            ", ".join(["%{}".format(idx + input_count) for idx in range(output_count)]),
        )
        if output_count > 1
        else "",
        # Return
        "return (%output)" if output_count >= 1 else "",
    )

    # print(inputStr)
    # print("=============")

    try:
        graph = torch._C.parse_ir(inputStr)
        cu = torch._C.CompilationUnit()
        func = cu.create_function(n.name, graph)
    except Exception as e:
        print("TorchScript error: ", n.id, e, input_types, "\n", inputStr)
        return None, None
    return func, output_count


def generate_prefix(eg_input, cuda):
    template_prefix = """import gc
import json
import time

import torch
from param_bench.train.compute.python.tools.eg_replay_utils import (
    build_fbgemm_func,
    build_torchscript_func,
    generate_fbgemm_tensors,
    is_fbgemm_backward,
    is_fbgemm_forward,
    is_qualified,
)

from param_bench.train.compute.python.tools.execution_graph import ExecutionGraph
from param_bench.train.compute.python.tools.utility import trace_handler


global dfs_traverse
global funcs
global skip_node_names
global fbgemm_backward_ops

funcs = {{}}
skip_node_names = ["DataLoader", "aten::set_", "fb::", "c10d::allreduce_", "pyspeech::"]
fbgemm_backward_ops = []


def dfs_traverse(node):
    for child in node.children:
        if any(x in child.name for x in skip_node_names):
            continue
        if is_qualified(child):
            if is_fbgemm_forward(child):
                func, output_count = build_fbgemm_func(child, \"{cuda}\")
                fbgemm_backward_ops.append((func.backward, child.id))
                funcs[child.id] = (func.forward, output_count)
            elif is_fbgemm_backward(child):
                assert fbgemm_backward_ops
                backward_op, forward_id = fbgemm_backward_ops.pop(-1)
                funcs[child.id] = (backward_op, len(child.output_types))
            else:
                func, output_count = build_torchscript_func(child)
                funcs[child.id] = (func, output_count)
        else:
            dfs_traverse(child)


with open(\"{eg_input}\", 'r') as f:
    exgr = ExecutionGraph(json.load(f))
nodes = exgr.get_nodes(clean=True)
node = nodes[1]
dfs_traverse(node)
gc.collect()
torch.cuda.empty_cache()

"""
    return template_prefix.format(eg_input=eg_input, cuda=cuda)


def generate_suffix(warmup_iter, replay_iter):
    template_suffix = """
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    on_trace_ready=trace_handler,
) as prof:
    for iter in range({warmup_iter} + {replay_iter}):
        if iter == {warmup_iter}:
            start_ns = time.time_ns()
        run_ops()
        torch.cuda.synchronize()
        prof.step()
    print("Execution finished!")
    print("Avg execution time per iteration is {{}}ms".format((time.time_ns() - start_ns) / {replay_iter} / 1000000.0))

"""
    return template_suffix.format(warmup_iter=warmup_iter, replay_iter=replay_iter)
