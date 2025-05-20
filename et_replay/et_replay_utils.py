import os
import re

import torch

from et_replay.execution_trace import NodeType
from fbgemm_gpu.split_table_batched_embeddings_ops import PoolingMode, WeightDecayMode
from param_bench.train.compute.python.lib.pytorch.config_util import create_op_args
from param_bench.train.compute.python.workloads.pytorch.split_table_batched_embeddings_ops import (
    SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator,
    SplitTableBatchedEmbeddingBagsCodegenOp,
)


# TODO: Add all torch dtypes to here
TORCH_DTYPES_RNG = {
    "bool": (torch.bool, torch.ones),
    "int8": (torch.int8, torch.ones),
    "half": (torch.half, torch.randn),
    "int": (torch.int, torch.ones),
    "long": (torch.int64, torch.ones),
    "long int": (torch.int64, torch.ones),
    "float": (torch.float32, torch.randn),
    "double": (torch.float64, torch.randn),
    "signed char": (torch.int8, torch.ones),
    "unsigned char": (torch.uint8, torch.ones),
    "c10::Half": (torch.half, torch.randn),
    "c10::BFloat16": (torch.bfloat16, torch.randn),
    "c10::complex<float>": (torch.complex32, torch.randn),
}

TORCH_DTYPES_RNG_str = {
    "bool": ("torch.bool", "torch.ones"),
    "int8": ("torch.int8", "torch.ones"),
    "half": ("torch.half", "torch.randn"),
    "int": ("torch.int", "torch.ones"),
    "long": ("torch.int64", "torch.ones"),
    "long int": ("torch.int64", "torch.ones"),
    "float": ("torch.float32", "torch.randn"),
    "double": ("torch.float64", "torch.randn"),
    "signed char": ("torch.int8", "torch.ones"),
    "unsigned char": ("torch.uint8", "torch.ones"),
    "c10::Half": ("torch.half", "torch.randn"),
    "c10::BFloat16": ("torch.bfloat16", "torch.randn"),
    "c10::complex<float>": ("torch.complex32", "torch.randn"),
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
    "signed char": 1,
    "unsigned char": 1,
    "c10::Half": 2,
    "c10::BFloat16": 2,
    "c10::complex<float>": 8,
}


def is_tensor_list(n, idx, is_input):
    types_list = n.input_types if is_input else n.output_types
    return isinstance(idx, int) and "GenericList[Tensor" in types_list[idx]


def is_tensor(n, idx, is_input):
    types_list = n.input_types if is_input else n.output_types
    return (
        isinstance(idx, int)
        and "Tensor" in types_list[idx]
        and "GenericList" not in types_list[idx]
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
            or (
                "fbgemm::" in op.name
                and "fbgemm::split_embedding_codegen_lookup_" not in op.name
            )
        )
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
        or (op.name == "record_param_comms" and op.inputs[3] == "init")
        or (op.name == "aten::view" and "aten::view.dtype" in op.op_schema)
    )


def is_qualified(op):
    return not skip_op(op) and (
        is_backward_aten(op)
        or is_fbgemm_backward(op)
        or (is_op(op, strict=True) and not is_backward_parent(op))
    )


def get_input_tensors(n):
    return n.get_input_tensors()


def get_output_tensors(n):
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
    return f"exact_{opt}"  # Workaround, should be more accurate


def get_fbgemm_info(n, rows_per_table):
    num_tables = n.input_shapes[6][0] - 1
    rows = [rows_per_table] * num_tables
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

    pooling_mode = [n.inputs[13]] * num_tables

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
        pooling_mode,
        weighted,
        weights_precision,
        optimizer,
        lr,
        eps,
        weight_decay,
        weight_decay_mode,
    )


def build_fbgemm_func(n, device, rows_per_table):
    assert n.parent is not None
    (
        rows,
        num_tables,
        dims,
        _,
        pooling_mode,
        weighted,
        weights_precision,
        optimizer,
        lr,
        eps,
        weight_decay,
        weight_decay_mode,
    ) = get_fbgemm_info(n, rows_per_table)
    op = SplitTableBatchedEmbeddingBagsCodegenOp()
    op.device = device
    if num_tables == 1:
        op.build(
            num_tables,
            rows[0],
            dims[0],
            PoolingMode(pooling_mode[0]),
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
            PoolingMode(pooling_mode[0]),
            weighted,
            weights_precision,
            optimizer,
            lr,
            eps,
            weight_decay,
            weight_decay_mode,
        )
    return op, len(n.outputs)


def generate_fbgemm_tensors(n, device, rows_per_table, pooling_factor, alpha):
    assert n.parent is not None
    (
        rows,
        num_tables,
        dims,
        batch_size,
        _,
        weighted,
        weights_precision,
        optimizer,
        _,
        _,
        _,
        _,
    ) = get_fbgemm_info(n, rows_per_table)

    if num_tables == 1:
        rows = rows[0]
        dims = dims[0]
        pooling_factors = pooling_factor
    else:
        pooling_factors = [pooling_factor] * num_tables

    data_generator_config = create_op_args(
        [
            {"type": "int", "name": "num_tables", "value": num_tables},
            {"type": "int", "name": "rows", "value": rows},
            {"type": "int", "name": "dim", "value": dims},
            {"type": "int", "name": "batch_size", "value": batch_size},
            {"type": "int", "name": "pooling_factor", "value": pooling_factors},
            {"type": "bool", "name": "weighted", "value": weighted},
            {"type": "str", "name": "weights_precision", "value": weights_precision},
        ],
        {"optimizer": {"type": "str", "value": optimizer}},
    )

    input_data_gen = SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator()

    (input_args, input_kwargs) = input_data_gen.get_data(
        data_generator_config, device, alpha
    )
    if is_fbgemm_forward_unweighted(n):
        input_args.pop(-1)

    return input_args, input_kwargs  # Discard weights if not needed


def build_torchscript_func(n):
    input_count = len(n.input_types)
    output_count = len(n.output_types)

    if (
        n.op_schema == ""
        or n.name == "aten::record_stream"
        or n.name.startswith("aten::_foreach")
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
        ", ".join([f"%{idx}: {t}" for idx, t in enumerate(input_types)]),
        # Op
        (
            "%output: {}".format(output_types[0] if output_count == 1 else "NoneType")
            if output_count <= 1
            else ", ".join(
                [f"%{idx + input_count}: {t}" for idx, t in enumerate(output_types)]
            )
        ),
        n.name,
        ", ".join([f"%{idx}" for idx in range(input_count)]),
        # Tuple handling
        (
            "%output : ({}) = prim::TupleConstruct({})".format(
                ", ".join(["Tensor" for _ in range(output_count)]),
                ", ".join([f"%{idx + input_count}" for idx in range(output_count)]),
            )
            if output_count > 1
            else ""
        ),
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


def build_triton_func(n, resources_dir, async_compile, device):
    with open(os.path.join(resources_dir, n.kernel_file)) as f:
        code = f.read()

    func = None
    # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES controls whether each triton
    # kernel is given a unique name or not, if it is not, then the
    # kernel name will be "triton_" for all triton kernels.
    try:
        func = async_compile.triton(n.name, code, device_str=device)
    except Exception:
        func = async_compile.triton("triton_", code, device_str=device)

    return func, 0


def generate_prefix(label, skip_nodes, et_input, cuda, compute_only, tf32, rows):
    template_prefix = """import gc
import argparse
import json
import logging
import os
import time
from datetime import datetime
from et_replay.comm import comms_utils

import torch
from et_replay.comm import commsTraceReplay
from et_replay.et_replay_utils import (
    build_fbgemm_func,
    build_torchscript_func,
    generate_fbgemm_tensors,
    is_fbgemm_backward,
    is_fbgemm_forward,
    is_qualified,
)

from et_replay.execution_trace import ExecutionTrace
from et_replay.utils import trace_handler


print("PyTorch version: ", torch.__version__)

tf32 = {tf32}
if tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

global dfs_traverse
global funcs
global skip_node_names
global fbgemm_backward_ops
global sorted_nodes

funcs = {{}}
skip_node_names = [{skip_nodes}]
fbgemm_backward_ops = []
sorted_nodes = []

def dfs_traverse(node):
    for child in node.children:
        if "{label}" and "{label}" in child.name:
            sorted_nodes.append(child)

        if any(x in child.name for x in skip_node_names):
            continue

        if is_qualified(child):
            sorted_nodes.append(child)
        else:
            dfs_traverse(child)

if "://" in \"{et_input}\":
    try:
        from param_bench.et_replay.comm.vendor_internal.fb_internals import (
            read_remote_trace,
        )
    except ImportError:
        logging.info("FB internals not present")
        exit(1)
    else:
        et, _ = read_remote_trace(\"{et_input}\")
        extr = ExecutionTrace(json.load(et))
else:
    with open(\"{et_input}\", 'r') as f:
        extr = ExecutionTrace(json.load(f))

nodes = extr.get_nodes(clean=True)
node = nodes[1]
dfs_traverse(node)

operators_count = [0]
sorted_nodes = sorted(sorted_nodes, key=lambda x: x.id)

for i in range(len(sorted_nodes)):
    if "{label}" and "{label}" in sorted_nodes[i].name:
        operators_count.append(i)
if len(operators_count) > 1:
    sorted_nodes = sorted_nodes[
        operators_count[1] + 1 : operators_count[2]
    ]

print("#Operators to execute: ", len(sorted_nodes))

for node in sorted_nodes:
    if is_fbgemm_forward(node):
        func, output_count = build_fbgemm_func(node, \"{cuda}\", {rows})
        fbgemm_backward_ops.append((func.backward, node.id))
        funcs[node.id] = (func.forward, output_count)
    elif is_fbgemm_backward(node):
        assert fbgemm_backward_ops
        backward_op, forward_id = fbgemm_backward_ops.pop(-1)
        funcs[node.id] = (backward_op, len(node.output_types))
    else:
        func, output_count = build_torchscript_func(node)
        funcs[node.id] = (func, output_count)

compute_only = {compute_only}
if not compute_only:
    comms_env_params = comms_utils.read_comms_env_vars()
    global traceBench
    traceBench = commsTraceReplay.commsTraceReplayBench()

    parser = argparse.ArgumentParser(
        description="PARAM-Comms Trace Replay Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    args = traceBench.readArgs(parser)
    traceBench.setTraceFile(args, comms_env_params)
    traceBench.checkArgs(args)

    time.sleep(1)
    bootstrap_info = comms_utils.bootstrap_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, comms_env_params
    )
    global commsParams
    commsParams = comms_utils.commsParamsHolderBase(args)
    traceBench.initBackend(bootstrap_info, commsParams)
    traceBench.initBench(commsParams, args)
    traceBench.replayInit(commsParams)

"""
    return template_prefix.format(
        label=label,
        skip_nodes=skip_nodes,
        et_input=et_input,
        cuda=cuda,
        compute_only=str(compute_only),
        tf32=str(tf32),
        rows=rows,
    )


def generate_suffix(warmup_iter, replay_iter, cuda_id, profile_replay):
    template_suffix = """
start_time = datetime.now()

if {cuda_id} != -1:
    s1 = torch.cuda.Stream(device=(torch.device("cuda:{cuda_id}")))
else:
    s1 = torch.cuda.Stream(device=(torch.device("cuda")))

profile_replay = {profile_replay}
if profile_replay:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup={warmup_iter},
            active={replay_iter}
        ),
        record_shapes=True,
        on_trace_ready=trace_handler,
    ) as prof:
        for iter in range({warmup_iter} + {replay_iter}):
            if iter == {warmup_iter}:
                start_ns = time.time_ns()
            try:
                run_ops(iter, s1)
            except Exception as e:
                print(os.getpid(), e)
                exit(1)
            # if {cuda_id} != -1:
            #     torch.cuda.synchronize(torch.device("cuda:{cuda_id}"))
            # else:
            #     torch.cuda.synchronize(torch.device("cuda"))
            prof.step()
else:
    for iter in range({warmup_iter} + {replay_iter}):
        if iter == {warmup_iter}:
            start_ns = time.time_ns()
        try:
            run_ops(iter, s1)
        except Exception as e:
            print(os.getpid(), e)
            exit(1)
        # if {cuda_id} != -1:
        #     torch.cuda.synchronize(torch.device("cuda:{cuda_id}"))
        # else:
        #     torch.cuda.synchronize(torch.device("cuda"))

print("Execution finished!")
print("Avg execution time per iteration is {{}}ms".format((time.time_ns() - start_ns) / {replay_iter} / 1000000.0))
end_time = datetime.now()

try:
    from param_bench.et_replay.vendor_internal.fb_internal import (
        generate_query_url,
    )
except ImportError:
    logging.info("FB internals not present")
else:
    generate_query_url(start_time, end_time, {cuda_id})

"""
    return template_suffix.format(
        warmup_iter=warmup_iter,
        replay_iter=replay_iter,
        cuda_id=cuda_id,
        profile_replay=profile_replay,
    )
