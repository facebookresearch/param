import torch
import io
import json
import re
from .execution_graph import NodeType

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
    "unsigned char": (torch.int8, torch.ones)
}


def is_tensor(n, ip):
    return isinstance(ip, int) and 'Tensor' in n.input_types[ip]


def is_output_tensor(n, ip):
    return isinstance(ip, int) and 'Tensor' in n.output_types[ip]


def is_op(node, strict=False):
    if not strict:
        return node.type == NodeType.OPERATOR
    return node.type == NodeType.OPERATOR and (
                node.parent is not None and \
                node.parent.type != NodeType.OPERATOR\
            )


def has_backward_parent(op):
    if not op.parent or op.parent.id == op.id: # Top op
        return False
    if is_backward_parent(op):
        return True
    return has_backward_parent(op.parent)


def is_backward_parent(op):
    return "autograd::engine::evaluate_function: " in op.name or \
            "Optimizer.step" in op.name


def is_backward_aten(op):
    return op.name.startswith("aten::") and \
            has_backward_parent(op)


def is_qualified(op):
    return is_backward_aten(op) or is_op(op)

def get_tmp_trace_filename():
    import datetime
    import uuid
    trace_fn = "tmp_" + datetime.datetime.today().strftime("%Y%m%d")+ "_" + uuid.uuid4().hex[:7] + ".json"
    return trace_fn

def trace_handler(prof):
    prof.export_chrome_trace("/tmp/" + get_tmp_trace_filename())


def another_trace_handler():
    def handle_fn(prof):
        # print(prof.key_averages().table(
        #     sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace("/tmp/" + get_tmp_trace_filename())
    return handle_fn


def execution_graph_handler(output_file_name):
    print(f"pytroch execution graph output: {output_file_name}")
    found_root_node = False
    with io.open(output_file_name, 'r') as f:
        eg_graph = json.load(f)
        assert "nodes" in eg_graph
        nodes = eg_graph["nodes"]
        for n in nodes:
            assert "name" in n
            if "__ROOT_PROCESS__" in n["name"]:
                found_root_node = True

    assert found_root_node


def build_torchscript_func(n):
    input_count = len(n.input_types)
    output_count = len(n.output_types)

    if "pyspeech" in n.op_schema or n.op_schema == "":
        return None, None

    tmp = n.op_schema.split(') -> ')
    # items = [item for item in tmp[0].split(',') if item != ' *']
    types = [item for item in tmp[0].split(' ') if ',' not in item][:-1]
    # print(n.name, n.id, types)
    types = [re.sub(r'\[[0-9]\]', '[]', t) for t in types] # e.g. int[2] -> int[]
    # print(n.name, n.id, types)
    input_types = ['Tensor' if 'Tensor(' in t else t for t in types if ('*)' not in t and '->' not in t)] # e.g. Tensor(float) -> Tensor; exception: aten::unbind(Tensor(a -> *) self, ...
    # print(n.name, n.id, input_types)
    input_types[0] = re.sub(r'^.*?\(', '', input_types[0]) # Strip the op name, e.g. aten::zeros(int[] -> int[]
    # print(n.name, n.id, input_types)
    output_types = tmp[-1].lstrip(' (').rstrip(')').split(', ') # e.g. (Tensor, Tensor) -> [Tensor, Tensor]
    # print(n.id, input_types, output_types)
    output_types = [t if t == 'Tensor[]' or 'Tensor' not in t else 'Tensor' for t in output_types]
    # print(n.id, input_types, output_types)

    inputStr = """
        graph({}):
            {} = {}({})
            {}
            return (%output)
    """.format(
        ", ".join(["%{}: {}".format(idx, t) for idx, t in enumerate(input_types)]),
        "%output: {}".format(output_types[0]) if output_count == 1 else \
            ", ".join(["%{}: {}".format(idx + input_count, t) for idx, t in enumerate(output_types)]),
        n.name,
        ", ".join(["%{}".format(idx) for idx in range(input_count)]),
        "%output : ({}) = prim::TupleConstruct({})".format(
            ", ".join(["Tensor" for _ in range(output_count)]),
            ", ".join(["%{}".format(idx + input_count) for idx in range(output_count)])
        ) if output_count > 1 else "",
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
