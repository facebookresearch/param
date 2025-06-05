# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import torch

from et_replay.execution_trace import NodeType


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


def get_input_tensors(n):
    return n.get_input_tensors()


def get_output_tensors(n):
    return n.get_output_tensors()


def c10_type_to_str(t):
    if "c10::Half" in t:
        return "fp16"
    return "fp32"
    # raise ValueError("c10 type not supported!")


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
