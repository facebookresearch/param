from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import argparse
import json
import logging
import sys
from enum import Enum
from typing import Dict, List, TextIO, Tuple

import pydot

from cea.ml_perf_model.gpu.scripts.execution_graph_data import (
    Node,
    NodeType,
    TensorNode,
)
from util.graphml import GraphML

FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


class ExecutionGraph:
    def __init__(self, event):
        self.nodes = {}
        self.tensors = {}
        self.tensor_id_map = {}
        self.tensor_storage_map = {}
        self.proc_group = {}
        pid = event["pid"]
        self.proc_group = {pid: {}}
        nodes_list = event["nodes"]
        for x in nodes_list:
            id = x["id"]
            tid = x["tid"]
            self.nodes[id] = Node(
                x["name"],
                id,
                x["rf_id"],
                x["parent"],
                x["fw_parent"],
                x["seq_id"],
                pid,
                tid,
                x["fw_tid"],
                x["scope"],
                x["op_schema"],
                x["inputs"],
                x["input_types"],
                x["input_shapes"],
                x["outputs"],
                x["output_types"],
                x["output_shapes"],
            )
            input_tensors = self.nodes[id].get_input_tensors()
            output_tensors = self.nodes[id].get_output_tensors()

            # track the various process and threads we have
            if x["name"] == "[pytorch|profiler|execution_graph|process]":
                self.proc_group[pid][tid] = id

            # build tensor refernece table
            for (type, tensor_info, shape) in input_tensors:
                tensor_key = tuple(tensor_info)
                if tensor_key not in self.tensors:
                    dtype = type[7:-1]  # "Tensor(*)"
                    tensor_node = self._register_tensor_node(
                        tensor_key, tensor_info, dtype
                    )
                else:
                    tensor_node = self.tensors[tensor_key]

                tensor_node.add_sink(id)
                tensor_node.add_shape(shape)

            for (type, tensor_info, shape) in output_tensors:
                tensor_key = tuple(tensor_info)
                if tensor_key not in self.tensors:
                    dtype = type[7:-1]
                    tensor_node = self._register_tensor_node(
                        tensor_key, tensor_info, dtype
                    )
                else:
                    tensor_node = self.tensors[tensor_key]

                tensor_node.add_source(id)
                tensor_node.add_shape(shape)

        # populate parent and children nodes
        for n in self.nodes.values():
            # skip root node
            if n.id != 1:
                self.nodes[n.parent_id].add_child(n)
                n.set_parent(self.nodes[n.parent_id])
        # sort children nodes by id
        for n in self.nodes.values():
            n.sort_children()

    def _register_tensor_node(
        self, tensor_key: Tuple[int], tensor_info: Dict[int], dtype: str
    ):
        tensor_node = TensorNode(
            tensor_info[0],
            tensor_info[1],
            tensor_info[2],
            tensor_info[3],
            tensor_info[4],
            dtype,
            tensor_info[5],
        )
        self.tensors[tensor_key] = tensor_node
        self.tensor_id_map.setdefault(tensor_node.id, [])
        self.tensor_id_map[tensor_node.id].append(tensor_node)
        self.tensor_storage_map.setdefault(tensor_node.storage_id, [])
        self.tensor_storage_map[tensor_node.storage_id].append(tensor_node)

        return tensor_node

    def _get_dependency_id_for_tensor_info(self, tensor_info: List[int]):
        return self.tensors[tuple(tensor_info)].get_dependency_id()

    def print_op_stats(self, detail: bool = False, json_format: bool = False):
        def get_param(value, type, shape):
            type = type.lower()
            SCALAR_TYPES = {"int", "long", "float", "double", "bool"}
            param = {"type": type}
            if type.startswith("genericlist"):
                param = {"type": "genericlist"}
                param["value"] = []
                type_list = type[12:-1].split(",")
                param_list = zip(value, type_list, shape)
                for (v, t, s) in param_list:
                    param["value"].append(get_param(v, t, s))
                param["size"] = len(value)
            elif type in SCALAR_TYPES or type == "device":
                param["value"] = value
            elif type.startswith("tensor"):
                param["type"] = "tensor"
                param["dtype"] = type[7:-1]
                param["shape"] = shape
                param["requires_grad"] = False
            return param

        def convert_inputs(inputs, types, shapes):
            input_info = zip(inputs, types, shapes)
            params = []
            for (value, type, shape) in input_info:
                params.append(get_param(value, type, shape))
            return params

        ops = {}
        for n in self.nodes.values():
            if detail:
                is_op = n.type == NodeType.OPERATOR
            else:
                is_op = (
                    n.type == NodeType.OPERATOR and n.parent.type != NodeType.OPERATOR
                )
            if is_op:
                if n.name in ops:
                    ops[n.name]["count"] += 1
                    if n.op_schema in ops[n.name]["op_schema"]:
                        ops[n.name]["op_schema"][n.op_schema].append(
                            convert_inputs(n.inputs, n.input_types, n.input_shapes)
                        )
                    else:
                        ops[n.name]["op_schema"][n.op_schema] = [
                            convert_inputs(n.inputs, n.input_types, n.input_shapes)
                        ]
                else:
                    ops[n.name] = {"count": 1}
                    ops[n.name]["build_data_generator"] = "PyTorch:DefaultDataGenerator"
                    ops[n.name]["input_data_generator"] = "PyTorch:DefaultDataGenerator"
                    ops[n.name]["op_schema"] = {
                        n.op_schema: [
                            convert_inputs(n.inputs, n.input_types, n.input_shapes)
                        ]
                    }

        for _op_name, op_info in ops.items():
            op_info["config"] = []
            for op_schema, op_inputs in op_info["op_schema"].items():
                op_info["config"].append(
                    {
                        "build": [{"args": [{"type": "str", "value": op_schema}]}],
                        "input": [],
                    }
                )
                # Remove input dupliates.
                # use json to serialize list of dict to string for set
                unique = {json.dumps({"args": x}, sort_keys=True) for x in op_inputs}
                op_info["config"][-1]["input"] = list(map(json.loads, unique))

            # Remove temp schema book keeping entries
            op_info.pop("op_schema")

        if json_format:
            print(json.dumps(ops, indent=2))
        else:
            print("### OP STATS ###")
            for key, val in sorted(ops.items()):
                print(f"op: {key}")
                print(f"  count: {val['count']}")
                for config in val["config"]:
                    print(f"  build: {config['build']}")
                    for input in config["input"]:
                        print(f"  input: {input}")

    def gen_graphviz(self, file_name):
        dot = pydot.Dot(graph_type="digraph")
        for id, n in self.nodes.items():
            dot.add_node(
                pydot.Node(
                    id,
                    label=f"{n.name} ({n.id})",
                    shape="box",
                    style="filled",
                    fillcolor="#fffbed",
                )
            )
        for _tensor_key, tensor_node in self.tensors.items():
            id = tensor_node.get_dependency_id()
            dot.add_node(
                pydot.Node(
                    id,
                    label=f"T{tensor_node.id} ({tensor_node.storage_id}+{tensor_node.offset})",
                    style="filled",
                    fillcolor="#e8faff",
                )
            )
        nodes = len(self.nodes) + len(self.tensors)
        edges = 0
        for id, n in self.nodes.items():
            dot.add_edge(pydot.Edge(n.parent_id, id, arrowhead="odiamond"))
            edges += 1
            for (_, tensor_info, _) in n.get_input_tensors():
                input_id = self._get_dependency_id_for_tensor_info(tensor_info)
                dot.add_edge(pydot.Edge(input_id, id))
                edges += 1
            for (_, tensor_info, _) in n.get_output_tensors():
                output_id = self._get_dependency_id_for_tensor_info(tensor_info)
                dot.add_edge(pydot.Edge(id, output_id))
                edges += 1
        dot.write_svg(file_name, prog="dot")
        logging.info(f"nodes: {nodes}")
        logging.info(f"edges: {edges}")

    def gen_graphml(self, file_name):
        graphml = GraphML(self)
        graphml.write("execution graph", file_name)

    def gen_graph(self, file_name, type=None):
        dot_max_nodes = 300
        if (
            len(self.nodes) < dot_max_nodes or type == "graphviz"
        ) and type != "graphml":
            out_name = f"{file_name}.svg"
            self.gen_graphviz(out_name)
        else:
            out_name = f"{file_name}.graphml"
            self.gen_graphml(out_name)

        print(f"Execution graph written to {out_name}")

    def gen_tfjs_json(self, file_name):
        """Function to export graph to Tensorflow.js style JSON format.

        This can be run by adding optional flag --tfjs-json
        """

        tfjs_nodes = []

        for n in self.nodes.values():
            tfjs_node = {}
            tfjs_node["name"] = f"{n.id}"
            tfjs_node["op"] = f"{n.name} ({n.id})"
            tfjs_node["input"] = []
            if n.parent is not None:
                tfjs_node["input"].append(f"{n.parent.id}")

            for (_, input_info, _) in n.get_input_tensors():
                id = self._get_dependency_id_for_tensor_info(input_info)
                tfjs_node["input"].append(f"{id}")

            tfjs_nodes.append(tfjs_node)

        for tensor_key, t in self.tensors.items():
            tfjs_node = {}
            id = self.tensors[tensor_key].get_dependency_id()
            tfjs_node["name"] = f"{id}"
            tfjs_node["op"] = f"T{t.id} ({t.storage_id}+{t.offset})"
            tfjs_node["input"] = [f"{self.nodes[elem].id}" for elem in t.sources]
            # tfjs_node["shape"] = list(list(t.shapes)[0])
            tfjs_nodes.append(tfjs_node)

        output = {"modelTopology": {"node": tfjs_nodes}, "weightsManifest": []}
        out_name = f"{file_name}.json"
        with open(out_name, "w") as outfile:
            json.dump(output, outfile)

        print(f"Execution graph written to {out_name}")

    def print_tensors(self, detail: bool = False):
        print("### TENSORS ###")
        for _, tensor_node in self.tensors.items():
            print("Tensor:")
            self.print_tensor_node(tensor_node, detail)

    def _print_tree_preorder(self, n, indent, pid, tid, detail: bool):
        def print_inputs_outputs(node: Node):
            inputs = list(n.get_inputs())
            print(f"{indent}    arg: {inputs}")
            outputs = list(n.get_outputs())
            print(f"{indent}    out: {outputs}")

        if n.type == NodeType.OPERATOR:
            print(
                f"{indent}({n.parent_id}|{n.id}:{n.rf_id}:{n.seq_id}) {n.name} [{n.op_schema}]"
            )
            print_inputs_outputs(n)
            if not detail:
                return

        else:
            print(f"{indent}({n.parent_id}|{n.id}:{n.rf_id}:{n.seq_id}) {n.name}")
            print_inputs_outputs(n)

        for c in n.children:
            self._print_tree_preorder(c, indent + "  ", pid, tid, detail)

    def print_tree(self, detail: bool = False):
        print("### Execution Tree ###")
        for pid, threads in self.proc_group.items():
            print(f"process: {pid}")
            for tid in sorted(threads):
                print(f"  thread: {tid}")
                thread_node = self.nodes[threads[tid]]
                self._print_tree_preorder(thread_node, "    ", pid, tid, detail)

    def node_depend(self, id: int):
        n = self.nodes[id]
        print("Type: Node")
        print("-" * 15)
        print("            id:", id)
        print("          name:", n.name)
        print("         rf_id:", n.parent_id)
        print("     parent_id:", n.parent_id)
        print("           tid:", n.tid)
        print("        fw_tid:", n.fw_tid)
        print("  fw_parent_id:", n.fw_parent_id)
        print("         scope:", n.scope)
        print("        inputs:")
        input_tensors = n.get_input_tensors()
        if input_tensors:
            for (dtype, tensor_info, shape) in input_tensors:
                prev_id = 0
                for s in self.tensors[tuple(tensor_info)].sources:
                    if s < id and s > prev_id:
                        prev_id = s
                if prev_id:
                    print(
                        f"{' '*16}{tensor_info}: {dtype} {shape} <-- {prev_id} ({self.nodes[prev_id].name})"
                    )
                else:
                    print(f"{' '*16}{tensor_info}: {dtype} {shape}")
        else:
            print(f"{' '*16}None")

        print("       outputs:")
        output_tensors = n.get_output_tensors()
        if output_tensors:
            for (dtype, tensor_info, shape) in output_tensors:
                next_id = sys.maxsize
                for s in self.tensors[tuple(tensor_info)].sinks:
                    # We could have cycle (s == id), where an op read and write to
                    # the same tensor.
                    if s > id and s < next_id:
                        next_id = s
                if next_id != sys.maxsize:
                    print(
                        f"{' '*16}{tensor_info}: {dtype} {shape} --> {next_id} ({self.nodes[next_id].name})"
                    )
                else:
                    print(f"{' '*16}{tensor_info}: {dtype} {shape}")
        else:
            print(f"{' '*16}None")

    def tensor_id_depend(self, id: int):
        print("Type: Tensor")
        tensor_list = self.tensor_id_map[id]
        for tensor_node in tensor_list:
            self.print_tensor_node(tensor_node, True)

    def tensor_storage_depend(self, storage_id: int):
        print("Type: Storage")
        tensor_list = self.tensor_storage_map[storage_id]
        for tensor_node in tensor_list:
            self.print_tensor_node(tensor_node, True)

    def print_tensor_node(self, tensor_node: TensorNode, detail: bool = False):
        print("-" * 13)
        print("   tensor_id:", tensor_node.id)
        print("  storage_id:", tensor_node.storage_id)
        print("      offset:", tensor_node.offset)
        print("    num_elem:", tensor_node.num_elem)
        print("  elem_bytes:", tensor_node.elem_bytes)
        print("   elem_type:", tensor_node.dtype)
        print("      device:", tensor_node.device)
        print("      shapes:", tensor_node.shapes)
        if detail:
            sources = {}
            for node_id in tensor_node.sources:
                sources[node_id] = self.nodes[node_id].name
            print("     sources:", sources)
            sinks = {}
            for node_id in tensor_node.sinks:
                sinks[node_id] = self.nodes[node_id].name
            print("       sinks:", sinks)


def main():
    parser = argparse.ArgumentParser(
        description="Execution graph building and analysis"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="input execution graph json file."
    )
    parser.add_argument(
        "--graph",
        dest="graph",
        default=False,
        action="store_true",
        help="generate a graph output using either graphviz (dot) or graphml.",
    )
    parser.add_argument(
        "--graphviz",
        dest="graphviz",
        default=False,
        action="store_true",
        help="generate graph output in graphviz (dot) format.",
    )
    parser.add_argument(
        "--graphml",
        dest="graphml",
        default=False,
        action="store_true",
        help="generate graph output in graphml format.",
    )
    parser.add_argument(
        "--tfjs-json",
        dest="tfjs_json",
        default=False,
        action="store_true",
        help="generate graph output in model.json format of Tensorflow.js for importing in Netron.",
    )
    parser.add_argument(
        "--list-tensor",
        dest="list_tensor",
        default=False,
        action="store_true",
        help="list all tensors and its shape and type information.",
    )
    parser.add_argument(
        "--list-op",
        dest="list_op",
        default=False,
        action="store_true",
        help="list all the ops in the execution graph.",
    )
    parser.add_argument(
        "--node",
        type=int,
        dest="node",
        default=-1,
        action="store",
        help="query information about a node and its dependency given a node ID.",
    )
    parser.add_argument(
        "--detail",
        dest="detail",
        default=False,
        action="store_true",
        help="combined with some other options, will show more detailed information.",
    )
    parser.add_argument(
        "--json",
        dest="json",
        default=False,
        action="store_true",
        help="for --list-op option, generate outputs in JSON format.",
    )
    parser.add_argument(
        "--tree",
        dest="tree",
        default=False,
        action="store_true",
        help="generate an execution tree by process, threads, and order of node execution.",
    )
    args = parser.parse_args()

    execution_json: str = args.input

    with open(execution_json) as execution_data:
        execution_data: TextIO
        execution_graph: ExecutionGraph = ExecutionGraph(json.load(execution_data))
        if args.list_op:
            execution_graph.print_op_stats(args.detail, args.json)
        if args.list_tensor:
            execution_graph.print_tensors(args.detail)
        if args.tree:
            execution_graph.print_tree(args.detail)
        if args.node != -1:
            if args.node in execution_graph.nodes:
                execution_graph.node_depend(args.node)
            elif args.node in execution_graph.tensor_id_map:
                execution_graph.tensor_id_depend(args.node)
            elif args.node in execution_graph.tensor_storage_map:
                execution_graph.tensor_storage_depend(args.node)
            else:
                logging.error(f"node {args.node} not found.")

        if args.graph or args.graphviz or args.graphml or args.tfjs_json:
            out_file: str = "execution_graph"
            if args.graphviz:
                execution_graph.gen_graph(out_file, "graphviz")
            elif args.graphml:
                execution_graph.gen_graph(out_file, "graphml")
            elif args.tfjs_json:
                execution_graph.gen_tfjs_json(out_file)
            else:
                execution_graph.gen_graph(out_file)


if __name__ == "__main__":
    main()
