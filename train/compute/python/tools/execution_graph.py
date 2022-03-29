from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    annotations,
)

import argparse
import json
import logging
import sys
from enum import Enum
from typing import Dict, Set, List, Any, Callable, Iterable, Type, TextIO

import pydot
from .lib.graphml import GraphML

FORMAT = "[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

# OPERATOR: nodes actually does something
# LABEL: nodes used as markers
NodeType = Enum("NodeType", "OPERATOR LABEL")

"""
TensorNode

Contains information about a tensor object (TensorImpl in PyTorch). Each has an
unique ID, element data type, source node IDs, sink nodes IDs, and all the
shapes associated with this tensor used in different contexts. Their producer
consumer relationship is:

source node --> tensor --> sink node

It's important to note that a tensor object maybe shared and used in different
ways in the PyTorch computation graph.
"""


class TensorNode:
    def __init__(self, id: int, dtype: str):
        self.id: int = id
        self.dtype = dtype
        self.sources: Set = set()
        self.sinks: Set = set()
        self.shapes: Set = set()

    def add_source(self, id: int):
        self.sources.add(id)

    def add_sink(self, id: int):
        self.sinks.add(id)

    def add_shape(self, shape: List[Any]):
        self.shapes.add(tuple(shape))


"""
Node

Contains all the information about a non-tensor node in the PyTorch computation
graph.

A node has an unique ID. This ID is in the order of execution in the original
graph. Special nodes:
- A single label node __ROOT_PROCESS__ has node ID 1 and is the root of the execution
graph.
- Each thread has its __ROOT_THREAD__ node with an unique ID.

All the input tensors will have ID < node ID.
"""


class Node:
    def __init__(
        self,
        name: str,
        id: int,
        parent_id: int,
        fw_parent_id: int,
        pid: int,
        tid: int,
        fw_tid: int,
        scope: int,
        inputs: List[Any],
        input_types: List[str],
        input_shapes: List[Any],
        outputs: List[Any],
        output_types: List[str],
        output_shapes: List[Any],
    ):
        self.name: str = name
        self.parent_id: int = parent_id
        self.parent: Node = None
        self.children: List[Node] = []
        self.id: int = id
        self.pid: int = pid
        self.tid: int = tid
        self.fw_tid: int = fw_tid
        self.fw_parent_id: int = fw_parent_id
        self.scope: int = scope
        self.type: str = self.detect_type(name, inputs, outputs)
        self.inputs: List[Any] = inputs
        self.input_types: List[str] = input_types
        self.input_shapes: List[Any] = input_shapes
        self.outputs: List[Any] = outputs
        self.output_types: List[str] = output_types
        self.output_shapes: List[Any] = output_shapes

    def get_inputs(self) -> Iterable:
        return zip(self.input_types, self.inputs, self.input_shapes)

    def get_outputs(self) -> Iterable:
        return zip(self.output_types, self.outputs, self.output_shapes)

    def set_parent(self, parent: Node):
        assert parent.id == self.parent_id
        self.parent = parent

    def add_child(self, child: Node):
        self.children.append(child)

    def is_leaf_op(self) -> bool:
        return not self.children

    def _get_base_op(self, node) -> Node:
        if node.parent.type == NodeType.LABEL:
            return self
        return self._get_base_op(node.parent)

    def get_base_op(self) -> Node:
        return self._get_base_op(self)

    def detect_type(self, name: str, inputs: List[Any], outputs: List[Any]) -> NodeType:
        if (
            (name.startswith("##") or name.startswith("__"))
            and not inputs
            and not outputs
        ):
            return NodeType.LABEL
        else:
            return NodeType.OPERATOR

    def get_tensors(self, param_list: Iterable) -> List[tuple]:
        tensors = []
        for (type, input, shape) in param_list:
            if type.startswith("Tensor"):
                tensors.append((type, input, shape))
            # GenericList could have tensor elements
            if type.startswith("GenericList"):
                elem_type = type[12:-1].split(",")
                tensors.extend(self.get_tensors(zip(elem_type, input, shape)))
        return tensors

    def get_input_tensors(self) -> List[tuple]:
        return self.get_tensors(self.get_inputs())

    def get_output_tensors(self) -> List[tuple]:
        return self.get_tensors(self.get_outputs())

    def sort_children(self):
        self.children.sort(key=lambda x: x.id)


class ExecutionGraph:
    def __init__(self, json):
        self.nodes = {}
        self.tensors = {}
        self.proc_group = {}
        pid = json["pid"]
        self.proc_group = {pid: {}}
        nodes_list = json["nodes"]
        for x in nodes_list:
            id = x["id"]
            tid = x["tid"]
            self.nodes[id] = Node(
                x["name"],
                id,
                x["parent"],
                x["fw_parent"],
                pid,
                tid,
                x["fw_tid"],
                x["scope"],
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
            if x["name"] == "__ROOT_THREAD__":
                self.proc_group[pid][tid] = id

            # build tensor refernece table
            for (type, t_id, shape) in input_tensors:
                if t_id not in self.tensors:
                    dtype = type[7:-1]
                    self.tensors[t_id] = TensorNode(t_id, dtype)
                self.tensors[t_id].add_sink(id)
                self.tensors[t_id].add_shape(shape)

            for (type, t_id, shape) in output_tensors:
                if t_id not in self.tensors:
                    dtype = type[7:-1]
                    self.tensors[t_id] = TensorNode(t_id, dtype)
                self.tensors[t_id].add_source(id)
                self.tensors[t_id].add_shape(shape)

        # populate parent and children nodes
        for n in self.nodes.values():
            # skip root node
            if n.id != 1:
                self.nodes[n.parent_id].add_child(n)
                n.set_parent(self.nodes[n.parent_id])
        # sort children nodes by id
        for n in self.nodes.values():
            n.sort_children()

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
                    ops[n.name]["inputs"].append(
                        convert_inputs(n.inputs, n.input_types, n.input_shapes)
                    )
                else:
                    ops[n.name] = {"count": 1}
                    ops[n.name]["inputs"] = [
                        convert_inputs(n.inputs, n.input_types, n.input_shapes)
                    ]
        # Remove dupliates
        for attr in ops.values():
            # use json to serialize list of dict to string for set
            unique = {json.dumps(x, sort_keys=True) for x in attr["inputs"]}
            attr["inputs"] = list(map(json.loads, unique))
        if json_format:
            print(json.dumps(ops, indent=2, sort_keys=True))
        else:
            print("### OP STATS ###")
            for key, val in sorted(ops.items()):
                print(f"op: {key}")
                print(f"  count: {val['count']}")
                print("  unique inputs:")
                for i in val["inputs"]:
                    print(f"  input: {i}")

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
        for id in self.tensors:
            dot.add_node(
                pydot.Node(id, label=f"T{id}", style="filled", fillcolor="#e8faff")
            )
        nodes = len(self.nodes) + len(self.tensors)
        edges = 0
        for id, n in self.nodes.items():
            dot.add_edge(pydot.Edge(n.parent_id, id, arrowhead="odiamond"))
            edges += 1
            for (_, input, _) in n.get_input_tensors():
                dot.add_edge(pydot.Edge(input, id))
                edges += 1
            for (_, output, _) in n.get_output_tensors():
                dot.add_edge(pydot.Edge(id, output))
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

            for (_, input_id, _) in n.get_input_tensors():
                tfjs_node["input"].append(f"{input_id}")

            tfjs_nodes.append(tfjs_node)

        for id, t in self.tensors.items():
            tfjs_node = {}
            tfjs_node["name"] = f"{id}"
            tfjs_node["op"] = f"T{id}"
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
        for id, t in self.tensors.items():
            if detail:
                print(f"ID {id}:")
                print("     type:", t.dtype)
                print("   shapes:", t.shapes)
                print("  sources:", t.sources)
                print("    sinks:", t.sinks)
            else:
                print(f"id = {id}:")
                print("     type:", t.dtype)
                print("   shapes:", t.shapes)

    def _print_tree_preorder(self, n, indent, pid, tid, detail: bool):
        if n.type == NodeType.OPERATOR:
            print(f"{indent}({n.parent_id}:{n.id}) {n.name}")
            inputs = list(n.get_inputs())
            print(f"{indent}    arg: {inputs}")
            outputs = list(n.get_outputs())
            print(f"{indent}    out: {outputs}")
            if not detail:
                return
        else:
            print(f"{indent}({n.id}) {n.name}")
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
        print(f"ID {id}: Operator")
        print("          name:", n.name)
        print("           tid:", n.tid)
        print("     parent_id:", n.parent_id)
        print("        fw_tid:", n.fw_tid)
        print("  fw_parent_id:", n.fw_parent_id)
        print("         scope:", n.scope)
        print("        inputs:")
        for (dtype, tensor_id, shape) in n.get_input_tensors():
            prev_id = 0
            for s in self.tensors[tensor_id].sources:
                if s < id and s > prev_id:
                    prev_id = s
            if prev_id:
                print(
                    f"{' '*16}{tensor_id}: {dtype} {shape} <-- {prev_id} ({self.nodes[prev_id].name})"
                )
            else:
                print(f"{' '*16}{tensor_id}: {dtype} {shape}")

        print("       outputs:")
        for (dtype, tensor_id, shape) in n.get_output_tensors():
            next_id = sys.maxsize
            for s in self.tensors[tensor_id].sinks:
                # We could have cycle (s == id), where an op read and write to
                # the same tensor.
                if s > id and s < next_id:
                    next_id = s
            if next_id != sys.maxsize:
                print(
                    f"{' '*16}{tensor_id}: {dtype} {shape} --> {next_id} ({self.nodes[next_id].name})"
                )
            else:
                print(f"{' '*16}{tensor_id}: {dtype} {shape}")

    def tensor_depend(self, id: int):
        t = self.tensors[id]
        print(f"ID {id}: Tensor")
        print("     type:", t.dtype)
        print("   shapes:", t.shapes)
        sources = {}
        for node_id in t.sources:
            sources[node_id] = self.nodes[node_id].name
        print("  sources:", sources)
        sinks = {}
        for node_id in t.sinks:
            sinks[node_id] = self.nodes[node_id].name
        print("    sinks:", sinks)


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
            elif args.node in execution_graph.tensors:
                execution_graph.tensor_depend(args.node)
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
