from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    annotations,
)

import logging
from typing import TYPE_CHECKING, Dict, Set, List, Any, Callable, Iterable, Type, TextIO
from xml.sax.saxutils import escape

if TYPE_CHECKING:
    from execution_graph import ExecutionGraph

"""
Generates GraphML file.
"""


class GraphML:
    def __init__(self, execution_graph: ExecutionGraph):
        self.nodes: List = []
        self.edges: List = []
        # construct op nodes and edges
        for id, n in execution_graph.nodes.items():
            self._create_node(id, f"{n.name} ({n.id})", n.name)
        for tensor in execution_graph.tensors.values():
            self._create_tensor_node(tensor)
        for id, n in execution_graph.nodes.items():
            self._create_edge(n.parent_id, id)
            for (_, input, _) in n.get_input_tensors():
                self._create_edge(input, id)
            for (_, output, _) in n.get_output_tensors():
                self._create_edge(id, output)

        logging.info(f"nodes: {len(self.nodes)}")
        logging.info(f"edges: {len(self.edges)}")

    def _create_node(
        self,
        node_id: int,
        name: str = "",
        type: str = "",
        input: str = "",
        output: str = "",
        arg: str = "",
        device: str = "",
        engine: str = "",
        is_grad: str = "",
        info: str = "",
    ):
        self.nodes.append(
            {
                "id": node_id,
                "name": name,
                "type": type,
                "input": input,
                "output": output,
                "arg": arg,
                "is_grad": is_grad,
                "device": device,
                "engine": engine,
                "info": info,
            }
        )

    def _create_tensor_node(self, tensor):
        self._create_node(
            tensor.id,
            f"T{tensor.id}",
            type="Tensor",
            input=tensor.sources,
            output=tensor.sinks,
        )

    def _create_edge(self, source, target):
        self.edges.append({"source": source, "target": target})

    def write(self, name, file_name):
        """
        Given the graph information, write a GraphML file.
        Parameters:
        name (string): Name of the network.
        file_name: Output file name.
        """

        def write_header():
            out.write('<?xml version="1.0" encoding="UTF-8"?>\n')

        def graphml_begin():
            graphml = (
                '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns '
                'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
                '<key id="name" attr.name="name" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="type" attr.name="type" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="input" attr.name="input" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="output" attr.name="output" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="arg" attr.name="arg" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="device" attr.name="device" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="engine" attr.name="engine" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="is_grad" attr.name="is_grad" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
                '<key id="info" attr.name="info" '
                'attr.type="string" for="node" namespace="all::node"/>\n'
            )
            out.write(graphml)

        def graphml_end():
            out.write("</graphml>\n")

        def write_graph():
            out.write(f'<graph id="{name}" edgedefault="directed">\n')
            for node in self.nodes:
                write_node(node)
            for id, edge in enumerate(self.edges):
                write_edge(id, edge["source"], edge["target"])
            out.write("</graph>\n")

        def write_node(node):
            out.write(f'<node id="{node["id"]}">\n')
            for name, data in node.items():
                if name != "id" and data:
                    data = escape(str(data))
                    out.write(f'  <data key="{name}">{data}</data>\n')
            out.write("</node>\n")

        def write_edge(id, source, target):
            out.write(f'<edge id="e_{id}" source="{source}" target="{target}"/>\n')

        with open(file_name, "wt") as out:
            write_header()
            graphml_begin()
            write_graph()
            graphml_end()
