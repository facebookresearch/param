from __future__ import annotations

import gzip
import json

from et_replay.execution_trace import ExecutionTrace


class TraceValidator:
    def __init__(self, execution_trace: ExecutionTrace):
        self.et = execution_trace

    def _ops(self):
        return (n for n in self.et.nodes.values() if n.is_op())

    def _validate_ops(self) -> bool:
        """Make sure the pytorch operators are valid"""
        ops = self._ops()
        for op in ops:
            if op.name == "":
                print(f"op should have valid name, node id = {op.id}")

            # if len(list(op.get_outputs())) + len(list(op.get_inputs())) == 0:
            #    print(f"op should have outputs or inputs, node = {op.name}")
            #    FIXME see "autograd::engine::evaluate_function: DivBackward1"
            #    currently let's skip this
            #    return False
        return True

    def _validate_tree(self) -> bool:
        """TBD validate that the generated datastructure is a tree
        with parent/child relationship. We can use pydot or networkx libs for this
        """
        return True

    def _validate_param_comms(self) -> bool:
        """Check if param comms has correct attributes"""

        if self.et.schema_pytorch() < (1, 0, 2):
            return True

        def check_comms_node_pre_1_1_0(n) -> bool:
            """Roughly based on commsTraceParser"""
            # https://github.com/facebookresearch/param/blob/main/train/comms/pt/commsTraceParser.py#L256

            has_pg_id = False
            # Slightly hacky but find a argument with tuple type
            for arg in n.get_inputs():
                if arg[0] == "Tuple[String,String]":
                    print(f" {n.name}, process group args = {arg}")
                    has_pg_id = True
            return has_pg_id

        def check_comms_node_1_1_0(n) -> bool:
            """New elements are added as per
            https://github.com/pytorch/pytorch/issues/124674
            """
            # TODO check for node.commArgs dataclass
            print(n.commArgs)
            return True

        check_comms_node = (
            check_comms_node_1_1_0
            if self.et.schema_pytorch() >= (1, 1, 0)
            else check_comms_node_pre_1_1_0
        )

        return all(
            check_comms_node(n)
            for n in self.et.nodes.values()
            if n.is_op() and n.name == "record_param_comms"
        )

    def _validate_triton(self) -> bool:
        """Make sure triton kernels have correct values
        TODO update for checking if kernel files are captured.
        """
        return True

    def validate(self) -> bool:
        return all(
            [
                self._validate_ops(),
                self._validate_tree(),
                self._validate_param_comms(),
                self._validate_triton(),
            ]
        )

    def num_ops(self) -> int:
        return len(list(self._ops()))

    def num_comm_ops(self) -> int:
        return sum(1 for op in self._ops() if op.name == "record_param_comms")

    def num_triton_ops(self) -> int:
        return sum(1 for op in self._ops() if "triton" in op.name)


def main():
    import sys

    execution_json = sys.argv[1]

    with (
        gzip.open(execution_json, "rb")
        if execution_json.endswith("gz")
        else open(execution_json)
    ) as execution_data:
        execution_trace: ExecutionTrace = ExecutionTrace(json.load(execution_data))
        t = TraceValidator(execution_trace)
        print(
            f"num ops = {t.num_ops()}, num comms = {t.num_comm_ops()}, "
            f"num triton ops = {t.num_triton_ops()}"
        )
        print("Trace validation result = ", t.validate())


if __name__ == "__main__":
    main()
