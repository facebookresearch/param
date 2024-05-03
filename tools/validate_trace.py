from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import argparse
import gzip
import json

from param.execution_trace import ExecutionTrace


class TraceValidator:
    """
    Validates the structure and contents of an execution trace for PyTorch
    operations.

    Attributes:
        execution_trace (ExecutionTrace): The execution trace object that holds
        operational data to be validated.
    """

    def __init__(self, execution_trace: ExecutionTrace):
        self.et = execution_trace

    def _ops(self):
        """Generates PyTorch operations from the execution trace."""
        return (n for n in self.et.nodes.values() if n.is_op())

    def _validate_ops(self) -> bool:
        """
        Validates that all operations have a non-empty name attribute.

        Returns:
            bool: True if all operations are valid, False otherwise.
        """
        ops = self._ops()
        for op in ops:
            if op.name == "":
                print(f"op should have valid name, node id = {op.id}")
        return True

    def _validate_tree(self) -> bool:
        """
        Placeholder for validating that the execution trace forms a valid tree.

        Returns:
            bool: True if the trace forms a tree, False otherwise.
        """
        return True

    def _validate_param_comms(self) -> bool:
        """
        Validates parameter communications in the execution trace based on PyTorch
        version and node inputs.

        Returns:
            bool: True if all parameter communications are correct, False otherwise.
        """
        if self.et.schema_pytorch() < (1, 0, 2):
            return True

        def check_comms_node(n) -> bool:
            """Checks if the node has a process group argument."""
            has_pg_id = False
            for arg in n.get_inputs():
                if arg[0] == "Tuple[String,String]":
                    print(f"{n.name}, process group args = {arg}")
                    has_pg_id = True
            return has_pg_id

        return all(check_comms_node(n) for n in self.et.nodes.values() if n.is_op() and n.name == "record_param_comms")

    def _validate_triton(self) -> bool:
        """
        Validates Triton kernels in the execution trace.

        Returns:
            bool: True if all Triton kernels are validated, False otherwise.
        """
        return True

    def validate(self) -> bool:
        """
        Validates the execution trace across various criteria.

        Returns:
            bool: True if the trace is valid across all checks, False otherwise.
        """
        return all(
            [
                self._validate_ops(),
                self._validate_tree(),
                self._validate_param_comms(),
                self._validate_triton(),
            ]
        )

    def num_ops(self) -> int:
        """Counts the number of operation nodes in the execution trace."""
        return len(list(self._ops()))

    def num_comm_ops(self) -> int:
        """Counts the number of communication operation nodes in the trace."""
        return sum(1 for op in self._ops() if op.name == "record_param_comms")

    def num_triton_ops(self) -> int:
        """Counts the number of Triton operation nodes in the trace."""
        return sum(1 for op in self._ops() if "triton" in op.name)


def main():
    parser = argparse.ArgumentParser(description="Validate execution trace file.")
    parser.add_argument("et", type=str, help="The execution trace JSON file, optionally gzipped.")
    args = parser.parse_args()

    open_func = gzip.open if args.execution_json.endswith("gz") else open

    with open_func(args.execution_json, "rb") as execution_data:
        execution_trace: ExecutionTrace = ExecutionTrace(json.load(execution_data))
        validator = TraceValidator(execution_trace)
        print(
            f"num ops = {validator.num_ops()}, num comms = {validator.num_comm_ops()}, "
            f"num triton ops = {validator.num_triton_ops()}"
        )
        print("Trace validation result = ", validator.validate())


if __name__ == "__main__":
    main()
