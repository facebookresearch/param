import gzip
import json
import os
import unittest

from param_bench.train.compute.python.tools.execution_trace import ExecutionTrace
from param_bench.train.compute.python.tools.validate_trace import TraceValidator

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class TestTraceLoadAndValidate(unittest.TestCase):
    def setUp(self):
        self.trace_base = os.path.join(CURR_DIR, "data")

    def _test_and_validate_trace(self, trace_file):
        with (
            gzip.open(trace_file, "rb")
            if trace_file.endswith("gz")
            else open(trace_file, "r")
        ) as execution_data:
            execution_trace: ExecutionTrace = ExecutionTrace(json.load(execution_data))
            t = TraceValidator(execution_trace)
            self.assertTrue(t.validate())
            return t, execution_trace

    def test_trace_load_resnet_1gpu(self):
        et_file = os.path.join(
            self.trace_base, "1.0.3-chakra.0.0.4/resnet_1gpu_et.json.gz"
        )
        t, et = self._test_and_validate_trace(et_file)
        self.assertGreater(t.num_ops(), 1000)
        self.assertEqual(t.num_comm_ops(), 12)
        self.assertEqual(t.num_triton_ops(), 0)


if __name__ == "__main__":
    unittest.main()
