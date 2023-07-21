import os
import unittest

from param_bench.train.compute.python.tools.eg_replay import ExgrReplayManager
from param_bench.train.compute.python.tools.utility import load_execution_trace_file

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class TestExecutionTraceReplay(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(CURR_DIR, "data")

    def test_compute_only_replay(self):
        replay_manager = ExgrReplayManager()
        replay_manager.compute_only = True
        replay_manager.trace_file = os.path.join(self.config_path, "resnet_et.json.gz")
        replay_manager.eg = load_execution_trace_file(replay_manager.trace_file)

        replay_manager.benchTime()

        self.assertTrue(True) # just check to see if replay ran without failure

if __name__ == "__main__":
    unittest.main()
