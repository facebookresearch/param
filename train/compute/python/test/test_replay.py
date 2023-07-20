import json
import unittest
from ..tools.eg_replay import ExgrReplayManager
from ..tools.execution_graph import ExecutionGraph

class TestExecutionTraceReplay(unittest.TestCase):
    def test_compute_only_replay(self):
        replay_manager = ExgrReplayManager()
        replay_manager.compute_only = True
        replay_manager.trace_file = './data/resnet_et.json'

        with open(replay_manager.trace_file, "r") as f:
            replay_manager.eg = ExecutionGraph(json.load(f))

        replay_manager.benchTime()

        self.assertTrue(True) # just check to see if replay ran without failure

if __name__ == "__main__":
    unittest.main()
