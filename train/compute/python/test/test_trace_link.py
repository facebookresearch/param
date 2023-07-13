import unittest
from param_bench.train.compute.python.tools.trace_link import trace_analysis, exact_match, approximate_match


class TestTraceLink(unittest.TestCase):
    def test_exact_match(self):
        et_file = './data/linear_et.json'
        kineto_file = './data/linear_kineto.json'
        # Annotation to slice multiple iterations in kineto trace
        annotation = 'Optimizer.step#SGD.step'

        et_nodes, kineto_et_events = trace_analysis(et_file, kineto_file, annotation)
        et_enhanced = exact_match(kineto_et_events, et_nodes)
        self.assertTrue(et_enhanced)

    def test_approximate_match(self):
        et_file = './data/resnet_et.json'
        kineto_file = './data/resnet_kineto.json'
        # Annotation to slice multiple iterations in kineto trace
        annotation = 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'

        et_nodes, kineto_et_events = trace_analysis(et_file, kineto_file, annotation)
        et_enhanced = approximate_match(kineto_et_events, et_nodes)
        self.assertTrue(et_enhanced)


if __name__ == "__main__":
    unittest.main()
