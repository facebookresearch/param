import logging
import os
import unittest

from param_bench.train.compute.python.lib import pytorch as lib_pytorch

from param_bench.train.compute.python.lib.config import BenchmarkConfig
from param_bench.train.compute.python.lib.init_helper import load_modules
from param_bench.train.compute.python.lib.pytorch.benchmark import (
    Benchmark,
    make_default_benchmark,
)
from param_bench.train.compute.python.lib.pytorch.config_util import (
    get_benchmark_options,
)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class TestBenchmarkLoad(unittest.TestCase):
    def setUp(self):
        self.config_path = os.path.join(
            CURR_DIR, "pytorch", "configs", "test_native_basic_ops.json"
        )
        # Load PyTorch implementations for data generator and operators.
        load_modules(lib_pytorch)

    def test_json_load_benchmark(self):
        run_options = get_benchmark_options()
        bench_config = BenchmarkConfig(run_options)
        bench_config.load_json_file(self.config_path)
        benchmark = make_default_benchmark(bench_config)
        self.assertTrue(isinstance(benchmark, Benchmark))
        self.assertTrue(len(benchmark.run_options) > 0)
        self.assertTrue(len(benchmark.bench_config.bench_config) > 0)


if __name__ == "__main__":
    unittest.main()
