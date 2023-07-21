import unittest

from param_bench.train.compute.python.lib.config import make_op_config
from param_bench.train.compute.python.lib.pytorch.config_util import create_op_info
from param_bench.train.compute.python.workloads.pytorch import (  # noqa
    split_table_batched_embeddings_ops,
)


class TestSplitTableBatchedEmbeddingOps(unittest.TestCase):
    @unittest.skip("fbgemm is failing.")
    def test_build_op(self):
        op_name = "SplitTableBatchedEmbeddingBagsCodegen"
        op_info = create_op_info()
        op_info[
            "input_data_generator"
        ] = "SplitTableBatchedEmbeddingBagsCodegenInputDataGenerator"
        op_config = make_op_config(op_name, op_info, "cpu")
        op_config.op.cleanup()

        op_config.op.build(
            1,
            [1000],
            [64],
            0,  # PoolingMode.SUM
            False,
            "fp16",
            "exact_adagrad",
        )
        self.assertEqual(op_config.op.op.embedding_specs[0][0], 1000)
        self.assertEqual(op_config.op.op.embedding_specs[0][1], 64)

        op_config.op.build(
            1,
            2000,
            128,
            0,  # PoolingMode.SUM
            False,
            "fp16",
            "exact_adagrad",
        )
        self.assertEqual(op_config.op.op.embedding_specs[0][0], 2000)
        self.assertEqual(op_config.op.op.embedding_specs[0][1], 128)

        op_config.op.build(
            2,
            [1000, 2000],
            [64, 128],
            0,  # PoolingMode.SUM
            False,
            "fp16",
            "exact_adagrad",
        )
        self.assertEqual(op_config.op.op.embedding_specs[0][0], 1000)
        self.assertEqual(op_config.op.op.embedding_specs[1][0], 2000)
        self.assertEqual(op_config.op.op.embedding_specs[0][1], 64)
        self.assertEqual(op_config.op.op.embedding_specs[1][1], 128)


if __name__ == "__main__":
    unittest.main()
