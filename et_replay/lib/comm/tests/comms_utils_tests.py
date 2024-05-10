import os
import unittest

import torch

from param_bench.train.comms.pt import comms_utils
from param_bench.train.comms.pt.tests.mocks.backend_mock import MockBackendFunction
from param_bench.train.comms.pt.tests.test_utils import (
    bootstrap_info_test,
    commsParamsTest,
)


class TestParseSize(unittest.TestCase):
    """
    Test and see if sizes are being parsed correctly.
    """

    def test_gb_size(self):
        sizeStr = "2GB"
        size = comms_utils.parsesize(sizeStr)
        # size is in bytes
        self.assertEqual(2147483648, size)

    def test_mb_size(self):
        sizeStr = "3MB"
        size = comms_utils.parsesize(sizeStr)
        self.assertEqual(3145728, size)

    def test_kb_size(self):
        sizeStr = "5KB"
        size = comms_utils.parsesize(sizeStr)
        self.assertEqual(5120, size)

    def test_single_size(self):
        sizeStr = "1024"
        size = comms_utils.parsesize(sizeStr)
        self.assertEqual(1024, size)


class TestParseRankList(unittest.TestCase):
    """
    Test differently formatted strings with ranks in them.

    TODO: Test graceful exits and error logging.
    """

    def test_comma_separated(self):
        comma_rank_list = "0,2,4,6"
        bootstrap_info = bootstrap_info_test()
        bootstrap_info.world_size = 8
        parsed_rank_list = comms_utils.parseRankList(comma_rank_list)
        # We should have 4 ranks returned.
        self.assertEqual(4, len(parsed_rank_list))
        # We should have ranks 0,2,4,6. They should be in this order as well.
        for i in range(4):
            self.assertEqual(i * 2, parsed_rank_list[i])

    def test_range_ranks(self):
        range_rank_list = "0:7"  # This is inclusive end.
        bootstrap_info = bootstrap_info_test()
        bootstrap_info.world_size = 8
        parsed_rank_list = comms_utils.parseRankList(range_rank_list)
        # We should have 8 ranks returned.
        self.assertEqual(8, len(parsed_rank_list))
        # We should have ranks 0-7 inclusive, in order.
        for i in range(8):
            self.assertEqual(i, parsed_rank_list[i])

    def test_single_rank(self):
        single_rank = "5"
        bootstrap_info = bootstrap_info_test()
        bootstrap_info.world_size = 8
        parsed_rank_list = comms_utils.parseRankList(single_rank)
        # We should have 1 rank returned.
        self.assertEqual(1, len(parsed_rank_list))
        # We should have rank 5.
        self.assertEqual(5, parsed_rank_list[0])


class TestGetAlgBW(unittest.TestCase):
    """
    Test if algorithmic bandwidth is being calculated properly.
    """

    def test_no_iterations(self):
        elapsedTimeNs = 30000
        dataSize = 90000  # bytes
        numIters = 0
        (avgIterNS, algBW) = comms_utils.getAlgBW(elapsedTimeNs, dataSize, numIters)
        # If we had no iterations, then we have no avg iteration time or algBW.
        self.assertEqual(0.0, avgIterNS, algBW)

    def test_iterations(self):
        elapsedTimeNs = 30000
        dataSize = 90000  # bytes
        numIters = 3
        (avgIterNS, algBW) = comms_utils.getAlgBW(elapsedTimeNs, dataSize, numIters)
        # avgIterNS = elapsedTimeNS / numIters = 10000
        self.assertEqual(10000.0, avgIterNS)
        # algBW = dataSize / avgIterNs = 9
        self.assertEqual(9.0, algBW)


class TestGetSizes(unittest.TestCase):
    """
    Test size getting between iterations.
    """

    def test_get_sizes_with_stepfactor(self):
        beginSize = 32
        endSize = 1024
        stepFactor = 2
        # Start at 32, end at 1024 by increasing by a factor of 2 after each iteration.
        correct_list = [32, 64, 128, 256, 512, 1024]
        result_list = comms_utils.getSizes(beginSize, endSize, stepFactor, stepBytes=0)
        # Lists should have same size and items in the same order.
        self.assertEqual(len(correct_list), len(result_list))
        self.assertTrue(correct_list == result_list)

    def test_get_sizes_with_stepbytes(self):
        beginSize = 32
        endSize = 256
        stepFactor = 2
        stepBytes = 32
        # Start at 32, end at 256 by increasing 32 bytes after each iteration.
        correct_list = [32, 64, 96, 128, 160, 192, 224, 256]
        result_list = comms_utils.getSizes(beginSize, endSize, stepFactor, stepBytes)
        # Lists should have same size and items in the same order.
        self.assertEqual(len(correct_list), len(result_list))
        self.assertTrue(correct_list == result_list)


class TestFixBeginSize(unittest.TestCase):
    """
    Test fix begin size to ensure that we have one member/rank.
    """

    def test_all_to_all(self):
        commsParams = commsParamsTest()
        commsParams.collective = "all_to_all"
        commsParams.beginSize = 0
        commsParams.element_size = 2
        commsParams.bitwidth = 32
        world_size = 16
        comms_utils.fixBeginSize(commsParams, world_size)
        # beginSize / element_size < world_size, so the new begin size should be element_size * world_size
        self.assertEqual(32, commsParams.beginSize)

    def test_all_to_all_quantized(self):
        commsParams = commsParamsTest()
        commsParams.collective = "all_to_all"
        commsParams.beginSize = 0
        commsParams.element_size = 2
        commsParams.bitwidth = 31  # Bitwidth less than 32 triggers quantization
        commsParams.quant_a2a_embedding_dim = 2
        world_size = 16
        comms_utils.fixBeginSize(commsParams, world_size)
        # (beginSize / element_size / world_size) < quant_a2a_embedding_dim, so the new begin size should be element_size * world_size * quant_a2a_embedding_dim
        self.assertEqual(64, commsParams.beginSize)

    def test_all_reduce(self):
        commsParams = commsParamsTest()
        commsParams.collective = "all_reduce"
        commsParams.beginSize = 0
        commsParams.element_size = 2
        world_size = 16
        comms_utils.fixBeginSize(commsParams, world_size)
        # For reduce collectives, beginSize should >= element_size
        self.assertEqual(2, commsParams.beginSize)


class TestGetRankDetails(unittest.TestCase):
    """
    Test if we are getting the rank details from backend correctly.
    """

    def test_mock_backend(self):
        mockBackend = MockBackendFunction()
        mockTuple = (
            mockBackend.local_rank,
            mockBackend.global_rank,
            mockBackend.world_size,
            mockBackend.group,
            mockBackend.device,
            mockBackend.device,
        )
        self.assertEqual(comms_utils.get_rank_details(mockBackend), mockTuple)


class TestEnv2Int(unittest.TestCase):
    """
    Test to see if we are getting environment variables and parsing them correctly to int.
    """

    def test_env_var_found(self):
        os.environ["TEST"] = "100"
        env_list = ["TEST"]
        # We should find TEST in env vars and return 100
        self.assertEqual(100, comms_utils.env2int(env_list))

    def test_env_var_not_found(self):
        env_list = ["DNE"]
        # We won't find DNE in env vars, so return default value of -3.
        self.assertEqual(-3, comms_utils.env2int(env_list, -3))


class TestReadCommEnvVars(unittest.TestCase):
    """
    Test to see if we are reading env vars related to comms correctly.
    """

    def test_read_comm_env_vars(self):
        os.environ["WORLD_SIZE"] = "16"
        os.environ["LOCAL_SIZE"] = "8"
        os.environ["RANK"] = "4"
        os.environ["LOCAL_RANK"] = "0"
        comm_env_vars = comms_utils.read_comms_env_vars()
        # We should only have read 4 env vars.
        self.assertEqual(4, len(comm_env_vars))
        # Check the values of the comm_env_vars.
        self.assertEqual(16, comm_env_vars["world_size"])
        self.assertEqual(8, comm_env_vars["local_size"])
        self.assertEqual(4, comm_env_vars["global_rank"])
        self.assertEqual(0, comm_env_vars["local_rank"])


class TestParamToCommName(unittest.TestCase):
    """
    Test to see if we are converting comm names properly.
    TODO: Test names that are not in supported_comms and check for gracefulexit().
    """

    def test_no_change(self):
        testName = "all_to_all"
        result = comms_utils.paramToCommName(testName)
        self.assertEqual("all_to_all", result)

    def test_change(self):
        testName = "all12345to___a3l1l"  # weird way of typing all_to_all
        result = comms_utils.paramToCommName(testName)
        self.assertEqual("all_to_all", result)


class TestEnsureTensorFlush(unittest.TestCase):
    """
    Run the function to see if it completes without errors. We want to call item() on last
    tensor to ensure flush.
    """

    def test_list_tensors(self):
        tensor_list = [torch.ones(3)]
        last_tensor_value = comms_utils.ensureTensorFlush(tensor_list)
        self.assertEqual(1, last_tensor_value)

    def test_tensors(self):
        tensors = torch.ones(3)
        last_tensor_value = comms_utils.ensureTensorFlush(tensors)
        self.assertEqual(1, last_tensor_value)


if __name__ == "__main__":
    unittest.main()
