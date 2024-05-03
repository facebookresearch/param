import os

import torch
from comm_utils import commsArgs

from param.comm import comm_utils
from param.comm.backend.mock import MockBackendFunction
from param.comm.tests.test_utils import (
    bootstrap_info_test,
    commsParamsTest,
)


class testArgs:  # default args to run tests with
    def __init__(self):
        self.trace_file = ""
        self.use_remote_trace = False
        self.dry_run = False
        self.auto_shrink = False
        self.max_msg_cnt = 0  # 0 means no limit
        self.num_msg = 0
        self.z = 0
        self.no_warm_up = True
        self.allow_ops = ""
        self.output_path = "/tmp/paramReplayedTrace"
        self.colls_per_batch = -1
        self.use_timestamp = False
        self.rebalance_policy = ""


class commsParamsTest:
    def __init__(self):
        # A holding object for common input parameters, add as needed to test
        self.nw_stack = "pytorch_dist"
        self.dtype = "Int"
        self.backend = "nccl"
        self.device = "cpu"
        self.blockingFlag = 1
        # quantization
        self.bitwidth = 32
        self.quant_a2a_embedding_dim = 1
        self.quant_threshold = 1
        self.dcheck = 1
        self.num_pgs = 1


class bootstrap_info_test:
    def __init__(self):
        self.global_rank = 0
        self.local_rank = 0
        self.world_size = 16

        self.master_ip = "localhost"
        self.master_port = "25555"
        self.num_tpu_cores = 16


def createCommsArgs(**kwargs) -> commsArgs:
    """
    Test utility to create comms args from a dict of values.
    """
    curComm = commsArgs()
    for key, value in kwargs.items():
        setattr(curComm, key, value)

    return curComm


class TestParseSize(unittest.TestCase):
    """
    Test and see if sizes are being parsed correctly.
    """

    def test_gb_size(self):
        sizeStr = "2GB"
        size = comm_utils.parsesize(sizeStr)
        # size is in bytes
        self.assertEqual(2147483648, size)

    def test_mb_size(self):
        sizeStr = "3MB"
        size = comm_utils.parsesize(sizeStr)
        self.assertEqual(3145728, size)

    def test_kb_size(self):
        sizeStr = "5KB"
        size = comm_utils.parsesize(sizeStr)
        self.assertEqual(5120, size)

    def test_single_size(self):
        sizeStr = "1024"
        size = comm_utils.parsesize(sizeStr)
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
        parsed_rank_list = comm_utils.parseRankList(comma_rank_list)
        # We should have 4 ranks returned.
        self.assertEqual(4, len(parsed_rank_list))
        # We should have ranks 0,2,4,6. They should be in this order as well.
        for i in range(4):
            self.assertEqual(i * 2, parsed_rank_list[i])

    def test_range_ranks(self):
        range_rank_list = "0:7"  # This is inclusive end.
        bootstrap_info = bootstrap_info_test()
        bootstrap_info.world_size = 8
        parsed_rank_list = comm_utils.parseRankList(range_rank_list)
        # We should have 8 ranks returned.
        self.assertEqual(8, len(parsed_rank_list))
        # We should have ranks 0-7 inclusive, in order.
        for i in range(8):
            self.assertEqual(i, parsed_rank_list[i])

    def test_single_rank(self):
        single_rank = "5"
        bootstrap_info = bootstrap_info_test()
        bootstrap_info.world_size = 8
        parsed_rank_list = comm_utils.parseRankList(single_rank)
        # We should have 1 rank returned.
        self.assertEqual(1, len(parsed_rank_list))
        # We should have rank 5.
        self.assertEqual(5, parsed_rank_list[0])


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
        result_list = comm_utils.getSizes(beginSize, endSize, stepFactor, stepBytes=0)
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
        result_list = comm_utils.getSizes(beginSize, endSize, stepFactor, stepBytes)
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
        comm_utils.fixBeginSize(commsParams, world_size)
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
        comm_utils.fixBeginSize(commsParams, world_size)
        # (beginSize / element_size / world_size) < quant_a2a_embedding_dim, so the new begin size should be element_size * world_size * quant_a2a_embedding_dim
        self.assertEqual(64, commsParams.beginSize)

    def test_all_reduce(self):
        commsParams = commsParamsTest()
        commsParams.collective = "all_reduce"
        commsParams.beginSize = 0
        commsParams.element_size = 2
        world_size = 16
        comm_utils.fixBeginSize(commsParams, world_size)
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
        self.assertEqual(comm_utils.get_rank_details(mockBackend), mockTuple)


class TestEnv2Int(unittest.TestCase):
    """
    Test to see if we are getting environment variables and parsing them correctly to int.
    """

    def test_env_var_found(self):
        os.environ["TEST"] = "100"
        env_list = ["TEST"]
        # We should find TEST in env vars and return 100
        self.assertEqual(100, comm_utils.env2int(env_list))

    def test_env_var_not_found(self):
        env_list = ["DNE"]
        # We won't find DNE in env vars, so return default value of -3.
        self.assertEqual(-3, comm_utils.env2int(env_list, -3))


class TestReadCommEnvVars(unittest.TestCase):
    """
    Test to see if we are reading env vars related to comms correctly.
    """

    def test_read_comm_env_vars(self):
        os.environ["WORLD_SIZE"] = "16"
        os.environ["LOCAL_SIZE"] = "8"
        os.environ["RANK"] = "4"
        os.environ["LOCAL_RANK"] = "0"
        comm_env_vars = comm_utils.read_env_vars()
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
        result = comm_utils.paramToCommName(testName)
        self.assertEqual("all_to_all", result)

    def test_change(self):
        testName = "all12345to___a3l1l"  # weird way of typing all_to_all
        result = comm_utils.paramToCommName(testName)
        self.assertEqual("all_to_all", result)


if __name__ == "__main__":
    unittest.main()
