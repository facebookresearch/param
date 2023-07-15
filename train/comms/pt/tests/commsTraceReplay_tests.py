import unittest
from unittest import mock

import torch

from comms_utils import commsArgs

from param_bench.train.comms.pt.commsTraceReplay import commsTraceReplayBench
from param_bench.train.comms.pt.tests.mocks.backend_mock import MockBackendFunction
from param_bench.train.comms.pt.tests.test_utils import (
    commsParamsTest,
    createCommsArgs,
    testArgs,
)


class TestPrepComms(unittest.TestCase):
    """
    Test prepComms() to verify that correct tensors are being generated for different
    collective communications.
    """

    def test_no_tensor(self):
        # wait and barrier require no tensors
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = commsArgs()
        curComm.comms = "wait"
        (iptensor, optensor) = testBench.prepComms(curComm, None)
        self.assertEqual(0, len(iptensor))
        self.assertEqual(0, len(optensor))
        curComm.comms = "barrier"
        (iptensor, optensor) = testBench.prepComms(curComm, None)
        self.assertEqual(0, len(iptensor))
        self.assertEqual(0, len(optensor))

    def test_tensor_no_shrink(self):
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = commsArgs(comms="recv", dtype="Int", inMsgSize=1, outMsgSize=1)
        testBench.shrink = False
        testBench.collectiveArgs.world_size = 1
        (iptensor, optensor) = testBench.prepComms(curComm, commsParams)
        # tensor length needs to match world_size
        self.assertEqual(1, len(iptensor))
        self.assertEqual(1, len(optensor))
        # both input and output tensors should be equal to 1
        self.assertEqual(1, iptensor[0])
        self.assertEqual(1, optensor[0])

    def test_tensor_shrink_alltoallv(self):
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = commsArgs(
            comms="all_to_allv",
            dtype="Int",
            inMsgSize=4,
            outMsgSize=4,
            inSplit=[1, 1, 1, 1],
            outSplit=[1, 1, 1, 1],
            worldSize=4,
        )
        testBench.shrink = True
        testBench.collectiveArgs.world_size = 1
        (iptensor, optensor) = testBench.prepComms(curComm, commsParams)
        # tensor length should shrink to world size
        self.assertEqual(1, len(iptensor))
        self.assertEqual(1, len(optensor))
        # both input and output tensors should be equal to 1 for all_to_allv
        self.assertEqual(1, iptensor[0])
        self.assertEqual(1, optensor[0])

    def test_tensor_shrink_allgather(self):
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = commsArgs(
            comms="all_gather", dtype="Int", inMsgSize=4, outMsgSize=4, worldSize=4
        )
        testBench.shrink = True
        testBench.collectiveArgs.world_size = 1
        (iptensor, optensor) = testBench.prepComms(curComm, commsParams)
        # tensor length should shrink to world size
        self.assertEqual(1, len(iptensor))
        self.assertEqual(1, len(optensor))


class TestReplayTrace(unittest.TestCase):
    """
    Make sure function runs without failure for now.
    Future work can be to use unittest.mock to validate printed output.
    """

    def test_warm_up_bench(self):
        test_trace = [
            createCommsArgs(
                comms="test", inMsgSize=1, outMsgSize=1, markerStack=["test_stack"]
            ),
            createCommsArgs(comms="all_gather", inMsgSize=2, outmsgSize=2),
            createCommsArgs(comms="wait", markerStack=["test_stack"]),
        ]
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        testBench.comms_trace = test_trace
        commsParams = commsParamsTest()
        testBench.replayTrace(commsParams, True)
        self.assertTrue(True)  # just check to see if warmup ran without failure

    def test_replay(self):
        test_trace = [
            createCommsArgs(
                comms="test", inMsgSize=1, outMsgSize=1, markerStack=["test_stack"]
            ),
            createCommsArgs(comms="all_gather", inMsgSize=2, outmsgSize=2),
            createCommsArgs(comms="wait", markerStack=["test_stack"]),
        ]
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        testBench.comms_trace = test_trace
        commsParams = commsParamsTest()
        testBench.replayTrace(commsParams)
        self.assertTrue(True)  # just check to see if replay ran without failure


class TestRunComms(unittest.TestCase):
    """
    Make sure function returns a latency and global latency.
    If nonblocking, both of these should be the equal.
    """

    def test_blocking_run(self):
        testBench = commsTraceReplayBench()
        testBench.is_blocking = True
        testBench.backendFuncs = MockBackendFunction()
        collName = "all_gather"
        curComm = commsArgs(req=0)
        (latency, global_latency) = testBench.runComms(collName, curComm, "test_stack")
        self.assertIsNotNone(latency)
        self.assertIsNotNone(global_latency)
        self.assertNotEqual(latency, global_latency)

    def test_non_blocking_run(self):
        testBench = commsTraceReplayBench()
        testBench.is_blocking = False
        testBench.backendFuncs = MockBackendFunction()
        collName = "all_gather"
        curComm = commsArgs(req=0)
        (latency, global_latency) = testBench.runComms(collName, curComm, "test_stack")
        self.assertIsNotNone(latency)
        self.assertIsNotNone(global_latency)
        self.assertEqual(latency, global_latency)


class TestinitTraceStat(unittest.TestCase):
    """
    Test initTraceStat to see if trace stats
    are being initialized properly on the first run.
    """

    def test_dry_run(self):
        test_trace = [
            createCommsArgs(
                comms="test", inMsgSize=1, outMsgSize=1, markerStack=["test_stack"]
            ),
            createCommsArgs(comms="all_gather", inMsgSize=2, outMsgSize=2),
            createCommsArgs(comms="wait", markerStack=["test_stack"]),
        ]
        testBench = commsTraceReplayBench()
        testBench.comms_trace = test_trace
        testBench.is_dry_run = True
        testBench.initTraceStat()
        # Only 2 messages had msg sizes
        self.assertEqual(2, len(testBench.collInMsgBytes))
        self.assertEqual(2, len(testBench.collOutMsgBytes))
        # The sum of the sizes of all all_gather msgs is 2 for in and out
        self.assertEqual(2, sum(testBench.collInMsgBytes["all_gather"]))
        self.assertEqual(2, sum(testBench.collOutMsgBytes["all_gather"]))
        # Dry run records comm blocks. We have two colls in test_stack
        self.assertEqual(2, len(testBench.comms_blocks["test_stack"]))
        # check values of comm_blocks
        self.assertEqual(
            "test", testBench.comms_blocks["test_stack"][0]["comms"]
        )  # first comm in "test_stack" is test
        self.assertEqual(1, testBench.comms_blocks["test_stack"][0]["in_msg_size"])
        self.assertEqual(1, testBench.comms_blocks["test_stack"][0]["out_msg_size"])

        self.assertEqual(
            "wait", testBench.comms_blocks["test_stack"][1]["comms"]
        )  # second comm in "test_stack" is wait

    def test_not_dry_run(self):
        test_trace = [
            createCommsArgs(
                comms="test", inMsgSize=1, outMsgSize=1, markerStack=["test_stack"]
            ),
            createCommsArgs(comms="all_gather", inMsgSize=2, outMsgSize=2),
            createCommsArgs(comms="wait", markerStack=["test_stack"]),
        ]
        testBench = commsTraceReplayBench()
        testBench.comms_trace = test_trace
        testBench.initTraceStat()
        # Only 2 messages had msg sizes
        self.assertEqual(2, len(testBench.collInMsgBytes))
        self.assertEqual(2, len(testBench.collOutMsgBytes))
        # The sum of the sizes of all all_gather msgs is 2 for in and out
        self.assertEqual(2, sum(testBench.collInMsgBytes["all_gather"]))
        self.assertEqual(2, sum(testBench.collOutMsgBytes["all_gather"]))
        # Not dry run does not record comm blocks.
        self.assertEqual(0, len(testBench.comms_blocks["test_stack"]))


class TestInitBench(unittest.TestCase):
    """
    Test initBench to see if replay parameters are being set properly.
    """

    def test_init_bench(self):
        testBench = commsTraceReplayBench()
        commsParams = commsParamsTest()
        args = testArgs()
        args.use_timestamp = True
        args.num_msg = 1000
        args.auto_shrink = False
        args.do_warm_up = False
        testBench.initBench(commsParams, args)
        # check if parameters are being set
        self.assertEqual(True, args.use_timestamp, testBench.use_timestamp)
        self.assertEqual(1000, args.num_msg, testBench.max_msg_cnt)
        self.assertEqual(False, args.auto_shrink, testBench.shrink)
        self.assertEqual(False, args.do_warm_up, not testBench.do_warm_up)


class TestRebalanceSplit(unittest.TestCase):
    """
    Test rebalance split function based on different policies.
    """

    def test_equal_policy(self):
        testBench = commsTraceReplayBench()
        testBench.collectiveArgs.device = "cpu"
        testBench.collectiveArgs.world_size = 2
        testBench.rebalance_policy = "equal"

        testComm = commsArgs()
        testComm.comms = "all_to_allv"
        testComm.inMsgSize = 5
        testComm.outMsgSize = 3
        testComm.inSplit = [3, 2]
        testComm.outSplit = [1, 2]

        ipTensor = torch.tensor(
            [16], dtype=torch.int
        )  # Mock a second rank to have inMsgSize 11
        testBench.backendFuncs = MockBackendFunction()
        testBench.backendFuncs.mock_collective = mock.MagicMock(
            side_effect=(
                lambda collectiveArgs: setattr(collectiveArgs, "ipTensor", ipTensor)
            )
        )

        testBench.rebalanceSplit(testComm)
        # Mock all_reduce wil return 16, so inMsgSize, outMsgSize should be equal to 8 since we are assuming world_size = 2.
        # inSplit and outSplit should be [4, 4]
        print(f"ipTensor after: {testBench.collectiveArgs.ipTensor}")
        self.assertEqual(8, testComm.inMsgSize)
        self.assertEqual(8, testComm.outMsgSize)
        self.assertEqual([4, 4], testComm.inSplit)
        self.assertEqual([4, 4], testComm.outSplit)

    def test_unsupported_policy(self):
        testBench = commsTraceReplayBench()
        testBench.rebalance_policy = (
            "unsupported"  # any str that isn't in supported is considered unsupported
        )

        testComm = commsArgs()
        testComm.comms = "all_to_allv"
        testComm.inMsgSize = 5
        testComm.outMsgSize = 3
        testComm.worldSize = 2
        testComm.inSplit = [3, 2]
        testComm.outSplit = [1, 2]

        testBench.rebalanceSplit(testComm)

        # should be no change
        self.assertEqual(5, testComm.inMsgSize)
        self.assertEqual(3, testComm.outMsgSize)
        self.assertEqual([3, 2], testComm.inSplit)
        self.assertEqual([1, 2], testComm.outSplit)


if __name__ == "__main__":
    unittest.main()
