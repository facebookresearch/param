import unittest

from param_bench.train.comms.pt.commsTraceReplay import commsTraceReplayBench
from param_bench.train.comms.pt.tests.test_utils import testArgs, commsParamsTest
from param_bench.train.comms.pt.tests.mocks.backend_mock import MockBackendFunction

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
        curComm = {}
        curComm["comms"] = "wait"
        (iptensor, optensor) = testBench.prepComms(curComm, None)
        self.assertEqual(0, len(iptensor), len(optensor))
        curComm["comms"] = "barrier"
        (iptensor, optensor) = testBench.prepComms(curComm, None)
        self.assertEqual(0, len(iptensor), len(optensor))

    def test_tensor_no_shrink(self):
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = {}
        curComm["comms"] = "recv"
        curComm["dtype"] = "Int"
        curComm["in_msg_size"] = 1
        curComm["out_msg_size"] = 1
        testBench.shrink = False
        testBench.collectiveArgs.world_size = 1
        (iptensor, optensor) = testBench.prepComms(curComm, commsParams)
        # tensor length needs to match world_size
        self.assertEqual(1, len(iptensor), len(optensor))
        # both input and output tensors should be equal to 1
        self.assertEqual(1, iptensor[0], optensor[0])

    def test_tensor_shrink_alltoallv(self):
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = {}
        curComm["comms"] = "all_to_allv"
        curComm["dtype"] = "Int"
        curComm["in_msg_size"] = 4
        curComm["out_msg_size"] = 4
        curComm["in_split"] = [1, 1, 1, 1]
        curComm["out_split"] = [1, 1, 1, 1]
        curComm["world_size"] = 4
        testBench.shrink = True
        testBench.collectiveArgs.world_size = 1
        (iptensor, optensor) = testBench.prepComms(curComm, commsParams)
        # tensor length should shrink to world size
        self.assertEqual(1, len(iptensor), len(optensor))
        # both input and output tensors should be equal to 1 for all_to_allv
        self.assertEqual(1, iptensor[0], optensor[0])

    def test_tensor_shrink_allgather(self):
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        commsParams = commsParamsTest()
        commsParams.dcheck = 1
        commsParams.device = "cpu"
        curComm = {}
        curComm["comms"] = "all_gather"
        curComm["dtype"] = "Int"
        curComm["in_msg_size"] = 4
        curComm["out_msg_size"] = 4
        curComm["world_size"] = 4
        testBench.shrink = True
        testBench.collectiveArgs.world_size = 1
        (iptensor, optensor) = testBench.prepComms(curComm, commsParams)
        # tensor length should shrink to world size
        self.assertEqual(1, len(iptensor), len(optensor))

class TestWarmUpBench(unittest.TestCase):
    """
    Make sure function runs without failure for now.
    Future work can be to use unittest.mock to validate printed output.
    """

    def test_warm_up_bench(self):
        test_trace = [
                        {"comms": "test", "in_msg_size": 1,
                         "out_msg_size": 1, "marker_stack": ["test_stack"]},
                        {"comms": "all_gather", "in_msg_size": 2,
                         "out_msg_size": 2},
                        {"comms": "wait", "marker_stack": ["test_stack"]}
                     ]
        testBench = commsTraceReplayBench()
        testBench.backendFuncs = MockBackendFunction()
        testBench.comms_trace = test_trace
        commsParams = commsParamsTest()
        testBench.warmUpBench(commsParams)
        self.assertTrue(True) # just check to see if warmUpBench ran without failure


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
        curComm = {}
        curComm["req"] = 0
        (latency, global_latency) = testBench.runComms(collName, curComm, "test_stack")
        self.assertIsNotNone(latency)
        self.assertIsNotNone(global_latency)
        self.assertNotEqual(latency, global_latency)

    def test_non_blocking_run(self):
        testBench = commsTraceReplayBench()
        testBench.is_blocking = False
        testBench.backendFuncs = MockBackendFunction()
        collName = "all_gather"
        curComm = {}
        curComm["req"] = 0
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
                        {"comms": "test", "in_msg_size": 1,
                         "out_msg_size": 1, "marker_stack": ["test_stack"]},
                        {"comms": "all_gather", "in_msg_size": 2,
                         "out_msg_size": 2},
                        {"comms": "wait", "marker_stack": ["test_stack"]}
                     ]
        testBench = commsTraceReplayBench()
        testBench.comms_trace = test_trace
        testBench.is_dry_run = True
        testBench.initTraceStat()
        # Only 2 messages had msg sizes
        self.assertEqual(2, len(testBench.collInMsgSizes),
                        len(testBench.collOutMsgSizes))
        # The sum of the sizes of all all_gather msgs is 2 for in and out
        self.assertEqual(2, sum(testBench.collInMsgSizes["all_gather"]),
                        sum(testBench.collOutMsgSizes["all_gather"]))
        # Dry run records comm blocks. We have two colls in test_stack
        self.assertEqual(2, len(testBench.comms_blocks["test_stack"]))
        # check values of comm_blocks
        self.assertEqual("test", testBench.comms_blocks["test_stack"][0]["comms"]) # first comm in "test_stack" is test
        self.assertEqual(1, testBench.comms_blocks["test_stack"][0]["in_msg_size"], testBench.comms_blocks["test_stack"][0]["out_msg_size"])

        self.assertEqual("wait", testBench.comms_blocks["test_stack"][1]["comms"]) # second comm in "test_stack" is wait

    def test_not_dry_run(self):
        test_trace = [
                        {"comms": "test", "in_msg_size": 1,
                         "out_msg_size": 1, "marker_stack": ["test_stack"]},
                        {"comms": "all_gather", "in_msg_size": 2,
                         "out_msg_size": 2},
                        {"comms": "wait", "marker_stack": ["test_stack"]}
                     ]
        testBench = commsTraceReplayBench()
        testBench.comms_trace = test_trace
        testBench.initTraceStat()
        # Only 2 messages had msg sizes
        self.assertEqual(2, len(testBench.collInMsgSizes),
                        len(testBench.collOutMsgSizes))
        # The sum of the sizes of all all_gather msgs is 2 for in and out
        self.assertEqual(2, sum(testBench.collInMsgSizes["all_gather"]),
                        sum(testBench.collOutMsgSizes["all_gather"]))
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
        args.no_warm_up = False
        testBench.initBench(commsParams, args)
        # check if parameters are being set
        self.assertEqual(True, args.use_timestamp, testBench.use_timestamp)
        self.assertEqual(1000, args.num_msg, testBench.max_msg_cnt)
        self.assertEqual(False, args.auto_shrink, testBench.shrink)
        self.assertEqual(False, args.no_warm_up, not testBench.do_warm_up)

if __name__ == '__main__':
    unittest.main()
