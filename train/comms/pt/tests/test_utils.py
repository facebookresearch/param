"""
This file contains test classes with default values for comms unit tests.
Feel free to add additional classes or modify existing ones as needed for new tests.
"""
from comms_utils import commsArgs


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
