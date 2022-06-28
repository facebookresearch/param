"""
This file contains test classes with default values for comms unit tests.
Feel free to add additional classes or modify existing ones as needed for new tests.
"""

class testArgs: # default args to run tests with
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
