import torch

class MockBackendFunction: # Mock backend function
    # TODO: Add configurable options.
    def __init__(self):
        self.collectiveFunc = {
            "all_to_all": self.all_to_all,
            "all_to_allv": self.all_to_allv,
            "all_reduce": self.all_reduce,
            "broadcast": self.broadcast,
            "all_gather": self.all_gather,
            "reduce": self.reduce,
            "barrier": self.barrier,
            "recv": self.recv,
            "noop": self.noop,
        }

        self.device = "cpu"
        self.world_size = 1
        self.local_rank = 0
        self.global_rank = 0
        self.group = "default"

    def noop(self, collectiveArgs=None, retFlag=False, pair=False):
        """no-op for the case we want to skip comms/compute"""
        pass

    def sayHello(self, global_rank, local_rank, world_size, master_ip):
        pass

    # Collectives
    def all_gather(self, collectiveArgs, retFlag=False):
        pass

    def all_reduce(self, collectiveArgs, retFlag=False):
        pass

    def broadcast(self, collectiveArgs, retFlag=False):
        pass

    def reduce(self, collectiveArgs, retFlag=False):
        pass

    def all_to_all(self, collectiveArgs, retFlag=False):
        pass

    def all_to_allv(self, collectiveArgs, retFlag=False):
        pass

    def recv(self, collectiveArgs, retFlag=False):
        pass

    def complete_accel_ops(self, collectiveArgs, devSync=False):
        pass

    def barrier(self, collectiveArgs, name="dummy"):
        pass

    def sync_barrier(self, collectiveArgs, desc="world"):
        self.barrier(collectiveArgs, name=desc)

    def get_reduce_op(self, opName):
        pass

    # Compute functions

    def gemm(self, collectiveArgs):
        pass

    # Memory related

    def get_mem_size(self, collectiveArgs):
        pass

    def alloc_embedding_tables(self, n, m, curRankDevice, dtype):
        pass

    def alloc_empty(self, sizeArr, dtype, curRankDevice):
        pass

    def clear_memory(self, collectiveArgs):
        pass

    # Getting world-size and other information.

    def get_local_rank(self):
        return self.local_rank

    def get_global_rank(self):
        return self.global_rank

    def get_world_size(self):
        return self.world_size

    def get_device(self):
        return self.device

    def get_hw_device(self):
        return self.device

    def get_default_group(self):
        return self.group

    def get_groups(self):
        pass

    # Init functions

    def initialize_backend(self, master_ip, master_port, backend="gloo"):
        pass

    def benchmark_comms(self):
        pass

    def alloc_ones(
        self, sizeArr, curRankDevice="cpu", dtype=torch.int32, scaleFactor=1.0
    ):
        ipTensor = torch.ones(sizeArr, device=curRankDevice, dtype=dtype)
        if scaleFactor != 1.0:
            ipTensor = ipTensor * scaleFactor
        return ipTensor

    def alloc_random(
        self, sizeArr, curRankDevice="cpu", dtype=torch.int32, scaleFactor=1.0
    ):
        return self.alloc_ones(sizeArr, "cpu", dtype, 1.0) # just return arrays of 1 for testing
