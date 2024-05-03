import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from param.comm.backend.base_backend import BaseBackend


class PyTorchTPUBackend(BaseBackend):
    """Extends BaseBackend to implement backend functionalities specific to PyTorch on TPU."""

    def __init__(self, bootstrap_info, comms_params):
        self.bootstrap_info = bootstrap_info
        self.comms_params = comms_params

    def say_hello(self):
        """Prints a startup message with device and network information."""
        my_host = os.uname()[1]
        device = self.get_device()
        hw_device = self.get_hw_device()
        global_rank = self.get_global_rank()
        local_rank = self.get_local_rank()
        world_size = self.get_world_size()
        master_ip = self.bootstrap_info.master_ip
        print(
            f"\tRunning on host: {my_host} g-rank: {global_rank}, "
            f"l-rank: {local_rank} world_size: {world_size} "
            f"master_ip: {master_ip} device: {device} ({hw_device})"
        )

    def all_reduce(self, collective_args, ret_flag=False):
        """Performs an all_reduce operation on TPU."""
        ret_obj = xm.all_reduce(collective_args.op, [collective_args.ip_tensor])
        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)
        if ret_flag:
            return ret_obj

    def reduce(self, collective_args, ret_flag=False):
        """Raises NotImplementedError as reduce is not implemented yet on TPU."""
        raise NotImplementedError("reduce: not implemented yet on TPU")

    def all_to_all(self, collective_args, ret_flag=False):
        """Performs an all-to-all operation on TPU."""
        ret_obj = xm.all_to_all(collective_args.ip_tensor, 0, 0, collective_args.world_size)
        collective_args.op_tensor = ret_obj
        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)
        if ret_flag:
            return ret_obj

    def all_gather(self, collective_args, ret_flag=False):
        """Performs an all_gather operation on TPU."""
        ret_obj = xm.all_gather(collective_args.ip_tensor, dim=0)
        collective_args.op_tensor = ret_obj
        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)
        if ret_flag:
            return ret_obj

    def complete_accel_ops(self, collective_args):
        """Marks the completion of TPU operations for synchronization."""
        xm.mark_step()

    def get_reduce_op(self, op_name):
        """Returns the reduce operation based on the operation name."""
        if op_name == "sum":
            return xm.REDUCE_SUM
        elif op_name == "max":
            return xm.REDUCE_MAX
        else:
            return xm.REDUCE_SUM

    def barrier(self, collective_args, name="world"):
        """Synchronizes all processes in the environment."""
        xm.rendezvous(name)

    def gemm(self, collective_args):
        """Performs a general matrix multiplication (GEMM)."""
        collective_args.mm_out = torch.mm(collective_args.mm_in1, collective_args.mm_in2)

    def alloc_random(self, size_arr, cur_rank_device, dtype, scale_factor=1.0):
        """Allocates a tensor with random values scaled by the given factor."""
        if dtype in (torch.int32, torch.long):
            ip_tensor = torch.randint(0, 1000, size_arr, device=cur_rank_device, dtype=dtype)
        else:
            ip_tensor = torch.rand(size_arr, device=cur_rank_device, dtype=dtype)
        if scale_factor != 1.0:
            ip_tensor /= scale_factor
        return ip_tensor

    def alloc_empty(self, size_arr, dtype, cur_rank_device):
        """Allocates an uninitialized tensor."""
        return torch.empty(size_arr, device=cur_rank_device, dtype=dtype)

    def clear_memory(self, collective_args):
        """Clears allocated memory, no operation for now."""
        pass

    def get_local_rank(self):
        """Returns the local rank of the process on the TPU device."""
        return xm.get_local_ordinal()

    def get_global_rank(self):
        """Returns the global rank of the process across all TPU devices."""
        return xm.get_ordinal()

    def get_world_size(self):
        """Returns the total number of processes participating in the environment."""
        return xm.xrt_world_size()

    def get_device(self):
        """Returns the current device in use as a string."""
        return xm.xla_device()

    def get_hw_device(self):
        """Returns the hardware device information."""
        return xm._xla_real_device(xm.xla_device())

    def benchmark_comms(self, bench_time, comms_params):
        """Runs communication benchmarks across TPU cores."""
        xmp.spawn(
            fn=bench_time,
            args=(comms_params, self),
            nprocs=self.bootstrap_info.num_tpu_cores,
        )

    def __del__(self):
        """Clean-up code for the backend, no operation for now."""
        pass
