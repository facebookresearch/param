# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .base_backend import BaseBackend
from .coll_args import CollArgs

try:
    from param.comm.fb.internals import (
        all_to_all_internal,
        all_to_allv_internal,
        extend_distributed,
    )

    has_ext_dist = True
except ImportError:
    try:
        import extend_distributed

        has_ext_dist = True
    except ImportError:
        has_ext_dist = False

logger = logging.getLogger(__name__)


def _downcast(input: torch.Tensor, bitwidth: int) -> torch.Tensor:
    if bitwidth == 16:
        return input.to(torch.float16)
    elif bitwidth == 8:
        return input.to(torch.int8)
    else:
        raise NotImplementedError("Unsupported bitwidth. Set --bitwidth to 8/16/32")


def _dequantize(obj: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if obj is None:
        return None
    elif isinstance(obj, torch.Tensor):
        return obj if obj.dtype == torch.float32 else obj.to(torch.float32)
    else:
        result_tensor = obj.value()[0]
        return (
            result_tensor
            if result_tensor.dtype == torch.float32
            else result_tensor.to(torch.float32)
        )


class PyTorchDistBackend(BaseBackend):
    """
    PyTorch implementation of the BaseBackend for distributed operations.
    """

    def __init__(
        self,
        master_ip: str,
        master_port: str,
        world_size: int,
        local_size: int,
        global_rank: int,
        local_rank: int,
        comm_params: Dict,
    ) -> None:
        super().__init__()
        self.master_ip = master_ip
        self.master_port = master_port
        self.world_size = world_size
        self.local_size = local_size
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.comm_params = comm_params
        self.use_ext_dist = comm_params.get("use_ext_dist", False)
        self.collective_func: Dict[str, Callable] = {
            "wait": self.wait,
            "send": self.send,
            "recv": self.recv,
            "isend": self.isend,
            "irecv": self.irecv,
            "batch_isend_irecv": self.batch_isend_irecv,
            "pt2pt": self.noop,
        }
        backend = comm_params.get("backend", "gloo")
        if backend == "ucc":
            try:
                import torch_ucc
            except ImportError:
                try:
                    from ucc_plugin import initialize_ucc_plugin

                    initialize_ucc_plugin(backend)
                except ImportError:
                    raise RuntimeError("Unable to import initialize_ucc_plugin")
        if backend == "fairring":
            try:
                import fairring
            except ImportError:
                raise RuntimeError("Unable to import Fairring")

    def initialize_backend(
        self,
        master_ip: str,
        master_port: str,
        backend: str = "gloo",
        eager_mode: bool = False,
    ) -> None:
        self.set_device(self.local_rank, self.global_rank)
        if has_ext_dist and self.use_ext_dist:
            extend_distributed.init_distributed(
                rank=self.global_rank,
                size=self.world_size,
                backend=backend,
                init_method=self.comm_params.get("init_method", None),
            )
            self.tcp_store = extend_distributed.my_store
        else:
            self.use_ext_dist = False

        if self.tcp_store is None:
            self.initialize_tcpstore(master_ip, master_port)

        if not dist.is_initialized():
            dist.init_process_group(
                backend,
                rank=self.global_rank,
                world_size=self.world_size,
                store=(
                    self.tcp_store if not self.comm_params.get("init_method") else None
                ),
                init_method=self.comm_params.get("init_method"),
                device_id=(
                    torch.device(f"cuda:{self.local_rank}") if eager_mode else None
                ),
            )
        self.groups = {0: self.get_default_group()}
        self.num_pgs = len(self.groups)
        self.round_robin_group = cycle(list(self.groups.values()))

    def say_hello(
        self, global_rank: int, local_rank: int, world_size: int, master_ip: str
    ) -> None:
        myhost = os.uname()[1]
        device = self.get_device()
        hello_msg = (
            f"[Rank {global_rank:3}] host {myhost}, device: {device}, "
            f"local_rank: {local_rank}, world_size: {world_size}, master_ip: {master_ip}"
        )
        self.store_set(f"hello_msg_{global_rank}", hello_msg)
        if global_rank == 0:
            for rank in range(0, world_size):
                rank_hello_msg = self.store_get(f"hello_msg_{rank}").decode()
                print(f"Hello from Rank {rank}: {rank_hello_msg}")

    def alloc_ones(
        self,
        size_arr: int,
        cur_rank_device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        tensor = torch.ones(size_arr, device=cur_rank_device, dtype=dtype)
        return tensor * scale_factor

    def alloc_random(
        self,
        size_arr: int,
        cur_rank_device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        scale_factor: float = 1.0,
    ) -> torch.Tensor:
        if dtype in (
            torch.int8,
            torch.uint8,
            torch.short,
            torch.int16,
            torch.int32,
            torch.long,
        ):
            input_tensor = torch.randint(
                0, 10, size=(size_arr,), device=cur_rank_device, dtype=dtype
            )
        elif dtype == torch.bool:
            input_tensor = torch.rand(size_arr, device=cur_rank_device) < 0.5
        else:
            input_tensor = torch.rand(size_arr, device=cur_rank_device, dtype=dtype)
            if scale_factor != 0:
                input_tensor /= scale_factor
        return input_tensor

    def alloc_empty(
        self, size_arr: int, dtype: torch.dtype, cur_rank_device: str
    ) -> torch.Tensor:
        return torch.empty(size_arr, device=cur_rank_device, dtype=dtype)

    def clear_memory(self, collective_args: CollArgs) -> None:
        del collective_args.input_tensor
        del collective_args.output_tensor
        if collective_args.input_tensor_pair is not None:
            del collective_args.input_tensor_pair
            del collective_args.output_tensor_pair
        torch.cuda.empty_cache()

    def all_reduce(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        quantized = (
            _downcast(collective_args.input_tensor, collective_args.allreduce_qcomm)
            if collective_args.allreduce_qcomm != 32 and not collective_args.pair
            else collective_args.input_tensor
        )

        if self.use_ext_dist:
            ret_obj = collective_args.group.all_reduce(
                tensor=quantized,
                op=collective_args.op,
                async_op=collective_args.async_op,
            )
        else:
            ret_obj = dist.all_reduce(
                quantized,
                op=collective_args.op,
                group=collective_args.group,
                async_op=collective_args.async_op,
            )

        if (
            id(quantized) != id(collective_args.input_tensor)
            and not collective_args.pair
        ):
            ret_obj = (
                ret_obj.get_future().then(_dequantize)
                if collective_args.async_op
                else _dequantize(quantized)
            )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def reduce(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        quantized = (
            _downcast(collective_args.input_tensor, collective_args.reduce_qcomm)
            if collective_args.reduce_qcomm != 32 and not collective_args.pair
            else collective_args.input_tensor
        )

        ret_obj = dist.reduce(
            quantized,
            dst=collective_args.src_or_dst,
            op=collective_args.op,
            group=self.get_collective_group(collective_args),
            async_op=collective_args.async_op,
        )

        if collective_args.reduce_qcomm != 32 and not collective_args.pair:
            ret_obj = (
                ret_obj.get_future().then(_dequantize)
                if collective_args.async_op
                else _dequantize(quantized)
            )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def all_to_all(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        if collective_args.all2all_qcomm and not collective_args.pair:
            collective_args.use_ext_dist = self.use_ext_dist
            work = all_to_all_internal(collective_args)
        else:
            work = dist.all_to_all(
                (
                    collective_args.output_tensor
                    if not collective_args.pair
                    else collective_args.output_tensor_pair
                ),
                (
                    collective_args.input_tensor
                    if not collective_args.pair
                    else collective_args.input_tensor_pair
                ),
                group=self.get_collective_group(collective_args),
                async_op=collective_args.async_op,
            )

        if collective_args.async_op:
            collective_args.wait_obj.append(work)

        if ret_flag:
            return work

    def all_to_allv(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        if (
            collective_args.all2all_qcomm
            and collective_args.input_tensor.dtype == torch.float32
            and (
                collective_args.output_tensor.nelement()
                >= collective_args.quant_threshold
                or collective_args.input_tensor.nelement()
                >= collective_args.quant_threshold
            )
            and not collective_args.pair
        ):
            work = all_to_allv_internal(collective_args)
        elif self.use_ext_dist:
            work = collective_args.group.alltoall_single(
                (
                    collective_args.output_tensor
                    if not collective_args.pair
                    else collective_args.output_tensor_pair
                ),
                (
                    collective_args.input_tensor
                    if not collective_args.pair
                    else collective_args.input_tensor_pair
                ),
                (
                    collective_args.output_tensor_split
                    if not collective_args.pair
                    else collective_args.output_tensor_split_pair
                ),
                (
                    collective_args.input_tensor_split
                    if not collective_args.pair
                    else collective_args.input_tensor_split_pair
                ),
                async_op=collective_args.async_op,
            )
        else:
            work = dist.all_to_all_single(
                (
                    collective_args.output_tensor
                    if not collective_args.pair
                    else collective_args.output_tensor_pair
                ),
                (
                    collective_args.input_tensor
                    if not collective_args.pair
                    else collective_args.input_tensor_pair
                ),
                (
                    collective_args.output_tensor_split
                    if not collective_args.pair
                    else collective_args.output_tensor_split_pair
                ),
                (
                    collective_args.input_tensor_split
                    if not collective_args.pair
                    else collective_args.input_tensor_split_pair
                ),
                group=collective_args.group,
                async_op=collective_args.async_op,
            )

        if collective_args.async_op:
            collective_args.wait_obj.append(work)

        if ret_flag:
            return work

    def gather(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        input_tensors = (
            collective_args.input_tensor_pair
            if collective_args.pair
            else collective_args.input_tensor
        )
        output_tensors = (
            collective_args.output_tensor_pair
            if collective_args.pair
            else collective_args.output_tensor
        )

        ret_obj = dist.gather(
            gather_list=(
                output_tensors
                if collective_args.global_rank == collective_args.src_or_dst
                else None
            ),
            tensor=input_tensors,
            dst=collective_args.src_or_dst,
            group=self.get_collective_group(collective_args),
            async_op=collective_args.async_op,
        )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def scatter(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        input_tensors = (
            collective_args.input_tensor_pair
            if collective_args.pair
            else collective_args.input_tensor
        )
        output_tensors = (
            collective_args.output_tensor_pair
            if collective_args.pair
            else collective_args.output_tensor
        )

        ret_obj = dist.scatter(
            tensor=output_tensors,
            scatter_list=(
                input_tensors
                if collective_args.global_rank == collective_args.src_or_dst
                else None
            ),
            src=collective_args.src_or_dst,
            group=self.get_collective_group(collective_args),
            async_op=collective_args.async_op,
        )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def reduce_scatter(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        input_tensor = (
            collective_args.input_tensor_pair
            if collective_args.pair
            else collective_args.input_tensor
        )
        output_tensor = (
            collective_args.output_tensor_pair
            if collective_args.pair
            else collective_args.output_tensor
        )

        if self.use_ext_dist:
            ret_obj = collective_args.group.reduce_scatter(
                output=output_tensor,
                input_list=input_tensor,
                op=collective_args.op,
                async_op=collective_args.async_op,
            )
        else:
            ret_obj = dist.reduce_scatter(
                output=output_tensor,
                input_list=input_tensor,
                op=collective_args.op,
                group=collective_args.group,
                async_op=collective_args.async_op,
            )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def reduce_scatter_base(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        input_tensor = (
            collective_args.input_tensor_pair
            if collective_args.pair
            else collective_args.input_tensor
        )
        output_tensor = (
            collective_args.output_tensor_pair
            if collective_args.pair
            else collective_args.output_tensor
        )

        ret_obj = dist.reduce_scatter_tensor(
            output=output_tensor,
            input=input_tensor,
            op=collective_args.op,
            group=self.get_collective_group(collective_args),
            async_op=collective_args.async_op,
        )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def broadcast(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        ret_obj = dist.broadcast(
            tensor=(
                collective_args.output_tensor
                if not collective_args.pair
                else collective_args.output_tensor_pair
            ),
            src=collective_args.src_or_dst,
            group=self.get_collective_group(collective_args),
            async_op=collective_args.async_op,
        )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def all_gather(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        ret_obj = (
            collective_args.group.all_gather(
                tensor_list=(
                    collective_args.output_tensor
                    if not collective_args.pair
                    else collective_args.output_tensor_pair
                ),
                tensor=(
                    collective_args.input_tensor
                    if not collective_args.pair
                    else collective_args.input_tensor_pair
                ),
                async_op=collective_args.async_op,
            )
            if self.use_ext_dist
            else dist.all_gather(
                tensor_list=(
                    collective_args.output_tensor
                    if not collective_args.pair
                    else collective_args.output_tensor_pair
                ),
                tensor=(
                    collective_args.input_tensor
                    if not collective_args.pair
                    else collective_args.input_tensor_pair
                ),
                group=collective_args.group,
                async_op=collective_args.async_op,
            )
        )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def all_gather_base(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[torch.Tensor]:
        input_tensor = (
            collective_args.input_tensor_pair
            if collective_args.pair
            else collective_args.input_tensor
        )
        output_tensor = (
            collective_args.output_tensor_pair
            if collective_args.pair
            else collective_args.output_tensor
        )

        ret_obj = dist.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=self.get_collective_group(collective_args),
            async_op=collective_args.async_op,
        )

        if collective_args.async_op:
            collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def incast(self, collective_args: CollArgs) -> None:
        if collective_args.global_rank == collective_args.src_or_dst:
            for idx, src_rank in enumerate(collective_args.src_ranks):
                ret_obj = dist.irecv(
                    tensor=collective_args.output_tensor[idx],
                    src=src_rank,
                    group=self.get_collective_group(collective_args),
                    tag=0,
                )
                collective_args.wait_obj.append(ret_obj)
            if not collective_args.async_op:
                self.complete_accel_ops(collective_args, dev_sync=False)
        elif collective_args.global_rank in collective_args.src_ranks:
            if collective_args.async_op:
                self.isend(collective_args, collective_args.src_or_dst)
            else:
                self.send(collective_args, collective_args.src_or_dst)

    def multicast(self, collective_args: CollArgs) -> None:
        if collective_args.global_rank == collective_args.src_or_dst:
            for dst_rank in collective_args.dst_ranks:
                self.isend(collective_args, dst_rank)
            if not collective_args.async_op:
                self.complete_accel_ops(collective_args, dev_sync=False)
        elif collective_args.global_rank in collective_args.dst_ranks:
            if collective_args.async_op:
                self.irecv(collective_args, collective_args.src_or_dst)
            else:
                self.recv(collective_args, collective_args.src_or_dst)

    def send(
        self,
        collective_args: CollArgs,
        dst_rank: int,
        ret_flag: bool = False,
        tag: int = 0,
    ) -> None:
        dist.send(
            tensor=collective_args.input_tensor,
            dst=dst_rank,
            group=self.get_collective_group(collective_args),
            tag=tag,
        )

    def recv(
        self,
        collective_args: CollArgs,
        src_rank: int,
        ret_flag: bool = False,
        tag: int = 0,
    ) -> None:
        dist.recv(
            tensor=collective_args.output_tensor,
            src=src_rank,
            group=self.get_collective_group(collective_args),
            tag=tag,
        )

    def isend(
        self,
        collective_args: CollArgs,
        dst_rank: int,
        ret_flag: bool = False,
        tag: int = 0,
    ) -> Optional[torch.Tensor]:
        ret_obj = dist.isend(
            tensor=collective_args.input_tensor,
            dst=dst_rank,
            group=self.get_collective_group(collective_args),
            tag=tag,
        )

        collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def irecv(
        self,
        collective_args: CollArgs,
        src_rank: int,
        ret_flag: bool = False,
        tag: int = 0,
    ) -> Optional[torch.Tensor]:
        ret_obj = dist.irecv(
            tensor=collective_args.output_tensor,
            src=src_rank,
            group=self.get_collective_group(collective_args),
            tag=tag,
        )

        collective_args.wait_obj.append(ret_obj)

        if ret_flag:
            return ret_obj

    def P2POp(
        self, collective_args: CollArgs, ret_flag: bool = False, tag: int = 0
    ) -> Optional[torch.distributed.P2POp]:
        if collective_args.collective in ("send", "isend"):
            op = dist.isend
            tensor = collective_args.input_tensor
            peer = collective_args.dst_rank
        elif collective_args.collective in ("recv", "irecv"):
            op = dist.irecv
            tensor = collective_args.output_tensor
            peer = collective_args.src_rank
        else:
            raise RuntimeError(f"Unknown operation type {collective_args.collective}")

        req = dist.P2POp(
            op=op,
            tensor=tensor,
            peer=peer,
            group=self.get_collective_group(collective_args),
            tag=tag,
        )

        collective_args.p2p_ops.append(req)

        if ret_flag:
            return req

    def batch_isend_irecv(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> Optional[List[Any]]:
        reqs = dist.batch_isend_irecv(collective_args.p2p_ops)

        collective_args.p2p_ops.clear()

        for req in reqs:
            collective_args.wait_obj.append(req)

        if ret_flag:
            return reqs

    def wait(self, collective_args: CollArgs, ret_flag: bool = False) -> None:
        if len(collective_args.wait_obj_ids) == 0:
            self.complete_single_op(collective_args)
            return

        if collective_args.collective_id in collective_args.wait_obj_ids:
            wait_obj = collective_args.wait_obj_ids[collective_args.collective_id]
            if wait_obj is not None:
                wait_obj.wait()

    def complete_accel_ops(
        self, collective_args: CollArgs, dev_sync: bool = True
    ) -> None:
        for wait_req in collective_args.wait_obj:
            if wait_req is not None:
                wait_req.wait()
        collective_args.wait_obj.clear()
        collective_args.wait_obj_ids.clear()

        if dev_sync:
            self.device_sync(collective_args)

    def complete_single_op(
        self, collective_args: CollArgs, ret_flag: bool = False
    ) -> None:
        if len(collective_args.wait_obj) > 0:
            wait_req = collective_args.wait_obj.pop(0)
            if wait_req is not None:
                wait_req.wait()
            self.device_sync(collective_args)

    def device_sync(self, collective_args: CollArgs) -> None:
        dev_str = self.comm_params.get("device", "cpu")
        if dev_str == "cuda":
            torch.cuda.synchronize(collective_args.device)

    def get_collective_group(self, collective_args: CollArgs) -> ProcessGroup:
        if self.use_ext_dist:
            return collective_args.group.my_pg
        else:
            return collective_args.group

    def gemm(self, collective_args: CollArgs) -> None:
        collective_args.mm_out = torch.mm(
            collective_args.mm_in1, collective_args.mm_in2
        )

    def add(self, collective_args: CollArgs) -> None:
        collective_args.comp_out = torch.add(
            collective_args.comp_in1, collective_args.comp_in2, alpha=2
        )

    def sub(self, collective_args: CollArgs) -> None:
        collective_args.comp_out = torch.sub(
            collective_args.comp_in1, collective_args.comp_in2, alpha=2
        )

    def add_num(self, collective_args: CollArgs) -> None:
        collective_args.comp_out = torch.add(collective_args.comp_in1, 20)

    def sub_num(self, collective_args: CollArgs) -> None:
        collective_args.comp_out = torch.sub(collective_args.comp_in1, 20)

    def copy(self, collective_args: CollArgs) -> None:
        collective_args.comp_in1.copy_(collective_args.comp_out)

    def get_default_group(self) -> ProcessGroup:
        if self.use_ext_dist:
            return extend_distributed.from_process_group(dist.GroupMember.WORLD)
        else:
            return dist.GroupMember.WORLD

    def get_groups(self) -> List[ProcessGroup]:
        return list(self.groups.values())

    def get_num_pgs(self) -> int:
        return self.num_pgs

    def get_new_pg(self, group_ranks: List[int], backend: str) -> ProcessGroup:
        if self.use_ext_dist:
            return extend_distributed.new_extend_process_group(
                ranks=group_ranks, backend=backend
            )
        else:
            return dist.new_group(ranks=group_ranks, backend=backend)

    def initialize_groups(self, backend: str = "gloo") -> None:
        groups = {}
        world_size = self.get_world_size()

        for pg_id, group_ranks in self.comm_params.get("group_ranks", {}).items():
            if len(group_ranks) > world_size:
                groups.clear()
                break
            if len(group_ranks) == world_size:
                pg = self.get_default_group()
            else:
                pg = self.get_new_pg(group_ranks=group_ranks, backend=backend)
                global_rank = self.get_global_rank()
                if global_rank in group_ranks:
                    logger.info(
                        f"initialize_groups: Rank {global_rank} creates new group pg_id {pg_id} {pg} with {group_ranks}"
                    )
            groups[pg_id] = pg

        if len(groups):
            self.groups = groups

        self.num_pgs = len(self.groups)
        self.round_robin_group = cycle(list(self.groups.values()))

    def set_device(self, local_rank: int, global_rank: int) -> None:
        dev_str = self.comm_params.get("device", "cpu")
        if dev_str.startswith("cuda"):
            if local_rank > torch.cuda.device_count():
                raise ValueError(
                    "Insufficient #GPUs: "
                    f"available {torch.cuda.device_count()} "
                    f"requested {local_rank}"
                )
            torch.cuda.set_device(local_rank)
        logger.info(f"rank {global_rank} set torch device to {dev_str}:{local_rank}")

    def get_device(self) -> str:
        dev_str = self.comm_params.get("device", "cpu")
        my_dev = torch.device(dev_str)
        if dev_str == "cuda":
            ordinal = self.local_rank
            if self.local_rank == -1:
                logger.warning(
                    "Cannot determine device ordinal since LOCAL_RANK is -1. Try GPU 0 and continue."
                )
                ordinal = 0
            my_dev = torch.device(f"cuda:{ordinal}")
        elif dev_str != "cpu":
            raise ValueError(f"{dev_str} is not a valid device option")
        return my_dev

    def get_hw_device(self) -> str:
        return self.get_device()

    def get_local_rank(self) -> int:
        return self.local_rank

    def get_local_size(self) -> int:
        return self.local_size

    def get_global_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def get_collective_group(self, collective_args: CollArgs) -> ProcessGroup:
        if self.use_ext_dist:
            return collective_args.group.my_pg
        else:
            return collective_args.group

    def initialize_tcpstore(self, master_ip: str, master_port: str) -> None:
        self.tcp_store = dist.TCPStore(
            master_ip,
            int(master_port),
            self.world_size,
            is_master=(self.global_rank == 0),
            use_libuv=True,
        )

    def store_get(self, key: str) -> bytes:
        return self.tcp_store.get(key)

    def store_set(self, key: str, val: str) -> None:
        self.tcp_store.set(key, val)

    def get_new_stream(self) -> Optional[torch.cuda.Stream]:
        if self.comm_params.get("device", "cpu") == "cuda":
            return torch.cuda.Stream(device=self.get_device(), priority=0)
        else:
            return None

    def get_new_event(self, enable_timing: bool = False) -> Optional[torch.cuda.Event]:
        if self.comm_params.get("device", "cpu") == "cuda":
            return torch.cuda.Event(enable_timing)
        else:
            return None

    def get_current_stream(
        self, device: Optional[torch.device]
    ) -> Optional[torch.cuda.Stream]:
        if self.comm_params.get("device", "cpu") == "cuda":
            return torch.cuda.current_stream(device)
        else:
            return None

    def switch_stream(
        self, stream: Optional[torch.cuda.Stream], device: Optional[torch.device]
    ) -> Optional[torch.cuda.Stream]:
        if device is None:
            device = self.get_device()
        if stream is not None and device.type == "cuda":
            cur_stream = torch.cuda.current_stream(device=device)
            torch.cuda.set_stream(stream)
            return cur_stream
        else:
            return None

    def sync_stream(
        self,
        stream: Optional[torch.cuda.Stream] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = self.get_device()
        if device.type == "cuda":
            cur_stream = (
                stream
                if stream is not None
                else torch.cuda.current_stream(device=device)
            )
            cur_stream.synchronize()

    def tensor_list_to_numpy(self, tensor_list: List[torch.Tensor]) -> np.ndarray:
        if isinstance(tensor_list, list):
            tensor_list = [t.cpu().detach().numpy() for t in tensor_list]
        return np.array(tensor_list)

    def __del__(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
