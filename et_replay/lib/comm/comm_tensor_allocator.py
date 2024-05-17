from typing import List, Tuple

import torch

from .comm_op_args import CommOpArgs
from .comm_trace_replay_args import CommTraceReplayArgs
from .utils import standardize_comm_name


class CommTensorAllocator:
    """
    A class to allocate tensors for various collective communication operations
    in a distributed computing environment.
    """

    def __init__(self, backend, collective_args, init_val=1):
        self.backend = backend
        self.collective_args = collective_args
        self.init_val = init_val

        self.operation_dispatcher = {
            "all_to_all_single": self._allocate_all_to_all_single,
            "all_to_allv": self._allocate_all_to_allv,
            "all_to_all": self._allocate_all_to_all,
            "all_gather": self._allocate_all_gather,
            "gather": self._allocate_all_gather,
            "all_gather_base": self._allocate_all_gather_base,
            "incast": self._allocate_incast,
            "reduce_scatter": self._allocate_reduce_scatter,
            "reduce_scatter_base": self._allocate_reduce_scatter_base,
            "scatter": self._allocate_reduce_scatter,
            "pt2pt": self._allocate_pt2pt,
        }

    def allocate(
        self,
        curr_comm: CommOpArgs,
        comm_trace_replay_args: CommTraceReplayArgs,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate the tensors for collective.

        Args:
            curr_comm: Current collective communication.
            comm_trace_replay_args: Holds parameters that affect tensor allocation.
        Returns:
            (iptensor, optensor): Appropriate input and output tensors for collective.
        """
        comm_op_name = standardize_comm_name(
            (
                curr_comm.comms
                if (curr_comm.comms is not None)
                else comm_trace_replay_args.collective
            ),
            supported_comms=self.backend.collectiveFunc.keys(),
        )

        if comm_op_name in ("wait", "barrier"):
            return ([], [])

        num_elements_in = curr_comm.in_msg_size
        num_elements_out = curr_comm.out_msg_size
        world_size = self.collective_args.world_size
        cur_device = comm_trace_replay_args.device
        dtype = comm_trace_replay_args.dtype
        scaleFactor = world_size
        output_tensor = []

        if allocate:
            if comm_trace_replay_args.dcheck == 1:
                input_tensor = self.backend.alloc_ones(
                    [num_elements_in], cur_device, dtype, scaleFactor=self.initVal
                )
            else:
                input_tensor = self.backend.alloc_random(
                    [num_elements_in], cur_device, dtype, scaleFactor
                )
        else:
            input_tensor = []

        method = self.operation_dispatcher.get(comm_op_name)
        if method:
            input_tensor, output_tensor = method(
                input_tensor,
                curr_comm,
                comm_trace_replay_args,
                num_elements_in,
                num_elements_out,
                world_size,
                cur_device,
                dtype,
                scaleFactor,
                allocate,
            )
        else:
            output_tensor = input_tensor

        return (input_tensor, output_tensor)

    def _allocate_all_to_all_single(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare tensors for all_to_all_single operation.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and output tensors for the operation.
        """
        if allocate:
            if comms_params.dcheck == 1:
                input_tensor = self.backend.alloc_ones(
                    [num_elements_in], cur_device, comms_params.dtype, self.init_val
                )
            else:
                input_tensor = self.backend.alloc_random(
                    [num_elements_in], cur_device, comms_params.dtype, scale_factor
                )
            output_tensor = self.backend.alloc_random(
                [num_elements_out], cur_device, dtype, scale_factor
            )
        else:
            output_tensor = None

        return (input_tensor, output_tensor)

    def _allocate_all_to_allv(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the all_to_allv mode.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and output tensors for the operation.
        """
        if allocate:
            output_tensor = self.backend.alloc_random(
                [num_elements_out], cur_device, dtype, scale_factor
            )
        else:
            output_tensor = []

        # all_to_allv requires tensors to specify split
        self.collective_args.op_tensor_split = (
            curr_comm.out_split
            if curr_comm.out_split is not None
            else [(num_elements_out // world_size) for _ in range(world_size)]
        )
        self.collective_args.input_tensor_split = (
            curr_comm.in_split
            if curr_comm.in_split is not None
            else [(num_elements_in // world_size) for _ in range(world_size)]
        )

        return (input_tensor, output_tensor)

    def _allocate_all_to_all(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepare tensors for all_to_all operation, which requires creating lists of tensors.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Lists of input and output tensors for the operation.
        """
        input_tensor_list = []
        output_tensor_list = []

        if allocate:
            if comms_params.dcheck == 1:
                for _ in range(world_size):
                    input_tensor_list.append(
                        self.backend.alloc_ones(
                            [(num_elements_in // world_size)],
                            cur_device,
                            comms_params.dtype,
                            self.init_val,
                        )
                    )
            else:
                for _ in range(world_size):
                    input_tensor_list.append(
                        self.backend.alloc_random(
                            [(num_elements_in // world_size)],
                            cur_device,
                            comms_params.dtype,
                            scale_factor,
                        )
                    )

            for _ in range(world_size):
                output_tensor_list.append(
                    self.backend.alloc_random(
                        [(num_elements_out // world_size)],
                        cur_device,
                        dtype,
                        scale_factor,
                    )
                )

        return (input_tensor_list, output_tensor_list)

    def _allocate_all_gather(
        self,
        input_tensor: torch.tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare tensors for all_gather operation, which requires a single input tensor and a list of output tensors.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input tensor and list of output tensors for the operation.
        """
        output_tensor_list = []

        if not comms_params.size_from_trace:
            num_elements_in = num_elements_in // world_size

        if allocate:
            if comms_params.dcheck == 1:
                input_tensor = self.backend.alloc_ones(
                    [num_elements_in],
                    cur_device,
                    dtype,
                    scaleFactor=self.init_val,
                )
            else:
                input_tensor = self.backend.alloc_random(
                    [num_elements_in], cur_device, dtype, scale_factor
                )
            # allgather requires a tensor list, e.g., List[torch.Tensor]
            for _ in range(world_size):
                output_tensor_list.append(
                    self.backend.alloc_random(
                        [num_elements_in], cur_device, dtype, scale_factor
                    )
                )
        return (input_tensor, output_tensor_list)

    def _allocate_all_gather_base(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a single all gather operation with a flat output tensor.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input tensor and flat output tensor for the operation.
        """
        if not comms_params.size_from_trace:
            num_elements_out = num_elements_in
            num_elements_in = num_elements_in // world_size

        if allocate:
            if comms_params.dcheck == 1:
                input_tensor = self.backend.alloc_ones(
                    num_elements_in, cur_device, dtype, self.init_val
                )
            else:
                input_tensor = self.backend.alloc_random(
                    [num_elements_in], cur_device, dtype, scale_factor
                )
            output_tensor = self.backend.alloc_random(
                [num_elements_out], cur_device, dtype, scale_factor
            )
        else:
            input_tensor = []
            output_tensor = None

        return (input_tensor, output_tensor)

    def _allocate_incast(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Prepare tensors for incast operation, which requires a list of tensors for the operation.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: The input tensor and list of output tensors for the operation.
        """
        output_tensor_list = []

        if allocate:
            for _ in curr_comm.src_ranks:
                output_tensor_list.append(
                    self.backend.alloc_random(
                        [num_elements_out], cur_device, dtype, scale_factor
                    )
                )

        return (input_tensor, output_tensor_list)

    def _allocate_reduce_scatter(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Prepare tensors for reduce scatter operation.
        This operation typically involves partitioning the input tensor into equal parts among all devices.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in each part of the input tensor.
            num_elements_out: Number of elements in each output part after reduction.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: List of input tensors and a single output tensor for the operation.
        """
        input_tensor_list = []
        if not comms_params.size_from_trace:
            num_elements_in = num_elements_out // world_size
            num_elements_out = num_elements_out // world_size

        if allocate:
            if comms_params.dcheck == 1:
                for _ in range(world_size):
                    input_tensor_list.append(
                        self.backend.alloc_ones(
                            [num_elements_in],
                            cur_device,
                            comms_params.dtype,
                            self.init_val,
                        )
                    )
            else:
                for _ in range(world_size):
                    input_tensor_list.append(
                        self.backend.alloc_random(
                            [num_elements_in],
                            cur_device,
                            comms_params.dtype,
                            scale_factor,
                        )
                    )
            output_tensor = self.backend.alloc_random(
                [num_elements_out], cur_device, dtype, scale_factor
            )
        else:
            output_tensor = None

        return (input_tensor_list, output_tensor)

    def _allocate_reduce_scatter_base(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare tensors for the base version of the reduce scatter operation.
        This operation typically involves partitioning the input tensor into equal
        parts among all devices and then performing a reduction operation.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in each part of the input tensor.
            num_elements_out: Number of elements in each output part after reduction.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: List of input tensors and a single
            output tensor for the operation.
        """
        if not comms_params.size_from_trace:
            num_elements_in = num_elements_out
            num_elements_out = num_elements_out // world_size

        if allocate:
            if comms_params.dcheck == 1:
                input_tensor = self.backend.alloc_ones(
                    num_elements_in, cur_device, comms_params.dtype, self.init_val
                )
            else:
                input_tensor = self.backend.alloc_random(
                    [num_elements_in], cur_device, comms_params.dtype, scale_factor
                )
            output_tensor = self.backend.alloc_random(
                [num_elements_out], cur_device, dtype, scale_factor
            )
        else:
            input_tensor = []
            output_tensor = None

        return (input_tensor, output_tensor)

    def _allocate_pt2pt(
        self,
        input_tensor: torch.Tensor,
        curr_comm: CommOpArgs,
        comms_params: CommTraceReplayArgs,
        num_elements_in: int,
        num_elements_out: int,
        world_size: int,
        cur_device: str,
        dtype: torch.dtype,
        scale_factor: float,
        allocate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare tensors for point-to-point (Pt2Pt) communication operations such
        as send and receive.

        Args:
            input_tensor: Input tensor for the operation.
            curr_comm: Current communication parameters.
            comms_params: Communication parameters holder.
            num_elements_in: Number of elements in the input tensor.
            num_elements_out: Number of elements in the output tensor.
            world_size: Number of devices involved in the operation.
            cur_device: Device where the operation is performed.
            dtype: Data type of the operation.
            scale_factor: Scaling factor for the random tensor generation.
            allocate: Flag indicating whether to allocate tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The input tensor and the output
            tensor for the operation.
        """
        output_tensor = None
        if allocate:
            output_tensor = self.backend.alloc_random(
                [num_elements_out], cur_device, dtype, scale_factor
            )
        return (input_tensor, output_tensor)
