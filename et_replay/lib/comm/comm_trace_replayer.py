import argparse
import logging
import time
from typing import Any, Dict, List, Tuple

import torch

from ..utils import read_mpi_env_vars
from .backend import BaseBackend, CollArgs, PyTorchDistBackend
from .comm_op_args import CommOpArgs
from .comm_tensor_allocator import CommTensorAllocator
from .comm_trace_reader import CommTraceReader
from .comm_trace_replay_args import CommTraceReplayArgs
from .param_profile import ParamProfile
from .param_timer import ParamTimer

try:
    from trainer_iteration_wrapper import setTrainingIteration  # @manual
except ImportError:
    pass

logger = logging.getLogger(__name__)

# sleep for 20ms to wait for next collective
LOOP_TIMER_S = 0.02


def get_rank_details(
    backend: BaseBackend,
) -> Tuple[int, int, int, Any, str, str]:
    """
    Returns the details of the rank for the current backendFunction.

    Args:
        backend: Backend we are gathering information from.
    Returns:
        (local_rank, global_rank, world_size, group, curDevice, curHwDevice): Returns the values of these in the provided backendFunction.
    """
    local_rank = backend.get_local_rank()
    global_rank = backend.get_global_rank()
    world_size = backend.get_world_size()
    group = backend.get_default_group()
    curDevice = backend.get_device()
    curHwDevice = backend.get_hw_device()

    return (local_rank, global_rank, world_size, group, curDevice, curHwDevice)


class CommTraceReplayer:
    """
    A class to replay and benchmark generated traces for collective communications.

    This class will read a provided trace and replay it based on runtime parameters
    specified in the command line. At the end of a replay, the benchmarks for the run
    will be recorded in different JSON files in the specified out_path, if provided.
    The goal of this class is to help scale AI training optimizations by studying the
    behaviors of AI backends.
    """

    def __init__(self, args):
        (world_size, local_size, global_rank, local_rank) = read_mpi_env_vars()
        comm_trace_replay_args = CommTraceReplayArgs(args)
        self.init_backend(
            args.master_ip,
            args.master_port,
            world_size,
            local_size,
            global_rank,
            local_rank,
            comm_trace_replay_args,
        )
        self.init_replay(comm_trace_replay_args, args)

        self.global_rank = self.backend.get_global_rank()
        logger.info(f"[Rank-{self.global_rank}] reading {self.trace_type} trace")
        self.report = (
            True
            if self.global_rank == 0
            or (
                comm_trace_replay_args.enable_local_report
                and self.backend.get_local_rank() == 0
            )
            else False
        )
        self.trace_reader = CommTraceReader(self.backend.get_world_size())
        self.comm_trace = self.trace_reader.read_trace(
            remote_path=self.trace_file, rank=self.global_rank
        )

        self.init_trace_stats()

        self.supported_network_stacks = ["pytorch-dist"]
        self.dtype_map = {
            "float": torch.float32,
            "float32": torch.float32,
            "float16": torch.half,
            "float64": torch.double,
            "double": torch.double,
            "int32": torch.int32,
            "int": torch.int32,
            "long": torch.long,
            "bfloat16": torch.bfloat16,
            "bool": torch.bool,
            "half": torch.half,
            "byte": torch.uint8,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "short": torch.short,
            "char": torch.int8,
        }
        self.backend = None
        self.collective_args = CollArgs()
        self.init_val = 1
        self.report = False
        self.comm_trace = {}
        self.is_dry_run = False
        self.shrink = False
        self.max_msg_cnt = 0  # 0 means no limit
        self.is_blocking = True
        self.do_warm_up = True
        self.reuse_tensors = False

        self.allow_list = ""
        self.out_path = ""
        self.output_ranks = None
        self.colls_per_batch = -1
        self.use_timestamp = False
        self.num_replays = 1
        self.profiler_num_replays_start = 0
        self.profiler_num_replays = 10

        self.coll_in_msg_bytes: Dict[str, List] = {}

        self.comms_blocks: Dict[str, List] = {}
        self.replay_iter = 0

        self.et_to_tensors = {}

        self.tensor_allocator = CommTensorAllocator()
        self.stats = CommStats()

    def init_backend(
        self,
        master_ip,
        master_port,
        world_size,
        local_size,
        global_rank,
        local_rank,
        comm_trace_replay_args,
    ) -> None:
        """
        Initializes backend.

        Args:
            bootstrap_info: Holds current environment information.
            comm_trace_replay_args: Holds comms params to pass into backend for initialization.

        Returns:
            None
        """
        # init backend and corresponding function pointers
        if comm_trace_replay_args.network_stack == "pytorch-dist":
            self.backend = PyTorchDistBackend(bootstrap_info, comm_trace_replay_args)
        else:
            logger.error("Unsupported NW stack! ")
            comm_utils.gracefulExit()

        self.backend.initialize_backend(
            master_ip,
            master_port,
            backend=comm_trace_replay_args.backend,
        )
        self.backend.sayHello()

    def init_replay(
        self, comm_trace_replay_args: CommTraceReplayArgs, args: argparse.Namespace
    ) -> None:
        """
        Initializes replay parameters based on provided command line arguments and comms parameters.

        Args:
            comm_trace_replay_args: Holds comms params to initialize quantized communication context.
            args: Namespace containing command line args that will set replay parameters.

        Returns:
            None
        """
        self.is_dry_run = args.dry_run
        self.shrink = args.auto_shrink
        self.max_msg_cnt = args.max_msg_cnt
        self.is_blocking = args.z
        self.do_warm_up = args.do_warm_up
        self.reuse_tensors = args.reuse_tensors
        self.allow_list = args.allow_ops
        if args.output_ranks == "all":
            self.output_ranks = list(range(self.backend.get_world_size()))
        else:
            self.output_ranks = comm_utils.parse_rank_list(args.output_ranks)
        self.out_path = args.output_path
        self.colls_per_batch = args.colls

    def init_trace_stats(self):
        """
        Perform a first pass on the trace to gather statistics on message count, message sizes,
        and record how many collectives each block has.
        """
        self.max_msg_cnt = (
            len(self.comm_trace) if self.max_msg_cnt == 0 else self.max_msg_cnt
        )
        self.stats.update_message_count(len(self.comm_trace))
        for curr_comm in self.comm_trace[: self.stats.max_msg_cnt]:
            if curr_comm.compute is not None:
                continue

            coll_name = standardize_comm_name(curr_comm.comms)
            dtype_size = torch.tensor(
                [], dtype=self.dtype_map[curr_comm.dtype]
            ).element_size()
            self.stats.record_communication(
                coll_name, curr_comm.in_msg_size, curr_comm.out_msg_size, dtype_size
            )

            # Get information sorted by code block
            curr_blocks = (
                curr_comm.marker_stack if curr_comm.marker_stack is not None else []
            )
            for curr_block in curr_blocks:
                if curr_block not in self.stats.comms_blocks:
                    self.stats.comms_blocks[curr_block] = []
                if self.is_dry_run and coll_name not in ("wait", "barrier"):
                    self.stats.comms_blocks[curr_block].append(
                        {
                            "comms": coll_name,
                            "in_msg_size": curr_comm.in_msg_size,
                            "out_msg_size": curr_comm.out_msg_size,
                        }
                    )
                else:
                    self.stats.comms_blocks[curr_block].append({"comms": coll_name})

    def run(self, comm_trace_replay_args: CommTraceReplayArgs) -> None:
        """
        Run the comms-replay benchmark:
        1) Each rank reads its trace
        2) First pass of the trace to ensure the format is valid and get basic stats
        3) Execute communication replay [Skip if on dry-run mode]
        4) Report stats and performance (if not dry-run)

        Args:
            comm_trace_replay_args: Holds comms params to pass into inner functions.

        Returns:
            None
        """
        if not self.is_dry_run:
            self.set_bench(comm_trace_replay_args)
            self.bench_time(comm_trace_replay_args)

        if self.report:
            self.report_bench_time()

        if not self.is_dry_run:
            self.backend.barrier(self.collective_args)
            self.backend.complete_accel_ops(self.collective_args)

    def set_bench(
        self,
        comm_trace_replay_args: CommTraceReplayArgs,
    ) -> None:
        """
        Initializes replay basic collective info.

        Args:
            comm_trace_replay_args: Holds comms params to pass into backend for initialization.

        Returns:
            None
        """
        # init process groups
        for curr_comm in self.comm_trace[: self.max_msg_cnt]:
            # record process group info
            if curr_comm.comms == "init":
                comm_trace_replay_args.group_ranks[curr_comm.pg_id] = (
                    curr_comm.group_ranks
                )
        self.backend.initialize_groups(comm_trace_replay_args.backend)

        # set basic collective info
        local_rank, global_rank, world_size, group, cur_device, cur_hw_device = (
            get_rank_details(self.backend)
        )

        self.collective_args.group = group  # default group
        self.collective_args.groups = self.backend.get_groups()
        self.collective_args.num_pgs = self.backend.get_num_pgs()
        self.collective_args.device = cur_device
        self.collective_args.world_size = world_size
        self.collective_args.global_rank = global_rank
        self.collective_args.backend_funcs = self.backend
        self.collective_args.src_or_dst = (
            0  # assuming it's always sum for reduce/allreduce
        )
        self.collective_args.op = self.backend.get_reduce_op("sum")
        self.collective_args.async_op = not self.is_blocking
        self.collective_args.input_tensor = None
        self.collective_args.output_tensor = None
        self.collective_args.enable_profiler = comm_trace_replay_args.enable_profiler

        # set of collectives to be replayed
        if self.allow_list in ("all", "default", "*"):
            self.allow_list = self.backend.collective_func.keys()
        else:
            self.allow_list = [
                self.param_to_comm_name(op) for op in self.allow_list.split(",")
            ]

    def bench_time(self, comm_trace_replay_args: CommTraceReplayArgs) -> None:
        """
        Run all collectives in the current rank and record timing metrics for benchmarking.

        Args:
            comm_trace_replay_args: Holds comms params to pass into prep_comms() to acquire appropriate tensors
                          and perform data validation in blocking runs.

        Returns:
            None
        """
        if comm_trace_replay_args.enable_profiler:
            # num of iterations to skip
            numWarmupIters = (
                1 if self.do_warm_up else 0
            ) + self.profiler_num_replays_start
            # num of iterations to profile, at most num_replays iterations
            numProfileIters = (
                self.profiler_num_replays
                if self.profiler_num_replays < self.num_replays
                else self.num_replays
            )

        # warm-up
        if self.do_warm_up:
            self.replay_iter = -1
            self.replay_trace(
                comm_trace_replay_args=comm_trace_replay_args, warmup=True
            )
        self.reset_comms()

        # sync everything before starting real runs
        with ParamProfile(description="# PARAM replay warmup post-replay global sync"):
            self.backend.sync_barrier(self.collective_args)

        if self.backend.get_global_rank() == 0:
            logger.info(
                f"\n+ {self.max_msg_cnt} messages in the trace...replaying (if present) {list(self.allow_list)}"
            )
            for coll, sizes in self.coll_in_msg_bytes.items():
                logger.info(f"\t{coll}: {len(sizes)}")

        trace_start_time = time.monotonic_ns()
        for i in range(self.num_replays):
            if self.backend.get_global_rank() == 0:
                logger.info(f"Replay #{i}")

            # set training iteration number in NCCL
            try:
                setTrainingIteration(i + 1)
            except NameError:
                pass

            # replay comms trace
            self.replay_iter = i
            self.replay_trace(
                comm_trace_replay_args=comm_trace_replay_args, warmup=False
            )
            self.reset_comms()

            # make sure all ops are completed
            with ParamProfile(
                description=f"# PARAM replay {self.replay_iter} post-replay global sync"
            ):
                self.backend.sync_barrier(self.collective_args)

        # record how long it took for trace-replay to complete
        trace_end_time = time.monotonic_ns()
        self.stats.total_trace_latency = (
            trace_end_time - trace_start_time
        ) / 1e3  # make it us

        # cleanup any memory left in use
        self.backend.clear_memory(self.collective_args)

    def reset_comms(self):
        """
        Reset collective group to default PG
        """
        self.collective_args.group = self.backend.get_default_group()
        self.world_size = self.backend.get_world_size()

    def report_bench_time(self):
        # TODO
        pass

    def replay_trace(
        self,
        comm_trace_replay_args: CommTraceReplayArgs,
        warmup: bool = False,
    ) -> None:
        """
        Replay comms trace. This function handles the entire process of replaying
        communication traces for benchmarking purposes.

        Args:
            comm_trace_replay_args: Run-time parameters for replay.
            warmup: Indicates whether this is a warm-up run to prepare the system.

        Returns:
            None
        """
        log_label = "[Warm-up]" if warmup else f"[Replay {self.replay_iter}]"
        start_time = time.monotonic_ns()
        for cnt, curr_comm in enumerate(self.comm_trace[: self.max_msg_cnt]):
            curr_blocks = (
                curr_comm.marker_stack if curr_comm.marker_stack is not None else []
            )
            curr_block_stack = (
                " ".join(curr_blocks) if len(curr_blocks) > 0 else "Unnamed/Unknown"
            )

            if curr_comm.compute is not None:
                # Skip processing for compute operations
                continue

            # Get the name of the collective from the comm object
            coll_name = self.param_to_comm_name(curr_comm.comms)
            (group_rank, group_desc) = self.get_comm_group_info(
                curr_comm, comm_trace_replay_args
            )
            # Skip comm if the local process doesn't belong to the PG or encounter an unexpected collective
            if coll_name not in self.allow_list or group_rank == -1:
                continue

            if group_rank >= 0:
                comm_desc = f"{str(curr_comm.comms)}: NumElemsIn={curr_comm.in_msg_size}, NumElemsOut={curr_comm.out_msg_size}, Dtype={curr_comm.dtype}"
                if curr_comm.comms == "all_to_allv":
                    comm_desc += f", InSplit={curr_comm.in_split}, OutSplit={curr_comm.out_split}"
                if curr_comm.comms in SupportedP2pOps:
                    comm_desc += f", Src_Rank={curr_comm.src_rank}, Dst_Rank={curr_comm.dst_rank}"
                logger.info(
                    f"{log_label}[Rank {self.collective_args.global_rank:3}] [{cnt+1} / {self.max_msg_cnt}] Replaying {comm_desc} with {group_desc}"
                )

            # Prepare the tensors
            (self.collective_args.input_tensor, self.collective_args.output_tensor) = (
                self.prep_comms(
                    curr_comm, comm_trace_replay_args, not self.reuse_tensors
                )
            )

            # Wait for collective timestamp if enabled
            if not warmup and self.use_timestamp:
                self.wait_for_timestamp(curr_comm, start_time)

            # Execute the collective operation
            (latency, global_latency) = self.run_comms(
                coll_name, curr_comm, curr_block_stack
            )

            # Perform data validation check on the final output_tensor
            if (
                self.is_blocking
                and comm_trace_replay_args.dcheck == 1
                and coll_name not in ("wait", "barrier")
            ):
                comm_trace_replay_args.collective = coll_name
                comm_trace_replay_args.src_or_dst = (
                    curr_comm.root if curr_comm.root is not None else 0
                )
                self.dcheck(
                    comm_trace_replay_args,
                    curr_comm.out_msg_size,
                    self.collective_args.output_tensor,
                )

            if self.backend.get_global_rank() == 0:
                logger.info(
                    f"{log_label}[{cnt+1} / {self.max_msg_cnt}] Replayed {coll_name} in block [{curr_block_stack}]... {global_latency:.2f} us"
                )

    def get_comm_group_info(
        self, curr_comm: CommOpArgs, comm_trace_replay_args: CommTraceReplayArgs
    ) -> (int, str):
        """
        Return the group infomation of the current process group
        including group rank of the local process, and a description string for logging purpose.
        A -1 group rank indicates an invalid process group on the local process.
        """

        # If a PG is associated, the process needs to be included in the PG (group_rank != -1);
        # otherwise invalid communication to the local process.
        if curr_comm.pg_id is not None and not self.shrink:
            group = self.collective_args.groups[curr_comm.pg_id]
            groupDesc = f"PG: id={curr_comm.pg_id}, world_ranks={comm_trace_replay_args.group_ranks[curr_comm.pg_id]}"
        else:
            group = self.backend.get_default_group()
            groupDesc = "PG: default group"

        return (self.backend.get_group_rank(group), groupDesc)

    def wait_for_timestamp(self, curr_comm: CommOpArgs, start_time: float) -> None:
        """
        Sleep until enough time has passed to match the collective's timestamp, based on the start time.

        Args:
            curr_comm: Current collective to sleep/wait for.
            start_time: Start time when replay began.

        Returns:
            None
        """
        if curr_comm.start_time_ns is not None:  # for backwards compatibility
            while time.monotonic_ns() - start_time <= curr_comm.start_time_ns:
                time_diff = curr_comm.start_time_ns - (time.monotonic_ns() - start_time)
                if time_diff / 1e9 >= LOOP_TIMER_S:  # Convert ns to seconds
                    time.sleep(LOOP_TIMER_S)

    def prep_comms(
        self,
        curr_comm: CommOpArgs,
        comm_trace_replay_args: CommTraceReplayArgs,
        regenerate_tensors: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares the appropriate tensors for the current collective communication.

        Args:
            curr_comm: The current communication that we are preparing the correct tensor for.
            comm_trace_replay_args: Holds the comms param arguments that will determine tensor attributes.
            regenerate_tensors: when an id is being replayed multiple times, setting this to false will use tensors from previous runs

        Returns:
            Tuple[torch.Tensor, torch.Tensor] if the current communication requires tensors, None otherwise.
        """
        comm_op = self.param_to_comm_name(curr_comm.comms)
        if comm_op in ("wait", "barrier", "batch_isend_irecv"):
            return ([], [])

        # Prepare process group for hard-coded traces
        if curr_comm.pg_id is not None and not self.shrink:
            self.collective_args.group = self.collective_args.groups[curr_comm.pg_id]
            self.collective_args.world_size = (
                curr_comm.world_size
            )  # match world size to the size of the current PG
        else:  # use default process group if no pg_id is provided or shrink is enabled
            self.collective_args.group = self.backend.get_default_group()
            self.world_size = self.backend.get_world_size()

        # For all_to_allv, we can shrink the size if running on smaller scale
        if self.shrink and comm_op == "all_to_allv":
            new_num_elems_in = (
                curr_comm.in_msg_size // curr_comm.world_size
            ) * self.world_size
            new_num_elems_out = (
                curr_comm.out_msg_size // curr_comm.world_size
            ) * self.world_size
            curr_comm.in_msg_size = new_num_elems_in
            curr_comm.out_msg_size = new_num_elems_out

        if not curr_comm.id or regenerate_tensors:
            return self.tensor_allocator.allocate(curr_comm, comm_trace_replay_args)
        else:
            comm_op_hash = self.hash_et_comm_op(curr_comm)
            if comm_op_hash in self.et_to_tensors:
                # Reuse tensors from previous runs if available
                (input_tensor, output_tensor) = self.et_to_tensors[comm_op_hash]
            else:
                (input_tensor, output_tensor) = self.tensor_allocator.allocate(
                    curr_comm, comm_trace_replay_args, True
                )
                self.et_to_tensors[comm_op_hash] = (input_tensor, output_tensor)
        return (input_tensor, output_tensor)

    def hash_et_comm_op(self, comm_op: CommOpArgs) -> int:
        """
        Hash the current collective communication into a unique integer for tensors reuse

        """
        op = None
        if comm_op.comms in SupportedP2pOps:
            op = (
                comm_op.comms,
                comm_op.src_rank,
                comm_op.dst_rank,
                comm_op.in_msg_size,
                comm_op.out_msg_size,
            )
        elif comm_op.in_split or comm_op.out_split:
            op = (
                comm_op.comms,
                comm_op.pg_id,
                comm_op.in_msg_size,
                comm_op.out_msg_size,
                comm_op.in_split,
                comm_op.out_split,
            )
        else:
            op = (
                comm_op.comms,
                comm_op.pg_id,
                comm_op.in_msg_size,
                comm_op.out_msg_size,
            )

        return hash(op)

    def run_comms(
        self, coll_name: str, curr_comm: CommOpArgs, curr_block_stack: str
    ) -> Tuple[float, float]:
        """
        Replays collective communication operation and records metrics for benchmarking.

        Args:
            coll_name: Name of the collective that is going to be replayed.
            curr_comm: Object containing information on the current collective.
            curr_block_stack: String containing the marker_stack(s) that this collective is a part of.

        Returns:
            Tuple[float, float]: Latency and global latency, which are the timings of how long
                                 the replay or posting (if non-blocking) of the collective took.
        """
        coll_timer = ParamTimer()

        if self.is_blocking:
            with ParamProfile(
                description=f"# PARAM replay {self.replay_iter} pre-comm barrier # "
                + curr_block_stack
            ):
                self.backend.sync_barrier(self.collective_args)

        # replay the collective
        with ParamProfile(
            timer=coll_timer,
            description=f"# PARAM replay {self.replay_iter}:" + curr_block_stack,
        ):
            if coll_name in self.backend.collective_func.keys():
                # handle point-to-point separately
                if coll_name in SupportedP2pOps:
                    self.collective_args.src_rank = curr_comm.src_rank
                    self.collective_args.dst_rank = curr_comm.dst_rank
                    if curr_comm.batch_p2p:
                        self.collective_args.collective = coll_name
                        self.backend.P2POp(self.collective_args, ret_flag=True)

                ret_obj = self.backend.collective_func[coll_name](
                    self.collective_args, ret_flag=True
                )
            else:
                logger.warn(
                    f"Unsupported collective name: {coll_name}. Skipping replaying the collective"
                )

            # if blocking, post outstanding ops and wait for them to complete
            if self.is_blocking:
                self.backend.complete_accel_ops(self.collective_args)

        # For non-blocking, latency and global_latency are the same
        global_latency = latency = coll_timer.getTimeUS()

        if self.is_blocking:
            with ParamProfile(
                description=f"# PARAM replay {self.replay_iter} post-comm barrier # "
                + curr_block_stack
            ) as bt:
                self.backend.sync_barrier(self.collective_args)

            # We sync the global_latency for blocking
            global_latency = latency + (bt.intervalNS / 1e3)  # Convert ns to us

        return (latency, global_latency)

    def dcheck(self, comm_trace_replay_args, cur_size, tensor):
        """
        Data validation check for collectives. Will raise an exception if invalid.

        Args:
            comm_trace_replay_args: Contains collective information.
            cur_size: Current size in bytes.
            tensor: Tensor to validate.

        Returns:
            None
        """
        exp_res = self.init_val
        if (
            comm_trace_replay_args.collective
            in ("all_reduce", "reduce_scatter", "reduce_scatter_base")
        ) or (
            self.backend.get_global_rank() == comm_trace_replay_args.src_or_dst
            and comm_trace_replay_args.collective == "reduce"
        ):
            exp_res = (
                self.init_val
                if tensor.dtype == torch.bool
                else self.collective_args.world_size * self.init_val
            )

        if (
            comm_trace_replay_args.collective in ("incast", "reduce", "gather")
            and self.backend.get_global_rank() != comm_trace_replay_args.src_or_dst
        ):
            return

        if isinstance(tensor, list):
            for rank, t in enumerate(tensor):
                if not torch.all(torch.eq(t, exp_res)):
                    for index, val in enumerate(t):
                        if val != exp_res:
                            raise ValueError(
                                f"[{cur_size}-bytes {comm_trace_replay_args.collective}] Wrong value at [{rank}][{index}] = {t[index]}, expected {exp_res}\n {tensor}"
                            )
        else:
            if not torch.all(torch.eq(tensor, exp_res)):
                for index, val in enumerate(tensor):
                    if val != exp_res:
                        raise ValueError(
                            f"[{cur_size}-bytes {comm_trace_replay_args.collective}] Wrong value at [{index}] = {tensor[index]}, expected {exp_res}\n {tensor}"
                        )
