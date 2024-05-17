from argparse import Namespace


class CommTraceReplayArgs:
    """
    Class holding arguments for communication trace replay parameters.

    Attributes:
        network_stack: Network stack to be used.
        dtype: Data type for communication.
        backend: Backend to be used.
        device: Device on which to run the communication.
        blocking_flag: Flag to indicate if blocking communication is used.
        dcheck: Flag for debug checks.
        group_ranks: Dictionary recording ranks each process group will work on.
        use_ext_dist: Flag to indicate if external distribution is used.
        size_from_trace: Flag to determine if size should be inferred from trace.
        init_method: Initialization method to be used.
        enable_local_report: Flag to enable local report generation.
        enable_profiler: Flag to enable profiling.
        use_perf_logger: Flag to enable performance logging.
        ibv_devices: List of Infiniband devices.
        init_only: Flag to indicate if only initialization is required.
    """

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the CommTraceReplayArgs with provided arguments.

        Args:
            args: Argument parser object with necessary parameters.
        """
        self.network_stack: str = args.network_stack
        self.dtype: str = args.dtype
        self.backend: str = args.backend
        self.device: str = args.device
        self.blocking_flag: bool = args.z
        self.dcheck: bool = args.c
        self.group_ranks: dict = {}
        self.use_ext_dist: bool = args.use_ext_dist
        self.size_from_trace: bool = False
        self.init_method: str = args.init_method
        self.enable_local_report: bool = args.enable_local_report
        self.enable_profiler: bool = args.enable_profiler
        self.use_perf_logger: bool = args.use_perf_logger
        self.ibv_devices: list = args.ibv_devices
        self.init_only: bool = args.init_only
