try:
    from param_bench.train.comms.pt.fb.internals import (
        fbInitProfiler,
        fbSampleProfiler,
        fbStartProfiler,
    )

    has_fb_internal_libs = True
except ImportError:
    has_fb_internal_libs = False

import logging
logger = logging.getLogger(__name__)

class MetaProfiler():
    def __init__(self, rank, device, profiler_num_replays_start, profiler_num_replays, num_replays):
        self.rank = rank
        self.device = device
        # num of iterations to skip
        self.num_warmup_iters = self.profiler_num_replays_start
        # num of iterations to profile, at most num_replays iterations
        self.num_profile_iters = (
            profiler_num_replays
            if profiler_num_replays_start + profiler_num_replays <= num_replays
            else num_replays - profiler_num_replays_start
        )

    def __enter__(self):
        MetaProfiler.start_profiler(
            rank=self.rank,
            device=self.device,
            numWarmupIters=self.num_warmup_iters,
            numIters=self.num_profile_iters
        )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        MetaProfiler.sample_profiler(stop=True)

    def step(self):
        MetaProfiler.sample_profiler()

    @classmethod
    def start_profiler(cls, rank: int, device: str, numWarmupIters: int, numIters: int) -> bool:
        """
        Starts internal profiler with given parameters.

        Args:
            rank: Global rank.
            device: Type of device "cuda", "cpu", etc.
            numWarmupIters: Number of warmup iterations.
            numIters: Number of real iterations.
        Returns:
            bool: Returns if internal profile was able to start or not.
        """
        if not has_fb_internal_libs:
            raise RuntimeError('FB internal libs are not supported')

        fbInitProfiler(
            rank=rank,
            device=device,
            warmup=numWarmupIters,
            iters=numIters,
        )
        fbStartProfiler()
        return True

    @classmethod
    def sample_profiler(cls, stop: bool = False) -> None:
        """
        Starts internal sample profiler.

        Args:
            stop: Bool to be passed into sample profiler.
        Returns:
            None
        """
        if not has_fb_internal_libs:
            raise RuntimeError('FB internal libs are not supported')

        fbSampleProfiler(stop)