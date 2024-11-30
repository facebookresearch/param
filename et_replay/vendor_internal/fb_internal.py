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


def fbStartProfiler(rank: int, device: str, numWarmupIters: int, numIters: int) -> bool:
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


def fbSampleProfiler(stop: bool = False) -> None:
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