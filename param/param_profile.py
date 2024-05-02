# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from torch.autograd.profiler import record_function

logger = logging.getLogger(__name__)


@dataclass
class ParamTimer:
    """
    A simple timer class for profiling execution time in nanoseconds.

    Attributes:
        elapsed_time_ns (float): The elapsed time in nanoseconds.
    """

    elapsed_time_ns: float = 0.0

    def reset(self, new_time: float = 0.0) -> None:
        """
        Reset the timer to a new time or zero if no new time is provided.

        Args:
            new_time (float): The new starting time in nanoseconds. Default is 0.0.
        """
        self.elapsed_time_ns = new_time

    def incr_time_ns(self, time_ns: float) -> None:
        """
        Increment the timer by a specified amount of nanoseconds.

        Args:
            time_ns (float): The amount of time in nanoseconds to add.
        """
        self.elapsed_time_ns += time_ns

    def get_time_us(self) -> float:
        """
        Get the elapsed time in microseconds.

        Returns:
            float: The elapsed time in microseconds.
        """
        return self.elapsed_time_ns / 1e3

    def get_time_ns(self) -> float:
        """
        Get the elapsed time in nanoseconds.

        Returns:
            float: The elapsed time in nanoseconds.
        """
        return self.elapsed_time_ns


class ParamProfile(record_function):
    """
    A profiler class that extends the PyTorch record_function profiler with custom
    timing capabilities to measure execution intervals using a ParamTimer.

    Attributes:
        description (str): A description of the profiling context.
        timer (Optional[ParamTimer]): An instance of ParamTimer to log execution times.
    """

    def __init__(self, timer: Optional[ParamTimer] = None, description: str = "") -> None:
        """
        Initialize the ParamProfile.

        Args:
            timer (Optional[ParamTimer]): The ParamTimer instance to use for timing.
            description (str): A descriptive label for the profiling block.
        """
        self.description = description
        self.timer = timer
        super().__init__(name=description)

    def __enter__(self) -> ParamProfile:
        """
        Start the profiling interval.

        Returns:
            ParamProfile: A reference to the instance itself.
        """
        super().__enter__()
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        End the profiling interval and log the execution time.

        Args:
            exc_type: The type of the exception if an exception was raised.
            exc_value: The value of the exception if an exception was raised.
            traceback: The traceback if an exception was raised.
        """
        self.end = time.monotonic()
        self.interval_ns = (self.end - self.start) * 1e9  # Convert seconds to nanoseconds
        if isinstance(self.timer, ParamTimer):
            self.timer.incr_time_ns(self.interval_ns)
        logger.debug(f"{self.description} took {self.interval_ns} ns")
        super().__exit__(exc_type, exc_value, traceback)
