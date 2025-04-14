# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from torch.autograd.profiler import record_function

logger = logging.getLogger(__name__)


class paramProfile(record_function):
    """Inherit from PyTorch profiler to enable autoguard profiling while measuring the time interval in PARAM"""

    def __init__(self, timer: paramTimer | None = None, description: str = "") -> None:
        super().__init__(name=description)
        self.description = description
        self.timer = timer
        self.start = 0.0
        self.end = 0.0
        self.intervalNS = 0.0

    def __enter__(self) -> paramProfile:
        super().__enter__()
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.end = time.monotonic()
        self.intervalNS = (self.end - self.start) * 1e9  # keeping time in NS
        # if given a valid paramTimer object, directly update the measured time interval
        if isinstance(self.timer, paramTimer):
            self.timer.incrTimeNS(self.intervalNS)
        logger.debug(f"{self.description} took {self.intervalNS} ns")
        super().__exit__(exc_type, exc_value, traceback)


@dataclass
class paramTimer:
    """
    Timer for param profiler.
    """

    elapsedTimeNS: float = 0.0  # keeping time in NS

    def reset(self, newTime: float = 0.0) -> None:
        self.elapsedTimeNS = newTime

    def incrTimeNS(self, timeNS: float) -> None:
        self.elapsedTimeNS += timeNS

    def getTimeUS(self) -> float:
        return self.elapsedTimeNS / 1e3

    def getTimeNS(self) -> float:
        return self.elapsedTimeNS
