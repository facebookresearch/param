# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any

from torch.autograd.profiler import record_function

from .param_timer import ParamTimer

logger = logging.getLogger(__name__)


class ParamProfile(record_function):
    """Inherit from PyTorch profiler to enable autoguard profiling while measuring the time interval in PARAM"""

    def __init__(self, timer: ParamTimer = None, description: str = "") -> None:
        self.description = description
        self.timer = timer
        super().__init__(name=description)

    def __enter__(self) -> "ParamProfile":
        super().__enter__()
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.end = time.monotonic()
        self.intervalNS = (self.end - self.start) * 1e9  # keeping time in NS
        # if given a valid ParamTimer object, directly update the measured time interval
        if isinstance(self.timer, ParamTimer):
            self.timer.incrTimeNS(self.intervalNS)
        logger.debug(f"{self.description} took {self.intervalNS} ns")
        super().__exit__(exc_type, exc_value, traceback)
