# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from enum import Enum
from typing import abstractmethod, Dict, Optional

from param_bench.train.comms.pt.pytorch_backend_utils import backendFunctions

logger = logging.getLogger(__name__)


class benchType(Enum):
    Collective = 0
    Pt2Pt = 1
    QuantCollective = 2


@dataclass
class commsPerfMetrics:
    """
    Base Class for storing performance metrics for communication op.
    """

    commsOp: str = None
    Datatype: str = None
    BenchCommsType: int = None
    Backend: str = None
    Tags: str = ""
    InputSize: float = 0.0
    OutputSize: float = 0.0
    NumElements: int = 0
    NumElements_pair: int = 0


@dataclass
class commsQuantCollPerfMetrics(commsPerfMetrics):
    """
    Class for storing performance metrics for a collective with quentization enabled.
    """

    p95_latency_us: float = 0.0
    quant_p95_latency_us: float = 0.0
    dequant_p95_latency_us: float = 0.0
    quant_comms_p95_latency_us: float = 0.0
    TFLOPs: Optional[float] = 0.0

    def __post_init__(self):
        self.BenchCommsType = benchType.QuantCollective


@dataclass
class commsCollPerfMetrics(commsPerfMetrics):
    """
    Class for storing performance metrics for a collective.
    """

    p50_latency_us: float = 0.0
    p75_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0
    AlgoBW_GBs: float = 0.0
    BusBW_GBs: float = 0.0
    TFLOPs: Optional[float] = 0.0

    def __post_init__(self):
        self.BenchCommsType = benchType.Collective


@dataclass
class commsPt2PtPerfMetrics(commsPerfMetrics):
    """
    Class for storing performance metrics for a point-to-point.
    """

    p50_latency_us: float = 0.0
    p75_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    AvgUniBW_GBs: float = 0.0
    AvgBiBW_GBs: float = 0.0
    TotalUniBW_GBs: float = 0.0
    TotalBiBW_GBs: float = 0.0

    def __post_init__(self):
        self.BenchCommsType = benchType.Pt2Pt


class commsPerfLogger:
    """
    Helper class for logging performance metrics.
    """

    def __init__(self, loggerName: str):
        self.name = loggerName

    @abstractmethod
    def logPerf(
        self,
        benchmarkName: str,
        metrics: commsPerfMetrics,
        backendFuncs: backendFunctions,
        **kwargs,
    ):
        """
        Log performance metrics for the collective.
        Args:
            benchmarkName: Name of benchmark, e.g., "comms" or "replay".
            metrics: Performance metrics for this collective.
            backendFuncs: Backend function/object used in this benchmark.
        Returns:
            None
        """
        pass


customized_perf_loggers: Dict[str, commsPerfLogger] = {}


def register_perf_logger(
    name: str,
    func: commsPerfLogger,
) -> None:
    global customized_perf_loggers
    customized_perf_loggers[name] = func
    logger.info(f"Registered custom perf logger {name}")
