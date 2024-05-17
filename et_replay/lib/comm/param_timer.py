from dataclasses import dataclass


@dataclass
class ParamTimer:
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
