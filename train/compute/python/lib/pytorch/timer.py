import time

import torch

# Timer in seconds
class Timer:
    def __init__(self, device: str):
        self.device: str = device
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()

    # Return result in milliseconds.
    def elapsed_time(self) -> float:
        return (self.end_time - self.start_time) * 1000.0
