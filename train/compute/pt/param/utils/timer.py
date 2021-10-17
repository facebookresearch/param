

import time
import torch

# Timer in seconds
class Timer:
    def __init__(self, device: str):
        self.device: str = device
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.device == "cpu":
            self.start_time = time.perf_counter()
        else:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            self.start_time = 0
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cpu":
            self.end_time = time.perf_counter()
        else:
            self.end_event.record()
            torch.cuda.synchronize()
            self.end_time = self.start_event.elapsed_time(self.end_event) * 1.0e-3

    # returns time in seconds
    def elapsed_time(self):
        return self.end_time - self.start_time