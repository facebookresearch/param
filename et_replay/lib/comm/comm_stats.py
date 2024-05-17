from typing import Dict, List, Set


class CommStats:
    """
    Class to handle statistics collection for communication operations during
    the replay of traces for collective communications.
    """

    def __init__(self):
        self.num_msg: int = 0
        self.max_msg_cnt: int = 0
        self.coll_in_msg_bytes: Dict[str, List[int]] = {}
        self.coll_out_msg_bytes: Dict[str, List[int]] = {}
        self.coll_lat: Dict[str, List[float]] = {}
        self.coll_in_uni_msg_bytes: Dict[str, Set[int]] = {}
        self.coll_out_uni_msg_bytes: Dict[str, Set[int]] = {}
        self.total_trace_latency: float = 0.0
        self.comms_blocks: Dict[str, int] = {}

    def record_communication(
        self, coll_name: str, in_msg_size: int, out_msg_size: int, dtype_size: int
    ) -> None:
        """
        Record statistics for a single communication.

        Args:
            coll_name: Name of the collective operation.
            in_msg_size: Input message size in elements.
            out_msg_size: Output message size in elements.
            dtype_size: Size of the data type per element.
        """
        if coll_name not in self.coll_in_msg_bytes:
            self.coll_in_msg_bytes[coll_name] = []
            self.coll_out_msg_bytes[coll_name] = []
            self.coll_lat[coll_name] = []
            self.coll_in_uni_msg_bytes[coll_name] = set()
            self.coll_out_uni_msg_bytes[coll_name] = set()

        self.coll_in_msg_bytes[coll_name].append(in_msg_size * dtype_size)
        self.coll_out_msg_bytes[coll_name].append(out_msg_size * dtype_size)
        self.coll_in_uni_msg_bytes[coll_name].add(in_msg_size * dtype_size)
        self.coll_out_uni_msg_bytes[coll_name].add(out_msg_size * dtype_size)

    def update_total_latency(self, latency: float) -> None:
        """
        Update the total latency for all operations.

        Args:
            latency: Latency to add to the total in microseconds.
        """
        self.total_trace_latency += latency

    def get_stats_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Generates a summary of collected statistics.

        Returns:
            A dictionary summarizing the statistics for each collective operation.
        """
        summary: Dict[str, Dict[str, float]] = {}
        for coll_name, sizes in self.coll_in_msg_bytes.items():
            summary[coll_name] = {
                "total_in_bytes": sum(sizes),
                "total_out_bytes": sum(self.coll_out_msg_bytes[coll_name]),
                "average_latency_us": (
                    sum(self.coll_lat[coll_name]) / len(self.coll_lat[coll_name])
                    if self.coll_lat[coll_name]
                    else 0
                ),
                "max_latency_us": (
                    max(self.coll_lat[coll_name]) if self.coll_lat[coll_name] else 0
                ),
                "min_latency_us": (
                    min(self.coll_lat[coll_name]) if self.coll_lat[coll_name] else 0
                ),
            }
        return summary

    def print_detailed_stats(self) -> None:
        """
        Prints detailed statistics for debugging and analysis purposes.
        """
        print("\nDetailed Communication Statistics:")
        for coll_name, stats in self.get_stats_summary().items():
            print(f"\n{coll_name}:")
            print(f"  Total input bytes: {stats['total_in_bytes']}")
            print(f"  Total output bytes: {stats['total_out_bytes']}")
            print(f"  Average latency: {stats['average_latency_us']} us")
            print(f"  Max latency: {stats['max_latency_us']} us")
            print(f"  Min latency: {stats['min_latency_us']} us")

    def reset(self) -> None:
        """
        Resets all collected statistics.
        """
        self.num_msg = 0
        self.max_msg_cnt = 0
        self.coll_in_msg_bytes.clear()
        self.coll_out_msg_bytes.clear()
        self.coll_lat.clear()
        self.coll_in_uni_msg_bytes.clear()
        self.coll_out_uni_msg_bytes.clear()
        self.total_trace_latency = 0.0
        self.comms_blocks.clear()

    def update_message_count(self, trace_length: int) -> None:
        """
        Update the total and maximum message count based on the trace length.

        Args:
            trace_length: The total number of messages in the trace.
        """
        self.num_msg = trace_length
        self.max_msg_cnt = trace_length if self.max_msg_cnt == 0 else self.max_msg_cnt
