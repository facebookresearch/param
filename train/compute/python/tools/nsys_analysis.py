import argparse
import json
import logging
import os
import sqlite3
from typing import List, Dict, Tuple, Any

from ..lib.init_helper import init_logging, get_logger

logger = get_logger()

def print_rows(rows):
    for row in rows:
        print(row)


class OperatorEvent:
    def __init__(self, name, id):
        self.event_data = {
            "op_name": name,
            "id": id,
            "ranges": {},
            "analysis": {"T1": [], "T2": [], "T3": [], "T4": []},
        }

    def add_op_event(self, range_id, name, start, end):
        if range_id not in self.event_data["ranges"]:
            self.event_data["ranges"][range_id] = {
                "name": name,
                "start": start,
                "end": end,
                "cuda_events": [],
            }

    def add_cuda_event(
        self,
        range_id,
        correlation_id,
        kernel_name,
        kernel_start,
        kernel_end,
        runtime_name,
        runtime_start,
        runtime_end,
    ):
        self.event_data["ranges"][range_id]["cuda_events"].append(
            (
                correlation_id,
                {
                    "name": kernel_name,
                    "start": kernel_start,
                    "end": kernel_end,
                },
                {
                    "name": runtime_name,
                    "start": runtime_start,
                    "end": runtime_end,
                },
            )
        )

    def __str__(self):
        return json.dumps(self.event_data)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return self.event_data


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if "to_json" in dir(o):
            return o.to_json()
        return json.JSONEncoder.default(self, o)


queries: Dict[str, str] = {
    "nvtx": """
    SELECT NVTX_EVENTS.rangeId,
        NVTX_EVENTS.text,
        NVTX_EVENTS.start,
        NVTX_EVENTS.end,
        CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
        KERNEL_NAME.value,
        CUPTI_ACTIVITY_KIND_KERNEL.start,
        CUPTI_ACTIVITY_KIND_KERNEL.end,
        RUNTIME_NAME.value,
        CUPTI_ACTIVITY_KIND_RUNTIME.start,
        CUPTI_ACTIVITY_KIND_RUNTIME.end
        FROM NVTX_EVENTS
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON NVTX_EVENTS.eventType == 60 AND
            NVTX_EVENTS.globalTid == CUPTI_ACTIVITY_KIND_RUNTIME.globalTid AND
            NVTX_EVENTS.start <= CUPTI_ACTIVITY_KIND_RUNTIME.start AND
            NVTX_EVENTS.end >= CUPTI_ACTIVITY_KIND_RUNTIME.end
        JOIN CUPTI_ACTIVITY_KIND_KERNEL ON CUPTI_ACTIVITY_KIND_KERNEL.correlationId == CUPTI_ACTIVITY_KIND_RUNTIME.correlationId
        JOIN StringIds AS KERNEL_NAME ON KERNEL_NAME.id = CUPTI_ACTIVITY_KIND_KERNEL.demangledName
        JOIN StringIds AS RUNTIME_NAME ON RUNTIME_NAME.id = CUPTI_ACTIVITY_KIND_RUNTIME.nameId
        WHERE
            CUPTI_ACTIVITY_KIND_KERNEL.correlationId == CUPTI_ACTIVITY_KIND_RUNTIME.correlationId AND
            NVTX_EVENTS.text <> "param_bench:measure" AND NVTX_EVENTS.text <> "param_bench:warmup" AND NVTX_EVENTS.text NOT LIKE "%:warmup:%"
        ORDER BY CUPTI_ACTIVITY_KIND_KERNEL.start
    """,
}


def build_events(event_info: List[Tuple[Any]]):
    op_events: Dict[str, Dict[str, OperatorEvent]] = {}

    for event in event_info:
        range_id = event[0]
        nvtx_label = event[1]
        op_info = nvtx_label.split(":")
        op_name = op_info[0]
        stage = op_info[1]
        id = f"{op_info[2]}:{op_info[3]}:{op_info[4]}"
        pass_name = op_info[5]
        logger.debug(
            f"op_name: {op_name}, range_id: {range_id}, stage: {stage}, id: {id}, pass: {pass_name}"
        )
        logger.debug(f"  start: {event[2]}, end: {event[3]}, dur: {event[3]-event[2]}")
        correlation_id = event[4]
        logger.debug(f"    correlation_id: {correlation_id}")
        kernel_name = event[5]
        logger.debug(f"    kernel: {kernel_name}")
        logger.debug(
            f"      start: {event[6]}, end: {event[7]}, dur: {event[7]-event[6]}"
        )
        runtime_name = event[8]
        logger.debug(f"    runtime: {runtime_name}")
        logger.debug(
            f"      start: {event[9]}, end: {event[10]}, dur: {event[10]-event[9]}"
        )
        op_events.setdefault(op_name, {})
        op_events[op_name].setdefault(id, OperatorEvent(op_name, id))
        op_events[op_name][id].add_op_event(
            range_id, pass_name, event[2], event[3]
        )
        op_events[op_name][id].add_cuda_event(
            range_id,
            correlation_id,
            kernel_name,
            event[6],
            event[7],
            runtime_name,
            event[9],
            event[10],
        )

    return op_events


def analyze_events(op_events):
    for name, run_info in op_events.items():
        for id, op_event in run_info.items():
            T1 = op_event.event_data["analysis"]["T1"]  # total time
            T2 = op_event.event_data["analysis"]["T2"]  # op_start to first kernel_start
            T3 = op_event.event_data["analysis"]["T3"]  # total of kernel latencies
            T4 = op_event.event_data["analysis"]["T4"]  # last kernel_end to op_end
            for range_id, data in op_event.event_data["ranges"].items():
                op_start = data["start"]
                op_end = data["end"]
                T1.append(op_end - op_start)
                assert len(data["cuda_events"]) > 0
                (_, first_kernel_event, _) = data["cuda_events"][0]
                (_, last_kernel_event, _) = data["cuda_events"][-1]
                T2.append(first_kernel_event["start"] - op_start)
                T4.append(op_end - last_kernel_event["end"])
                total_kernel_time = 0
                for cuda_event in data["cuda_events"]:
                    (_, kernel_event, _) = cuda_event
                    total_kernel_time += kernel_event["end"] - kernel_event["start"]
                T3.append(total_kernel_time)


def main():

    parser = argparse.ArgumentParser(
        description="Microbenchmarks NSight System Analysis"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None, help="The nsys sqlite file."
    )
    parser.add_argument(
        "-l", "--log_level", default="INFO", help="Log level"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="output file name",
    )
    args = parser.parse_args()

    logger = init_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    con = sqlite3.connect(args.file)

    cur = con.cursor()
    rows = list(cur.execute(queries["nvtx"]))
    con.close()
    op_events = build_events(rows)
    analyze_events(op_events)
    if args.output:
        out_file_name = args.output
    else:
        out_file_name = os.path.splitext(args.file)[0] + ".json"
    with open(out_file_name, "w") as out_file:
        for op, run_info in op_events.items():
            for _, op_range_info in run_info.items():
                print(json.dumps(op_range_info, cls=CustomEncoder), file=out_file)
    logger.info(f"Output written to: {out_file_name}")


if __name__ == "__main__":
    main()
