import argparse
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Tuple

from ..lib.init_helper import get_logger, init_logging

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
            "analysis": {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []},
        }

    def add_op_event(self, range_id, name, start, end):
        if range_id not in self.event_data["ranges"]:
            self.event_data["ranges"][range_id] = {
                "name": name,
                "start": start,
                "end": end,
                "cuda_kernel": [],
                "cuda_sync": [],
            }

    def add_cuda_kernel(
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
        self.event_data["ranges"][range_id]["cuda_kernel"].append(
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

    def add_cuda_sync(
        self,
        range_id,
        correlation_id,
        runtime_name,
        runtime_start,
        runtime_end,
    ):
        self.event_data["ranges"][range_id]["cuda_sync"].append(
            (
                correlation_id,
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


def find_overlap_intervals(r1, r2):
    overlaps = []
    i = 0
    j = 0
    n = len(r1)
    m = len(r2)
    while i < n and j < m:
        # find overlap
        left = max(r1[i][0], r2[j][0])
        right = min(r1[i][1], r2[j][1])
        if left < right:
            overlaps.append([left, right])

        # go to next interval
        if r1[i][1] < r2[j][1]:
            i += 1
        else:
            j += 1

    return overlaps


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if "to_json" in dir(o):
            return o.to_json()
        return json.JSONEncoder.default(self, o)


queries: Dict[str, str] = {
    "cuda_kernel": """
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
            NVTX_EVENTS.text <> "param_bench|measure" AND
            NVTX_EVENTS.text <> "param_bench|warmup" AND
            NVTX_EVENTS.text NOT LIKE "%|warmup|%"
        ORDER BY CUPTI_ACTIVITY_KIND_KERNEL.start
    """,
    "cuda_sync": """
    SELECT NVTX_EVENTS.rangeId,
       NVTX_EVENTS.text,
       NVTX_EVENTS.start,
       NVTX_EVENTS.end,
       CUPTI_ACTIVITY_KIND_RUNTIME.correlationId,
       RUNTIME_NAME.value,
       CUPTI_ACTIVITY_KIND_RUNTIME.start,
       CUPTI_ACTIVITY_KIND_RUNTIME.end
    FROM NVTX_EVENTS
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME ON
        NVTX_EVENTS.eventType == 60 AND
        NVTX_EVENTS.globalTid == CUPTI_ACTIVITY_KIND_RUNTIME.globalTid AND
        NVTX_EVENTS.start <= CUPTI_ACTIVITY_KIND_RUNTIME.start AND
        NVTX_EVENTS.end >= CUPTI_ACTIVITY_KIND_RUNTIME.end
        JOIN StringIds AS RUNTIME_NAME ON CUPTI_ACTIVITY_KIND_RUNTIME.nameId=RUNTIME_NAME.id
    WHERE
        NVTX_EVENTS.text <> "param_bench|measure" AND
        NVTX_EVENTS.text <> "param_bench|warmup" AND
        NVTX_EVENTS.text NOT LIKE "%|warmup|%" AND
        RUNTIME_NAME.value LIKE "%cudaDeviceSynchronize%"
    ORDER BY NVTX_EVENTS.start
    """,
}


def create_op_event_range(
    op_events: Dict[str, Dict[str, OperatorEvent]],
    op_name,
    id,
    range_id,
    pass_name,
    start_time,
    end_time,
):
    # check if op event is already created
    try:
        x = op_events[op_name][id].event_data["ranges"][range_id]
        logger.debug(f"found key {op_name} {id}")
    except KeyError:
        logger.debug(f"creating key {op_name} {id}")
        op_events.setdefault(op_name, {})
        op_events[op_name].setdefault(id, OperatorEvent(op_name, id))
        op_events[op_name][id].add_op_event(range_id, pass_name, start_time, end_time)
    return op_events[op_name][id]


def parse_kernel_events(
    event_info: List[Tuple[Any]], op_events: Dict[str, Dict[str, OperatorEvent]]
):
    for event in event_info:
        range_id = event[0]
        nvtx_label = event[1]
        op_info = nvtx_label.split("|")
        op_name = op_info[0]
        stage = op_info[1]
        id = f"{op_info[2]}|{op_info[3]}|{op_info[4]}"
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

        op_event = create_op_event_range(
            op_events, op_name, id, range_id, pass_name, event[2], event[3]
        )
        logger.debug(op_event)
        op_event.add_cuda_kernel(
            range_id,
            correlation_id,
            kernel_name,
            event[6],
            event[7],
            runtime_name,
            event[9],
            event[10],
        )

    return op_event


def parse_sync_events(
    event_info: List[Tuple[Any]], op_events: Dict[str, Dict[str, OperatorEvent]]
):
    for event in event_info:
        range_id = event[0]
        nvtx_label = event[1]
        op_info = nvtx_label.split("|")
        op_name = op_info[0]
        stage = op_info[1]
        id = f"{op_info[2]}|{op_info[3]}|{op_info[4]}"
        pass_name = op_info[5]
        logger.debug(
            f"op_name: {op_name}, range_id: {range_id}, stage: {stage}, id: {id}, pass: {pass_name}"
        )
        correlation_id = event[4]
        runtime_name = event[5]
        op_event = create_op_event_range(
            op_events, op_name, id, range_id, pass_name, event[2], event[3]
        )
        logger.debug(op_event)
        op_event.add_cuda_sync(
            range_id,
            correlation_id,
            runtime_name,
            event[6],
            event[7],
        )

    return op_event


def analyze_events(op_events):
    for _name, run_info in op_events.items():
        for id, op_event in run_info.items():
            T1 = op_event.event_data["analysis"]["T1"]  # total time
            T2 = op_event.event_data["analysis"]["T2"]  # op_start to first kernel_start
            T3 = op_event.event_data["analysis"]["T3"]  # total of kernel latencies
            T4 = op_event.event_data["analysis"]["T4"]  # last kernel_end to op_end
            T5 = op_event.event_data["analysis"]["T5"]  # last cudaDeviceSynchronize
            for _range_id, data in op_event.event_data["ranges"].items():
                op_start = data["start"]
                op_end = data["end"]
                T1.append(op_end - op_start)
                assert len(data["cuda_kernel"]) > 0
                assert len(data["cuda_sync"]) > 0
                (_, first_kernel_event, _) = data["cuda_kernel"][0]
                (_, last_kernel_event, _) = data["cuda_kernel"][-1]
                (_, last_sync_event) = data["cuda_sync"][-1]
                T2.append(first_kernel_event["start"] - op_start)
                T4.append(op_end - last_kernel_event["end"])
                T5.append(last_sync_event["end"] - last_sync_event["start"])
                total_kernel_time = 0
                for cuda_event in data["cuda_kernel"]:
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
    parser.add_argument("-l", "--log_level", default="INFO", help="Log level")
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
    cuda_kernel_rows = list(cur.execute(queries["cuda_kernel"]))
    cuda_sync_rows = list(cur.execute(queries["cuda_sync"]))
    con.close()

    op_events: Dict[str, Dict[str, OperatorEvent]] = {}
    parse_kernel_events(cuda_kernel_rows, op_events)
    parse_sync_events(cuda_sync_rows, op_events)
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
