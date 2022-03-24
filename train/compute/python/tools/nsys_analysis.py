import argparse
import json
import sqlite3
from typing import Dict

def print_rows(rows):
    for row in rows:
        print(row)


class OperatorEvent:
    def __init__(self, name, id, start, end):
        self.event_data = {
            "op_name": name,
            "id": id,
            "start": start,
            "end": end,
            "ranges": {},
            "analysis": {"T1": [], "T2": [], "T3": [], "T4": []}
        }

    def add_cuda_event(self, range_id, correlation_id, event_type, name, start, end):
        assert event_type in {"kernel", "runtime"}
        self.event_data["ranges"].setdefault(range_id, {}).setdefault(correlation_id, {}).setdefault(event_type, None)
        self.event_data["ranges"][range_id][correlation_id][event_type]= {
                "name": name,
                "start": start,
                "end": end,
            }
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
"kernels":
    """
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

def main():

    parser = argparse.ArgumentParser(description="Microbenchmarks")
    parser.add_argument(
        "-f", "--file", type=str, default=None, help="The nsys sqlite file."
    )
    parser.add_argument(
        "-o",
        "--output_prefix",
        type=str,
        default=None,
        help="output file prefix",
    )

    args = parser.parse_args()

    con = sqlite3.connect(args.file)

    cur = con.cursor()
    rows = list(cur.execute(queries["kernels"]))
    # print_rows(rows)

    op_events: Dict[str, Dict[str, OperatorEvent]] = {}

    for row in rows:
        range_id = row[0]
        nvtx_label = row[1]
        op_info = nvtx_label.split(":")
        op_name = op_info[0]
        stage = op_info[1]
        id = f"{op_info[2]}:{op_info[3]}:{op_info[4]}"
        pass_name = op_info[5]
        print(f"op_name: {op_name}, range_id: {range_id}, stage: {stage}, id: {id}, pass: {pass_name}")
        print(f"  start: {row[2]}, end: {row[3]}, dur: {row[3]-row[2]}")
        correlation_id = row[4]
        print(f"    correlation_id: {correlation_id}")
        kernel_name = row[5]
        print(f"    kernel: {kernel_name}")
        print(f"      start: {row[6]}, end: {row[7]}, dur: {row[7]-row[6]}")
        runtime_name = row[8]
        print(f"    runtime: {runtime_name}")
        print(f"      start: {row[9]}, end: {row[10]}, dur: {row[10]-row[9]}")
        op_events.setdefault(op_name, {})
        op_events[op_name].setdefault(id, OperatorEvent(op_name, id, row[2], row[3]))
        op_events[op_name][id].add_cuda_event(range_id, correlation_id, "kernel", kernel_name, row[6], row[7])
        op_events[op_name][id].add_cuda_event(range_id, correlation_id, "runtime", runtime_name, row[9], row[10])

    print(json.dumps(op_events, indent=2, cls=CustomEncoder))





    # rows = cur.execute("select * from where text='measure'")
    con.close()


if __name__ == "__main__":
    main()
