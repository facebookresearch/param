import ast
import json
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict

import numpy as np
from intervaltree import Interval, IntervalTree

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# refer to:
# https://github.com/pytorch/pytorch/blob/2cc01cc6d3ad2aff47e8460667ba654b2e4c9f21/c10/core/ScalarType.h#L61
_dtype_size_map: Dict[str, int] = {
    "Byte": 1,
    "Char": 1,
    "Short": 2,
    "Int": 4,
    "Long": 8,
    "Half": 2,
    "Float": 4,
    "Double": 8,
    "ComplexHalf": 4,
    "ComplexFloat": 8,
    "ComplexDouble": 16,
    "Bool": 1,
    "QInt8": 1,
    "QUInt8": 1,
    "QInt32": 4,
    "BFloat16": 2,
    "QUInt4x2": 1,
    "QUInt2x4": 1,
    "Bits1x8": 1,
    "Bits2x4": 1,
    "Bits4x2": 1,
    "Bits8": 1,
    "Bits16": 2,
    "Float8_e5m2": 1,
    "Float8_e4m3fn": 1,
    "Float8_e5m2fnuz": 1,
    "Float8_e4m3fnuz": 1,
    "UInt16": 2,
    "UInt32": 4,
    "UInt64": 8,
}

# refer to: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
_busbw_correction_factors_tbl: Dict[str, Callable[[int], float]] = {
    "all_reduce": (lambda n: 2 * (n - 1) / n),
    "all_gather": (lambda n: (n - 1) / n),
    "all_to_all": (lambda n: (n - 1) / n),
    "reduce_scatter": (lambda n: (n - 1) / n),
    "reduce": (lambda n: 1),
    "scatter": (lambda n: (n - 1) / n),
    "gather": (lambda n: (n - 1) / n),
    "broadcast": (lambda n: 1),
    "send": (lambda n: 1),
    "recv": (lambda n: 1),
}

# map collective name of event to key string for bw calculation
_collname_to_busbw_corr_factor_func: Dict[str, Callable[[int], float]] = {
    "allreduce": _busbw_correction_factors_tbl["all_reduce"],
    "all_gather": _busbw_correction_factors_tbl["all_gather"],
    "_allgather_base": _busbw_correction_factors_tbl["all_gather"],
    "reduce_scatter": _busbw_correction_factors_tbl["reduce_scatter"],
    "_reduce_scatter_base": _busbw_correction_factors_tbl["reduce_scatter"],
    "all_to_all": _busbw_correction_factors_tbl["all_to_all"],
    "all_to_allv": _busbw_correction_factors_tbl["all_to_all"],
    "broadcast": _busbw_correction_factors_tbl["broadcast"],
    "reduce": _busbw_correction_factors_tbl["reduce"],
    "gather": _busbw_correction_factors_tbl["gather"],
    "scatter": _busbw_correction_factors_tbl["scatter"],
    "send": _busbw_correction_factors_tbl["send"],
    "recv": _busbw_correction_factors_tbl["recv"],
}


def _get_dict_value(d, k, err_msg):
    if k not in d:
        raise ValueError(err_msg)
    return d.get(k)


def _calculate_event_data_size(evt):
    return (
        max(evt["args"]["In msg nelems"], evt["args"]["Out msg nelems"])
        * _dtype_size_map[evt["args"]["dtype"]]
    )


def _calculate_algbw(evt: Dict[str, Any]) -> float:
    duration_us = _get_dict_value(evt, "dur", f'Missing "dur" in event: {evt}')
    total_bytes = _calculate_event_data_size(evt)

    # NCCL tests use 1024^3 to convert B to GB (but not 1e9)
    # https://github.com/NVIDIA/nccl-tests/blob/8dfeab9eb9bdfdf13503e71e1f33e7f8a208b540/src/common.cu#L102
    # but it uses 1e9 to convert bw from B/s to GB/s
    # https://github.com/NVIDIA/nccl-tests/blob/8dfeab9eb9bdfdf13503e71e1f33e7f8a208b540/src/all_gather.cu#L41
    return round((total_bytes / duration_us) / 1e3, 2)


def _get_event_busbw_factor(evt):
    coll_name = _get_dict_value(
        evt["args"], "Collective name", f'Missing "Collective name" in event: {evt}'
    )

    # barrier is implemented using AllReduce
    if coll_name in [
        "barrier",
    ]:
        return 0

    group_size = _get_dict_value(
        evt["args"], "Group size", f'Missing "Group size" in event: {evt}'
    )
    correction_factor_func = _get_dict_value(
        _collname_to_busbw_corr_factor_func,
        coll_name,
        f"Unsupported collective op for busbw calculation: {coll_name}",
    )

    return correction_factor_func(group_size)


def calculate_bw_(trace_data):
    nccl_events = [
        i
        for i in trace_data["traceEvents"]
        if i.get("cat", "") == "kernel" and i["name"].startswith("ncclDevKernel_")
    ]
    for evt in nccl_events:
        try:
            coll_name = _get_dict_value(
                evt["args"],
                "Collective name",
                f'Missing "Collective name" in event: {evt}',
            )

            # barrier is implemented using AllReduce
            if coll_name in [
                "barrier",
            ]:
                continue

            algbw = _calculate_algbw(evt)
            busbw_factor = _get_event_busbw_factor(evt)
            busbw = round(algbw * busbw_factor, 2)

            evt["args"]["algbw (GB/sec)"] = algbw
            evt["args"]["busbw (GB/sec)"] = busbw
            evt["args"]["busbw_factor"] = busbw_factor
        except ValueError as e:
            logger.error("Error processing event: %s", e)


def calculate_sbw(trace_data):
    # calculate shared bw per rank
    nccl_events = [
        i
        for i in trace_data["traceEvents"]
        if i.get("cat", "") == "kernel"
        and i["name"].startswith("ncclDevKernel_")
        and "busbw_factor" in i["args"]
    ]

    if not len(nccl_events):
        return 0

    total_data_size = sum(
        [
            _calculate_event_data_size(evt) * _get_event_busbw_factor(evt)
            for evt in nccl_events
        ]
    )

    time_range_tree = IntervalTree(
        [Interval(evt["ts"], evt["ts"] + evt["dur"]) for evt in nccl_events]
    )
    time_range_tree.merge_overlaps()

    begin_time_point = min([i.begin for i in time_range_tree])
    end_time_point = max([i.end for i in time_range_tree])

    sorted_tr = sorted(time_range_tree)
    total_idle_time = (
        sum(
            [
                sorted_tr[i + 1].begin - sorted_tr[i].end
                for i in range(len(sorted_tr) - 1)
            ]
        )
        if len(sorted_tr) > 1
        else 0
    )

    return total_data_size / (end_time_point - begin_time_point - total_idle_time) / 1e3


def pick_iter_e2e_time_(trace_data, tl):
    tl.extend(
        [
            evt["dur"]
            for evt in trace_data["traceEvents"]
            if evt.get("cat", "") == "user_annotation"
            and evt["name"].startswith("ProfilerStep#")
        ]
    )


def pick_comm_bw_(trace_data, comm_bw_data):
    rank = trace_data["distributedInfo"]["rank"]
    nccl_events = [
        i
        for i in trace_data["traceEvents"]
        if i.get("cat", "") == "kernel"
        and i["name"].startswith("ncclDevKernel_")
        and "algbw (GB/sec)" in i["args"]
    ]
    for evt in nccl_events:
        knl_name = evt["name"][: evt["name"].index("(")]
        data_size = _calculate_event_data_size(evt)
        ranks_count = evt["args"]["Group size"]

        ranks = ast.literal_eval(evt["args"]["Process Group Ranks"])
        pg_id = int(evt["args"]["Process Group Name"])
        pg = tuple([*ranks, pg_id]) if rank == min(ranks) else None

        comm_bw_data[(knl_name, data_size, ranks_count)].append(
            [
                evt["dur"],
                evt["args"]["algbw (GB/sec)"],
                evt["args"]["busbw (GB/sec)"],
                pg,
            ]
        )


def analyze_profiler_trace(trace_dir: str, report_dir: str):
    """
    Analyse input PyTorch profiler trace (i.e. Kineto trace) and generate report.

    Args:
        trace_dir (str): dir path of input traces, where trace name should be in "rank-n.json" format.
        report_dir (str): dir path for generated reports
    """
    logger.info(
        f'Parse profiler trace from "{trace_dir}" and generate reports to "{report_dir}"'
    )

    processed_trace_dir = os.path.join(report_dir, "profiler_trace_processed")
    pathlib.Path(processed_trace_dir).mkdir(parents=True, exist_ok=True)

    # list of iteration time in all ranks
    iter_e2e_time = []

    # list of shared bw
    sbw_lst = []

    # key is (kernel_name, data size, ranks number)
    # value is list of [dur, algbw, busbw, pg]
    comm_bw_data = defaultdict(list)

    for fpath in os.scandir(trace_dir):
        if not fpath.is_file():
            continue

        with open(fpath.path, "r", encoding="utf-8") as f:
            trace = json.load(f)

        calculate_bw_(trace)
        with open(
            os.path.join(processed_trace_dir, fpath.name), "w", encoding="utf-8"
        ) as f:
            json.dump(trace, f)

        sbw_lst.append(calculate_sbw(trace))

        pick_iter_e2e_time_(trace, iter_e2e_time)
        pick_comm_bw_(trace, comm_bw_data)

    comm_bw_summary = {}
    for k, v in comm_bw_data.items():
        t_lst = [i[0] for i in v]
        busbw_lst = [i[2] for i in v]
        pg_set = set([i[3] for i in v if i[3]])
        comm_bw_summary[k] = [
            len(pg_set),
            np.average(t_lst),
            np.average(busbw_lst),
            np.percentile(busbw_lst, 1),
            np.percentile(busbw_lst, 50),
            np.percentile(busbw_lst, 90),
            np.percentile(busbw_lst, 99),
        ]
    comm_bw_summary = dict(sorted(comm_bw_summary.items()))

    # dump summary report
    with open(
        os.path.join(report_dir, "profiler_trace_summary_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            f"avg. E2ETime of iters among all ranks: {sum(iter_e2e_time) / len(iter_e2e_time) / 1e3 :.3f} ms\n"
        )
        f.write(
            f"avg. SharedBW (i.e. sum(data_size * busbw_factor) / GPU_comm_busy_time  per rank) among all ranks: {sum(sbw_lst) / len(sbw_lst) :.3f} GB/s\n"
        )

        f.write(
            f'\n{" ":>70s}|{" ":>5s}|{"AVG.":^19s}|{"p01":^8s}|{"p50":^8s}|{"p90":^8s}|{"p99":^8s}|\n'
        )

        f.write(
            f'{"kernel":>50s} {"size":>12s} {"#rks":>6s}|{"#pgs":>5s}|{"  dur":>10s} '
        )
        for i in range(5):  # average, p01, p50, p90, p99
            f.write(f'{" busbw":>8s}|')
        f.write("\n")

        f.write(
            f'{"      ":>50s} {" (B)":>12s} {"    ":>6s}|{"    ":>5s}|{" (ms)":>10s} '
        )
        for i in range(5):  # average, p50, p90, p99
            f.write(f'{"(GB/s)":>8s}|')
        f.write("\n")

        for k, v in comm_bw_summary.items():
            f.write(f"{k[0]:>50s} {k[1]:>12d} {k[2]:>6d}|{v[0]:>5d}|{v[1]/1e3:>10.3f} ")
            for i in range(2, len(v)):
                f.write(f"{v[i]:>8.2f}|")
            f.write("\n")
