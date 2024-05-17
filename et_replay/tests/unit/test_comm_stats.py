from et_replay.comm.comm_stats import CommStats


def test_initial_state():
    comm_stats = CommStats()
    assert comm_stats.num_msg == 0
    assert comm_stats.max_msg_cnt == 0
    assert comm_stats.coll_in_msg_bytes == {}
    assert comm_stats.coll_out_msg_bytes == {}
    assert comm_stats.coll_lat == {}
    assert comm_stats.coll_in_uni_msg_bytes == {}
    assert comm_stats.coll_out_uni_msg_bytes == {}
    assert comm_stats.total_trace_latency == 0.0
    assert comm_stats.comms_blocks == {}


def test_record_communication():
    comm_stats = CommStats()
    comm_stats.record_communication("allreduce", 100, 200, 4)

    assert "allreduce" in comm_stats.coll_in_msg_bytes
    assert comm_stats.coll_in_msg_bytes["allreduce"] == [400]
    assert comm_stats.coll_out_msg_bytes["allreduce"] == [800]
    assert comm_stats.coll_in_uni_msg_bytes["allreduce"] == {400}
    assert comm_stats.coll_out_uni_msg_bytes["allreduce"] == {800}


def test_update_total_latency():
    comm_stats = CommStats()
    comm_stats.update_total_latency(100.0)
    comm_stats.update_total_latency(50.0)

    assert comm_stats.total_trace_latency == 150.0


def test_get_stats_summary():
    comm_stats = CommStats()
    comm_stats.record_communication("allreduce", 100, 200, 4)
    comm_stats.update_total_latency(100.0)
    comm_stats.coll_lat["allreduce"].append(50.0)
    comm_stats.coll_lat["allreduce"].append(150.0)

    summary = comm_stats.get_stats_summary()
    assert summary["allreduce"]["total_in_bytes"] == 400
    assert summary["allreduce"]["total_out_bytes"] == 800
    assert summary["allreduce"]["average_latency_us"] == 100.0
    assert summary["allreduce"]["max_latency_us"] == 150.0
    assert summary["allreduce"]["min_latency_us"] == 50.0


def test_reset():
    comm_stats = CommStats()
    comm_stats.record_communication("allreduce", 100, 200, 4)
    comm_stats.update_total_latency(100.0)
    comm_stats.reset()

    assert comm_stats.num_msg == 0
    assert comm_stats.max_msg_cnt == 0
    assert comm_stats.coll_in_msg_bytes == {}
    assert comm_stats.coll_out_msg_bytes == {}
    assert comm_stats.coll_lat == {}
    assert comm_stats.coll_in_uni_msg_bytes == {}
    assert comm_stats.coll_out_uni_msg_bytes == {}
    assert comm_stats.total_trace_latency == 0.0
    assert comm_stats.comms_blocks == {}


def test_update_message_count():
    comm_stats = CommStats()
    comm_stats.update_message_count(5)

    assert comm_stats.num_msg == 5
    assert comm_stats.max_msg_cnt == 5

    comm_stats.update_message_count(10)

    assert comm_stats.num_msg == 10
    assert comm_stats.max_msg_cnt == 5
