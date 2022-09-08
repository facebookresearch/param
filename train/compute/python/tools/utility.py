def get_tmp_trace_filename():
    import datetime
    import uuid

    trace_fn = (
        "tmp_"
        + datetime.datetime.today().strftime("%Y%m%d")
        + "_"
        + uuid.uuid4().hex[:7]
        + ".json"
    )
    return trace_fn


def trace_handler(prof):
    fn = get_tmp_trace_filename()
    prof.export_chrome_trace("/tmp/" + fn)
    print(f"Chrome profile trace written to /tmp/{fn}")
    upload_trace(fn)


def generate_query_url(start_time, end_time, cuda_id):
    return


def upload_trace(fn):
    return
