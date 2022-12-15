import logging


def get_tmp_trace_filename():
    import datetime
    import os
    import uuid

    trace_fn = (
        "tmp_"
        + datetime.datetime.today().strftime("%Y%m%d")
        + "_"
        + uuid.uuid4().hex[:7]
        + "_"
        + str(os.getpid())
        + ".json"
    )
    return trace_fn


def trace_handler(prof):
    fn = get_tmp_trace_filename()
    prof.export_chrome_trace("/tmp/" + fn)
    logging.warning(f"Chrome profile trace written to /tmp/{fn}")
    # try:
    #     from param_bench.train.compute.python.tools.internals import upload_trace

    #     upload_trace(fn)
    # except ImportError:
    #     logging.info("FB internals not present")
    # except Exception as e:
    #     logging.info(f"Upload trace error: {e}")
    #     pass
