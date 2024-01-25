# Using Execution Trace in PARAM Benchmark

This section includes how to collect Chakra Execution Trace from a PyTorch training workload, as well as how to run PARAM replay on top of the collected ET.


## Execution Trace Collection
Execution Trace collection logic has to be added in the main training loop. This includes three steps:

### Step 1: Set up Execution Trace Observer
The first step is to create a Execution Trace Observer object and register a. temporary file for ET store.

```
from torch.profiler import ExecutionTraceObserver

et_ob = ExecutionTraceObserver()
fp = tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False)
fp.close()
et_ob.register_callback(fp.name)
```

### Step 2: Define your function to dump Execution Trace
You have to define a function to store/dump/upload your collected ET trace for further use. Here is an example:

```
def dump_execution_trace(tmp_et_path):
    et_dir.mkdir(exist_ok=True, parents=True)
    et_path = DUMP_DIR / f"rank-{global_rank}.et.json.gz"
    with open(tmp_et_path) as fin:
        with gzip.open(et_path, "wt") as fout:
            fout.writelines(fin)
    os.remove(tmp_et_path)
    print(f"Finished Rank {global_rank} ET collection at {et_path}")
```

### Step 3: Collect Execution Trace in the training loop
This is the key step to collect ET. You have to insert the collection logic into the main training loop of your workload.
TWO parameters have to be set:
- ET_START_ITER: the iteration to start ET collection
- ET_END_ITER: the iteration to stop ET collection

```
<START of training loop>
while step < TRAINING_STEPS:
    ...
    ...
    # Collect Execution Trace Logic

    # Start ET collection
    if et_ob and step == ET_START_ITER:
        et_ob.start()

        # First record process group(PG) mapping
        pg_config_info = (
            torch.distributed.distributed_c10d._world.pg_config_info
        )
        rf_handle = torch.autograd._record_function_with_args_enter(
            "## process_group:init ##", json.dumps(pg_config_info)
        )
        torch.autograd._record_function_with_args_exit(rf_handle)

    # Stop ET collection
    elif et_ob and state.step == ET_END_ITER:
        et_ob.stop()
        tmp_et_path = et_ob.get_output_file_path()
        et_ob.unregister_callback()
        dump_execution_trace(tmp_et_path)

    ...
    ...
    step += 1
<END of training loop>
```

Note that process group information collection is not automatically covered by ET observer, because process_group initialization happens before the main training loop. Therefore, you have to manually add pg information collection, as the code shown above.




## PARAM Comms Replay on Execution Trace
Execution Trace now is fully supported in PARAM benchmark. In order to replay an ET trace, just need to specify `--trace-type=et` and the benchmark will parse your ET and replay the collective communication operators.

An example command:

```
/bin/mpirun -np 8 commsTraceReplay.par --trace-path <ET-PATH> --trace-type et
```
