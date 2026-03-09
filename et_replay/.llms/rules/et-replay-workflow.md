---
description: ET Replay project workflow - build, test, and replay Chakra execution traces
oncalls:
  - cea_perf
  - hardware_foundation
---

# ET Replay Workflow

## Project Overview

ET Replay (Execution Trace Replay) replays Chakra Execution Traces from ML models for performance analysis.

## Quick Commands

```bash
# Build
buck2 build //param_bench/et_replay:et_replay

# Test
buck2 test //param_bench/et_replay:test_execution_trace

# Replay a trace
buck2 run //param_bench/et_replay:et_replay -- --input <trace.json> -c --profile-replay

# Lint
arc lint -a
```

## Key Files

| File | Purpose |
|------|---------|
| `tools/et_replay.py` | Main entry point for trace replay |
| `tools/comm_replay.py` | Communication collective replay |
| `execution_trace.py` | Trace parsing and data structures |
| `comm/backend/base_backend.py` | Abstract backend interface |

## Code Standards

- **Python 3.11+** with type hints required
- **Line length**: 88 characters
- **Copyright header**: Apache 2.0 (see existing files)
- **Logging**: Use `logging` module, not print statements

## Vendor Internal Pattern

For Meta-internal features, use try/except pattern:

```python
try:
    from et_replay.vendor_internal.fb_internal import internal_feature
    HAS_INTERNAL = True
except ImportError:
    HAS_INTERNAL = False

def my_function():
    if HAS_INTERNAL:
        internal_feature()
```

## Adding New Backends

1. Create new file in `comm/backend/`
2. Inherit from `BaseBackend` in `base_backend.py`
3. Implement required abstract methods
4. Register in BUCK dependencies

## Common Issues

- **Trace not loading**: Check if gzipped (`gzip -d trace.json.gz`)
- **Missing deps**: Run `buck2 build` first to fetch dependencies
- **Kineto traces**: Only generated with `--profile-replay` flag
