#!/usr/bin/bash
# Entry point for torchx scripts

set -eE

# shellcheck disable=SC1091

# Source environment setup script to handle architecture-specific library paths
# This script detects whether we're on aarch64 (GB200) or x86_64 and sets
# LD_PRELOAD, LD_LIBRARY_PATH, and PYTHONPATH accordingly
if [[ -f "${WORKSPACE_DIR}/param_bench/env_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE_DIR}/param_bench/env_setup.sh"
else
  # Fallback to x86 paths if env_setup.sh is not found
  export LD_PRELOAD="${PRELOAD_PATH:=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_DIR}/lib"
  export PYTHONPATH="${PYTHONPATH}:${TORCHX_RUN_PYTHONPATH}"
fi

# shellcheck disable=SC1091
source "${CONDA_DIR}/bin/activate"
cd "${WORKSPACE_DIR}"
python3 -X faulthandler "$@"
