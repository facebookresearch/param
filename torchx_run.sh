#!/usr/bin/bash
# Entry point for torchx scripts

set -eE

# shellcheck disable=SC1091

export LD_PRELOAD="${PRELOAD_PATH:=/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_DIR}/lib"
export PYTHONPATH="${PYTHONPATH}:${TORCHX_RUN_PYTHONPATH}"

# shellcheck disable=SC1091
source "${CONDA_DIR}/bin/activate"
cd "${WORKSPACE_DIR}"
python3 -X faulthandler "$@"
