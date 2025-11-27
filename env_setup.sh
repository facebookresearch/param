#!/bin/bash

set -x

# Sets up environment variables for the conda env to be functional
# env activation should be executed explicitly.

# Detect architecture and set appropriate platform path
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
  # aarch64 is used for hosts that have GB200s which run on ARM64 architecture
  FBCODE_PLATFORM="/usr/local/fbcode/platform010-aarch64/lib"
  # For GB200s we need to use the following order:
  # 1. Try to use the CONDA lib dir.
  # 2. Fallback to the systems lib.
  # 3. Use the fbcode built libs if nothing else works out.
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/lib64:${FBCODE_PLATFORM}"
else
  # x86_64 is used by default
  FBCODE_PLATFORM="/usr/local/fbcode/platform010/lib"
fi

LIBCUDA="${FBCODE_PLATFORM}/libcuda.so"
export LD_PRELOAD="${PRELOAD_PATH:+:$PRELOAD_PATH}"

# Adding non-existent library to LD_PRELOAD results in a lot of spam
if [[ -f "${LIBCUDA}" ]]; then
  export LD_PRELOAD="$LIBCUDA:${FBCODE_PLATFORM}/libnvidia-ml.so${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# Set library path with conda directory if available
if [[ -n "${CONDA_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Set PYTHONPATH if WORKSPACE_DIR is available
if [[ -n "${WORKSPACE_DIR}" ]]; then
  export PYTHONPATH="$WORKSPACE_DIR${PYTHONPATH:+:$PYTHONPATH}${TORCHX_RUN_PYTHONPATH:+:$TORCHX_RUN_PYTHONPATH}"
fi

# There's a bug in triton which prevents libcuda assertion from passing even if it is in the path
# https://github.com/openai/triton/issues/2507
# Workaround by setting TRITON_LIBCUDA_PATH
export TRITON_LIBCUDA_PATH=${FBCODE_PLATFORM}
