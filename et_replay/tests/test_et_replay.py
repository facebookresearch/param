# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tarfile
import tempfile
import unittest

import torch
from et_replay.tools.et_replay import ExgrReplayManager


CURR_DIR = os.path.dirname(os.path.realpath(__file__))

ET_TEST_FILES = [
    "resnet_1gpu_et.json.tar.gz",
    "hf_GPT2_et.json.tar.gz",
]

# pt2_et.json.tar.gz requires SM_80 architecture
SM_80_TEST_FILES = [
    "pt2_et.json.tar.gz",
]


def is_sm_80_available() -> bool:
    """Check if CUDA device has SM_80 (Ampere) architecture or higher."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 8


def get_et_files(include_sm_80: bool = False):
    full_path_et_test_files = []
    inputs_dir = os.path.join(CURR_DIR, "inputs")
    test_files = ET_TEST_FILES.copy()
    if include_sm_80:
        test_files.extend(SM_80_TEST_FILES)
    for f in test_files:
        tar_file = os.path.join(inputs_dir, f)
        tmp_dir = tempfile.mkdtemp()
        with tarfile.open(tar_file) as tar_ref:
            tar_ref.extractall(tmp_dir)
        full_path = os.path.join(tmp_dir, f.replace(".tar.gz", ""))
        full_path_et_test_files.append(full_path)
    return full_path_et_test_files


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class ETReplayIntegrationTest(unittest.TestCase):
    def test_et_replay_integration_test(self) -> None:
        full_path_et_test_files = get_et_files(include_sm_80=is_sm_80_available())
        for f in full_path_et_test_files:
            print(f"\n\nRunning {f}")
            replay_manager = ExgrReplayManager()
            replay_manager.readComputeArgs(check_args=False)
            replay_manager.args.input = f
            replay_manager.args.replay_mode = "comp"
            replay_manager.initBench()
            result = replay_manager.benchTime()
            del replay_manager
            if result["execution finished"] is False:
                self.fail(f"Failed to run {f}")


if __name__ == "__main__":
    unittest.main()
