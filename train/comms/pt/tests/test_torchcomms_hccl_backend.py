# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock, patch

import torch


class TestPyTorchMtiaTorchcommsBackendRegistration(unittest.TestCase):
    """Test that the HCCL torchcomms backend is registered correctly."""

    def test_hccl_torchcomms_is_correct_class(self):
        from param_bench.train.comms.pt.fb.pytorch_mtia_torchcomms_backend import (
            PyTorchMtiaTorchcommsBackend,
        )
        from param_bench.train.comms.pt.pytorch_backend_utils import customized_backend

        self.assertIs(
            customized_backend["hccl_torchcomms"], PyTorchMtiaTorchcommsBackend
        )

    def test_mtia_in_supported_devices(self):
        import param_bench.train.comms.pt.fb.pytorch_mtia_torchcomms_backend  # noqa: F401 trigger registration
        from param_bench.train.comms.pt.pytorch_backend_utils import supportedDevices

        self.assertIn("mtia", supportedDevices)


class TestPyTorchMtiaTorchcommsBackendTest(unittest.TestCase):
    """Tests for PyTorchMtiaTorchcommsBackend: device, streams, memory."""

    @staticmethod
    def make_backend(device="mtia", backend="hccl"):
        """Create a backend instance with mocked MTIA/HCCL initialization."""
        from param_bench.train.comms.pt.fb.pytorch_mtia_torchcomms_backend import (
            PyTorchMtiaTorchcommsBackend,
        )

        bootstrap_info = MagicMock()
        bootstrap_info.local_rank = 0
        bootstrap_info.global_rank = 0
        bootstrap_info.world_size = 2
        bootstrap_info.local_size = 2

        commsParams = MagicMock()
        commsParams.device = device
        commsParams.backend = backend
        commsParams.init_device = False
        commsParams.__contains__ = MagicMock(return_value=False)

        with (
            patch("torch.mtia.init"),
            patch(
                "param_bench.train.comms.pt.fb.pytorch_mtia_torchcomms_backend.PyTorchTorchcommsBackend.__init__",
                return_value=None,
            ),
        ):
            obj = PyTorchMtiaTorchcommsBackend.__new__(PyTorchMtiaTorchcommsBackend)
            obj.bootstrap_info = bootstrap_info
            obj.commsParams = commsParams
            obj.torchcomm = None
            obj.groupRanks = {}
            obj.groups = {}
            obj.tcp_store = None
            obj.reduce_op_map = {}
        return obj

    # ── Device selection ──

    @patch("torch.mtia.device_count", return_value=4)
    def test_get_device_mtia(self, mock_count):
        backend = self.make_backend(device="mtia")
        backend.bootstrap_info.local_rank = 2
        dev = backend.get_device()
        self.assertEqual(dev, torch.device("mtia:2"))

    @patch("torch.mtia.device_count", return_value=4)
    def test_get_device_mtia_ordinal_wraps(self, mock_count):
        backend = self.make_backend(device="mtia")
        backend.bootstrap_info.local_rank = 6
        dev = backend.get_device()
        self.assertEqual(dev, torch.device("mtia:2"))

    @patch("torch.mtia.device_count", return_value=4)
    def test_get_device_mtia_fallback_rank_negative(self, mock_count):
        backend = self.make_backend(device="mtia")
        backend.bootstrap_info.local_rank = -1
        dev = backend.get_device()
        self.assertEqual(dev, torch.device("mtia:0"))

    @patch("torch.mtia.device_count", return_value=4)
    def test_get_device_via_torchcomm(self, mock_count):
        backend = self.make_backend(device="mtia")
        mock_torchcomm = MagicMock()
        mock_torchcomm.get_device.return_value = torch.device("mtia:1")
        backend.torchcomm = mock_torchcomm
        dev = backend.get_device()
        self.assertEqual(dev, torch.device("mtia:1"))
        mock_torchcomm.get_device.assert_called_once()

    def test_init_rejects_non_mtia_device(self):
        from param_bench.train.comms.pt.fb.pytorch_mtia_torchcomms_backend import (
            PyTorchMtiaTorchcommsBackend,
        )

        bootstrap_info = MagicMock()
        commsParams = MagicMock()
        commsParams.device = "cuda"

        def fake_parent_init(self_inner, bi, cp):
            self_inner.bootstrap_info = bi
            self_inner.commsParams = cp

        with patch(
            "param_bench.train.comms.pt.fb.pytorch_mtia_torchcomms_backend.PyTorchTorchcommsBackend.__init__",
            fake_parent_init,
        ):
            with self.assertRaises(ValueError):
                PyTorchMtiaTorchcommsBackend(bootstrap_info, commsParams)

    @patch("torch.mtia.synchronize")
    @patch("torch.mtia.device_count", return_value=4)
    def test_device_sync_mtia(self, mock_count, mock_sync):
        backend = self.make_backend(device="mtia")
        backend.device_sync(MagicMock())
        mock_sync.assert_called_once_with(torch.device("mtia:0"))

    @patch("torch.mtia.empty_cache")
    @patch("torch.mtia.device_count", return_value=4)
    def test_clear_memory_mtia(self, mock_count, mock_cache):
        backend = self.make_backend(device="mtia")
        collectiveArgs = MagicMock()
        collectiveArgs.ipTensor_pair = [MagicMock()]
        collectiveArgs.opTensor_pair = [MagicMock()]
        backend.clear_memory(collectiveArgs)
        mock_cache.assert_called_once()

    # ── Streams ──

    @patch("torch.mtia.device_count", return_value=4)
    @patch("torch.mtia.Stream")
    def test_get_new_stream_mtia(self, mock_stream_cls, mock_count):
        backend = self.make_backend(device="mtia")
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream
        result = backend.get_new_stream()
        mock_stream_cls.assert_called_once_with(
            device=torch.device("mtia:0"), priority=0
        )
        self.assertEqual(result, mock_stream)

    @patch("torch.mtia.device_count", return_value=4)
    @patch("torch.mtia.current_stream")
    def test_sync_stream_mtia(self, mock_cur, mock_count):
        backend = self.make_backend(device="mtia")
        mock_stream = MagicMock()
        mock_cur.return_value = mock_stream
        device = torch.device("mtia:0")
        backend.sync_stream(device=device)
        mock_cur.assert_called_once_with(device=device)
        mock_stream.synchronize.assert_called_once()

    @patch("torch.mtia.current_stream")
    def test_get_current_stream(self, mock_cur):
        backend = self.make_backend(device="mtia")
        mock_stream = MagicMock()
        mock_cur.return_value = mock_stream
        device = torch.device("mtia:0")
        result = backend.get_current_stream(device)
        mock_cur.assert_called_once_with(device)
        self.assertEqual(result, mock_stream)

    @patch("torch.mtia.set_stream")
    @patch("torch.mtia.current_stream")
    @patch("torch.mtia.device_count", return_value=4)
    def test_switch_stream_mtia(self, mock_count, mock_cur, mock_set):
        backend = self.make_backend(device="mtia")
        old_stream = MagicMock()
        new_stream = MagicMock()
        mock_cur.return_value = old_stream
        device = torch.device("mtia:0")
        result = backend.switch_stream(new_stream, device)
        mock_set.assert_called_once_with(new_stream)
        self.assertEqual(result, old_stream)

    @patch("torch.mtia.device_count", return_value=4)
    def test_switch_stream_none_returns_none(self, mock_count):
        backend = self.make_backend(device="mtia")
        device = torch.device("mtia:0")
        result = backend.switch_stream(None, device)
        self.assertIsNone(result)

    @patch("torch.mtia.Event")
    @patch("torch.mtia.device_count", return_value=4)
    def test_get_new_event(self, mock_count, mock_event_cls):
        backend = self.make_backend(device="mtia")
        mock_event = MagicMock()
        mock_event_cls.return_value = mock_event
        result = backend.get_new_event(enable_timing=True)
        mock_event_cls.assert_called_once_with(
            torch.device("mtia:0"), enable_timing=True
        )
        self.assertEqual(result, mock_event)

    # ── Memory allocation ──

    def test_alloc_random_cpu_path(self):
        """alloc_random allocates on CPU."""
        backend = self.make_backend(device="cpu")
        tensor = backend.alloc_random([4, 4], curRankDevice="cpu")
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertEqual(tensor.shape, torch.Size([4, 4]))

    def test_alloc_random_int_dtype(self):
        backend = self.make_backend(device="cpu")
        tensor = backend.alloc_random([4], curRankDevice="cpu", dtype=torch.int32)
        self.assertEqual(tensor.dtype, torch.int32)

    def test_alloc_random_bool_dtype(self):
        backend = self.make_backend(device="cpu")
        tensor = backend.alloc_random([4], curRankDevice="cpu", dtype=torch.bool)
        self.assertEqual(tensor.dtype, torch.bool)

    def test_alloc_empty_cpu(self):
        backend = self.make_backend(device="cpu")
        tensor = backend.alloc_empty([2, 3], "cpu", torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertEqual(tensor.shape, torch.Size([2, 3]))


if __name__ == "__main__":
    unittest.main()
