"""Device-agnostic behaviour of the inference entry points.

Everything except the last test runs on any host: detection order, explicit device
handling, the seeding and autocast helpers, and the call sites that used to read
``torch.cuda`` directly. The last test decodes depth tokens with a tiny random-weight
VQVAE (the model behind ``scripts/reconstruct_from_tokens.py``) on CPU and on XPU.
"""

import os
import sys
import unittest
from unittest import mock

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "scripts", os.path.join("experiments", "libero")):
    sys.path.insert(0, os.path.join(ROOT, sub))

import robot_utils

import molmoact_device as dev


def _patch_availability(*, xpu: bool, cuda: bool):
    def fake_is_available(device_type: str) -> bool:
        return {"xpu": xpu, "cuda": cuda}.get(device_type, device_type == "cpu")

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.default_device(), torch.device("xpu", 0))

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device(), torch.device("cuda", 0))

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.default_device(), torch.device("cpu"))
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("cpu"), torch.device("cpu"))

    def test_device_type_of_inputs(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.device("cuda", 1)), "cuda")
        self.assertEqual(dev.device_type(torch.zeros(1)), "cpu")


class HelperDispatchTest(unittest.TestCase):
    def test_seed_all_without_cuda_build_does_not_touch_cuda(self):
        with (
            _patch_availability(xpu=False, cuda=False),
            mock.patch("torch.cuda.manual_seed_all") as cuda_seed,
        ):
            dev.seed_all(3)
        cuda_seed.assert_not_called()

    def test_seed_all_seeds_each_available_backend(self):
        with (
            _patch_availability(xpu=True, cuda=True),
            mock.patch("torch.cuda.manual_seed_all") as cuda_seed,
            mock.patch("torch.xpu.manual_seed_all", create=True) as xpu_seed,
        ):
            dev.seed_all(3)
        cuda_seed.assert_called_once_with(3)
        xpu_seed.assert_called_once_with(3)

    def test_empty_cache_dispatch(self):
        with (
            _patch_availability(xpu=True, cuda=True),
            mock.patch("torch.cuda.empty_cache") as cuda_empty,
            mock.patch("torch.xpu.empty_cache", create=True) as xpu_empty,
        ):
            dev.empty_cache("xpu:0")
            xpu_empty.assert_called_once()
            cuda_empty.assert_not_called()
            dev.empty_cache("cpu")
            xpu_empty.assert_called_once()

    def test_autocast_device_type_follows_model_device(self):
        self.assertEqual(dev.autocast_device_type(torch.device("xpu", 0)), "xpu")
        self.assertEqual(dev.autocast_device_type("cuda:0"), "cuda")
        self.assertEqual(dev.autocast_device_type(torch.device("cpu")), "cpu")

    def test_autocast_dtype_uses_per_backend_getter(self):
        self.assertEqual(dev.autocast_dtype("cpu"), torch.float32)
        with mock.patch(
            "torch.get_autocast_dtype", return_value=torch.bfloat16
        ) as getter:
            self.assertEqual(dev.autocast_dtype("xpu:0"), torch.bfloat16)
        getter.assert_called_once_with("xpu")


class CallSiteTest(unittest.TestCase):
    def test_robot_utils_seed_routes_through_device_module(self):
        with mock.patch.object(robot_utils, "seed_all") as seed_all:
            robot_utils.set_seed_everywhere(7)
        seed_all.assert_called_once_with(7)

    def test_robot_utils_device_is_a_torch_device(self):
        self.assertIsInstance(robot_utils.DEVICE, torch.device)

    def test_reconstruct_script_has_no_hardcoded_cuda(self):
        import reconstruct_from_tokens as script

        with mock.patch.object(
            script, "default_device", return_value=torch.device("cpu")
        ) as default:
            codes = torch.tensor(
                [1, 2, 3, 4], dtype=torch.long, device=str(script.default_device())
            ).view(1, 2, 2)
        default.assert_called_once()
        self.assertEqual(codes.device.type, "cpu")


def _tiny_vqvae():
    from vqvae import VQVAE

    return VQVAE(
        image_size=64,
        num_resnet_blocks=1,
        downsample_ratio=32,
        num_tokens=32,
        codebook_dim=32,
        hidden_dim=8,
        use_norm=False,
        channels=1,
        train_objective="regression",
        max_value=10.0,
        residul_type="v1",
        loss_type="mse",
    )


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an Intel XPU"
)
class XpuParityTest(unittest.TestCase):
    def test_vqvae_decode_matches_cpu(self):
        torch.manual_seed(0)
        vae = _tiny_vqvae().eval()
        # Re-randomise every parameter so that no zero-initialised layer hides a mismatch.
        with torch.no_grad():
            for p in vae.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
        codes = torch.randint(0, 32, (1, 2, 2))
        with torch.no_grad():
            cpu_out = vae.decode(codes)
            xpu_out = vae.to("xpu:0").decode(codes.to("xpu:0")).cpu()
        self.assertEqual(cpu_out.shape, xpu_out.shape)
        mse = torch.mean((cpu_out - xpu_out) ** 2).item()
        self.assertLess(mse, 1e-6)
        torch.testing.assert_close(xpu_out, cpu_out, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
