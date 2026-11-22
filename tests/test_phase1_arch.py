r"""Phase 1: architecture tests. Run: .venv\Scripts\python.exe -I tests\test_phase1_arch.py"""
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ecg_concept.inference import load_model  # noqa: E402
from ecg_concept.lightecgnet_paper import ArchConfig, DualHeadLightECGNet, LightECGNet, count_params, v2_state_to_dual  # noqa: E402
from ecg_concept.lightecgnetv2 import LightECGNetV2  # noqa: E402

CKPT = os.path.join(ROOT, "models", "pretrained", "lightv2_ft_fold0.pt")


class T_(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(3, 12, 5000, generator=g)

    def test_default_config_equals_v2_parameter_count_and_names(self):
        a, b = LightECGNet(45), LightECGNetV2(45)
        self.assertEqual(count_params(a), 376108)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))

    def test_default_loads_reference_checkpoint_and_is_bit_identical(self):
        ref = load_model(CKPT, "cpu")
        new = LightECGNet(45)
        new.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=True)
        new.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(ref(self.x), new(self.x)))
            f1, f2 = ref.forward_features(self.x), new.forward_features(self.x)
            for k in ("feature_map", "pooled", "hidden", "logits"):
                self.assertTrue(torch.equal(f1[k], f2[k]), k)

    def test_shapes(self):
        m = LightECGNet(45).eval()
        with torch.inference_mode():
            f = m.forward_features(self.x)
        self.assertEqual(tuple(f["feature_map"].shape), (3, 256, 625))
        self.assertEqual(tuple(f["pooled"].shape), (3, 512))
        self.assertEqual(tuple(f["logits"].shape), (3, 45))

    def test_paper_text_flags_change_exactly_what_they_claim(self):
        base = count_params(LightECGNet(45))
        nb = LightECGNet(45, ArchConfig(la_bias=False))
        self.assertEqual(sum(p.numel() for p in nb.lead_attn.parameters()), 144)                 # the paper's 144
        self.assertEqual(base - count_params(nb), 18)
        self.assertEqual(count_params(LightECGNet(45, ArchConfig(stem_proj_bn=False))), base - 128)
        s = LightECGNet(45, ArchConfig(pool_fusion="sum")).eval()
        with torch.inference_mode():
            self.assertEqual(tuple(s.forward_features(self.x)["pooled"].shape), (3, 256))
        self.assertEqual(count_params(LightECGNet(45, ArchConfig(head_dropout=0.5))), base)      # dropout has no parameters

    def test_two_layer_block_variants_run_and_are_larger(self):
        base = count_params(LightECGNet(45))
        ds = LightECGNet(45, ArchConfig(block_variant="two_ds", post_add_relu=False)).eval()
        self.assertGreater(count_params(ds), base)
        with torch.inference_mode():
            self.assertEqual(tuple(ds(self.x).shape), (3, 45))
        std = LightECGNet(45, ArchConfig(block_variant="two_conv", kernels=(3,) * 6, post_add_relu=False)).eval()
        with torch.inference_mode():
            self.assertTrue(torch.isfinite(std(self.x)).all())
        with self.assertRaises(ValueError):
            LightECGNet(45, ArchConfig(block_variant="v2", post_add_relu=False))

    def test_dual_head_shapes_params_and_isolation(self):
        d = DualHeadLightECGNet().eval()
        self.assertEqual(count_params(d.backbone), 376108 - 142893)
        self.assertEqual(count_params(d.heads["physionet"]), 142893)
        self.assertEqual(count_params(d.heads["ptbxl"]), 131328 + 257 * 5)
        with torch.inference_mode():
            out = d.forward_all(self.x)
            self.assertEqual((tuple(out["physionet"].shape), tuple(out["ptbxl"].shape)), ((3, 45), (3, 5)))
            self.assertTrue(torch.equal(d(self.x, "ptbxl"), out["ptbxl"]))
        d.train()
        d(self.x, "ptbxl").sum().backward()                                                       # the unused head receives no gradient
        self.assertTrue(all(p.grad is None for p in d.heads["physionet"].parameters()))
        self.assertTrue(any(p.grad is not None for p in d.backbone.parameters()))

    def test_v2_checkpoint_maps_onto_dual_head_bit_identically(self):
        state = torch.load(CKPT, map_location="cpu", weights_only=True)
        d = DualHeadLightECGNet()
        missing = d.load_state_dict(v2_state_to_dual(state), strict=False)
        self.assertEqual([k for k in missing.unexpected_keys], [])
        self.assertTrue(all(k.startswith("heads.ptbxl") for k in missing.missing_keys))
        d.eval()
        ref = load_model(CKPT, "cpu")
        with torch.inference_mode():
            self.assertTrue(torch.equal(ref(self.x), d(self.x, "physionet")))


if __name__ == "__main__":
    unittest.main(verbosity=1)
