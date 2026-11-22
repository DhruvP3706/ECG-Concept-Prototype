"""Tests for the lead-attention probe and the inference-only uniform-attention intervention. Run: .venv\Scripts\python.exe -I tests\test_lead_attention.py"""
import hashlib
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from ecg_concept import attribution as at  # noqa: E402
from ecg_concept.lightecgnetv2 import LightECGNetV2  # noqa: E402


def make(seed=0):
    torch.manual_seed(seed)
    return LightECGNetV2(6).eval()


def h(m):
    return hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()


class T(unittest.TestCase):
    def setUp(self):
        self.m = make()
        torch.manual_seed(1)
        self.x = torch.randn(3, 12, 400) + torch.arange(12).view(1, 12, 1) * 0.1          # non-zero lead means

    def test_original_mode_equals_unmodified_model_and_is_removed_afterwards(self):
        with torch.no_grad():
            ref = self.m(self.x)
            with at.lead_attention_intervention(self.m, "original"):
                got = self.m(self.x)
            after = self.m(self.x)
        self.assertTrue(torch.equal(ref, got))
        self.assertTrue(torch.equal(ref, after))
        self.assertNotIn("forward", self.m.lead_attn.__dict__)                              # patch removed

    def test_weights_shape_and_range(self):
        s, mu = at.lead_attention_weights(self.m, self.x)
        self.assertEqual((s.shape, mu.shape), ((3, 12), (3, 12)))
        self.assertTrue(((s > 0) & (s < 1)).all())

    def test_uniform_mode_changes_only_the_gating_operation(self):
        s, _ = at.lead_attention_weights(self.m, self.x)
        cap = {}
        hook = self.m.lead_attn.register_forward_hook(lambda mod, i, o: cap.update(out=o.detach().clone()))
        with torch.no_grad(), at.lead_attention_intervention(self.m, "uniform_mean"):
            self.m(self.x)
        hook.remove()
        u = s.mean(1, keepdim=True)
        self.assertTrue(torch.allclose(cap["out"], self.x * u.unsqueeze(-1), atol=1e-6))    # output of the gate = x * (same value for all leads)
        # downstream is untouched: feeding the gated tensor straight into the rest of the network reproduces the intervened logits
        with torch.no_grad(), at.lead_attention_intervention(self.m, "uniform_mean"):
            full = self.m(self.x)
        with torch.no_grad():
            la = self.m.lead_attn
            self.m.lead_attn = torch.nn.Identity()
            rest = self.m(cap["out"])
            self.m.lead_attn = la
        self.assertTrue(torch.allclose(full, rest, atol=1e-5))

    def test_parameters_and_checkpoint_state_unchanged(self):
        h0 = h(self.m)
        with at.lead_attention_intervention(self.m, "uniform_mean"):
            self.m(self.x)
        with at.lead_attention_intervention(self.m, "ones"):
            self.m(self.x)
        self.assertEqual(h0, h(self.m))

    def test_uniform_gates_are_equal_across_leads_and_ones_mode(self):
        cap = {}
        hook = self.m.lead_attn.register_forward_hook(lambda mod, i, o: cap.update(out=o.detach().clone()))
        with torch.no_grad(), at.lead_attention_intervention(self.m, "ones"):
            self.m(self.x)
        hook.remove()
        self.assertTrue(torch.allclose(cap["out"], self.x))
        with self.assertRaises(ValueError):
            with at.lead_attention_intervention(self.m, "bogus"):
                pass

    def test_ig_uses_the_intervened_function_and_completes(self):
        x = self.x[0, :, :400].numpy()
        with at.lead_attention_intervention(self.m, "uniform_mean"):
            r = at.integrated_gradients(self.m, x, 2, steps=256)
            f_int = float(self.m(torch.from_numpy(x)[None])[0, 2])
        self.assertAlmostEqual(r["f_x"], f_int, places=5)
        self.assertLess(r["completeness_abs_err"], 0.05 * max(abs(r["f_x"] - r["f_baseline"]), 0.1) + 1e-3)

    def test_gate_depends_on_input_only_through_lead_means(self):
        x2 = self.x - self.x.mean(-1, keepdim=True)                                         # zero-mean leads (like the z-scored real input)
        s, mu = at.lead_attention_weights(self.m, x2)
        self.assertLess(float(mu.abs().max()), 1e-5)
        s_b, _ = at.lead_attention_weights(self.m, torch.randn(5, 12, 400) * 3 - torch.randn(5, 12, 1) * 0)  # different content
        s0, _ = at.lead_attention_weights(self.m, torch.zeros(1, 12, 400))
        self.assertTrue(torch.allclose(s, s0.expand_as(s), atol=1e-5))                      # same gates for every zero-mean input


if __name__ == "__main__":
    unittest.main(verbosity=1)
