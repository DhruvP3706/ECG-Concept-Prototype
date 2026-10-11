"""Checkpoint loading and the recovered inference rule.

Rule (original `evaluate_model`): probs = sigmoid(logits); pred_c = probs_c > thresh_c
(strictly greater, per-class thresholds saved next to the checkpoint, 0.5 if none).
"""
import numpy as np
import torch

from .lightecgnetv2 import LightECGNetV2


def load_model(ckpt_path, device="cpu", num_classes=45):
    """Safe load (weights_only=True) with strict key matching; returns model in eval mode."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = LightECGNetV2(num_classes)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_thresholds(path):
    return np.load(path, allow_pickle=False)


@torch.inference_mode()
def predict_probs(model, x):
    """x: (B,12,5000) float32 tensor on the model's device -> sigmoid probabilities (B,45), float32."""
    return torch.sigmoid(model(x).float())


def apply_thresholds(probs, thresholds):
    return (np.asarray(probs) > np.asarray(thresholds)[None, :]).astype(int)
