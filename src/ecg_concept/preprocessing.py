"""Signal preprocessing recovered from the original training notebook
(HonsExAI/LightECGNet_v2_FIXED_faster.ipynb, `load_ecg_mat`).

Pipeline: read 'val' -> (12, T) -> zero-pad / crop to 5000 -> per-lead
0.5-40 Hz Butterworth (order 3, zero-phase filtfilt, fs=500) -> per-lead z-score.

Quirk preserved by default: loadmat returns int16, and the original code writes
each filtered lead back into that int16 array (`signal[i] = bandpass_filter(...)`),
which truncates the filtered values toward zero to integer counts before the
z-score. `quantize_int16=False` skips that step (the float32 path used by the
later SHAP script); the two differ slightly.
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

FS = 500
SIGNAL_LEN = 5000
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def read_raw_mat(path):
    """Return the raw (12, T) int16 array stored under 'val' (1000 counts = 1 mV)."""
    signal = loadmat(path).get("val")
    if signal is None:
        raise ValueError(f"No 'val' array in {path}")
    return signal if signal.shape[0] == 12 else signal.T


def preprocess(raw, target_len=SIGNAL_LEN, quantize_int16=True):
    """Apply the training-time preprocessing to a (12, T) array. Returns float32 (12, target_len)."""
    signal = np.array(raw, copy=True)
    cur = signal.shape[1]
    if cur < target_len:
        signal = np.concatenate([signal, np.zeros((12, target_len - cur), dtype=signal.dtype)], axis=1)
    elif cur > target_len:
        signal = signal[:, :target_len]

    nyq = 0.5 * FS
    b, a = butter(3, [0.5 / nyq, 40.0 / nyq], btype="band")
    if quantize_int16:
        for i in range(12):                      # in-place write into the int16 array, as in training
            signal[i] = filtfilt(b, a, signal[i])
    else:
        signal = filtfilt(b, a, signal.astype(np.float64), axis=-1)

    signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
    return signal.astype(np.float32)


def load_ecg(path, quantize_int16=True):
    return preprocess(read_raw_mat(path), quantize_int16=quantize_int16)
