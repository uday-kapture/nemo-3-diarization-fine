"""Phone-channel degradation, loudness-matched mixing and Silero VAD."""
from functools import cache
from math import log

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from silero_vad import get_speech_timestamps, load_silero_vad

from nemo_3_diarization.labels import Intervals, merge

_A = 87.6
_K = 1 + log(_A)
_BLOCK = 600 * 16000  # VAD block, samples at 16 kHz


def alaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """Continuous A-law compand, 8-bit sign-magnitude quantise, expand."""
    a = np.abs(x)
    y = np.where(a < 1 / _A, _A * a / _K, (1 + np.log(np.maximum(_A * a, 1))) / _K)
    y = np.round(y * 127) / 127
    return np.sign(x) * np.where(y < 1 / _K, y * _K / _A, np.exp(y * _K - 1) / _A)


def phone_channel(x: np.ndarray, sr: int = 48000) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    y = resample_poly(sosfiltfilt(sos, x), 8000, sr)
    return alaw_roundtrip(y).astype(np.float32)


def mix(tracks: list[np.ndarray], speech: list[Intervals], sr: int = 8000) -> np.ndarray:
    """Scale each track to -26 dBFS RMS over its speech (window-relative seconds), sum, cap peak at 0.9."""
    out = np.zeros(max(len(t) for t in tracks))
    for t, iv in zip(tracks, speech):
        seg = np.concatenate([t[int(s * sr):int(e * sr)] for s, e in iv]) if iv else t[:0]
        rms = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
        out[:len(t)] += t * (10 ** (-26 / 20) / rms if rms >= 1e-6 else 1.0)
    peak = np.max(np.abs(out))
    return (out * (0.9 / peak) if peak > 0.9 else out).astype(np.float32)


@cache
def _model():
    return load_silero_vad()


def vad(x: np.ndarray, sr: int) -> Intervals:
    y = resample_poly(x, 16000, sr).astype(np.float32)
    iv: Intervals = []
    for i in range(0, len(y), _BLOCK):
        ts = get_speech_timestamps(torch.from_numpy(y[i:i + _BLOCK]), _model(), sampling_rate=16000)
        iv += [((i + t["start"]) / 16000, (i + t["end"]) / 16000) for t in ts]
    return merge(iv)
