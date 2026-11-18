import numpy as np

from nemo_3_diarization.labels import merge


def fake_vad(x, sr):
    f = int(0.05 * sr)
    n = len(x) // f
    loud = np.sqrt((x[: n * f].reshape(n, f) ** 2).mean(1)) > 0.01
    return merge([(i * 0.05, (i + 1) * 0.05) for i in np.flatnonzero(loud)])
