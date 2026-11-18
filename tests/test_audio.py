import numpy as np
from nemo_3_diarization.audio import alaw_roundtrip, phone_channel, mix, vad

def test_alaw_is_8bit_and_accurate():
    x = np.linspace(-1, 1, 20001)
    y = alaw_roundtrip(x)
    assert len(np.unique(y)) <= 256
    big = np.abs(x) > 0.02
    assert np.max(np.abs(y[big] - x[big]) / np.abs(x[big])) < 0.03
    assert np.max(np.abs(y[~big] - x[~big])) < 0.0006

def test_phone_channel_rate_and_band():
    t = np.arange(48000 * 2) / 48000
    y = phone_channel(0.5 * np.sin(2 * np.pi * 1000 * t), 48000)
    assert y.dtype == np.float32 and len(y) == 16000
    assert abs(np.sqrt(np.mean(y[2000:-2000] ** 2)) - 0.5 / np.sqrt(2)) < 0.035
    n = np.random.default_rng(0).standard_normal(48000 * 2) * 0.1
    p = np.abs(np.fft.rfft(phone_channel(n, 48000))) ** 2
    f = np.fft.rfftfreq(16000, 1 / 8000)
    assert p[f > 3700].sum() / p.sum() < 0.02 and p[f < 150].sum() / p.sum() < 0.02

def test_mix_sets_speech_level_and_peak():
    t = np.arange(8000) / 8000
    y = mix([0.01 * np.sin(2 * np.pi * 300 * t)], [[(0.0, 1.0)]])
    assert abs(20 * np.log10(np.sqrt(np.mean(y ** 2))) + 26) < 0.2
    loud = mix([np.ones(8000), np.ones(8000)], [[], []])   # no speech → unscaled, sum 2.0 must be clamped
    assert abs(np.max(np.abs(loud)) - 0.9) < 1e-6

def test_mix_silent_track_no_nan():
    y = mix([np.zeros(8000), 0.1 * np.ones(8000)], [[(0.0, 1.0)], [(0.0, 1.0)]])
    assert np.all(np.isfinite(y))

def test_vad_silence_is_empty():
    assert vad(np.zeros(8000 * 5, dtype=np.float32), 8000) == []

def test_vad_times_are_sample_exact_with_block_offsets(monkeypatch):
    from nemo_3_diarization import audio
    monkeypatch.setattr(audio, "_BLOCK", 16000 * 2)
    monkeypatch.setattr(audio, "_model", lambda: None)
    monkeypatch.setattr(audio, "get_speech_timestamps", lambda *a, **k: [{"start": 1601, "end": 8003}])
    got = audio.vad(np.zeros(8000 * 5, dtype=np.float32), 8000)  # 5 s -> 3 blocks, offsets 0/2/4 s
    want = [(o + 1601 / 16000, o + 8003 / 16000) for o in (0.0, 2.0, 4.0)]
    assert np.allclose(got, want, atol=1e-9)
