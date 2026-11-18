import json
import sys

import numpy as np
import pytest
import soundfile as sf
from conftest import fake_vad

from nemo_3_diarization.labels import read_rttm
from nemo_3_diarization.prod_testset import main, process_recording

SR, DUR = 8000, 6


def make_stereo(path, ch0=((0, 2),), ch1=((3, 5),), sr=SR, amp=0.1):
    rng = np.random.default_rng(0)
    x = np.zeros((DUR * sr, 2), dtype=np.float32)
    for c, bursts in enumerate([ch0, ch1]):
        for s, e in bursts:
            x[int(s * sr):int(e * sr), c] = amp * rng.standard_normal(int((e - s) * sr))
    sf.write(path, x, sr, subtype="PCM_16")
    return path


def close(a, b):
    return len(a) == len(b) and all(abs(x - y) < 0.1 for p, q in zip(a, b) for x, y in zip(p, q))


def test_stereo_to_mono_with_two_speakers(tmp_path):
    row = process_recording(make_stereo(tmp_path / "call1.wav"), tmp_path / "out", vad_fn=fake_vad)
    assert row["num_speakers"] == 2 and row["split"] == "test" and row["kind"] == "prod"
    assert row["duration"] == DUR and row["audio_filepath"].endswith("call1.wav")
    info = sf.info(row["audio_filepath"])
    assert info.channels == 1 and info.samplerate == 8000 and info.subtype == "PCM_16"
    rt = read_rttm(row["rttm_filepath"])
    assert close(rt["ch0"], [(0, 2)]) and close(rt["ch1"], [(3, 5)])


def test_silent_channel_counts_one_speaker(tmp_path):
    row = process_recording(make_stereo(tmp_path / "c.wav", ch1=()), tmp_path / "out", vad_fn=fake_vad)
    assert row["num_speakers"] == 1 and set(read_rttm(row["rttm_filepath"])) == {"ch0"}


def test_mono_input_rejected(tmp_path):
    sf.write(tmp_path / "m.wav", np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(ValueError, match="expected 2 channels"):
        process_recording(tmp_path / "m.wav", tmp_path / "out", vad_fn=fake_vad)


def test_non_8k_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected 8000 Hz"):
        process_recording(make_stereo(tmp_path / "c.wav", sr=16000), tmp_path / "out", vad_fn=fake_vad)


def test_clipping_mix_scaled_to_099(tmp_path):
    row = process_recording(make_stereo(tmp_path / "c.wav", ch1=((0, 2),), amp=0.9), tmp_path / "out", vad_fn=fake_vad)
    assert abs(np.abs(sf.read(row["audio_filepath"])[0]).max() - 0.99) < 1e-3


def test_main_isolates_failures(tmp_path, monkeypatch, capsys):
    (tmp_path / "in").mkdir()
    make_stereo(tmp_path / "in" / "a.wav", ch0=(), ch1=())
    sf.write(tmp_path / "in" / "b.wav", np.zeros(SR, dtype=np.float32), SR)  # mono -> fails
    monkeypatch.setattr(sys, "argv", ["x", "--in-dir", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out")])
    main()
    rows = [json.loads(line) for line in (tmp_path / "out" / "test.json").read_text().splitlines()]
    assert [r["audio_filepath"].endswith("a.wav") for r in rows] == [True]
    assert "FAILED b: ValueError('expected 2 channels')" in capsys.readouterr().out
