import numpy as np
import soundfile as sf

from nemo_3_diarization.labels import merge, read_rttm, split_of
from nemo_3_diarization.prepare import process_segment

SR, DUR = 48000, 23


def find_conv(split):
    return next(f"c{i}" for i in range(1000) if split_of(f"c{i}") == split)


def fake_vad(x, sr):
    f = int(0.05 * sr)
    n = len(x) // f
    loud = np.sqrt((x[: n * f].reshape(n, f) ** 2).mean(1)) > 0.01
    return merge([(i * 0.05, (i + 1) * 0.05) for i in np.flatnonzero(loud)])


def make_segment(root, conversation_id, silence=()):
    bursts = {"a": [(0, 5.5), (12, 16)], "b": [(4.5, 10), (17, 21)], "c": []}
    rng = np.random.default_rng(0)
    tracks, transcript = {}, []
    (root / "audio").mkdir()
    for spk, bs in bursts.items():
        x = np.zeros(DUR * SR, dtype=np.float32)
        for s, e in bs:
            if spk not in silence:
                x[int(s * SR):int(e * SR)] = 0.1 * rng.standard_normal(int((e - s) * SR))
            transcript.append({"speaker": spk, "start": s, "end": e, "text": "x", "words": []})
        tracks[spk] = f"audio/{spk}.flac"
        sf.write(root / tracks[spk], x, SR)
    return {
        "segment_id": "s1", "conversation_id": conversation_id, "duration": DUR, "tracks": tracks,
        "muted_spans": [{"start": 15.0, "end": 15.5, "reason": "x"}],
        "speakers": [{"speaker": k, "on_call": [{"start": 0, "end": DUR}]} for k in bursts],
        "transcript": transcript,
    }


def test_process_segment_pair_window(tmp_path):
    seg = make_segment(tmp_path, conversation_id=find_conv("train"))
    rows = process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0)
    assert [r["kind"] for r in rows] == ["pair"]          # window 10–20 touches the mute, tail 20–23 dropped
    r = rows[0]
    assert r["num_speakers"] == 2 and r["split"] == "train" and r["duration"] == 10.0
    assert sf.info(r["audio_filepath"]).samplerate == 8000
    assert set(read_rttm(r["rttm_filepath"])) == {"a", "b"}


def test_process_segment_test_split_adds_all_kind(tmp_path):
    seg = make_segment(tmp_path, conversation_id=find_conv("test"))
    kinds = sorted(r["kind"] for r in process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0))
    assert kinds == ["all", "pair"]


def test_process_segment_single_speaker_yields_nothing(tmp_path):
    seg = make_segment(tmp_path, conversation_id=find_conv("train"), silence=["b"])
    assert process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0) == []
