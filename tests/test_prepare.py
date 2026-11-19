import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from conftest import fake_vad

from nemo_3_diarization.labels import merge, read_rttm, split_of
from nemo_3_diarization import prepare
from nemo_3_diarization.prepare import _vad_track, _work, process_segment

SR, DUR = 48000, 23


def find_conv(split):
    return next(f"c{i}" for i in range(1000) if split_of(f"c{i}") == split)


def make_segment(root, conversation_id, silence=(), bursts=None, muted=True):
    bursts = bursts or {"a": [(0, 5.5), (12, 16)], "b": [(4.5, 10), (17, 21)], "c": [(0.5, 5)]}
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
        "muted_spans": [{"start": 15.0, "end": 15.5, "reason": "x"}] if muted else [],
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
    rows = process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0)
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["all", "pair"]
    by = {r["kind"]: r for r in rows}
    assert by["all"]["num_speakers"] == 3 and by["pair"]["num_speakers"] == 2
    assert by["all"]["audio_filepath"].endswith("s1_0_all.wav") and by["pair"]["audio_filepath"].endswith("s1_0.wav")
    assert set(read_rttm(by["all"]["rttm_filepath"])) == {"a", "b", "c"}


def test_process_segment_single_speaker_yields_nothing(tmp_path):
    seg = make_segment(tmp_path, conversation_id=find_conv("train"), silence=["b"])
    assert process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0) == []


def test_no_all_row_without_pair_row(tmp_path):
    seg = make_segment(tmp_path, conversation_id=find_conv("test"), silence=["b"])
    assert process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0) == []


def test_window_at_nonzero_start_aligns_audio_and_rttm(tmp_path):
    bursts = {"a": [(10.5, 14), (15, 19)], "b": [(11, 13), (14.5, 19.5)], "c": []}  # c: silent track
    seg = make_segment(tmp_path, find_conv("train"), bursts=bursts, muted=False)
    (r,) = process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0)
    assert r["audio_filepath"].endswith("s1_10.wav")  # second window: start=10
    rt = read_rttm(r["rttm_filepath"])
    assert rt["a"][0] == (0.5, 4.0) and rt["b"][-1] == (4.5, 9.5)  # window-relative, not file-relative
    y, sr = sf.read(r["audio_filepath"])
    speech = merge([iv for v in rt.values() for iv in v])
    inside = np.zeros(len(y), bool)
    for s, e in speech:
        inside[int(s * sr):int(e * sr)] = True
    outside = ~np.convolve(inside, np.ones(401), "same").astype(bool)  # 25 ms margin each side for filter ringing
    rms = lambda z: np.sqrt(np.mean(z ** 2))
    assert outside.sum() > 0 and 20 * np.log10(rms(y[inside]) / (rms(y[outside]) + 1e-12)) >= 10
    # a speaks alone 3–4 s; it sits at mix's -26 dBFS only if mix got window-relative speech
    assert abs(20 * np.log10(rms(y[int(3.1 * sr):int(3.9 * sr)])) + 26) < 1


def test_vad_track_adds_block_offsets(tmp_path, monkeypatch):
    x = np.zeros(10 * SR, dtype=np.float32)
    x[int(3.5 * SR):int(6.2 * SR)] = 0.1  # crosses the 4 s and 6 s block boundaries
    sf.write(tmp_path / "t.flac", x, SR)
    monkeypatch.setattr(prepare, "BLOCK", 2 * SR)
    (s, e), = _vad_track(tmp_path / "t.flac", fake_vad)
    assert abs(s - 3.5) < 0.06 and abs(e - 6.2) < 0.06


def test_work_reports_failure_instead_of_raising(tmp_path):
    sid, rows, err = _work(({"segment_id": "bad"}, tmp_path, tmp_path / "out"))
    assert sid == "bad" and rows == [] and "KeyError" in err


def fake_hub(monkeypatch, src):
    def download(repo_id, filename, *, repo_type, local_dir):
        dst = Path(local_dir) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        return str(shutil.copy(src / filename, dst))
    monkeypatch.setattr(prepare, "hf_hub_download", download)


def test_work_streams_tracks_and_deletes_them(tmp_path, monkeypatch):
    src, raw = tmp_path / "hub", tmp_path / "raw"
    src.mkdir(), raw.mkdir()
    seg = make_segment(src, conversation_id=find_conv("train"))
    fake_hub(monkeypatch, src)
    sid, rows, err = _work((seg, raw, tmp_path / "out", fake_vad, 10.0))
    assert err is None and len(rows) == 1 and Path(rows[0]["audio_filepath"]).exists()
    assert list(raw.iterdir()) == []  # the 48 kHz tracks are gone once the segment is done


def test_non_48k_track_fails_segment_and_deletes_tracks(tmp_path, monkeypatch):
    src, raw = tmp_path / "hub", tmp_path / "raw"
    src.mkdir(), raw.mkdir()
    seg = make_segment(src, conversation_id=find_conv("train"))
    sf.write(src / seg["tracks"]["a"], np.zeros(DUR * 16000, dtype=np.float32), 16000)
    fake_hub(monkeypatch, src)
    sid, rows, err = _work((seg, raw, tmp_path / "out", fake_vad, 10.0))
    assert sid == "s1" and rows == [] and "expected 48000 Hz" in err
    assert list(raw.iterdir()) == []


def test_untranscribed_speech_is_labelled(tmp_path):
    # Basis ASR leaves out short replies ("haan", "hmm"); speech on the speaker's own track is speech regardless
    seg = make_segment(tmp_path, conversation_id=find_conv("train"))
    seg["transcript"] = [t for t in seg["transcript"] if t["speaker"] != "b"]
    (r,) = process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0)
    assert read_rttm(r["rttm_filepath"])["b"] == [(4.5, 10.0)]


def test_speech_off_call_is_not_labelled(tmp_path):
    seg = make_segment(tmp_path, conversation_id=find_conv("train"))
    seg["speakers"][1]["on_call"] = [{"start": 0, "end": 7}]  # b: 2.5 s left in window 0-10, under the 5 s minimum
    assert process_segment(seg, tmp_path, tmp_path / "out", vad_fn=fake_vad, win=10.0) == []
