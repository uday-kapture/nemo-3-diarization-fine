"""Stereo 8 kHz production calls (one speaker per channel) -> mono-mix test set with per-channel VAD RTTM labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from nemo_3_diarization.audio import vad
from nemo_3_diarization.labels import rttm_lines


def process_recording(path: Path, out: Path, vad_fn=vad) -> dict:
    path, out = Path(path).resolve(), Path(out).resolve()
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] != 2:
        raise ValueError("expected 2 channels")
    if sr != 8000:
        raise ValueError("expected 8000 Hz")
    fid, dur = path.stem, len(x) / sr
    activity = {f"ch{c}": vad_fn(x[:, c], sr) for c in range(2)}
    spks = [k for k, iv in activity.items() if iv]
    y = x[:, 0] + x[:, 1]
    peak = np.abs(y).max()
    if peak > 1.0:
        y *= 0.99 / peak

    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "rttm").mkdir(parents=True, exist_ok=True)
    wav, rttm = out / "wav" / f"{fid}.wav", out / "rttm" / f"{fid}.rttm"
    sf.write(wav, y, 8000, subtype="PCM_16")
    rttm.write_text("".join(line + "\n" for line in rttm_lines(fid, activity, 0.0, dur, spks)))
    return {
        "audio_filepath": str(wav), "offset": 0, "duration": dur, "label": "infer", "text": "-",
        "num_speakers": len(spks), "rttm_filepath": str(rttm), "uem_filepath": None,
        "split": "test", "kind": "prod",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/prod"))
    a = ap.parse_args()
    out = a.out_dir.resolve()

    rows: list[dict] = []
    failed = 0
    for p in sorted(p for p in a.in_dir.iterdir() if p.suffix.lower() == ".wav"):
        try:
            rows.append(process_recording(p, out))
        except Exception as e:  # one bad recording must not lose the rest of the run
            failed += 1
            print(f"FAILED {p.stem}: {e!r}")  # repr keeps it on one line
    out.mkdir(parents=True, exist_ok=True)
    (out / "test.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"recordings: {len(rows)}")
    print(f"failed recordings: {failed}")


if __name__ == "__main__":
    main()
