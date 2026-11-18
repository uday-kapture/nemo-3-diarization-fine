"""Basis Conversations 1500 -> 8 kHz mono multi-speaker sessions, RTTM labels and NeMo manifests."""
import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import snapshot_download

from nemo_3_diarization.audio import mix, phone_channel, vad
from nemo_3_diarization.labels import (
    Intervals, intersect, merge, rttm_lines, speaker_activity, split_of, top_speakers, windows,
)

SR = 48000
REPO = "basis-ai/basis-conversations-1500"


def _vad_track(path: Path, vad_fn) -> Intervals:
    iv: Intervals = []
    off = 0
    for block in sf.blocks(path, blocksize=600 * SR, dtype="float32"):  # bounds memory on hour-long tracks
        iv += [(s + off / SR, e + off / SR) for s, e in vad_fn(block, SR)]
        off += len(block)
    return merge(iv)


def process_segment(seg: dict, root: Path, out: Path, vad_fn=vad, win: float = 105.0) -> list[dict]:
    root, out = Path(root).resolve(), Path(out).resolve()
    sid, split = seg["segment_id"], split_of(seg["conversation_id"])
    paths = {spk: root / p for spk, p in seg["tracks"].items()}
    rows: dict[str, Intervals] = {}
    for t in seg["transcript"]:
        rows.setdefault(t["speaker"], []).append((t["start"], t["end"]))
    on_call = {s["speaker"]: [(c["start"], c["end"]) for c in s["on_call"]] for s in seg["speakers"]}
    muted = [(m["start"], m["end"]) for m in seg["muted_spans"]]
    # speakers with no transcript rows have no labelled speech, so skip their VAD
    activity = {
        spk: speaker_activity(_vad_track(paths[spk], vad_fn), rows[spk], on_call.get(spk, [(0.0, seg["duration"])]))
        for spk in paths if spk in rows
    }

    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "rttm").mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for start, end in windows(seg["duration"], muted, win):
        phone: dict[str, object] = {}
        picks = [("pair", top_speakers(activity, start, end, 2, 5.0))]
        if split == "test":
            picks.append(("all", top_speakers(activity, start, end, 8, 0.5)))
        for kind, spks in picks:
            if len(spks) < 2:  # pair is capped at 2 by top_speakers
                continue
            for spk in spks:
                if spk not in phone:
                    x, _ = sf.read(paths[spk], start=int(start * SR), stop=int(end * SR), dtype="float32")
                    phone[spk] = phone_channel(x, SR)
            speech = [[(s - start, e - start) for s, e in intersect(activity[spk], [(start, end)])] for spk in spks]
            fid = f"{sid}_{int(start)}" + ("_all" if kind == "all" else "")
            wav, rttm = out / "wav" / f"{fid}.wav", out / "rttm" / f"{fid}.rttm"
            sf.write(wav, mix([phone[s] for s in spks], speech), 8000, subtype="PCM_16")
            rttm.write_text("\n".join(rttm_lines(fid, activity, start, end, spks)) + "\n")
            manifest.append({
                "audio_filepath": str(wav), "offset": 0, "duration": end - start, "label": "infer", "text": "-",
                "num_speakers": len(spks), "rttm_filepath": str(rttm), "uem_filepath": None,
                "split": split, "kind": kind,
            })
    return manifest


def _init():
    torch.set_num_threads(1)


def _work(args):
    return process_segment(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "en"])
    ap.add_argument("--raw-dir", type=Path, default=Path("data/basis-raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/basis"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-segments", type=int, default=0)
    a = ap.parse_args()
    raw, out = a.raw_dir.resolve(), a.out_dir.resolve()

    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=raw,
                      allow_patterns=[f"{lang}/segments.jsonl" for lang in a.langs])
    segs = []
    for lang in a.langs:
        with open(raw / lang / "segments.jsonl") as f:
            for line in f:
                s = json.loads(line)
                s["transcript"] = [{k: t[k] for k in ("speaker", "start", "end")} for t in s["transcript"]]  # drop words
                segs.append(s)
                if a.limit_segments and len(segs) >= a.limit_segments:
                    break
        if a.limit_segments and len(segs) >= a.limit_segments:
            break
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=raw,
                      allow_patterns=[p for s in segs for p in s["tracks"].values()])

    rows: list[dict] = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(_work, [(s, raw, out) for s in segs]):
            rows += r
    out.mkdir(parents=True, exist_ok=True)
    for name, pick in [(sp, lambda r, sp=sp: r["split"] == sp and r["kind"] == "pair") for sp in ("train", "val", "test")] + [
        ("test_all", lambda r: r["kind"] == "all")
    ]:
        (out / f"{name}.json").write_text("".join(json.dumps(r) + "\n" for r in rows if pick(r)))
    for (split, kind), n in sorted(Counter((r["split"], r["kind"]) for r in rows).items()):
        print(f"{split:5s} {kind:4s} {n}")


if __name__ == "__main__":
    main()
