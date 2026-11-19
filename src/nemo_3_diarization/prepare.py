"""Basis Conversations 1500 -> 8 kHz mono multi-speaker sessions, RTTM labels and NeMo manifests."""
import argparse
import json
import multiprocessing as mp
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from nemo_3_diarization.audio import mix, phone_channel, vad
from nemo_3_diarization.labels import (
    Intervals, intersect, merge, rttm_lines, split_of, top_speakers, windows,
)

SR = 48000
BLOCK = 600 * SR  # VAD block, samples at 48 kHz; bounds memory on hour-long tracks
REPO = "basis-ai/basis-conversations-1500"


def _vad_track(path: Path, vad_fn) -> Intervals:
    iv: Intervals = []
    off = 0
    for block in sf.blocks(path, blocksize=BLOCK, dtype="float32"):
        iv += [(s + off / SR, e + off / SR) for s, e in vad_fn(block, SR)]
        off += len(block)
    return merge(iv)


def process_segment(seg: dict, root: Path, out: Path, vad_fn=vad, win: float = 105.0) -> list[dict]:
    root, out = Path(root).resolve(), Path(out).resolve()
    sid, split = seg["segment_id"], split_of(seg["conversation_id"])
    paths = {spk: root / p for spk, p in seg["tracks"].items()}
    for p in paths.values():  # another rate would silently time-scale every label taken from this track
        if sf.info(p).samplerate != SR:
            raise ValueError(f"{p.name}: expected {SR} Hz")
    on_call = {s["speaker"]: [(c["start"], c["end"]) for c in s["on_call"]] for s in seg["speakers"]}
    muted = [(m["start"], m["end"]) for m in seg["muted_spans"]]
    # speech = VAD on the speaker's own track while on the call. Not gated by transcript rows: Basis ASR leaves out
    # short replies, and on real data that gate dropped ~7x more own speech than bleed.
    activity = {spk: intersect(_vad_track(p, vad_fn), on_call.get(spk, [(0.0, seg["duration"])])) for spk, p in paths.items()}

    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "rttm").mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for start, end in windows(seg["duration"], muted, win):
        phone: dict[str, object] = {}
        pair = top_speakers(activity, start, end, 2, 5.0)
        if len(pair) != 2:
            continue
        picks = [("pair", pair)]
        if split == "test":  # the all-speakers mix only accompanies a window that produced a pair row
            picks.append(("all", top_speakers(activity, start, end, 8, 0.5)))
        for kind, spks in picks:
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


def _stream_segment(seg: dict, raw: Path, out: Path, vad_fn=vad, win: float = 105.0) -> list[dict]:
    # one segment's 48 kHz tracks live on disk only while it is processed, so the full ~60 GB never does
    with tempfile.TemporaryDirectory(prefix="seg-", dir=raw) as tmp:
        for p in seg["tracks"].values():
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=tmp)
        return process_segment(seg, tmp, out, vad_fn, win)


def _work(args):
    try:
        return args[0]["segment_id"], _stream_segment(*args), None
    except Exception as e:  # one bad segment must not lose the rest of the run
        return args[0]["segment_id"], [], repr(e)  # repr keeps it on one line


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
                del s["transcript"]  # unused, and the bulk of each row
                segs.append(s)
                if a.limit_segments and len(segs) >= a.limit_segments:
                    break
        if a.limit_segments and len(segs) >= a.limit_segments:
            break
    for d in raw.glob("seg-*"):  # tracks left behind by a killed run
        shutil.rmtree(d)

    rows: list[dict] = []
    failed = 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for sid, r, err in pool.imap_unordered(_work, [(s, raw, out) for s in segs]):
            rows += r
            if err:
                failed += 1
                print(f"FAILED {sid}: {err}")
    out.mkdir(parents=True, exist_ok=True)
    for name, pick in [(sp, lambda r, sp=sp: r["split"] == sp and r["kind"] == "pair") for sp in ("train", "val", "test")] + [
        ("test_all", lambda r: r["kind"] == "all")
    ]:
        (out / f"{name}.json").write_text("".join(json.dumps(r) + "\n" for r in rows if pick(r)))
    for (split, kind), n in sorted(Counter((r["split"], r["kind"]) for r in rows).items()):
        print(f"{split:5s} {kind:4s} {n}")
    print(f"failed segments: {failed}")


if __name__ == "__main__":
    main()
