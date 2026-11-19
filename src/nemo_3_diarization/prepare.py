"""Basis Conversations 1500 -> 8 kHz mono multi-speaker sessions, RTTM labels and NeMo manifests."""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from nemo_3_diarization.audio import mix, phone_channel, vad
from nemo_3_diarization.notify import GChat, duration
from nemo_3_diarization.labels import (
    Intervals, intersect, merge, rttm_lines, split_of, top_speakers, windows,
)

SR = 48000
BLOCK = 600 * SR  # VAD block, samples at 48 kHz; bounds memory on hour-long tracks
REPO = "basis-ai/basis-conversations-1500"
HEARTBEAT = 300  # s without a finished segment before a "still working" line: the biggest take ~30 min alone
REPORT_EVERY = 0.1  # a Chat progress message per 10 % of the audio


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
    from huggingface_hub.utils import disable_progress_bars

    torch.set_num_threads(1)
    disable_progress_bars()  # one bar per file from every worker buries the progress lines


def _stream_segment(seg: dict, raw: Path, out: Path, vad_fn=vad, win: float = 105.0) -> list[dict]:
    # one segment's 48 kHz tracks live on disk only while it is processed, so the full ~60 GB never does
    with tempfile.TemporaryDirectory(prefix="seg-", dir=raw) as tmp:
        for p in seg["tracks"].values():
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=tmp)
        return process_segment(seg, tmp, out, vad_fn, win)


def _work(args):
    t0 = time.monotonic()
    try:
        return args[0]["segment_id"], _stream_segment(*args), None, time.monotonic() - t0
    except Exception as e:  # one bad segment must not lose the rest of the run
        return args[0]["segment_id"], [], repr(e), time.monotonic() - t0  # repr keeps it on one line


def progress_line(done: int, total: int, done_h: float, total_h: float, mixes: int, failed: int,
                  elapsed: float) -> str:
    # progress by audio hours (duration x tracks), not segments: segments range from minutes to 76 h of tracks
    eta = f" · ETA {duration(elapsed * (total_h - done_h) / done_h)}" if 0 < done_h < total_h else ""
    return (f"{done}/{total} segments, {done_h / total_h:.0%} of the audio · {mixes:,} mixes · {failed} failed · "
            f"{duration(elapsed)} elapsed{eta}")


def _run(segs: list[dict], raw: Path, out: Path, workers: int, chat: GChat, name: str, work=_work):
    hours = {s["segment_id"]: s["duration"] * len(s["tracks"]) / 3600 for s in segs}
    total_h = sum(hours.values())
    start = f"*{name}* started: {len(segs)} segments, {total_h:,.0f} h of speaker tracks, {workers} workers"
    print(start.replace("*", ""), flush=True)
    chat.send(start)
    rows: list[dict] = []
    failed, done_h, t0 = 0, 0.0, time.monotonic()
    last, next_report = t0, REPORT_EVERY
    ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init)
    try:
        pending = {ex.submit(work, (s, raw, out)): s["segment_id"] for s in segs}
        while pending:
            finished, _ = wait(pending, timeout=HEARTBEAT, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if not finished:
                running = [sid for f, sid in pending.items() if f.running()]
                print(f"still working: {len(pending)} segments left, last one finished {duration(now - last)} ago; "
                      f"running: {', '.join(running)}", flush=True)
                continue
            for f in finished:
                sid = pending.pop(f)
                _, r, err, secs = f.result()  # a worker that died (e.g. killed for memory) raises BrokenProcessPool
                rows += r
                done_h += hours[sid]
                last = now
                if err:
                    failed += 1
                    print(f"FAILED {sid}: {err}", flush=True)
                    if failed == 1:
                        chat.send(f"⚠️ *{name}* first failed segment {sid}: {err[:500]}")
                line = progress_line(len(segs) - len(pending), len(segs), done_h, total_h, len(rows), failed,
                                     now - t0)
                print(f"done {sid}: {len(r)} mixes in {duration(secs)} · {line}", flush=True)
                if pending and done_h / total_h >= next_report:
                    chat.send(f"*{name}* {line}")
                    next_report = (int(done_h / total_h / REPORT_EVERY) + 1) * REPORT_EVERY
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return rows, failed


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

    run_name = f"prepare-{out.name}-{time.strftime('%Y%m%d-%H%M')}"
    chat = GChat.from_env(thread=run_name)
    t0 = time.monotonic()
    try:
        rows, failed = _run(segs, raw, out, a.workers, chat, run_name)
    except BaseException as e:
        hint = (" A worker died, most likely killed for memory: rerun with fewer --workers."
                if isinstance(e, BrokenProcessPool) else "")
        print(f"prepare stopped: {type(e).__name__}.{hint}", file=sys.stderr, flush=True)
        chat.send(f"🔴 *{run_name}* stopped after {duration(time.monotonic() - t0)}: {type(e).__name__}.{hint}")
        raise
    out.mkdir(parents=True, exist_ok=True)
    for name, pick in [(sp, lambda r, sp=sp: r["split"] == sp and r["kind"] == "pair") for sp in ("train", "val", "test")] + [
        ("test_all", lambda r: r["kind"] == "all")
    ]:
        (out / f"{name}.json").write_text("".join(json.dumps(r) + "\n" for r in rows if pick(r)))
    counts = sorted(Counter((r["split"], r["kind"]) for r in rows).items())
    for (split, kind), n in counts:
        print(f"{split:5s} {kind:4s} {n}")
    print(f"failed segments: {failed}")
    chat.send(f"✅ *{run_name}* finished in {duration(time.monotonic() - t0)}: "
              + ", ".join(f"{split} {kind} {n:,}" for (split, kind), n in counts) + f"; {failed} failed segments")


if __name__ == "__main__":
    main()
