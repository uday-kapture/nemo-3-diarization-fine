"""Pure-Python interval, window, split and RTTM helpers. Times are seconds."""
import hashlib
from pathlib import Path

Intervals = list[tuple[float, float]]


def merge(iv: Intervals, pad: float = 0.0) -> Intervals:
    out: Intervals = []
    for s, e in sorted((max(0.0, s - pad), e + pad) for s, e in iv):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def intersect(a: Intervals, b: Intervals) -> Intervals:
    out: Intervals = []
    i = j = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(iv: Intervals, start: float, end: float) -> float:
    return sum(e - s for s, e in intersect(iv, [(start, end)]))


def speaker_activity(vad: Intervals, rows: Intervals, on_call: Intervals, pad: float = 0.3) -> Intervals:
    """Speech = VAD within the padded transcript rows, restricted to time on the call."""
    return intersect(intersect(vad, merge(rows, pad)), on_call)


def windows(duration: float, muted: Intervals, win: float = 105.0) -> Intervals:
    """Consecutive full windows (tail dropped) that have no positive-length overlap with a muted span."""
    ws = [(i * win, (i + 1) * win) for i in range(int(duration // win))]
    return [(s, e) for s, e in ws if not any(ms < e and me > s for ms, me in muted)]


def top_speakers(activity: dict[str, Intervals], start: float, end: float, n: int, min_speech: float) -> list[str]:
    speech = {k: total(v, start, end) for k, v in activity.items()}
    ranked = sorted((k for k, t in speech.items() if t >= min_speech), key=lambda k: (-speech[k], k))
    return ranked[:n]


def rttm_lines(file_id: str, activity: dict[str, Intervals], start: float, end: float, speakers: list[str]) -> list[str]:
    return [
        f"SPEAKER {file_id} 1 {s - start:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>"
        for spk in speakers
        for s, e in intersect(activity[spk], [(start, end)])
    ]


def read_rttm(path: str | Path) -> dict[str, Intervals]:
    out: dict[str, Intervals] = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            s, d = float(f[3]), float(f[4])
            out.setdefault(f[7], []).append((s, s + d))
    return out


def split_of(conversation_id: str) -> str:
    # shortcut: speaker overlap across splits, switch to speaker-disjoint if val/test scores look inflated
    h = int(hashlib.sha1(conversation_id.encode(), usedforsecurity=False).hexdigest(), 16) % 100
    return "train" if h < 90 else "val" if h < 95 else "test"
