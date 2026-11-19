"""DER of stock vs fine-tuned Nemotron 3 Diarization on test manifests, at the model card's streaming latencies."""
import argparse
import json
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from nemo_3_diarization import tracking
from nemo_3_diarization.labels import Intervals, merge, read_rttm

# spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period (80 ms frames, model card)
LATENCIES = {"1.04s": (264, 264, 9, 4, 222), "0.32s": (264, 264, 3, 1, 222)}
FIELDS = ("spkcache_len", "fifo_len", "chunk_len", "chunk_right_context", "spkcache_update_period")


def parse_segments(lines: list[str]) -> dict[str, Intervals]:
    out: dict[str, Intervals] = {}
    for line in lines:
        s, e, spk = line.split()
        out.setdefault(spk, []).append((float(s), float(e)))
    return out


def _annotation(spk: dict[str, Intervals]) -> Annotation:
    a = Annotation()
    for label, iv in spk.items():
        for i, (s, e) in enumerate(merge(iv)):  # pyannote counts overlapping same-speaker segments twice
            a[Segment(s, e), f"{label}_{i}"] = label
    return a


def score(ref: dict[str, Intervals], hyp: dict[str, Intervals]) -> dict[str, float]:
    # pyannote's collar is the total width around each reference boundary: 0.5 here is NeMo's / NIST's ±0.25 s.
    d = DiarizationErrorRate(collar=0.5, skip_overlap=False)(_annotation(ref), _annotation(hyp), detailed=True)
    return {"total": d["total"], "miss": d["missed detection"], "fa": d["false alarm"],
            "conf": d["confusion"], "der": d["diarization error rate"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="HF name or path to a .nemo")
    ap.add_argument("--manifests", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mlflow-experiment", default="Nemotron-diarization-fine-tune")
    ap.add_argument("--batch-size", type=int, default=32, help="files streamed together; 1 leaves the GPU idle")
    a = ap.parse_args()
    try:
        tracking.tracking_uri()
    except tracking.TrackingError as e:
        ap.error(str(e))
    manifests = {m: [json.loads(line) for line in m.read_text().splitlines() if line.strip()] for m in a.manifests}
    if empty := [str(m) for m, rows in manifests.items() if not rows]:
        ap.error(f"empty manifest(s): {', '.join(empty)}")  # diarize([]) would raise after minutes of model loading

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for name in a.models:
        # a .nemo from diar-train nests its evaluation under that training run in MLflow
        id_file = Path(name).parent / tracking.RUN_ID_FILE
        parent = id_file.read_text().strip() if name.endswith(".nemo") and id_file.exists() else None
        with tracking.run(a.mlflow_experiment, name, stage="evaluate", parent_run_id=parent):
            _evaluate(name, manifests, device, results, a.out, a.batch_size)

    print(f"\n{'model':<45} {'latency':<7} {'manifest':<30} {'DER':>7} {'miss':>7} {'FA':>7} {'conf':>7} {'spk acc':>7}")
    for r in results:
        print(f"{r['model'][-45:]:<45} {r['latency']:<7} {_short(r['manifest'])[-30:]:<30} "
              f"{r['der']:7.4f} {r['miss']:7.4f} {r['fa']:7.4f} {r['conf']:7.4f} {r['spk_count_acc']:7.4f}")


def _short(manifest: str) -> str:
    return "/".join(Path(manifest).parts[-2:])  # data/basis/test.json -> basis/test.json


def _evaluate(name: str, manifests: dict, device, results: list[dict], out: Path, batch_size: int) -> None:
    import mlflow
    from nemo.collections.asr.models import SortformerEncLabelModel

    mlflow.log_params({"model": name, "manifests": " ".join(map(str, manifests))[:6000], "batch_size": batch_size})
    load = SortformerEncLabelModel.restore_from if name.endswith(".nemo") else SortformerEncLabelModel.from_pretrained
    model = load(name, map_location=device).float().eval()
    for lat, values in LATENCIES.items():
        for field, v in zip(FIELDS, values):
            setattr(model.sortformer_modules, field, v)
        model._check_streaming_parameters()
        for m, rows in manifests.items():
            # 8 kHz WAVs are resampled to the preprocessor's 16 kHz by NeMo's loader (AudioSegment target_sr).
            hyps = model.diarize(audio=[r["audio_filepath"] for r in rows], batch_size=batch_size)
            sums = {"total": 0.0, "miss": 0.0, "fa": 0.0, "conf": 0.0}
            spk_ok = 0
            for r, lines in zip(rows, hyps, strict=True):
                hyp = parse_segments(lines)
                s = score(read_rttm(r["rttm_filepath"]), hyp)
                for k in sums:
                    sums[k] += s[k]
                spk_ok += len(hyp) == r["num_speakers"]
            t = sums["total"] or float("nan")  # all-silent references: DER is undefined, so NaN
            results.append({
                "model": name, "latency": lat, "manifest": str(m), "files": len(rows), "total_sec": sums["total"],
                "der": (sums["miss"] + sums["fa"] + sums["conf"]) / t,
                "miss": sums["miss"] / t, "fa": sums["fa"] / t, "conf": sums["conf"] / t,
                "spk_count_acc": spk_ok / len(rows),
            })
            out.write_text(json.dumps(results, indent=2))  # after every row, so a crash keeps what was scored
            r = results[-1]
            mlflow.log_metrics(tracking.numeric(
                {k: r[k] for k in ("der", "miss", "fa", "conf", "spk_count_acc")}, f"{lat}/{_short(r['manifest'])}/"))


if __name__ == "__main__":
    main()
