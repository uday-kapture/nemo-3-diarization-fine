# Nemotron 3 Diarization LoRA Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking.

**Goal:** Build a pipeline that turns Basis Conversations (Hindi and English) into 8 kHz 2-speaker phone-call
training data, LoRA fine-tunes `nvidia/Nemotron-3-Diarization` with NeMo, and scores stock vs fine-tuned.

**Architecture:**
- Four CLIs live in one package: `diar-prepare`, `diar-prod-testset`, `diar-train` and `diar-evaluate`.
- Pure logic is kept separate from IO, so it is unit-tested on a Mac without NeMo or a GPU:
  - interval maths, windows, splits and RTTM in `labels.py`;
  - phone channel, mixing and VAD in `audio.py`;
  - LoRA inject and merge in `lora.py`;
  - scoring in `evaluate.py`.
- NeMo is only imported inside `train.py` and `evaluate.main`, and runs on the H200 VM.

**Tech Stack:** Python 3.12, uv, NeMo (`nemo-toolkit[asr]>=3.0.0`, `train` extra), PEFT, silero-vad, soundfile,
scipy, numpy, pyannote.metrics, huggingface_hub, pytest.

**Spec:** `docs/superpowers/specs/2026-11-18-nemotron3-lora-diarization-design.md`

## Global Constraints

- **Python:** `requires-python = ">=3.12,<3.13"`, and `.python-version` is `3.12`.
- **Model:** `nvidia/Nemotron-3-Diarization`, `SortformerEncLabelModel`, 16 kHz input (NeMo resamples 8 kHz on
  load), max 8 speakers.
- **Windows:** 105 s, the checkpoint's `session_len_sec`. Drop any window touching a `muted_span`. Keep the top
  2 speakers by labeled speech, each needing at least 5 s.
- **Speaker activity:** Silero VAD ∩ (transcript rows padded ±0.3 s, merged) ∩ `on_call`.
- **Phone channel:** band-pass 300–3400 Hz (4th-order Butterworth), resample to 8 kHz, G.711 A-law (A=87.6,
  8-bit) round trip. Applied per track, before mixing.
- **Mix:** set each track to −26 dBFS RMS over its own speech, sum, then scale to peak 0.9 only if the peak is
  above 0.9. Write 8 kHz 16-bit PCM WAV.
- **Split:** by `sha1(conversation_id)` mod 100: <90 train, <95 val, else test.
- **LoRA:** all `nn.Linear` in `model.encoder`, r=16, alpha=32, dropout 0.05. Everything else is frozen.
  Start at lr 2e-4 with 200 warmup steps.
- **Evaluation:**
  - DER with ±0.25 s collar (pyannote `collar=0.5`, which is total width), overlap included.
  - Gate latency 1.04 s (`264,264,9,4,222`); also report 0.32 s (`264,264,3,1,222`).
  - Order is `spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period`.
- **Licence:** Basis and production audio never leave the VM. `data/` and `out/` are git-ignored.

## Review Focus

1. **A segment with fewer than 2 speakers who have ≥5 s of speech in every window.** `prepare` yields no rows
   for it and does not crash. Test: `test_process_segment_single_speaker_yields_nothing` (Task 3).
2. **A chosen track that is silent or near-silent in the window.** The mix scales it by 1, not by inf or NaN.
   Test: `test_mix_silent_track_no_nan` (Task 2).
3. **A window touching a muted span.** It is dropped even if the overlap is tiny. Test:
   `test_windows_drop_muted` (Task 1).
4. **The model predicts no speech at all for a file.** Scoring reports DER = 1.0 (all missed) instead of
   crashing. Test: `test_score_empty_hypothesis` (Task 7).
5. **A production recording that is mono, or has one silent channel.** A mono file raises
   `ValueError("expected 2 channels")`. A silent channel gives `num_speakers: 1`. Tests: Task 4.

---

### Task 1: Project setup + `labels.py` (intervals, windows, split, RTTM)

**Files:**
- Modify: `pyproject.toml`, `.python-version`, `.gitignore`, `src/nemo_3_diarization/__init__.py` (make empty)
- Create: `src/nemo_3_diarization/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Produces, with `Intervals = list[tuple[float, float]]` (sorted, non-overlapping, seconds):
  - `merge(iv: Intervals, pad: float = 0.0) -> Intervals`: pads each side, clamps start at 0, merges overlaps.
  - `intersect(a: Intervals, b: Intervals) -> Intervals`
  - `total(iv: Intervals, start: float, end: float) -> float`: length inside `[start, end)`.
  - `speaker_activity(vad: Intervals, rows: Intervals, on_call: Intervals, pad: float = 0.3) -> Intervals`
  - `windows(duration: float, muted: Intervals, win: float = 105.0) -> Intervals`: consecutive full windows
    (the tail is dropped), minus any window that touches a muted span.
  - `top_speakers(activity: dict[str, Intervals], start: float, end: float, n: int, min_speech: float) -> list[str]`:
    up to `n` speakers with at least `min_speech` of speech, sorted by speech descending, then by id.
  - `rttm_lines(file_id: str, activity: dict[str, Intervals], start: float, end: float, speakers: list[str]) -> list[str]`:
    clips to the window and shifts times to be window-relative. Format:
    `SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>`.
  - `read_rttm(path: str | Path) -> dict[str, Intervals]`
  - `split_of(conversation_id: str) -> str`: one of `"train"`, `"val"` or `"test"`.

- [ ] **Step 1: Branch, pin Python, dependencies, ignore data**

```bash
git checkout -b feat/lora-pipeline
echo 3.12 > .python-version
printf 'data/\nout/\n' >> .gitignore
```

- **`pyproject.toml`:**
  - set `requires-python = ">=3.12,<3.13"`;
  - set `dependencies = ["numpy", "scipy", "soundfile", "silero-vad", "huggingface_hub", "peft", "torch", "pyannote.metrics"]`;
  - add `[project.optional-dependencies] train = ["nemo-toolkit[asr]>=3.0.0"]`;
  - add `[dependency-groups] dev = ["pytest"]`;
  - delete the `nemo-3-diarization` script entry. Each later task adds its own script.
- **`__init__.py`:** empty it.

Run: `uv sync && uv run python -c "import peft, silero_vad, soundfile, scipy, pyannote.metrics"`
Expected: exits 0.

- [ ] **Step 2: Write the failing tests** in `tests/test_labels.py`

```python
from nemo_3_diarization.labels import *

def test_merge_pads_and_joins():
    assert merge([(1.0, 2.0), (2.5, 3.0)], pad=0.3) == [(0.7, 3.3)]
    assert merge([(0.1, 0.2)], pad=0.3) == [(0.0, 0.5)]

def test_intersect():
    assert intersect([(0, 5), (8, 10)], [(4, 9)]) == [(4, 5), (8, 9)]

def test_speaker_activity_requires_all_three():
    assert speaker_activity(vad=[(0, 10)], rows=[(2, 3)], on_call=[(0, 2.5)], pad=0.3) == [(1.7, 2.5)]

def test_windows_drop_muted():
    assert windows(230.0, muted=[], win=105.0) == [(0.0, 105.0), (105.0, 210.0)]
    assert windows(230.0, muted=[(104.99, 105.0)], win=105.0) == [(105.0, 210.0)]

def test_top_speakers():
    act = {"a": [(0, 10)], "b": [(10, 30)], "c": [(30, 33)]}
    assert top_speakers(act, 0, 105, n=2, min_speech=5) == ["b", "a"]
    assert top_speakers(act, 0, 105, n=8, min_speech=0.5) == ["b", "a", "c"]

def test_rttm_roundtrip(tmp_path):
    lines = rttm_lines("w1", {"a": [(100, 110)], "b": [(200, 220)]}, 105, 210, ["a", "b"])
    assert lines[0] == "SPEAKER w1 1 0.000 5.000 <NA> <NA> a <NA> <NA>"
    (tmp_path / "w1.rttm").write_text("\n".join(lines) + "\n")
    assert read_rttm(tmp_path / "w1.rttm") == {"a": [(0.0, 5.0)], "b": [(95.0, 105.0)]}

def test_split_is_deterministic_and_proportional():
    s = [split_of(f"conv{i}") for i in range(10000)]
    assert s == [split_of(f"conv{i}") for i in range(10000)]
    assert 0.87 < s.count("train") / 10000 < 0.93
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: nemo_3_diarization.labels`.

- [ ] **Step 4: Implement `labels.py`** with the signatures above. Use plain Python, with no numpy for intervals.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src tests docs
git commit -m "feat: project setup and label/window utilities"
```

---

### Task 2: `audio.py` (A-law, phone channel, mix, VAD)

**Files:**
- Create: `src/nemo_3_diarization/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `Intervals` from Task 1.
- Produces:
  - `alaw_roundtrip(x: np.ndarray) -> np.ndarray`: float in [-1, 1], same shape.
  - `phone_channel(x: np.ndarray, sr: int = 48000) -> np.ndarray`: float32 at 8 kHz.
  - `mix(tracks: list[np.ndarray], speech: list[Intervals], sr: int = 8000) -> np.ndarray`: float32. The
    `speech` intervals are window-relative.
  - `vad(x: np.ndarray, sr: int) -> Intervals`: Silero. Resamples to 16 kHz with `scipy.signal.resample_poly`,
    processes 600 s blocks with time offsets, then `merge()`s the result. The model loads lazily once per
    process.

- [ ] **Step 1: Write the failing tests** in `tests/test_audio.py`

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: nemo_3_diarization.audio`.

- [ ] **Step 3: Implement `audio.py`**

**`alaw_roundtrip`** (A = 87.6, k = 1 + ln A). This is the continuous A-law curve with 8-bit sign-magnitude
quantisation:

```
a = |x|
y = where(a < 1/A, A*a/k, (1 + ln(max(A*a, 1)))/k)
y = round(y * 127) / 127
a' = where(y < 1/k, y*k/A, exp(y*k - 1)/A)
return sign(x) * a'
```

**`phone_channel`:** `butter(4, [300, 3400], btype="band", fs=sr, output="sos")` + `sosfiltfilt`, then
`resample_poly(x, 8000, sr)`, then `alaw_roundtrip`, then `.astype(np.float32)`.

**`mix`:**
- Scale each track to RMS `10**(-26/20)`, measured over its `speech` samples.
- If the track has no speech samples, or that RMS is below 1e-6, scale it by 1.
- Sum the tracks.
- If the peak is above 0.9, scale the sum to peak 0.9.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nemo_3_diarization/audio.py tests/test_audio.py
git commit -m "feat: phone channel simulation, mixing and VAD"
```

---

### Task 3: `prepare.py` (Basis → 8 kHz 2-speaker sessions + manifests)

**Files:**
- Create: `src/nemo_3_diarization/prepare.py`
- Modify: `pyproject.toml` (add `diar-prepare = "nemo_3_diarization.prepare:main"`)
- Test: `tests/test_prepare.py`

**Interfaces:**
- Consumes: everything in `labels.py` and `audio.py`.
- Produces:
  - `process_segment(seg: dict, root: Path, out: Path, vad_fn=vad, win: float = 105.0) -> list[dict]`
    - `seg` is one `segments.jsonl` row. `root` is the download dir that `seg["tracks"]` paths are relative to.
    - Writes `out/wav/{id}.wav` and `out/rttm/{id}.rttm`.
    - Returns manifest rows:
      `{"audio_filepath", "offset": 0, "duration", "label": "infer", "text": "-", "num_speakers", "rttm_filepath", "uem_filepath": None, "split", "kind"}`.
    - `kind` is `"pair"` (top 2 speakers) or `"all"` (every speaker with ≥0.5 s of speech, up to 8).
      `"all"` rows are written only when `split == "test"`.
    - `id` is `f"{segment_id}_{int(start)}"`, with an `_all` suffix for `kind == "all"`.
  - **`diar-prepare` CLI:** `--langs hi en --raw-dir data/basis-raw --out-dir data/basis --workers 16 --limit-segments 0`.
    It writes `out/{train,val,test}.json` (pair rows) and `out/test_all.json` (all rows) as JSON lines.

- [ ] **Step 1: Write the failing tests** in `tests/test_prepare.py`
  - Build a fake segment in `tmp_path` with 3 tracks of 48 kHz FLAC, each 23 s long:
    - speaker `a`: noise bursts at 0–5.5 s and 12–16 s;
    - speaker `b`: bursts at 4.5–10 s and 17–21 s. Each speaker gets at least 5 s of speech in window 0–10,
      with 1 s of overlap;
    - speaker `c`: silent.
  - Transcript rows cover the bursts. `on_call` is the full length for everyone. One `muted_span` is at
    `(15.0, 15.5)`.
  - `vad_fn` is a fake: 50 ms energy frames above 0.01 RMS, merged.

```python
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
```

`find_conv(split)` returns the first `f"c{i}"` whose `split_of` is `split`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: nemo_3_diarization.prepare`.

- [ ] **Step 3: Implement `process_segment` and `main`**

**`process_segment`:**
1. Run VAD on each track. Read the track in 600 s blocks at 48 kHz, so memory stays bounded.
2. Build `speaker_activity` per speaker.
3. For each window from `windows()`, pick the speakers with `top_speakers()`.
4. For each speaker, read only the window span from the FLAC (`sf.read(start=, stop=)`), then apply
   `phone_channel`.
5. `mix` the tracks with their window-relative speech, then write the WAV and RTTM.

**`main`:**
1. `snapshot_download(repo_id="basis-ai/basis-conversations-1500", repo_type="dataset", local_dir=raw_dir, allow_patterns=[f"{l}/segments.jsonl" for l in langs])`.
2. Load the segments. Apply `--limit-segments` when it is above 0.
3. Make a second `snapshot_download` with `allow_patterns` set to those segments' track paths.
4. Run `multiprocessing.Pool(workers).imap_unordered` over the segments, with `torch.set_num_threads(1)` in the
   worker initializer.
5. Write the manifests and print counts per split and kind.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prepare.py -v`
Expected: 3 passed.

- [ ] **Step 5: Real-data smoke on the Mac**

Run: `uv run diar-prepare --langs hi --limit-segments 1 --workers 1 --out-dir data/smoke`
Expected:
- the counts print, with more than 0 pair rows;
- `data/smoke/rttm/*.rttm` is non-empty;
- opening one WAV next to its RTTM, the speech lines up with the RTTM times. Listen to it.

- [ ] **Step 6: Commit**

```bash
git add src/nemo_3_diarization/prepare.py tests/test_prepare.py pyproject.toml
git commit -m "feat: Basis to 8 kHz 2-speaker sessions with RTTM manifests"
```

---

### Task 4: `prod_testset.py` (stereo production recordings → mono-mix test set)

**Files:**
- Create: `src/nemo_3_diarization/prod_testset.py`
- Modify: `pyproject.toml` (add `diar-prod-testset = "nemo_3_diarization.prod_testset:main"`)
- Test: `tests/test_prod_testset.py`

**Interfaces:**
- Consumes: `vad`, `merge`, `rttm_lines`.
- Produces:
  - `process_recording(path: Path, out: Path, vad_fn=vad) -> dict`
    - Raises `ValueError("expected 2 channels")` unless the file is stereo.
    - Speakers are `"ch0"` and `"ch1"`.
    - Mono = `ch0 + ch1`, scaled to peak 0.99 only if it would clip, written as 8 kHz 16-bit WAV.
    - Returns a manifest row with the same keys as Task 3, plus `"split": "test", "kind": "prod"`, and
      `num_speakers` = the number of channels with any speech.
  - **`diar-prod-testset` CLI:** `--in-dir <dir of stereo wavs> --out-dir data/prod`. Writes
    `data/prod/test.json`.

- [ ] **Step 1: Write the failing tests**

```python
def test_stereo_to_mono_with_two_speakers(tmp_path): ...   # ch0 bursts 0–2 s, ch1 bursts 3–5 s
    # assert row["num_speakers"] == 2; sf.info(mono).channels == 1 and samplerate == 8000
    # assert read_rttm(row["rttm_filepath"]) has ch0 ≈ [(0, 2)], ch1 ≈ [(3, 5)] within 0.1 s
def test_silent_channel_counts_one_speaker(tmp_path): ...  # assert row["num_speakers"] == 1
def test_mono_input_rejected(tmp_path): ...                # pytest.raises(ValueError, match="expected 2 channels")
```

The tests use the same `fake_vad` as Task 3. Move it to `tests/conftest.py`.

- [ ] **Step 2: Run to verify they fail.** Run `uv run pytest tests/test_prod_testset.py -v`, expecting
  `ModuleNotFoundError`.
- [ ] **Step 3: Implement `process_recording` and `main`.**
- [ ] **Step 4: Run to verify they pass.** Expected: 3 passed.
- [ ] **Step 5: Commit** with `git commit -m "feat: production stereo recordings to mono-mix test set"`.

---

### Task 5: `lora.py` (inject, merge, save adapter)

**Files:**
- Create: `src/nemo_3_diarization/lora.py`
- Test: `tests/test_lora.py`

**Interfaces:**
- Produces:
  - `add_lora(module: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> list[str]`
    - Targets every `nn.Linear` in `module`, by unique last name component.
    - Uses `peft.inject_adapter_in_model(LoraConfig(...), module)`.
    - Sets `requires_grad=False` on every non-`lora_` parameter of `module`.
    - Returns the target names.
  - `merge_lora(module: nn.Module) -> None`: for each `peft.tuners.lora.LoraLayer`, call `.merge()`, then
    replace it in its parent with `.get_base_layer()`.
  - `lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]`: `peft.get_peft_model_state_dict(module)`.

- [ ] **Step 1: Write the failing tests**

```python
import torch, torch.nn as nn
from nemo_3_diarization.lora import add_lora, merge_lora, lora_state_dict

def toy():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

def test_only_lora_trainable():
    m = toy(); add_lora(m)
    assert all("lora_" in n for n, p in m.named_parameters() if p.requires_grad)
    assert any(p.requires_grad for p in m.parameters())

def test_merge_preserves_output_and_removes_lora():
    m = toy(); add_lora(m)
    for n, p in m.named_parameters():
        if "lora_B" in n: nn.init.normal_(p)
    x = torch.randn(5, 4); y = m(x)
    assert len(lora_state_dict(m)) > 0
    merge_lora(m)
    assert torch.allclose(m(x), y, atol=1e-5)
    assert not any("lora" in k for k in m.state_dict())
    assert all(type(l) in (nn.Linear, nn.ReLU) for l in m)
```

- [ ] **Step 2: Run to verify they fail.** Run `uv run pytest tests/test_lora.py -v`, expecting
  `ModuleNotFoundError`.
- [ ] **Step 3: Implement `lora.py`.**
- [ ] **Step 4: Run to verify they pass.** Expected: 2 passed.
- [ ] **Step 5: Commit** with `git commit -m "feat: LoRA inject/merge helpers"`.

---

### Task 6: `train.py` (NeMo LoRA training → merged `.nemo`)

**Files:**
- Create: `src/nemo_3_diarization/train.py`
- Modify: `pyproject.toml` (add `diar-train = "nemo_3_diarization.train:main"`)

**Interfaces:**
- Consumes: `add_lora`, `merge_lora`, `lora_state_dict`, and manifests from Task 3.
- Produces:
  - `out/nemotron3-diar-lora-merged.nemo`, loadable with `SortformerEncLabelModel.restore_from`;
  - `out/lora_adapter.pt`;
  - Lightning checkpoints in `out/ckpt/`.
- **CLI:** `--train-manifest data/basis/train.json --val-manifest data/basis/val.json --out-dir out --max-steps 3000 --val-every 250 --batch-size 32 --num-workers 16 --lr 2e-4 --rank 16 --alpha 32`.

- [ ] **Step 1: Implement `main()` in this order.** The order matters: LoRA must exist before the optimizer is
  built.
  1. `trainer = lightning.pytorch.Trainer(devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=..., val_check_interval=val_every, check_val_every_n_epoch=None, callbacks=[ModelCheckpoint(dirpath=out/"ckpt", monitor="val_loss", save_top_k=1)])`.
     After the first validation, confirm that the model logs `val_loss`. If the key differs, use the logged one.
  2. `model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")`, then
     `model.set_trainer(trainer)`.
  3. Freeze the whole model. Then call `add_lora(model.encoder, r, alpha)`.
  4. Under `open_dict(model.cfg)`, set `train_ds.manifest_filepath`, `validation_ds.manifest_filepath`,
     `batch_size` and `num_workers`. Then call `model.setup_training_data(model.cfg.train_ds)` and
     `model.setup_validation_data(model.cfg.validation_ds)`.
  5. `model.setup_optimization({"name": "adamw", "lr": lr, "betas": [0.9, 0.98], "weight_decay": 0.001, "sched": {"name": "InverseSquareRootAnnealing", "warmup_steps": 200, "min_lr": 1e-6}})`.
  6. Print the trainable parameter count. Expect about 1–2 % of the total.
  7. Run `trainer.fit(model)`.
  8. Load the best checkpoint's `state_dict` into `model`.
  9. Save `torch.save(lora_state_dict(model.encoder), out/"lora_adapter.pt")`.
  10. Run `merge_lora(model.encoder)`, then `model.save_to(out/"nemotron3-diar-lora-merged.nemo")`.

- [ ] **Step 2: VM smoke run.** The user runs this on the H200; this session has no VM access.
  Run: `uv sync --extra train && uv run diar-prepare --langs hi --limit-segments 3 --out-dir data/smoke && uv run diar-train --train-manifest data/smoke/train.json --val-manifest data/smoke/train.json --out-dir out/smoke --max-steps 20 --val-every 10`
  Expected:
  - the loss prints and falls or holds;
  - `out/smoke/nemotron3-diar-lora-merged.nemo` exists;
  - `python -c "from nemo.collections.asr.models import SortformerEncLabelModel as M; M.restore_from('out/smoke/nemotron3-diar-lora-merged.nemo')"`
    loads with no missing or unexpected keys.

- [ ] **Step 3: Commit** with `git commit -m "feat: NeMo LoRA training with merged .nemo export"`.

---

### Task 7: `evaluate.py` (DER stock vs fine-tuned)

**Files:**
- Create: `src/nemo_3_diarization/evaluate.py`
- Modify: `pyproject.toml` (add `diar-evaluate = "nemo_3_diarization.evaluate:main"`)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `read_rttm`, and the manifests from Tasks 3 and 4.
- Produces:
  - `LATENCIES = {"1.04s": (264, 264, 9, 4, 222), "0.32s": (264, 264, 3, 1, 222)}`
  - `parse_segments(lines: list[str]) -> dict[str, Intervals]`: parses `diarize()` output strings of the form
    `"start end speaker_k"`.
  - `score(ref: dict[str, Intervals], hyp: dict[str, Intervals]) -> dict[str, float]`
    - Keys: `total`, `miss`, `fa`, `conf` in seconds, plus `der`.
    - Uses `pyannote.metrics.diarization.DiarizationErrorRate(collar=0.5, skip_overlap=False)(..., detailed=True)`.
  - **CLI:** `--models nvidia/Nemotron-3-Diarization out/nemotron3-diar-lora-merged.nemo --manifests data/basis/test.json data/basis/test_all.json [data/prod/test.json] --out out/results.json`.
    - A model path ending in `.nemo` loads with `restore_from`; anything else loads with `from_pretrained`.
    - For each latency, set the five `model.sortformer_modules` fields, then call `_check_streaming_parameters()`.
    - Run `model.diarize(audio=[paths], batch_size=1)`.
    - Aggregate DER as Σ(miss + fa + conf) / Σtotal per model × latency × manifest. Also report speaker-count
      accuracy against `num_speakers`.
    - Print a table and write the JSON.

- [ ] **Step 1: Write the failing tests**

```python
from nemo_3_diarization.evaluate import score, parse_segments

REF = {"a": [(0.0, 5.0)], "b": [(5.0, 10.0)]}

def test_score_perfect_and_permuted():
    assert score(REF, REF)["der"] == 0.0
    assert score(REF, {"x": [(0.0, 5.0)], "y": [(5.0, 10.0)]})["der"] == 0.0

def test_score_collar_forgives_quarter_second():
    assert score(REF, {"x": [(0.0, 5.2)], "y": [(5.2, 10.0)]})["der"] == 0.0

def test_score_empty_hypothesis():
    s = score(REF, {})
    assert s["der"] == 1.0 and s["miss"] > 0

def test_parse_segments():
    assert parse_segments(["0.00 1.50 speaker_0", "2.00 3.00 speaker_1"]) == {
        "speaker_0": [(0.0, 1.5)], "speaker_1": [(2.0, 3.0)]}
```

- [ ] **Step 2: Run to verify they fail.** Run `uv run pytest tests/test_evaluate.py -v`, expecting
  `ModuleNotFoundError`.
- [ ] **Step 3: Implement `evaluate.py`.** Import NeMo inside `main()` only.
- [ ] **Step 4: Run to verify they pass.** Expected: 4 passed.
- [ ] **Step 5: VM smoke (user).** Run
  `uv run diar-evaluate --models nvidia/Nemotron-3-Diarization out/smoke/nemotron3-diar-lora-merged.nemo --manifests data/smoke/train.json --out out/smoke/results.json`.
  Expected: a table with 2 models × 2 latencies, and DER values between 0 and 1.
- [ ] **Step 6: Commit** with `git commit -m "feat: stock vs fine-tuned DER evaluation"`.

---

### Task 8: README runbook + full VM run

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`** with the following sections:
  - What this is, with one paragraph and a link to the spec.
  - **VM setup:** `uv sync --extra train`; `hf auth login` with a token that has Basis access.
  - **Steps:**
    1. `diar-prepare --langs hi en --workers <cpu count>`. Expect about 60 GB of download and about 20 GB of
       output.
    2. Optional: `gsutil -m cp gs://kapturecx-ml/<path>/*.wav prod-raw/`, then
       `diar-prod-testset --in-dir prod-raw`.
    3. `diar-evaluate` on stock, to get the baseline.
    4. `diar-train`.
    5. `diar-evaluate` on stock and merged.
  - **Ship rule:** ship the merged model only if its DER at 1.04 s is no worse than stock on every manifest.
  - **Data rule:** Basis and production audio never leave the VM (licence).

- [ ] **Step 2: Run the full test suite.** Run `uv run pytest -v`, expecting 24 passed.
- [ ] **Step 3: Commit** with `git commit -m "docs: VM runbook"`.
- [ ] **Step 4: Full run on the H200 (user).** Follow the README. Record the baseline and fine-tuned tables
  from `out/results.json` in the PR description.
