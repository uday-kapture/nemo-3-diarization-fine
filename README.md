# Nemotron 3 Diarization: LoRA fine-tune for 8 kHz calls

This repo LoRA fine-tunes `nvidia/Nemotron-3-Diarization` on Basis Conversations 1500 (Hindi and English). The
Basis audio is first degraded to sound like an 8 kHz phone call. The repo then compares the stock and fine-tuned
models on Basis test mixes and, optionally, on real production calls. The output is a merged `.nemo` that loads
exactly like the stock model, with `SortformerEncLabelModel.restore_from(...)`. For the reasons behind each
choice, see the [design spec](docs/superpowers/specs/2026-11-18-nemotron3-lora-diarization-design.md).

This is an experiment. The stock model was already trained on phone audio. On one real 36 s production call it
scored 1.7 % DER at 1.04 s latency. The before/after evaluation decides whether the fine-tune is worth shipping.

## Data rule

Basis and production audio never leave the VM:
- the Basis licence forbids redistribution;
- production audio is customer data.

Do not push anything under `data/` to a hub, bucket or repo. `data/` and `out/` are git-ignored. Trained
weights contain no audio and may be shared (Basis licence §3).

## VM setup

You need:
- Linux with one NVIDIA GPU (built for an H200);
- an NVIDIA driver ≥ 580: the locked torch 2.15.0 is built for CUDA 13 (13.2 runtime); `nvidia-smi` shows the
  driver version;
- about 150 GB of free disk;
- outbound access to `github.com` and `huggingface.co`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # skip if uv is installed
git clone <repo-url> nemo-3-diarization && cd nemo-3-diarization
uv sync --extra train
```

`--extra train` installs NeMo from the NVIDIA-NeMo/Speech GitHub repo, pinned to a commit. PyPI
`nemo-toolkit` 3.0.0 cannot load this model's rope attention, which is why the VM needs GitHub access. uv
installs Python 3.12 itself.

A plain `uv sync` is enough for the unit tests, but it also **removes** NeMo. If you run it, run
`uv sync --extra train` again before training. `uv run` leaves NeMo in place.

Check the install:

```bash
uv run pytest
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

What good output looks like:
- `pytest` ends with `36 passed`. A silero `FutureWarning` and pyannote `'uem' was approximated` warnings are
  normal.
- The second command prints `True NVIDIA H200 ...`.

**Hugging Face login.** Basis (`basis-ai/basis-conversations-1500`) is gated. Log in with a token from an
account that has been granted access; you can request access on the dataset page. The model
`nvidia/Nemotron-3-Diarization` is public.

```bash
uv run hf auth login        # or: uvx --from huggingface_hub hf auth login
```

## Steps

### 1. Prepare Basis

Run a smoke first. It downloads 3 Hindi segments:

```bash
uv run diar-prepare --langs hi --limit-segments 3 --out-dir data/smoke
```

Before the full run, check that the labels line up with the real audio. Listen to one `data/smoke/wav/*.wav`
next to its `data/smoke/rttm/*.rttm`, or run this check:

```bash
uv run python - <<'EOF'
import json, numpy as np, soundfile as sf
from nemo_3_diarization.labels import merge, read_rttm
rms = lambda z: np.sqrt(np.mean(z ** 2)) + 1e-12
for line in open("data/smoke/train.json"):
    r = json.loads(line)
    y, sr = sf.read(r["audio_filepath"])
    inside = np.zeros(len(y), bool)
    for s, e in merge([iv for v in read_rttm(r["rttm_filepath"]).values() for iv in v]):
        inside[int(s * sr):int(e * sr)] = True
    print(f"{20 * np.log10(rms(y[inside]) / rms(y[~inside])):5.1f} dB  {r['audio_filepath']}")
EOF
```

It prints, per smoke mix, how much louder the audio is inside the RTTM speech than outside it. Every line should
be ≥ 10 dB. Lower values mean the labels are shifted or scaled against the audio: stop and investigate before the
full run. `nan` means the window has no non-speech; judge by the other lines.

Then the full run:

```bash
uv run diar-prepare --langs hi en --workers $(nproc)
```

This downloads about 60 GB (356 h of audio) to `data/basis-raw/`. It writes about 20 GB of 8 kHz mono WAV plus
RTTM labels to `data/basis/`, with these manifests:

| Manifest | Contents |
|---|---|
| `train.json`, `val.json`, `test.json` | 105 s mixes of the 2 most active speakers, split about 90/5/5 by conversation |
| `test_all.json` | the same test windows with every active speaker mixed in; catches a model that learned to stop at 2 speakers |

The run ends with one count line per split and a failure count:

```
test  all  <n>
test  pair <n>
train pair <n>
val   pair <n>
failed segments: <k>
```

- The spec estimated about 11.7k windows in total.
- A segment that fails prints `FAILED <segment_id>: <error>` and is skipped; the run continues. A handful is
  fine. Many failures point to a real problem, such as a full disk or a broken download.
- Re-running is safe: downloads are cached in `data/basis-raw/`, and outputs are overwritten.
- The smoke may have no `val` or `test` lines. That is fine, because the smoke steps below use only
  `data/smoke/train.json`.

### 2. Optional: production test set

This step needs read access to the `kapturecx-ml` GCS bucket. The input is stereo 8 kHz WAV production calls,
with one party per channel. Copy them under `data/` so they stay git-ignored:

```bash
mkdir -p data/prod-raw
gsutil -m cp 'gs://kapturecx-ml/<path>/*.wav' data/prod-raw/
uv run diar-prod-testset --in-dir data/prod-raw --out-dir data/prod
```

The script labels each channel with speech detection, sums the channels to mono as production does, and writes
`data/prod/test.json`.

Good output ends with `recordings: <n>` and `failed recordings: 0`. A mono file or one that is not 8 kHz prints
`FAILED <name>: ValueError('expected 2 channels')` or `ValueError('expected 8000 Hz')` and is skipped.

If you skip this step, drop `data/prod/test.json` from the evaluate commands below.

### 3. Baseline: evaluate the stock model

```bash
uv run diar-evaluate --models nvidia/Nemotron-3-Diarization \
  --manifests data/basis/test.json data/basis/test_all.json data/prod/test.json \
  --out out/baseline.json
```

How it scores:
- It runs every manifest at the two streaming latencies from the model card: 1.04 s, which is the ship gate, and
  0.32 s, which is reported too.
- DER uses a ±0.25 s collar with overlapped speech included, the model card's CALLHOME protocol.
- It runs on the GPU in fp32, one file at a time.

It prints one row per model, latency and manifest, and writes the same rows as JSON to `--out`. The numbers are
fractions: `0.0174` means 1.74 %. `spk acc` is the share of files where the model found the right number of
speakers. Example from the one production call above:

```
model                                         latency manifest                           DER    miss      FA    conf spk acc
nvidia/Nemotron-3-Diarization                 1.04s   prod/test.json                  0.0174  0.0174  0.0000  0.0000  1.0000
nvidia/Nemotron-3-Diarization                 0.32s   prod/test.json                  0.0221  0.0221  0.0000  0.0000  1.0000
```

### 4. Train

Training is GPU-only. `--accelerator cpu` does not train, because FlexAttention has no CPU backward pass.

Run a smoke first. It checks the GPU path, memory and the export in a few minutes:

```bash
uv run diar-train --train-manifest data/smoke/train.json --val-manifest data/smoke/train.json \
  --out-dir out/smoke --max-steps 20 --val-every 10
uv run python -c "from nemo.collections.asr.models import SortformerEncLabelModel as M; M.restore_from('out/smoke/nemotron3-diar-lora-merged.nemo')"
```

What good smoke output looks like:
- `LoRA targets (125)`, followed by the encoder linears;
- `trainable params: 4,087,808 / 103,318,355 (3.96 %)`;
- a `step <n> loss <x>` line per step;
- `'val_loss' reached ...` after each validation;
- `best checkpoint: ...`;
- the two files in `out/smoke/`;
- the `restore_from` command exits without error.

Keep `--max-steps` ≥ `--val-every`. With fewer steps no validation runs, so there is no best checkpoint: the script
prints `best checkpoint: NONE - exporting last weights` and exports the last weights. If training crashes or you
press Ctrl-C, the script still writes both output files, from the best checkpoint so far or else the last weights,
before it exits with the error.

Then the full run:

```bash
uv run diar-train --train-manifest data/basis/train.json --val-manifest data/basis/val.json
```

Defaults:

| Flag | Default |
|---|---|
| `--out-dir` | `out` |
| `--max-steps` | 3000 |
| `--val-every` | 250 |
| `--batch-size` | 32 |
| `--num-workers` | 16 |
| `--lr` | 2e-4 |
| `--rank` | 16 |
| `--alpha` | 32 |
| `--accelerator` | `gpu` |
| `--precision` | `bf16-mixed` |

What the run does:
- It freezes the base model and trains LoRA on every linear layer of the encoder.
- It validates every `--val-every` steps and keeps the checkpoint with the best `val_loss`.

`val_loss` should fall and then level off. If it is still falling at the last step, rerun with a larger
`--max-steps`. If you hit CUDA out of memory, lower `--batch-size`.

Outputs:
- `out/nemotron3-diar-lora-merged.nemo`: LoRA merged into the base weights. It is fp32, about 400 MB, and loads
  with `SortformerEncLabelModel.restore_from` exactly like stock. This is the file to ship.
- `out/lora_adapter.pt`: the LoRA weights on their own, about 16 MB.
- `out/ckpt/`: the best Lightning checkpoint.
- `out/lightning_logs/`: the Lightning logs.

### 5. Evaluate stock vs fine-tuned

Score only the merged model, with the same `--manifests` as step 3:

```bash
uv run diar-evaluate --models out/nemotron3-diar-lora-merged.nemo \
  --manifests data/basis/test.json data/basis/test_all.json data/prod/test.json \
  --out out/finetuned.json
```

Then compare it with the step-3 baseline and apply the ship rule below:

```bash
uv run python - <<'EOF'
import json
load = lambda p: {(r["latency"], r["manifest"]): r["der"] for r in json.load(open(p))}
stock, ft = load("out/baseline.json"), load("out/finetuned.json")
assert stock.keys() == ft.keys(), "both runs must finish on the same --manifests"
ship = True
for (lat, m), s in sorted(stock.items()):
    ok = ft[lat, m] <= s  # a NaN DER never passes
    ship &= ok or lat != "1.04s"
    print(f"{lat:<6} {m:<28} stock {s:.4f}  merged {ft[lat, m]:.4f}  {'PASS' if ok else 'FAIL'}"
          + ("" if lat == "1.04s" else "  (not gated)"))
print("SHIP" if ship else "DO NOT SHIP: keep the stock model")
EOF
```

It prints one line per latency and manifest and ends with `SHIP` only if the merged DER at 1.04 s is no worse
than stock's on every manifest. The 0.32 s lines are for information. Put both printed tables and this output in
the PR description.

`diar-evaluate` writes its JSON after every scored row, so a crashed run keeps what it finished; the `assert`
above stops on such a partial file. A DER of `nan` means that manifest's references hold no speech.

## Ship rule

Ship `out/nemotron3-diar-lora-merged.nemo` only if its DER at **1.04 s** is no worse than stock's on **every**
manifest: Basis 2-speaker, Basis all-speaker, and production when present. Otherwise keep the stock model.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync --extra train` cannot clone NeMo | The VM needs outbound access to github.com. |
| Gated-repo, 401 or 403 error from `diar-prepare` | The logged-in HF account has no Basis access. Request access, then run `uv run hf auth login` again. |
| `ModuleNotFoundError: No module named 'nemo'` | Run `uv sync --extra train`. A plain `uv sync` removes NeMo. |
| CUDA out of memory in `diar-train` | Lower `--batch-size`. |
| `torch.cuda.is_available()` prints `False` | Run `nvidia-smi`. If it fails or shows a driver below 580, install driver ≥ 580 (the locked torch needs CUDA 13) and reboot. |
