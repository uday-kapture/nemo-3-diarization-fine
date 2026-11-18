# Nemotron 3 Diarization: LoRA fine-tune on Basis Conversations for 8 kHz calls

Date: 2026-11-18. Status: draft for approval.

## Intent

Kapture runs speaker diarization ("who spoke when") on 8 kHz production calls. The goal is to LoRA fine-tune
`nvidia/Nemotron-3-Diarization` so it does better on that audio. Training data is Basis Conversations 1500
(`basis-ai/basis-conversations-1500`), made to sound like phone calls.

**Production audio**, from samples in `kapturecx/call-transcript-intent-data-v2`:
- Hindi and Hinglish loan and EMI calls between a Pipecat voice bot and a customer.
- Recordings are 8 kHz 16-bit PCM WAV, stereo, one party per channel, with no bleed between channels.
- Real bandwidth is under 3.4 kHz.
- The model runs **live, at low latency, on the mono mix** of the two channels.
- Production loads the `.nemo` file.

This is an experiment. The stock model already trained on 8 kHz phone calls (Fisher, CALLHOME) and Hindi
(DISPLACE). On 2-speaker CALLHOME it scores 5.98 % DER at 30.4 s latency and 7.75 % at 0.32 s. The user chose
to try Basis anyway. Whether
the fine-tune helps is decided by the before/after evaluation, not assumed.

## Goals

- One reproducible pipeline: download → speaker labels → 8 kHz 2-speaker mixes → NeMo manifests →
  LoRA training → merged `.nemo` → before/after scores.
- Runs end to end on one H200 VM.
- The merged checkpoint loads exactly like the stock one (`SortformerEncLabelModel.restore_from(...)`), so
  nothing downstream changes.

## Non-goals

- Training through the Transformers port (it is inference-only: no loss).
- Full fine-tuning, hyperparameter search, multi-GPU or multi-node training.
- Serving or deployment of the model.
- Languages other than Hindi and English in this first run.

## Anti-goals

- Shipping a merged model that scores worse than stock on any test set at the gate latency.
- Redistributing Basis audio: the licence forbids it. Prepared data stays on the VM and is never pushed to any
  hub or bucket. Trained weights and adapters may be shared or sold (licence §3).
- Anything that tries to identify a speaker (licence §6).

## Constraints (verified)

- **Model.** NeMo `SortformerEncLabelModel`, `nemo_version: 3.0.0`. Input 16 kHz mono, up to 8 speakers,
  `streaming_mode: true`. Encoder is `nemo.collections.asr.modules.TransformerEncoder` (31 layers, d_model 512,
  8 heads, rope). Losses come from the checkpoint config: PIL 0.75 + ATS 0.25, activity 0.5, phantom 0.1.
- **Training data format** (from the checkpoint's `train_ds`). Non-lhotse JSONL manifest with an RTTM per
  audio file, `session_len_sec: 105`, `subsegment_mode: true` (the loader crops sessions itself).
- **No LoRA in NeMo for this model class.** Grep of `sortformer_diar_models.py` finds no adapter or LoRA
  support, so LoRA is injected with PEFT.
- **Python 3.12.** NeMo asks for 3.12+, and this repo is pinned to 3.13. Pin 3.12 in `.python-version` and
  `requires-python` to avoid missing 3.13 wheels.
- **Data.** Hindi: 284 segments, 182 h, median 6 speakers per segment. English: 239 segments, 174 h, median 4.
  - Each speaker has their own 48 kHz FLAC track. All tracks in a segment share one clock and have the same
    length.
  - Per speaker, `on_call` spans mark when their mic was recording.
  - Per segment, `muted_spans` are digital silence on all tracks (1.9 h in total across Hindi and English).
  - Transcripts are automatic speech recognition on each speaker's own track. Word times are approximate.
  - Download is roughly 60 GB of audio.
- **Access.** The HF token on the VM must belong to an account with gated access. The UdayKapture account has
  access as of 2026-11-18.

## Approaches

| | Approach | Trade-off |
|---|---|---|
| **A (chosen)** | NeMo training loop + PEFT LoRA injected into the NeMo model | Uses NVIDIA's own loader, losses and streaming training. Costs a heavy Linux/CUDA install. |
| B | Transformers port + PEFT + HF Trainer | Lighter install, but we would rewrite the Sortformer loss and lose streaming training. A subtle loss bug would quietly make the model worse. |

**Labels.** Labels come from per-track speech detection, not transcript word times. The dataset authors warn
that the word times are approximate.

The user picked A.

## Design

### 1. `prepare` — Basis → 8 kHz 2-speaker training sessions

1. **Download.** Call `huggingface_hub.snapshot_download` with
   `allow_patterns=["{hi,en}/segments.jsonl", "{hi,en}/audio/*/*"]`, one pattern per language.
2. **Speaker activity per track.** Take the intersection of three things:
   - Silero VAD on the track, resampled to 16 kHz;
   - that speaker's transcript rows, each padded by ±0.3 s and merged;
   - that speaker's `on_call` spans.

   VAD gives tight speech boundaries. The transcript rows reject echo or bleed of other speakers, and noise.
   The dataset mutes some spans for echo, so bleed exists.
3. **Windows.**
   - Cut each segment into consecutive 105 s windows.
   - Drop any window that touches a `muted_span`.
   - In each window, keep the 2 speakers with the most labeled speech.
   - Keep the window only if both of them have at least 5 s of speech.
4. **Mix and phone channel.**
   - Phone-degrade each speaker's track on its own: band-limit to 300–3400 Hz, resample to 8 kHz, then apply a
     G.711 A-law encode/decode round trip. A-law is what Indian phone networks use.
   - Sum the two degraded tracks and peak-normalise.
   - Write 8 kHz 16-bit PCM WAV, the same format as production recordings.

   NeMo resamples to 16 kHz on load. That is the same path real 8 kHz production audio takes at inference.
5. **Labels and manifests.**
   - Write one RTTM per window, with times relative to the window.
   - Write one manifest line per window:
     `{"audio_filepath", "offset": 0, "duration", "rttm_filepath", "num_speakers": 2, ...}`.
     Add any other field NeMo's loader requires.
6. **Split.** Assign by hash of `conversation_id`: about 90 % train, 5 % val and 5 % test.
   - Speakers can recur across conversations. This is accepted because diarization does not learn speaker
     identity.
   - Mark the code with `shortcut: speaker overlap across splits, switch to speaker-disjoint if val/test scores
     look inflated`.
7. **All-speaker test set.** For each test window, also write a second mix that sums every active speaker's
   track, with a matching RTTM.
   - Reason (builder check): in 85 % of Hindi windows and 63 % of English windows, 3 or more people are
     speaking.
   - Training on 2-speaker mixes only could teach the model to stop finding a third speaker. This set catches
     that.
8. **Expected output.** About 11.7k windows and about 20 GB of 8 kHz WAV.

### 2. `train` — LoRA on the NeMo model

1. **Load.** `SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization")`.
2. **LoRA.**
   - Freeze every weight.
   - Inject PEFT LoRA into the `nn.Linear` layers of `model.encoder` (self-attention and feed-forward).
   - Starting values: r=16, alpha=32, dropout 0.05.
   - Only LoRA weights train.
3. **Data config.** Reuse the checkpoint's own `train_ds` and `validation_ds` blocks, overriding manifest paths,
   batch size and workers.
4. **Trainer.** Lightning, 1 GPU, bf16.
   - Starting values: lr 2e-4 with a short warmup. The checkpoint's 2,000-step warmup is meant for full
     training.
   - `max_steps` is set from the validation curve.
   - Validate periodically and keep the best checkpoint by validation loss.
5. **Output.**
   - Merge LoRA into the base weights and save a `.nemo`.
   - Also save the LoRA weights on their own.

### 3. `evaluate` — stock vs fine-tuned

- Score both models on every test manifest with the same streaming settings. The test manifests are:
  - Basis 2-speaker;
  - Basis all-speaker;
  - production mono-mix (section 3a), once it exists.
- **Latency settings** (from the model card, in 80 ms frames):

  | Setting | Role | SPKCACHE_LEN | FIFO_LEN | CHUNK_LEN | RIGHT_CONTEXT | UPDATE_PERIOD |
  |---|---|---|---|---|---|---|
  | Low latency, 1.04 s | the gate | 264 | 264 | 9 | 4 | 222 |
  | Ultra-low latency, 0.32 s | also reported | 264 | 264 | 3 | 1 | 222 |

  Do not use the checkpoint's default streaming config: it matches none of the card's settings.
- Report DER with a 0.25 s collar and overlap included, matching the model card's CALLHOME protocol. Also
  report speaker-count accuracy.
- Accepts any manifest + RTTM. Labeled production calls can be scored later with no code change.

### 3a. Production test set from stereo recordings

Production recordings already have bot and customer on separate channels, so labels come almost for free:
1. Run speech detection on each channel.
2. Write an RTTM with 2 speakers.
3. Sum the channels to mono, the same way production does.

- Use this set as the main ship gate, alongside the hand-labelled calls the user will add later.
- Prepared production audio stays on the VM, like the Basis data.
- **Blocker:** reading the recordings needs access to the `kapturecx-ml` GCS bucket. 2 of 3 sampled links
  refused anonymous access, and this Mac has no `gcloud`. Until access exists, this step is skipped and the
  gate is Basis-only.

### 4. Repo layout

```
pyproject.toml                      python 3.12; nemo-toolkit[asr] 3.0.x, peft, silero-vad, huggingface_hub, soundfile
src/nemo_3_diarization/prepare.py   console script: prepare
src/nemo_3_diarization/train.py     console script: train
src/nemo_3_diarization/evaluate.py  console script: evaluate
tests/test_prepare.py               labels / windowing / mixing on synthetic arrays
README.md                           H200 VM runbook: install, prepare, train, evaluate
```

`src/nemo_3_diarization/__init__.py` (hello-world `main`) and its `nemo-3-diarization` script entry are
replaced.

### 5. Verification

- **Mac:** unit tests for `prepare`. Run `prepare` on one real segment and listen to the mix against its RTTM.
- **VM:** a 20-step smoke run of `train` on a tiny manifest, then the full run, then `evaluate` on stock vs
  merged.

## Left to the builder

- Exact LoRA target module names: print `model.encoder` and pick its attention and feed-forward linears.
- LoRA rank, batch size, `max_steps`, `num_workers`, VAD threshold.
- Exact NeMo manifest fields and the NeMo DER utility to call.

## Decided by the user

- Approach A: NeMo + LoRA, trained on one H200 VM.
- The target is the live, low-latency mono mix of production calls.
- Production loads the `.nemo` file.
- Hand-labelled production calls will be added later as a test set.

## Calls made by the designer (user can correct)

- Training languages are Hindi and English. Training mixes have 2 speakers, to match bot + customer.
- The phone channel is simulated with A-law, not μ-law.
- 3a: label production stereo recordings per channel and use them as the main gate (needs bucket access).
- The gate latency is 1.04 s. 0.32 s is also reported.
- Every training voice is human. Basis has no bot voices, while production has one TTS voice per call. The
  stock model should handle a steady synthetic voice; the production test set will show if it does not.

## Security note (outside this pipeline)

- `kapturecx/call-transcript-intent-data-v2` is a public, ungated HF dataset of real customer call audio.
- At least one `kapturecx-ml` recording URL downloads without authentication.
- Both were reported to the user on 2026-11-18.
