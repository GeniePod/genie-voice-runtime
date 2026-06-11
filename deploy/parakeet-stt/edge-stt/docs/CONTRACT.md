# CONTRACT — the frozen grader-facing interface

> This interface is **frozen at M0**. Schemas do not change after this point; later milestones
> fill in implementations behind it. The README Deployment Guide **ends at `./transcribe`**;
> every step before it is a committed script under `scripts/`.

## 1. The entrypoint

A top-level executable shim **`./transcribe`** (thin wrapper, `chmod +x`):

```sh
#!/usr/bin/env bash
exec "${PYTHON:-python3}" -m src.entrypoint.transcribe "$@"
```

### Frozen CLI signature

```
./transcribe --mode {A|B|C} --input <dir|manifest.jsonl> --out hyps.jsonl \
             [--report report.json] [--limit N] [--device cuda] \
             [--replay {sequential|batched}] [--selfcheck]
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--mode` | yes | — | `A` offline-batched (RTF), `B` true-streaming, `C` multilingual/dynamic |
| `--input` | yes | — | a directory of `.wav`/`.flac`, or a `manifest.jsonl` (one obj/line, see §3) |
| `--out` | yes | — | path to write `hyps.jsonl` |
| `--report` | no | unset | when set, also emit a typed `report.json` (RTF/mem/WER) — first-class output, not a log scrape |
| `--limit N` | no | all | process only the first N manifest entries (M4 tag-smoke uses `--limit 5`) |
| `--device` | no | `cuda` | `cuda` or `cpu` |
| `--replay` | no | `sequential` | `sequential` (per-utterance, honest edge number) or `batched` (Mode-A throughput) |
| `--selfcheck` | no | — | run `scripts/preflight.sh` and exit (asserts CUDA/TRT/cuDNN versions — **no clock/MAXN assert**, shared device); M3-6 |

**Two input modes** (ROADMAP M0-2): a **`manifest.jsonl`** with `ref` fields → emits hyps **and** scores WER; a **bare directory of wav/flac** (no refs) → emits hyps only (no WER). The harness detects which by input type.

## 2. `hyps.jsonl` — transcripts (one JSON object per line)

Keyed to the manifest `id` so the WER join is unambiguous.

```json
{"id":"<utt_id>","text":"<hypothesis>","audio_sec":<float>,"infer_sec":<float>}
```

| Field | Type | Notes |
|---|---|---|
| `id` | string | matches the manifest `id` |
| `text` | string | raw hypothesis (un-normalized; normalization happens at scoring) |
| `audio_sec` | float | decoded audio duration |
| `infer_sec` | float | model inference wall time for this utterance (decode excluded) |

## 3. `manifest.jsonl` — the eval set (produced by `scripts/get-data.sh`)

```json
{"id":"<utt_id>","wav":"<abs path>","ref":"<reference text>","duration_sec":<float>}
```

(Exactly what `spike/prepare_data.py` already writes.)

## 4. `report.json` — per-run RTF / memory / WER (identical schema from both producers)

```json
{"mode":"A","n":300,"total_audio_sec":<f>,"wall_sec":<f>,"rtfx":<f>,"rtf":<f>,
 "wer_pct":<f>,"normalizer":"whisper",
 "mem":{"idle_total_mb":<f>,"peak_total_mb":<f>,"proc_rss_mb":<f>,"cuda_used_mb":<f>},
 "replay":"sequential","precision":"int8","engine":"...","device":"orin-nano-super-8gb",
 "limit":null}
```

All `mem.*` fields come from `src/common/memory_accountant.py` (the one accountant). `rtfx = total_audio_sec / wall_sec`; `rtf = 1/rtfx`.

## 5. Scoring split (ONE model, two producers)

- **`./transcribe --report`** — runs inference, emits `hyps.jsonl` **and** the per-run `report.json`. Inference-time RTF/mem come from here.
- **`scripts/score.sh`** — a thin wrapper: `python -m src.common.wer_rtf_report <hyps.jsonl> <manifest.jsonl> --report report.json`. Re-computes/joins **WER** over an existing `hyps.jsonl` **without re-running inference** (this is M4's `score.sh --report report.json`).

WER may be (re)computed by either path; both emit the **identical** `report.json` schema.

## 6. Audio decode contract

Input `.wav`/`.flac` decoded via **libsndfile** (`soundfile`) to **16 kHz mono PCM16**; resample with **soxr** only if `sr != 16000` — matching `scripts/prepare_data.py`. The **harness owns decode**; **RTF is measured over model inference only**, wall-clock reported separately.

## 7. CLI-only (no HTTP/WS twin)

Per the committed plan (ROADMAP M0-2), the deliverable is **CLI-only** — there is **no HTTP/WS
service twin**. The grader drives `./transcribe` directly. `transcribe_core()` stays the single
internal function all modes route through, but it is exposed **only** via the CLI.

## 8. Committed `scripts/` producers (README invokes these in order)

| Script | Role | Final form |
|---|---|---|
| `scripts/get-data.sh` | materialize LibriSpeech wavs + manifests from scratch | committed |
| `scripts/get-models.sh` | fetch model artifacts (parakeet.cpp GGUF + sherpa-onnx dirs) | committed |
| `scripts/build-parakeet.sh` | build the PRIMARY engine parakeet.cpp (ggml CUDA, sm_87) from source | committed |
| `scripts/build-sherpa.sh` | build the sherpa-onnx GPU binary (Modes B/C + alternative) | committed |
| `pip install -r requirements.lock` | install the locked Python deps (scoring + data-prep) | committed |
| `scripts/preflight.sh` | assert CUDA/cuDNN versions (`--selfcheck` shells this; no clock assert) | committed |
| `scripts/score.sh` | WER-join over an existing `hyps.jsonl` | committed |

**README rule:** the Deployment Guide ends at `./transcribe --mode ...`; every preceding step is one of the committed scripts above. Final wording delivered in M3-8.

---
*Frozen M0-T2. Device-target note: per the shared-device adaptation, the `device` field reports the actual on-device identity and the toolchain is used as-installed (see ASSUMPTIONS.md §Device).*
