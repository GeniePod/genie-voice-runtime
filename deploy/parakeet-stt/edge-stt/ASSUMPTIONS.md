# ASSUMPTIONS — measurement definitions & device reality

The grader runs **our** harness, so where the brief is ambiguous we pick an honest default,
**instrument both readings**, and never block on assessor confirmation. Each default is a
single flag/env so a clarification is a one-line change, not a rebuild.

---

## Device target (deviation from the written plan — read this first)

> **DEVICE (as-used, measured 2026-06-06):** Jetson Orin Nano Super 8 GB ·
> **L4T 36.4.7** (`R36 (release), REVISION: 4.7`) / JetPack 6.2.x ·
> **CUDA 12.6** · **TensorRT 10.3.0.30** · **cuDNN 9.3.0.75** · Python 3.10.12 · aarch64 ·
> power mode **25 W** (not MAXN SUPER).

The original ROADMAP pinned a clean reflash to **L4T 36.4.3** with a golden backup and a
clean-room rehearsal. **That is not done here:** the device is a **shared, in-use machine**
running other live workloads, so reflashing, power-mode/clock locking, and headless
idle-RAM isolation are out of scope. Consequences, recorded honestly:

- **Toolchain matches the plan anyway** — CUDA 12.6 / TRT 10.3 / cuDNN 9.3 are exactly the
  targeted versions; only the L4T point-release differs (4.7 vs 4.3). The 4.7 NVMap idle-RAM
  regression noted in the study means idle RAM is **higher and shared** — accounted for by
  measuring memory **as-observed** (see Memory below), never by freeing RAM.
- **No MAXN/clock lock** — RTF numbers are reported with the `25 W` power-mode caveat; they
  are conservative relative to a clock-locked MAXN SUPER run.
- **Preflight** asserts the **actual** versions (CUDA 12.6 / TRT 10.3 / cuDNN 9.3) rather than
  enforcing the 36.4.3 clean-room string.

---

## ENG — engine decision (Mode A primary = parakeet.cpp; sherpa-onnx alternative; TensorRT deferred)

- **Decision (2026-06-10, amends the 2026-06-09 sherpa decision):** Mode A's **primary engine is `parakeet.cpp`** (ggml CUDA backend, `tdt_ctc-110m`, built from source) — measured **1.808 % WER @ 61× (tdt, q5_k) / 2.383 % @ 107× (ctc, q8_0)**, +450 MB, after it beat the sherpa-onnx path (~25×) on RTF. **`sherpa-onnx-offline`** remains the **reproducible alternative** (`STT_ENGINE=sherpa`) and the **Mode B** (streaming Zipformer INT8) and **Mode C** (Whisper-tiny LID + Parakeet-v3 + TTL/LRU) engines. Reasoning: [`docs/STAGE2-MEMO.md`](docs/STAGE2-MEMO.md) + [`docs/TRADE-OFFS.md`](docs/TRADE-OFFS.md).
- **What this replaces:** the original Mode A was a bespoke **TensorRT 10.3 INT8 eager-batched label-looping TDT decode** built on-device. That path is **evaluated and deferred**, not built — parakeet.cpp reaches near-TRT throughput (107×) on a reproducible from-source build without the non-portable engine. Full rationale in [`docs/TRADE-OFFS.md`](docs/TRADE-OFFS.md).
- **Why TRT stays deferred:** (1) the **RTF ≤0.004 (250×) gate is unreachable either way** — TRT was only ~100–180×, a documented miss; the deliverable is the honest gap analysis, not the number. (2) The non-portable, device-keyed TRT engine + bespoke host-side batched-TDT decode is the **single biggest Stage-1 blind-replication hazard**. (3) **Accuracy is unaffected** — both decoders stay <3 % WER.
- **Consequence for RTF honesty:** Mode A's RTFx is the **best-achieved on-device number** (parakeet.cpp 107× ctc), reported with the **A100→Orin ~18–20× + 25 W / shared** gap analysis. Never presented as streaming.
- **Instrument:** `--mode A` runs **parakeet.cpp by default** (`PK_DECODER=tdt|ctc` picks the q5_k|q8_0 default; `PK_MODEL` overrides); `STT_ENGINE=sherpa` selects the sherpa-onnx alternative.
- **Preflight consequence:** `scripts/preflight.sh` hard-asserts only the **CUDA runtime** (CUDA 12.6 / cuDNN 9.3) and records **TensorRT warn-only** — reproduction must not fail for lack of the exact TRT version.
- **Override:** if a TRT Mode A is ever wanted, the deferred design + rationale live in [`docs/TRADE-OFFS.md`](docs/TRADE-OFFS.md) — it is a re-activation, not a rewrite.

## FND-MEM — what "memory" means

- **Decision:** headline = **total system RAM** (brief: "must accommodate OS overhead" → total
  is the literal read); **also** report **incremental process RSS over baseline** and CUDA usage.
- **Why:** the study shows a ~1.1–1.4 GB floor before weights; **≤1 GB total is infeasible as
  worded**, so we report total + floor + incremental and lead with that honesty.
- **Instrument:** `src/common/memory_accountant.py` samples `tegrastats`/`meminfo` total +
  `/proc/<pid>/status` RSS + `cudaMemGetInfo` together.
- **Shared-box rule:** the GATE-A floor is measured **as-observed** on the live machine (other
  projects resident) — we do **not** stop services or free RAM to get a cleaner number. The
  reported idle is therefore an upper bound, stated as such.
- **Pending-confirm:** **FND-MEM**. **Fallback:** if the assessor means incremental-only, the
  same report already carries `proc_rss_mb`/`delta_total_mb` — switch the headline field.

## FND-RTF — replay order

- **Decision:** **sequential per-utterance by default** (honest edge number); **also** report
  offline-**batched** RTFx for Mode A. `--replay {sequential|batched}`.
- **Why:** the grader may replay per-utterance, which kills the batched-250× premise; instrument
  both so neither is hidden.
- **Instrument:** `report.json.rtfx`/`rtf` + `--replay` field; streaming RTF reported separately,
  never conflated with offline.
- **Pending-confirm:** **FND-RTF**. **Fallback:** flip `--replay`; both paths emit identical schema.

## FND-SUBSET — which 300

- **Decision:** **first-300 sorted by utterance id** (matches `scripts/prepare_data.py`).
- **Why:** the brief's "default subset of 300" is unspecified; sorted-by-id is deterministic and
  reproducible byte-for-byte from `scripts/get-data.sh`.
- **Instrument:** `data/manifest.jsonl` (300 rows) + `data/manifest.sha256`.
- **Pending-confirm:** **FND-SUBSET** (note ±0.1–0.2 abs WER variance on N=300).
  **Fallback:** `prepare_data.py --n`/`--url` selects any other subset/split.

## FND-NORM — WER normalizer

- **Decision:** **Whisper-style normalizer** (Open ASR Leaderboard convention) via pinned
  `whisper-normalizer`; documented lowercase/punct-strip fallback if the package is absent.
- **Why:** matches the leaderboard numbers we cite (1.69 % v2 / 1.93 % v3).
- **Instrument:** `report.json.normalizer` records which normalizer actually ran
  (`whisper` vs `fallback-lower-punct`).
- **Pending-confirm:** **FND-NORM**. **Fallback:** swap the normalizer function; one call site.

---

## How to override

Every default above is a single flag or env var (`--replay`, `--limit`, `prepare_data.py --n/--url`,
the normalizer import). An assessor's clarification is a one-line change. The 4 `FND-` items are
the open questions; nothing in the build/run path blocks on a reply.

*GATE-A floor numbers (idle_total_mb, cuda_ctx_floor_mb, measured as-observed) will be appended
here once measured on-device.*
