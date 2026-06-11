# GPU dtype sweep — tdt_ctc-110m on the Jetson Orin Nano Super 8 GB

**Question:** does the GGUF quantization dtype change speed/accuracy/memory on the *GPU*? Upstream
parakeet.cpp only benched dtypes on a 20-core host **CPU** (BENCHMARK.md), where f16 was fastest and
the K-quants were *slower*. The Orin GPU has a different compute/bandwidth profile, so this is the
on-device truth.

**Method.** Every config run through the official harness (`PK_MODEL=<gguf> PK_DECODER=tdt|ctc
./transcribe --mode A`) → WER (Whisper-normalized) + RTFx (model-load-excluded) + peak memory
(`MemorySampler`), full LibriSpeech test-clean **300**, 25 W, **clean box** (GeniePod stopped), sm_87.
GGUFs are the true f32-derived variants from HF `mudler/parakeet-cpp-gguf`. Per-config reports:
`results/dtype/<dtype>_<decoder>.json`.

| dtype | GGUF size | decoder | WER % | RTFx | peak total MB |
|---|---|---|---|---|---|
| q4_k | 126 MB | ctc | 2.467 | 107.1 | 2232 |
| q5_k | 137 MB | ctc | 2.411 | 106.1 | 2245 |
| **q8_0** | 177 MB | ctc | **2.383** | **107.3** | 2279 |
| f16 | 256 MB | ctc | 2.369 | 107.2 | 2418 |
| q4_k | 126 MB | tdt | 2.018 | 61.9 | 2256 |
| **q5_k** | 137 MB | tdt | **1.808** | **61.4** | 2279 |
| q8_0 | 177 MB | tdt | 1.836 | 61.7 | 2319 |
| f16 | 256 MB | tdt | 1.808 | 60.9 | 2448 |

## Finding 1 — speed is **dtype-invariant** (quant is a memory lever, not a speed lever)

ctc sits at **~107×** and tdt at **~61×** across *every* dtype: q4_k (126 MB) is no faster than f16
(256 MB). Combined with the earlier thread sweep (also flat — `results`/the audit), this is a firm
diagnosis: at **batch=1 the engine is kernel-launch / small-matmul-latency bound**, not weight-
bandwidth or CPU bound. Shrinking weights frees **memory** (q4_k peak 2232 MB vs f16 2418 MB, a
~190 MB spread tracking the weight size) but never speeds anything up. **The only remaining lever for
higher RTFx is batching** — exactly the datacenter-throughput gap behind the 250× target.

## Finding 2 — q5_k is a free win for the accuracy (tdt) path

q5_k/tdt = **1.808 % WER @ 61× @ 137 MB** — it *matches* q8_0's accuracy (1.836 %, the delta is
within noise and nominally better), at **23 % smaller** and **identical speed**. f16 ties the WER but
is 256 MB. So the tdt default moves **q8_0 → q5_k** at zero speed/accuracy cost. The ctc (speed) path
keeps **q8_0** (f16 is 0.014 pt better but +86 MB; q4_k is smallest but slightly worse). If memory
tightens, **q4_k** (126 MB, +580 MB over idle) holds 2.018 %/2.467 % — still under the 3 % gate.

## Decision (folded into the submission)

- **tdt → `tdt_ctc-110m-q5_k.gguf`** (new default), **ctc → `tdt_ctc-110m-q8_0.gguf`** — per-decoder
  default in `src/modes/a_offline/parakeet_runner.py` (`_DECODER_GGUF`); override with `PK_MODEL`.
- `scripts/get-models.sh` fetches q8_0 + q5_k + q4_k.
- Memory ladder for "if constraints tighten" (Stage-2 Q4) gains a measured rung: **q5_k → q4_k**.
