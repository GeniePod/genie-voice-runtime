# parakeet.cpp evaluation — LibriSpeech 300, vs assessment targets

**Engine:** `parakeet.cpp` (ggml/llama.cpp-style CUDA backend, built from source on the Jetson) ·
**Model:** `tdt_ctc-110m-q8_0.gguf` (110 M params, INT8/q8_0, 170 MB on disk) · **GPU (sm_87)**.
**Eval:** the pinned **LibriSpeech test-clean first-300-by-id**, run **in isolation** (all GeniePod
services stopped → clean 1535 MB system baseline). Whisper-normalized WER; RTF = Σ proc / Σ audio.

> **Note:** the numbers below are the **q8_0** measurement (both decoders). A later **on-GPU dtype sweep**
> ([`../dtype/RESULTS.md`](../dtype/RESULTS.md)) found RTFx is *dtype-invariant* and moved the **tdt
> (accuracy) default to q5_k** — same accuracy (**1.808 %** ≈ 1.836 %), **23 % smaller** (137 MB), identical
> 61× speed. ctc keeps q8_0. The q8_0 tdt row below is the original, unchanged measurement.

## Results vs targets

| Metric | tdt decoder | ctc decoder | Target | Verdict |
|---|---|---|---|---|
| **WER** | **1.836 %** | **2.383 %** | ≤ 3.0 % | ✅ **PASS** (both) |
| **RTF** | 0.0166 (**60×**) | **0.00933 (107×)** | ≤ 0.004 (250×) | ✗ best 107× — see gap |
| **Memory (incremental)** | +450 MB | +450 MB | ≤ 1 GB total | ✅ footprint; total see below |
| Per-call latency | 126 ms median / 146 ms mean | — | — | matches the ~170 ms prod number (this excludes HTTP+sox) |
| load (one-time) | ~320 ms | ~440 ms | — | — |

- **WER ✅** — 1.84 % (tdt, more accurate) / 2.38 % (ctc, faster); both well under 3 %.
- **RTF** — **107× (ctc)** is the project best: **~4× sherpa-onnx (25×)** and **~14× the 0.6 B on ONNX
  Runtime (7.4×)** on identical hardware. The 250× target remains a documented miss (~18–20× A100→Orin
  gap + 25 W), but parakeet.cpp's ggml CUDA backend closes most of the gap **without** a bespoke TensorRT
  engine — i.e. near-TRT throughput on a reproducible from-source build.
- **Memory** — **+450 MB** active-inference incremental over a clean baseline (matches the ~410 MB field
  number). ≤ 1 GB *total* is infeasible (OS floor alone ~1.5 GB), but the model+inference footprint is small
  and fits beside other workloads.

## Why parakeet.cpp wins the RTF axis (the engine, reconciled)

The earlier sherpa-onnx exploration topped out at 25× because **ONNX Runtime's CUDA EP is the bottleneck**
(the 0.6 B was compute-bound at ~5–7× and didn't even benefit from batching). **parakeet.cpp uses a
ggml CUDA backend** (same lineage as the fast on-device LLM runtimes) which is far more efficient on the
Orin — **8–14× the throughput at the same INT8 precision** — and pairs it with a small **110 M** model. That
combination is what reaches 60–107× on a 25 W shared-arch device, where ORT could not.

## Operating points (current defaults)
- **Accuracy-first:** tdt decoder, **q5_k** default — **1.808 % WER, 61× RTF** (q8_0 measured 1.836 %/60×).
- **Speed-first:** ctc decoder, **q8_0** — **2.383 % WER, 107× RTF** (still < 3 %).

Both at **+450 MB** and ~110 M params — the strongest memory + RTF + accuracy balance measured in this project,
and the basis for the deployed GeniePod STT (`genie-parakeet` resident server on :8178, ~170 ms/call incl.
HTTP + sox resample).
