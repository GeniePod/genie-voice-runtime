# TRADE-OFFS — the three tensions, and why Mode A ships on parakeet.cpp (not TensorRT)

The brief asks for one engine that meets **five** constraints at once: WER ≤3% · RTF ≤0.004 (250×) · ≤1 GB *total* RAM · multilingual + dynamic model load · true streaming. **No single configuration meets all five** — they are in direct tension. This repo ships a **3-mode architecture behind one entrypoint** (`./transcribe --mode A|B|C`) and wins on **honest, measured trade-offs** rather than a single number that can't exist.

This document maps the three structural tensions to the Stage-2 architecture questions, and records the one consequential engineering decision that follows from them: **Mode A runs on `parakeet.cpp` (ggml CUDA), not a bespoke TensorRT engine** — with `sherpa-onnx` kept as the reproducible alternative and the Mode B/C engines.

---

## The three tensions

### Tension 1 — RTF (250×) vs. everything else
250× real-time is a **batched, offline, datacenter-class** number. It pulls toward aggressive batching + INT8 + a hand-tuned decode loop. Three forces pull the other way:
- **Streaming (Mode B) is batch=1 by definition** — it can never be batched, so its RTF lives in a separate table.
- **The device is a shared Orin Nano at 25 W** (no MAXN, no clock lock) — ~18–20× slower than the A100 the 250× figure implies.
- **Accuracy and memory** want the full-size model resident, not a shrunk one.

**Verdict:** 250× is **not reachable on this device in any honest configuration.** parakeet.cpp's ggml CUDA backend reaches **107× (ctc) / 61× (tdt)** — the project best, and near the low end of what a bespoke TRT engine was ever expected to reach — but still short of 250×. We report best-achieved RTFx + the A100→Orin gap analysis + the 25 W / shared caveat, and never present a batched offline number as a streaming one. (The dtype/thread sweeps showed the engine is launch-bound at batch=1, so the remaining gap is a *batching* gap — see `../results/dtype/RESULTS.md`.)

### Tension 2 — ≤1 GB *total* RAM vs. a ~1.1–1.4 GB floor
"Must accommodate OS overhead" reads literally as **total system RAM**. But the measured floor — OS + CUDA context + one loaded model — is already **~1.1–1.4 GB before the first frame of audio**, and on a *shared* box (other tenants resident) the as-observed floor is higher still.

**Verdict:** ≤1 GB *total* is **infeasible as worded.** We report **total + measured floor + incremental RSS**, lead with that honesty, and pick the lowest-footprint *accurate* engine — **parakeet.cpp `tdt_ctc-110m` (q5_k, +450 MB active-inference)**. Mode C keeps exactly **one** heavy model resident via TTL/LRU so resident RAM never sums two models.

### Tension 3 — true streaming vs. WER ≤3%
A cache-aware streaming Zipformer at **low look-ahead** trades accuracy for latency: test-clean WER drifts to **3.6–3.9%**, which **breaches 3%**. Raising look-ahead recovers WER but raises latency.

**Verdict:** we pick the look-ahead that best holds ≤3%, **disclose the breach per latency config** (with a bootstrap CI), and report streaming RTF separately. We do not hide a low-latency number behind an offline one.

---

## The decision that falls out of Tension 1 — defer TensorRT, ship parakeet.cpp for Mode A

The original plan built Mode A as a **TensorRT 10.3 INT8 engine with a bespoke host-side eager-batched label-looping TDT decode** — the encoder (and optionally the per-step joiner) through TRT, the token loop + TDT duration-skip hand-written in host code. It was scoped as the **single highest-risk, multi-day (~3–4 day) task** in the build, with no drop-in runtime.

**We evaluated it and deferred it.** Mode A now runs on **`parakeet.cpp`** (ggml CUDA backend, `tdt_ctc-110m`, built from source) — which reaches **107× RTF without any bespoke engine**. `sherpa-onnx-offline` remains the **reproducible alternative** (`STT_ENGINE=sherpa`) and the spine for Modes B and C.

### Why defer TRT — three reasons, in order of weight

1. **The RTF gate is unreachable either way (Tension 1).** TRT eager-batched was only ever expected to reach **~100–180×** vs the **250×** target — a documented miss. **parakeet.cpp's ggml CUDA backend already reaches ~107× on a reproducible from-source build** — at the low end of that TRT range, *without* the bespoke engine. A hand-tuned TRT decode would buy a *marginally smaller miss of an already-missed gate* at the cost of the riskiest task in the plan. That is a bad trade.

2. **The TRT path is the #1 Stage-1 blind-replication hazard.** The grader clones the repo onto *their* Orin and reproduces from the README alone. A TRT engine is **non-portable** — keyed to device-UUID + TRT version + model hash, rebuilt on-device from a shipped calibration cache — and the bespoke batched-TDT decode is brittle across TRT minor versions. parakeet.cpp instead builds from **pinned source** (v0.1.1, sm_87) and sherpa-onnx covers B/C — portable model dirs, documented aarch64 CUDA builds, stable CLIs, permissive licenses. The most reproducible story available.

3. **Accuracy — the gate that actually passes — is unaffected.** parakeet.cpp `tdt_ctc-110m` stays at **1.81–2.38% WER** (≤3% met) on either decoder. We give up nothing on the one headline gate we can win.

### What we give up, stated honestly
- A possibly **higher offline RTFx**. A bespoke TRT engine *might* reach ~180×; parakeet.cpp measures 107×. Both miss 250×, and the gap between them is now small — we report the real, reproducible number with its caveats.
- The chance to demonstrate a **hand-tuned TRT decode**. We keep that as a **Stage-2 discussion artifact**: we can speak to exactly how the eager-batched label-looping decode would work, why it reaches ~100–180×, and why the A100→Orin gap is ~18–20× — without shipping a fragile artifact the grader can't reproduce.

### Net effect on the build
- Removes the highest-risk long-pole (bespoke batched-TDT decode), the on-device TRT engine build, the custom GPU log-mel front-end, the custom batcher, the FP16-TRT dry-run, and the batched-vs-single **WER-parity** gate.
- Mode A is a clean `parakeet.cpp bench` driver (model-load excluded from RTF); Modes B/C are sherpa-onnx wrappers. The only Mode-A-specific check left is a sequential-vs-batched RTF comparison (throughput only — no accuracy divergence to defend).

---

## Tensions → Stage-2 questions

| Stage-2 question (architecture interview) | Which tension | Our answer |
|---|---|---|
| **How do you approach the RTF target?** | Tension 1 | 250× is a datacenter-batched number; on a 25 W shared Orin it is unreachable. parakeet.cpp's ggml CUDA backend reaches **107× (ctc)** — the project best, near the low end of TRT's expected ~100–180× — reported with the ~18–20× A100→Orin gap + the power/contention caveat. The remaining gap is a *batching* gap (the engine is launch-bound at batch=1). TRT eager-batched was evaluated and deferred — still a miss, at the cost of the riskiest, least-reproducible task. |
| **How do you fit the memory budget?** | Tension 2 | ≤1 GB *total* is infeasible as worded (a ~1.1–1.4 GB floor before weights). We report total + floor + incremental RSS, pick the lowest-footprint accurate engine (**parakeet.cpp 110 M q5_k, +450 MB**), and keep exactly one heavy model resident in Mode C (TTL/LRU). |
| **How do you do streaming and multilingual together?** | Tension 3 | Separate modes behind one entrypoint: Mode B = cache-aware streaming Zipformer (batch=1, look-ahead chosen to hold ≤3%, breach disclosed per config); Mode C = LID router (Whisper-tiny) → one resident recognizer (Parakeet-v3, 25 lang) under a programmatic single-resident cap. |
| **How do you make it reproducible?** | the decision above | parakeet.cpp from **pinned source** for Mode A + sherpa-onnx for B/C; portable model dirs; **locked deps** (`requirements.lock`); idempotent preflight hard-asserting the CUDA runtime (CUDA 12.6 / cuDNN 9.3) and recording TensorRT warn-only; no non-portable engines committed or built; a clean-clone reproduction rehearsal. Deferring TRT is the largest single reduction in replication risk. |

---

*Numbers cited here are stamped with the `L4T 36.4.7 / 25 W / shared` device caveat. The authoritative measured values live in [`../results/decisions.json`](../results/decisions.json), [`../results/feasibility.md`](../results/feasibility.md), [`../results/parakeet/RESULTS.md`](../results/parakeet/RESULTS.md), and [`../results/dtype/RESULTS.md`](../results/dtype/RESULTS.md). See [`../ASSUMPTIONS.md`](../ASSUMPTIONS.md) §ENG for the engine decision.*
