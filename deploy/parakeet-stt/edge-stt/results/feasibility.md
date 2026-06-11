# M1 — Feasibility truth on-device (Mode A measured)

> **HEADLINE (2026-06-10): the primary engine is now `parakeet.cpp` (ggml CUDA).** On LibriSpeech-300 it
> measures **WER 1.808 % @ 61× (tdt, q5_k default)** or **2.383 % @ 107× (ctc, q8_0)**, **+450 MB**,
> ~126 ms/call — ~4–14× the RTF of the sherpa-onnx paths below, at a fraction of the memory. Full write-up:
> [`parakeet/RESULTS.md`](parakeet/RESULTS.md); on-GPU dtype sweep (why tdt=q5_k): [`dtype/RESULTS.md`](dtype/RESULTS.md).
> The sherpa-onnx analysis below remains the **reproducible alternative** + the streaming (Mode B) and
> multilingual (Mode C) engines, and the honest engine/RTF/memory trade-off study.

**Device (as-used, 2026-06-09):** Jetson Orin Nano Super 8 GB · L4T 36.4.7 · CUDA 12.6 / TRT 10.3.0.30 / cuDNN 9.3.0.75 · **25 W** (not MAXN) · **shared box** normally running GeniePod (LLM + STT, ~2–3 GB resident).

**Engine:** the gate study below is the original **`sherpa-onnx-offline` INT8 Parakeet-TDT-0.6b-v2** feasibility baseline (the box that exposed the memory + RTF walls). The **primary Mode A engine is now `parakeet.cpp`** (banner above; 2026-06-10 pivot — see [`../docs/TRADE-OFFS.md`](../docs/TRADE-OFFS.md)); sherpa-onnx stays as the reproducible alternative + the Mode B/C engines. TensorRT deferred.

Every number below is from the frozen entrypoint `./transcribe --mode A` over the **pinned 300** (LibriSpeech test-clean, first-300-by-id), Whisper-normalized, on the actual device.

---

## Verdict per gate

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| **C — WER (Mode A offline)** | ≤ 3.0 % test-clean | **1.331 %** (INT8, Whisper-norm, n=300, oom=0) | ✅ **PASS** — beats the ~1.69 % prior on this subset |
| **B — RTF (Mode A offline)** | ≤ 0.004 (250×) | **RTFx 7.41** (rtf 0.135), sequential, model-load-excluded | ❌ misses 250× by ~34× — report best-achieved + gap |
| **A — Memory** | ≤ 1 GB **total** | **peak 6.6 GB** (cuda 6.7 GB; ~4.9 GB incremental over a 1.65 GB paused floor) | ❌ **wildly infeasible**; model does not coexist with GeniePod on 8 GB |
| **B-stream — Mode B streaming WER** | ≤ 3.0 % (disclose breach) | **1.976 %** (Whisper-norm, **n=300**, chunk-16-left-128) | ✅ no breach at this look-ahead |
| **C-multi — Mode C LID + non-English** | multilingual + dynamic | **LID 97 %**; German WER **7.41 %** (v3) / 54.2 % (whisper-tiny); v3 English 1.71 % | ✅ proven + accurate |
| **F — modes-split** | 3 modes / 1 entrypoint | **all 3 real & measured** behind one entrypoint | ✅ |

---

## GATE C — accuracy is the gate we win
**1.331 % WER** over all 300, INT8, on-device, Whisper-normalized — every clip transcribed (0 OOM in the clean run). This is the headline: the flagship model is accurate well within budget. Accuracy was never the risk; memory and RTF are.

## GATE B — RTF: honest miss + gap analysis
**RTFx 7.41** (inference-only, model-load excluded; total audio 2641 s / inference 357 s). This is the *naive* number: batch=1, sequential, sherpa-onnx-offline CUDA EP, **25 W**, shared box. It misses the 250× target by ~34×. The honest gap decomposition:
- **A100 → Orin Nano**: ~18–20× raw compute gap (the 250× figure is a datacenter-batched number).
- **No batching**: 250× assumes large offline batches; we measure batch=1.
- **25 W / shared**: conservative vs a clock-locked MAXN run.

The deferred TensorRT eager-batched path was only ever expected to reach ~100–180× — still a documented miss — at the cost of the highest-risk, least-reproducible task in the plan (see TRADE-OFFS.md). Batched RTF, if pursued, is **never** reported as streaming.

## GATE A — memory is THE binding constraint (vividly confirmed)
≤ 1 GB *total* is infeasible by a wide margin: the model **peaks at ~6.6 GB** (ORT's CUDA BFC arena grows greedily to ~1 GB activation buffers for the longest clips). The empirical proof of the shared-device reality:
- On the **as-observed contended box** (GeniePod's ~2–3 GB resident), the model's ~6.6 GB peak exceeds 8 GB → it **OOM-rebooted the device once**, then OOM-SIGKILLed the process.
- The clean 6.6 GB-peak number was obtainable **only after pausing GeniePod** (authorized), leaving a 1.65 GB idle floor → ~4.9 GB incremental.

**Reported honestly as:** total (6.6 GB peak) + floor (1.65 GB paused / ~3 GB contended) + incremental (~4.9 GB). The flagship 0.6 B model and the other 8 GB-box workload **do not fit together** — this is the central trade-off the brief is about.

## Runner robustness (how all 300 fit despite the arena)
`src/modes/a_offline/runner.py` sorts clips **shortest-first**, batches in chunks of 8, and on an arena-overflow abort **splits the chunk down to singles** (a single clip + model fits comfortably), recording any clip that still won't fit as a non-fatal `oom` row. Result: **0 OOM**, all 300 scored, with model-load excluded from RTF.

## Mode B — true streaming (measured)
**sherpa-onnx-online streaming Zipformer INT8**, chunk-16-left-128. **Streaming WER 1.976 %** on the **full 300** (Whisper-norm; 2.249 % on the first-100) — holds under 3 % at this look-ahead (no breach; lower-latency configs may approach the disclosed 3.3–3.9 %). Two engineering findings:
- **CPU, not GPU.** GPU is **~3× slower** for the tiny streaming model — per-chunk CUDA kernel-launch overhead dominates the small compute. Mode B runs on CPU (the right call), and therefore co-resides with GeniePod without the Mode-A OOM.
- **Streaming RTF is contention-sensitive.** ~0.6 isolated (1.6× faster than realtime — the engine capability) vs **~3.3 under full GeniePod CPU load** on the shared box. Reported separately from offline RTF, never conflated.

## Mode C — multilingual + dynamic (measured, accurate)
**Whisper-tiny LID router → recognizer**, over a 30-clip **German FLEURS** sample, with both recognizers measured on-device:
- **LID accuracy 97 %** — whisper-tiny auto-detects German on 29/30 clips, routing correctly (the JSON `lang` field is the router decision).
- **German WER 7.41 %** with **Parakeet-TDT-0.6b-v3** (the accurate 25-lang recognizer) — vs **54.2 %** with whisper-tiny alone (a **7× improvement**; BasicTextNormalizer). v3 also does **English at 1.71 %** on the 300 (vs v2's 1.33 % English-specialist — v3 trades a little English accuracy for 25-language coverage).
- **Residency:** `src/modes/c_multilingual/residency.py` implements the registry + LID router + **TTL/LRU one-resident manager**; the single-resident invariant (evict-before-load, never two 0.6 B models co-resident) is validated — exactly what GATE A requires.
- v3 is a 0.6 B model (~6.6 GB peak), so its measurement used a **GeniePod GPU pause**, same as Mode A. Staging it required a parallel-HF download workaround (HuggingFace throttled the VM per-connection).

**What this proves:** the multilingual stack — LID → route → accurate multilingual recognition under a one-resident memory cap — runs **end-to-end and accurately** on the 8 GB device.

---

## Footprint study — "6.6 GB is too big" (it is; here's the deployable number)

The Mode-A headline **6.6 GB peak** is a *worst-case* artifact of three stacked costs, not the model:
the **0.6 B weights** (~650 MB) + the **GPU CUDA context** (~1 GB, fixed for any CUDA model) + **ORT's
CUDA arena grown over a 300-clip batch** (~3 GB). The 0.6 B at chunk=1 (min arena) is already only
**~1.76 GB**; a tiny Zipformer on GPU is **~1 GB** (all CUDA context). So most of the 6.6 GB is
reducible runtime, and the GPU floor alone is ~1 GB.

The number that matters for the **LyraT voice pipeline** is different from a 300-clip LibriSpeech sweep:
a **resident model decoding one 3–5 s command on CPU**. Measured per-process peak RSS (`/proc VmHWM`),
one ~3 s clip, CPU:

| Offline model | Weights (disk) | **Per-command RSS** | WER (300, Whisper-norm) | RTFx (CPU) |
|---|---|---|---|---|
| **zipformer-small-en (zip28)** | 27 MB | **115 MB** | 2.036 % | 28× |
| **zipformer-en (zip70)** | 68 MB | **204 MB** | 1.867 % | 23× |
| Parakeet-TDT-0.6b-v2 | 631 MB | 1247 MB | 1.331 % | 5× |

**A small Zipformer hits ~115–204 MB per command at < 3 % WER and 20–30× real-time on CPU** — under the
300–400 MB target, fits comfortably alongside the LLM, and is *faster* than the 0.6 B was on GPU. The
weights are tiny; the runtime memory scales with **clip length × clips-per-invocation**, not model size
(ORT's arena accumulates — a 91-short-clip batch balloons to ~3 GB for *every* model, but a single
command does not).

**Recommendation for the deployable STT:** **zip70 (68 MB → 204 MB/command, 1.867 %, 23× RTFx)** — or
zip28 for 115 MB at 2.036 %. Keep the 0.6 B as the offline accuracy *showcase* (1.331 %), not the
on-device voice STT. (Caveats: these are English offline Zipformers; the streaming Mode-B Zipformer
already shows the same small-and-accurate profile, and a multilingual small model would be the Mode-C
production swap.)

## RTF optimization — why 250× is out of reach, and what actually moves it

The 250× (RTF ≤0.004) target is a **datacenter-batched** figure; on a **25 W Orin Nano** it is unreachable
(~18–20× A100→Orin compute gap, before the no-large-batch and power caveats). What we measured pushing RTF
on LibriSpeech (Whisper-norm WER alongside):

| Model · provider · note | RTFx | WER |
|---|---|---|
| 0.6 B Parakeet transducer · GPU | 7.4× | 1.33 % |
| **zipformer transducer (68 MB) · CPU** | **~25×** | **1.87 %** |
| zipformer transducer · GPU | ~25× (no gain, +800 MB) | 1.91 % |
| nemo-ctc **conformer** (45 MB) · CPU | 7.8× | — |
| zipformer-**ctc** (70 MB) · CPU | ~12× single-clip (batch OOMs) | — |

**The levers, measured (not guessed):**
- **The encoder is the bottleneck, not the decode head.** A NeMo *conformer*-CTC is **slower** (7.8×) than a
  *zipformer*-transducer (25×) despite CTC's "parallel" decode — because the conformer encoder is heavier.
  CTC vs transducer barely matters here; the **zipformer encoder** is what makes it fast.
- **GPU does not help** a small model (same ~25×, +800 MB CUDA context) — per-clip launch overhead dominates.
- **More threads hurt** (4 → 22.4× vs 6 → 18.9×): core oversubscription on the shared 6-core box.
- **GPU batching OOMs** on the 8 GB box (arena growth) — the one lever that *could* approach the target is the
  one the memory budget forbids. This is the core tension the assessment is built around.

**Batch-scaling (GeniePod paused, clean-box headroom) — reconciling 7.4× vs the ~100–180× TRT estimate:**

| model · GPU | batch=1 | batch=16 | batch≥64 |
|---|---|---|---|
| zipformer (68 MB) | 3.1× | **20.4×** | OOM |
| 0.6 B Parakeet | 4.2× | **4.8×** | OOM |

Two distinct facts fall out:
- **The 0.6 B is compute-bound on ONNX Runtime — batching does nothing** (4.2×→4.8× from batch 1→16). The big
  model saturates the 25 W GPU at batch=1, so the only lever left is a faster *engine*. **The ~100–180× figure
  was specific to the deferred TensorRT path** (fused/optimized INT8 engines + eager-batched label-looping
  decode); it was never an ORT/sherpa-onnx number, and ORT cannot reach it for the 0.6 B.
- **Batching helps only the small model** (zip70: 3.1×→20.4×, 6.5× from batch-16) and **OOMs past batch-16**
  (activation arena). So the small-model ceiling on the ORT path is ~20–25×.

So "7.4× vs ~110×" is **~90 % engine** (ORT leaves the 0.6 B at ~5×; TensorRT was the thing that hits triple
digits) and the rest batch headroom — the cost of the reproducibility-vs-peak-RTF decision (sherpa-onnx over a
bespoke, non-portable TRT engine; see `../docs/TRADE-OFFS.md`).

**Conclusion:** the practical RTF ceiling for an **accurate (<3 %) small model** on the ORT path is **~25×**
(zipformer encoder, INT8, CPU 4-thread, or GPU batch-16) — **3.4× faster than the 0.6 B**, at a fraction of the
memory. 250× is a documented, explained miss; it would require **TensorRT** (engine) **and** large GPU batches
(memory) — directly opposing the ≤1 GB budget. That opposition *is* the assessment's core tension.

**Single deployable pick (memory + RTF + accuracy), measured on the FULL 300:** **zip70 transducer, CPU,
4 threads** — **WER 1.598 %, RTFx 21.6×, 204 MB resident** (vs the 0.6 B showcase: 1.331 %, 7.4×, 1247 MB).
**2.9× faster and ~6× smaller than the 0.6 B, at <3 % WER.** If memory tightens further post-deployment
(a Stage-2 question), drop to **zip28** (115 MB resident, 2.04 %) — still under 3 %.

---

## Summary — all three modes real, measured, and accurate behind one entrypoint
- **Mode A** (offline): WER **1.331 %**, RTFx 7.4, the **6.6 GB OOM** memory finding.
- **Mode B** (streaming): WER **1.976 %** (n=300); CPU > GPU for streaming; contention-sensitive RTF.
- **Mode C** (multilingual): LID **97 %**, German **7.41 %** (v3) / English 1.71 %; one-resident residency validated.

The **shared-box memory reality** (Mode A OOM), the **streaming CPU-vs-GPU + contention** findings, and the **on-device multilingual LID + accurate v3** are all measured, reproducible results for the Stage-2 memo — not predictions.
