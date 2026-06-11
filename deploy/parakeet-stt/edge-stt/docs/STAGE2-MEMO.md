# Stage-2 memo — design reasoning & trade-offs

Every number here is measured on the **Jetson Orin Nano Super 8 GB** (L4T 36.4.7, **25 W**, sm_87),
LibriSpeech test-clean **300** (Whisper-normalized WER), in isolation (background services stopped for a
clean baseline). The four prompts from the brief, answered in order.

---

## 1. How did you reduce memory footprint without sacrificing accuracy?

The naive starting point — **Parakeet-TDT-0.6 B INT8 on ONNX Runtime — peaked at 6.6 GB** and OOM-rebooted
the device. Decomposing that peak was the key move, because **almost none of it is the model**:

| component | ~size | reducible? |
|---|---|---|
| OS / L4T floor | ~1.5 GB | no |
| GPU CUDA context (any CUDA model) | ~1 GB | drop on CPU |
| ORT activation arena (grows over a batch) | ~3 GB | drop with small batch / better engine |
| model weights | ~0.65 GB | use a smaller model |

So the levers were (a) **smaller model**, (b) **leaner engine**, (c) **bounded batching**. The endpoint —
**parakeet.cpp `tdt_ctc-110m` (q5_k, the tdt default) — runs at +450 MB active-inference with WER 1.808 %**
(vs the 0.6 B's 1.331 %). A ~0.48-point WER cost for a **~14× smaller footprint** and (below) **8× the speed**.
"Without sacrificing accuracy" is literal here: 1.81 % is far inside the 3 % budget, so the footprint win is free of
any *gate*-relevant accuracy loss. (≤ 1 GB *total* is infeasible — the OS floor alone is ~1.5 GB — so we
report total + floor + the +450 MB increment, and lead with that honesty.)

## 2. Why did you select your inference engine over alternatives?

We measured three engines on the same model class + hardware + INT8:

| engine | RTF (LibriSpeech-300) | note |
|---|---|---|
| ONNX Runtime CUDA EP (sherpa-onnx), 0.6 B | **7.4×** | compute-bound; batching gave *no* gain (4.2→4.8× at batch-16) |
| ONNX Runtime CUDA EP, small Zipformer | **25×** | encoder-light, but ORT caps it |
| **parakeet.cpp (ggml CUDA), 110 M** | **61× (tdt) / 107× (ctc)** | **chosen** |

**parakeet.cpp's ggml CUDA backend is ~4–14× faster than ONNX Runtime on identical hardware.** That is the
decisive fact: the engine, not just the model, was leaving most of the GPU on the table. ggml (the runtime
behind the fast on-device LLMs) reaches **near-TensorRT throughput** here. We chose it over:
- **TensorRT** — it *could* be faster still, but it requires a bespoke, device-keyed, non-portable engine +
  a hand-written batched decode (the #1 Stage-1 blind-replication hazard) and more memory (batching). parakeet.cpp
  gets ~107× on a **reproducible from-source build** (`scripts/build-parakeet.sh`, pinned v0.1.1 / sm_87) — the
  *only* concern that originally disqualified parakeet.cpp was its **prebuilt** binaries, which a clean native
  build removes.
- **sherpa-onnx / ORT** — kept as the **reproducible fallback** (`STT_ENGINE=sherpa`) and as the **streaming
  (Mode B)** and **multilingual (Mode C)** engines, where its model zoo is the pragmatic choice.

## 3. What trade-offs did you navigate around quantization, precision loss, or model size?

- **Quantization (measured on-GPU, not assumed):** we ran a full **dtype sweep** of the 110 M model on
  the Orin GPU — q4_k / q5_k / q8_0 / f16 × both decoders, full LibriSpeech-300 (`results/dtype/`). The
  decisive result: **RTFx is dtype-invariant** (ctc ~107×, tdt ~61× for *every* dtype — q4_k @ 126 MB is no
  faster than f16 @ 256 MB). At batch=1 the engine is **kernel-launch/latency-bound, not bandwidth-bound**,
  so **quantization is a memory lever, not a speed one**. That handed us a free win: the accuracy (tdt) path
  drops **q8_0 → q5_k** (1.808 % WER @ 137 MB — *matches* q8_0's 1.836 % at 23 % smaller, identical speed);
  the speed (ctc) path keeps q8_0. **q4_k** (126 MB, 2.018 %/2.467 %, still < 3 %) is the next rung down
  (answer 4). Upstream only benched dtypes on a host *CPU*; this is the on-device GPU truth.
- **Model size:** the **110 M vs 0.6 B** decision is the crux. 0.6 B = 1.33 % WER but 7.4×/6.6 GB; 110 M =
  1.81 % WER but **61–107×/+450 MB**. Since both clear 3 %, the assessment's *memory + RTF* emphasis makes the
  small model strictly better. We keep the 0.6 B documented as an accuracy *showcase*, not the deployable STT.
- **Decoder (a parakeet-specific lever):** the `tdt_ctc-110m` model exposes two decode heads — **tdt**
  (q5_k, **1.808 % @ 61×**) vs **ctc** (q8_0, **2.383 % @ 107×**). A clean ~0.57-point WER ⇄ ~1.75× RTF dial,
  both < 3 %. The gap is *structural*: **ctc** is one parallel matmul over all frames (no token context) →
  fast; **tdt** is an autoregressive transducer (the per-(t,u) joint graph is rebuilt every step — "the bulk
  of the cost" in the source) whose prediction net conditions each token on prior tokens (an implicit LM) →
  more accurate but sequential/latency-bound. tdt is the accuracy default; ctc is the speed default.
- **The 250× RTF gate is an honest miss** (best 107×). It is a datacenter-batched figure; on a 25 W Orin the
  A100→Orin gap is ~18–20× before the no-large-batch and power caveats. We report best-achieved + the gap, and
  never present a batched number as streaming. Closing the rest needs TensorRT **and** large GPU batches —
  which directly oppose the memory budget. **That opposition is the core tension the brief is built on**, and
  we resolve it toward memory + reproducibility, not a number that still misses.

## 4. How would the architecture adapt if memory constraints tightened further post-deployment?

A ladder of measured/known levers, cheapest first:
1. **Switch decoder tdt → ctc** — already supported; trims decode-side memory + ~1.8× faster, +0.55 WER.
2. **Lower quant → q4_k** — the tdt path *already* defaults to q5_k (137 MB). The next rung, **q4_k**
   (126 MB), is **measured** (`results/dtype/`): 2.018 % (tdt) / 2.467 % (ctc), both still < 3 %. (Quant
   buys memory, not speed here — RTFx is dtype-invariant on this GPU.)
3. **Smaller / CPU path** — the sherpa-onnx Zipformer fallback runs at **204 MB resident on CPU** (no CUDA
   context at all), 1.6 % WER, 21.6× — when GPU memory must go to another tenant.
4. **One-resident model manager** — `src/modes/c_multilingual/residency.py` (registry + LID router + TTL/LRU,
   evict-before-load) guarantees only one heavy model is resident at a time; the multilingual mode already
   proves the invariant. The same manager bounds memory if more languages/models are added.
5. **Streaming (Mode B)** processes audio in chunks with bounded state — fixed memory regardless of utterance
   length, for the always-on path.

The design is engine- and model-pluggable (`STT_ENGINE`, `PK_MODEL`, `PK_DECODER`, `MODE_A_MODEL_DIR`), so each
rung above is a config change, not a rebuild — which is the production-minded posture the brief asks for.

---

*Pointers: headline eval `results/parakeet/RESULTS.md`; engine/RTF/memory study `results/feasibility.md`;
machine-readable gates `results/decisions.json`; the three-tension framing `docs/TRADE-OFFS.md`.*
