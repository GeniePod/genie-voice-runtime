# Edge AI Speech-to-Text — Jetson Orin Nano Super 8 GB

An STT engine behind one CLI for the Jetson Orin Nano Super 8 GB. The **primary engine is parakeet.cpp**
(ggml CUDA) — the fastest, smallest, accurate option measured (**1.808 % WER @ 61×** with the q5_k tdt
default, or **2.383 % @ 107×** ctc, **+450 MB**). No single config meets all five brief constraints at once (WER ≤3 % · RTF ≤0.004 · ≤1 GB total ·
multilingual + dynamic · true streaming) — they are in direct tension — so a **3-mode** architecture (offline
/ streaming / multilingual) sits behind the CLI and the submission **reports honest, measured trade-offs**
(see [`docs/STAGE2-MEMO.md`](docs/STAGE2-MEMO.md) + [`docs/TRADE-OFFS.md`](docs/TRADE-OFFS.md)).

| Mode | Engine | Wins | Measured (LibriSpeech-300 / FLEURS-de) |
|---|---|---|---|
| **A** (primary) | offline — **parakeet.cpp** (ggml CUDA) `tdt_ctc-110m` (tdt=q5_k / ctc=q8_0) | **WER + RTF + memory** | **WER 1.808 %**, **RTF 0.0163 (61×)**, **+450 MB** · ctc decoder: 2.383 % @ **107×** |
| A (sherpa alt.) | sherpa-onnx-offline INT8 Zipformer (68 MB), CPU | reproducible fallback | 1.598 %, 21.6×, 204 MB resident |
| **B** | true streaming — sherpa-onnx-online INT8 Zipformer (chunk-16-left-128) | latency | **WER 1.976 %** (n=300), CPU |
| **C** | multilingual + dynamic — whisper-tiny LID + Parakeet-v3 (25-lang) | coverage | **LID 97 %**, German **7.41 %** (v3) |

> **Primary engine: parakeet.cpp (ggml CUDA).** This is the headline — `tdt_ctc-110m` reaches **61× RTF
> at 1.808 % WER** (decoder `tdt`, q5_k) or **107× at 2.383 %** (decoder `ctc`, q8_0), both < 3 %, at **+450 MB**
> and ~126 ms/call (results/parakeet/RESULTS.md). An on-GPU **dtype sweep** (`results/dtype/`) showed RTFx is
> *dtype-invariant* here — quant is a memory lever, not a speed one — so the tdt path runs q5_k for free
> (== q8_0 accuracy, 23 % smaller). The **ggml CUDA backend is ~4–14× faster than ONNX Runtime** on
> the same 25 W Orin — near-TensorRT throughput on a **reproducible from-source build** (no bespoke TRT
> engine). `./transcribe --mode A` uses it by default; pick the decoder with `PK_DECODER=tdt|ctc`. The
> sherpa-onnx Zipformer is the reproducible alternative (`STT_ENGINE=sherpa`), and Modes B/C (streaming +
> multilingual) stay on sherpa-onnx. **250× is still a documented miss** (~18–20× A100→Orin + 25 W), but
> parakeet.cpp closes most of the engine gap that sherpa-onnx could not. A bespoke **TensorRT** Mode A was
> evaluated and **deferred** (non-portable engine = the top blind-replication hazard, and it still misses
> 250×); parakeet.cpp gets near-TRT throughput without it. Reasoning: [`docs/STAGE2-MEMO.md`](docs/STAGE2-MEMO.md).

## Device assumptions (read first)

Used **as-is** on a **shared** Jetson — no reflash, no clock lock (see [`ASSUMPTIONS.md`](ASSUMPTIONS.md)):

- **L4T 36.4.7** · **CUDA 12.6** · **cuDNN 9.3.0.75** · Python 3.10 · aarch64 (sm_87). Build prereqs:
  `git`, `cmake`, a C++17 toolchain, `nvcc` on PATH (`/usr/local/cuda/bin`), `sox` (decode), and the
  **locked Python deps** in [`requirements.lock`](requirements.lock) (pinned, aarch64/cp310; `requirements.in`
  is the human-edited source). The GPU engines are built from source on-device, not pip-installed.
- Power mode **25 W** (not MAXN) — every RTF number carries this caveat.
- **Memory is the binding constraint.** ≤1 GB *total* is infeasible (the OS floor alone is ~1.5 GB), so we
  report total + floor + increment. The deployable parakeet.cpp engine adds only **+450 MB**; the 0.6 B
  alternative peaks ~6.6 GB and won't co-reside with a large workload on 8 GB. Run the eval on an otherwise-idle box.

## Deployment (every step is a committed script; the guide ends at `./transcribe`)

```sh
# 0. Verify the toolchain (CUDA 12.6 / nvcc / cuDNN 9.3; TensorRT warn-only — not used).
source scripts/preflight.sh           # or: ./transcribe --selfcheck

# 0b. Install the locked Python deps (scoring + data-prep; pinned for aarch64/cp310).
pip install --break-system-packages -r requirements.lock

# 1. Eval data: LibriSpeech test-clean first-300-by-id -> data/wav/ + data/manifest.jsonl
bash scripts/get-data.sh

# 2. Fetch models (parakeet.cpp GGUF + the sherpa-onnx models for B/C and the alt; HF, nothing committed).
bash scripts/get-models.sh            # -> ~/parakeet-models/ + ~/edge-stt/models/

# 3a. Build the PRIMARY engine: parakeet.cpp from source (ggml CUDA, sm_87). ~few min, -j4.
bash scripts/build-parakeet.sh        # -> ~/parakeet.cpp/build/examples/cli/parakeet-cli

# 3b. (Modes B/C + sherpa alternative) build the sherpa-onnx GPU binary. Pre-stage the ORT tarball once:
#     curl -L -o ~/Downloads/onnxruntime-linux-aarch64-gpu-cuda12-1.18.1.tar.bz2 \
#       https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.18.1/onnxruntime-linux-aarch64-gpu-cuda12-1.18.1.tar.bz2
bash scripts/build-sherpa.sh          # -> ~/edge-stt/sherpa-onnx/build/bin/{sherpa-onnx-offline,sherpa-onnx}

# 4. RUN. Both engines need the CUDA libs on LD_LIBRARY_PATH (sherpa also needs the ORT lib dir):
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$HOME/edge-stt/ort/onnxruntime-linux-aarch64-gpu-1.18.1/lib

# Mode A — parakeet.cpp (DEFAULT). PK_DECODER=tdt (1.81%/61x, q5_k) or ctc (2.38%/107x, q8_0); PK_MODEL overrides:
./transcribe --mode A --input data/manifest.jsonl --out hyps.jsonl --report report.json
PK_DECODER=ctc ./transcribe --mode A --input data/manifest.jsonl --out hyps_ctc.jsonl --report rctc.json
STT_ENGINE=sherpa ./transcribe --mode A --input data/manifest.jsonl --out hyps_sherpa.jsonl --report rs.json --device cpu  # alt

./transcribe --mode B --input data/manifest.jsonl --out hyps_B.jsonl --report rB.json --device cpu  # streaming
./transcribe --mode C --input <de_manifest> --out hyps_C.jsonl --report rC.json           # multilingual (LID auto)
```

`report.json` carries WER (Whisper-normalized) + RTFx + the memory accountant's `mem.*`. Re-score an
existing `hyps.jsonl` without re-running inference via `bash scripts/score.sh hyps.jsonl data/manifest.jsonl`.
The frozen CLI + schemas are in [`docs/CONTRACT.md`](docs/CONTRACT.md).

## Results & honesty

Measured numbers + the gate verdicts (WER pass; RTF/memory honest misses with gap analysis; streaming +
multilingual) are in [`results/feasibility.md`](results/feasibility.md) and machine-readable in
[`results/decisions.json`](results/decisions.json). The headline trade-offs map to the Stage-2 questions in
[`docs/TRADE-OFFS.md`](docs/TRADE-OFFS.md).

Full reasoning (the four Stage-2 questions — memory reduction, engine choice, quant/precision/size, adapting
if memory tightens) is in [`docs/STAGE2-MEMO.md`](docs/STAGE2-MEMO.md).

### Mode notes (engineering findings, all measured)
- **Mode A (parakeet.cpp, primary)** loads the GGUF once (`load_ms`, excluded from RTF) and benches the batch;
  per-file `proc_ms` is the inference-only time. `PK_DECODER` picks tdt (accurate) or ctc (fast). The
  sherpa-onnx alternative (`STT_ENGINE=sherpa`) sorts clips shortest-first and splits a chunk to singles if the
  ORT CUDA arena overflows.
- **Mode B** runs on **CPU** — GPU is ~3× *slower* for the tiny streaming model (per-chunk kernel-launch
  overhead). Streaming RTF is reported separately from offline, never conflated.
- **Mode C** uses whisper-tiny for both LID (the JSON `lang` field) and multilingual recognition; the accurate
  25-lang **Parakeet-v3** drops in behind the same router for production accuracy (German 7.41 %).
