# STT: whisper.cpp → parakeet.cpp evaluation and swap

This documents replacing the speech-to-text engine in the deployed voice pipeline
(`genie-claw` / GeniePod on the Jetson Orin Nano Super) from **whisper.cpp
(`ggml-small`)** to **parakeet.cpp (`tdt_ctc-110m`)**, and the head-to-head
comparison that motivated it. It closes the *"STT accuracy (WER) on real LyraT
captures vs clean reference"* open item in #2.

## Summary

On the target device, **parakeet `tdt_ctc-110m-q8_0` beats whisper `ggml-small`
on every axis** — accuracy, speed, and memory — and it is the *small* parakeet
model.

| Metric (lower=better unless noted) | whisper `ggml-small` (original) | parakeet `tdt_ctc-110m-q8` (current) |
|---|---|---|
| **WER**, 50 clean clips (Whisper-normalized) | 1.02 % | **0.82 %** |
| **RTFx** amortized (higher=better) | 19× | **116×** |
| RTFx naive, incl. per-file model load | 5.1× | 13.8× |
| **Per-call latency**, resident server | ~500 ms (config-stated) | **~170 ms** (measured 168–189 ms) |
| **Peak RAM** Δ over baseline | +726 MB | **+410 MB** |
| Model on disk | 466 MB | **170 MB** |
| Real LyraT capture | correct | correct |

Net: **more accurate, ~6× faster compute, ~3× lower per-call latency, ~300 MB
leaner.**

## Test setup

- **Device:** Jetson Orin Nano Super 8 GB · JetPack 6.2 (L4T 36.4.7) · CUDA 12.6 ·
  GPU `CUDA0` (Orin, compute 8.7). Unified 7.6 GB. Both engines measured with the
  genie LLM/STT services stopped so the GPU was uncontended and conditions were
  identical.
- **Eval set:** first 50 LibriSpeech `test-clean` clips (≈6.3 min), references from
  the corpus. WER via the Whisper normalizer (Open ASR Leaderboard convention,
  `whisper-normalizer`) + `jiwer`, applied identically to both engines.
- **whisper:** `/opt/geniepod/models/ggml-small.bin` via `whisper-cli` and the
  long-running `whisper-server` (`POST /inference`, the deployed integration).
- **parakeet:** `tdt_ctc-110m-q8_0.gguf` (177,796,224 bytes), parakeet.cpp
  `v0.1.1`, decoder `tdt`.

### Real-capture validation

Live ESP32-LyraT mic → I2S2 → ADMAIF1 → `plughw:APE,0` (24 kHz, per #2) → resampled
to 16 kHz mono → parakeet. Three live utterances transcribed correctly, e.g.
*"Hello, this is … okay, thank you."* and *"I'm a software engineer, can you
check me?"* (minor proper-noun spelling only). The clean-set WER is therefore a
ceiling; a labeled real-capture set is still needed to quantify the noisy-24 kHz
number at scale (tracked back into #2).

## What we changed

parakeet.cpp `v0.1.1` is **CLI-only** and reloads the 170 MB model on every
invocation (~1.3 s), which cannot match whisper-*server*'s resident ~500 ms. The
deployed integration point is whisper-server's `POST /inference` on
`127.0.0.1:8178` (genie-core calls it; config `whisper_port = 8178`). So:

1. **Resident `serve` mode** — patched parakeet.cpp's CLI to add a `serve`
   subcommand: load the model **once**, then read wav paths on stdin and emit one
   transcript per line. See [`deploy/parakeet-stt/serve-mode.patch`](../deploy/parakeet-stt/serve-mode.patch).
2. **Drop-in HTTP shim** — [`deploy/parakeet-stt/parakeet-stt-server.py`](../deploy/parakeet-stt/parakeet-stt-server.py)
   spawns `parakeet-cli serve` once and serves the **same** `POST /inference` →
   `{"text": ...}` contract on `:8178` (sox-resamples input to 16 kHz mono).
   Measured **~170 ms/call resident**.
3. **systemd service** — [`deploy/parakeet-stt/genie-parakeet.service`](../deploy/parakeet-stt/genie-parakeet.service)
   runs the shim (`Restart=always`, starts on boot, after `genie-ai-runtime`).
4. **Swap** — `systemctl disable --now genie-whisper` + `enable --now
   genie-parakeet`. **genie-core is unchanged** (it still posts to `:8178`;
   parakeet now answers). Fully reversible.

See [`deploy/parakeet-stt/README.md`](../deploy/parakeet-stt/README.md) for the
exact build + install + revert commands.

## Caveats

- Clean LibriSpeech subset — both engines score very low; the real win on noisy
  24 kHz LyraT captures is not yet quantified at scale (only the single live
  capture above).
- `tdt_ctc-110m` is the **lean** model. `tdt-0.6b-v2-q8` is more accurate
  (~1.7 % on the leaderboard) but ~904 MB and heavier to load/run — swap it in via
  the service's `PK_MODEL` env if accuracy outweighs footprint.
- Numbers are `q8_0` quantization on this exact device; treat as device-specific.

## Build notes (Jetson, parakeet.cpp v0.1.1)

- Build **natively** with `cmake` — **not** the `build-aarch64-linux-gnu.sh`
  cross-compile script. `nvcc` must be on `PATH` (`/usr/local/cuda/bin`); pass
  `-DPARAKEET_GGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87`.
- Runtime needs `LD_LIBRARY_PATH=/usr/local/cuda/lib64`.
- parakeet requires **16 kHz mono** input (it rejects 48 kHz/stereo; the shim
  resamples with `sox`).
- Prebuilt GGUF models: `huggingface.co/mudler/parakeet-cpp-gguf`.

## Optional: streaming

The cache-aware streaming model `realtime_eou_120m-v1` runs on this device
(`parakeet-cli transcribe --stream`) and produces incremental partials,
end-of-utterance (`[EOU]`) detection, and per-word timestamps — enabling
EOU-driven turn-taking instead of fixed-window recording. Validated but not wired
into the runtime; left as a follow-on.
