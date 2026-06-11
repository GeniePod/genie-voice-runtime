#!/usr/bin/env bash
# scripts/get-models.sh — the ONE model materializer. Fetches the sherpa-onnx model dirs for all
# three modes from HuggingFace into ~/edge-stt/models/ (override with MODELS_DIR). INT8 throughout.
#
# Engine decision (ROADMAP §Decision amendment + docs/TRADE-OFFS.md): sherpa-onnx for all three
# modes; no TensorRT engine build. Models are portable across machines — NO model is committed.
#
#   Mode A  Parakeet-TDT-0.6b-v2 int8   (offline; the WER/RTF/memory mode)
#   Mode B  streaming Zipformer EN int8 (chunk-16-left-128; true streaming)
#   Mode C  whisper-tiny int8 (LID + multilingual) + Parakeet-TDT-0.6b-v3 int8 (accurate 25-lang)
#
# Large files use PARALLEL range downloads (HF throttles per-connection; ~16x faster than a single
# stream). Each file is size-verified; a short/failed file is retried.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/edge-stt/models}"
HF="https://huggingface.co"
JOBS="${DL_JOBS:-8}"
mkdir -p "$MODELS_DIR"

# pdl <repo> <relpath> <outfile> — parallel range download with size verify + retry
pdl() {
  local repo="$1" rel="$2" out="$3"
  local url="$HF/$repo/resolve/main/$rel"
  local total; total=$(curl -sIL "$url" 2>/dev/null | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  if [ -z "${total:-}" ] || [ "$total" -lt 20000000 ]; then
    curl -fL --retry 10 --retry-all-errors -o "$out" "$url"; return
  fi
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" -eq "$total" ]; then echo "  [skip] $rel"; return; fi
  echo "  [pdl] $rel ($((total/1000000)) MB, -j$JOBS)"
  local pd; pd=$(mktemp -d); local n="$JOBS" ch=$(( (total + JOBS - 1) / JOBS )) i
  for i in $(seq 0 $((n-1))); do
    local s=$((i*ch)) e=$((i*ch+ch-1)); [ $e -ge $total ] && e=$((total-1)); local w=$((e-s+1))
    ( until curl -sL -r "${s}-${e}" -o "$pd/p$i" "$url" && [ "$(stat -c%s "$pd/p$i")" -eq "$w" ]; do sleep 3; done ) &
  done
  wait
  cat $(for i in $(seq 0 $((n-1))); do echo "$pd/p$i"; done) > "$out"; rm -rf "$pd"
  [ "$(stat -c%s "$out")" -eq "$total" ] || { echo "  [FAIL] $rel size mismatch"; return 1; }
}

fetch_repo() { # <repo> <dest_subdir> <file...>
  local repo="$1" dest="$MODELS_DIR/$2"; shift 2
  mkdir -p "$dest"
  echo "== $repo -> $dest"
  for f in "$@"; do pdl "$repo" "$f" "$dest/$(basename "$f")"; done
}

# ---- PRIMARY ENGINE: parakeet.cpp GGUF (the deployable headline; built by build-parakeet.sh) ----
# tdt_ctc-110m on LibriSpeech-300 -> WER 1.808%/61x (tdt, q5_k) or 2.383%/107x (ctc, q8_0), +450MB,
# 126ms/call. The ggml CUDA backend is the fastest path measured (~4-14x ONNX Runtime).
# DTYPE NOTE (results/dtype/RESULTS.md): on the Orin GPU, RTFx is dtype-INVARIANT (launch-bound at
# batch=1) -> quant is a MEMORY lever, not a speed one. So the accuracy (tdt) path defaults to q5_k
# (1.808% @ 137MB, == q8_0 accuracy, 23% smaller); the speed (ctc) path keeps q8_0; q4_k (126MB) is
# the if-memory-tightens fallback (2.018%/2.467%, still <3%). Per-decoder default in parakeet_runner.py.
PK_DIR="${PK_MODELS_DIR:-$HOME/parakeet-models}"; mkdir -p "$PK_DIR"
echo "== parakeet.cpp GGUF -> $PK_DIR"
pdl mudler/parakeet-cpp-gguf tdt_ctc-110m-q8_0.gguf "$PK_DIR/tdt_ctc-110m-q8_0.gguf"  # ctc default
pdl mudler/parakeet-cpp-gguf tdt_ctc-110m-q5_k.gguf "$PK_DIR/tdt_ctc-110m-q5_k.gguf"  # tdt default
pdl mudler/parakeet-cpp-gguf tdt_ctc-110m-q4_k.gguf "$PK_DIR/tdt_ctc-110m-q4_k.gguf"  # memory-tightens
# optional streaming EOU model:
# pdl mudler/parakeet-cpp-gguf realtime_eou_120m-v1-q8_0.gguf "$PK_DIR/realtime_eou_120m-v1-q8_0.gguf"

# ---- Mode A (sherpa-onnx alternative / accuracy showcase): Parakeet-TDT-0.6b-v2 int8 --------
fetch_repo csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 \
  sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 \
  encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt test_wavs/0.wav

# ---- Mode A DEPLOYABLE (recommended): small offline Zipformer transducers --------------------
# Best memory+RTF balance for the assessment: zip70 = 1.598% WER / 21.6x RTFx / 204MB resident on CPU
# (vs the 0.6b showcase's 1.331% / 7.4x / 1247MB). zip28 is the further-tightened fallback.
ZV=epoch-99-avg-1
fetch_repo csukuangfj/sherpa-onnx-zipformer-en-2023-06-26 \
  sherpa-onnx-zipformer-en-2023-06-26 \
  "encoder-$ZV.int8.onnx" "decoder-$ZV.int8.onnx" "joiner-$ZV.int8.onnx" tokens.txt
fetch_repo csukuangfj/sherpa-onnx-zipformer-small-en-2023-06-26 \
  sherpa-onnx-zipformer-small-en-2023-06-26 \
  "encoder-$ZV.int8.onnx" "decoder-$ZV.int8.onnx" "joiner-$ZV.int8.onnx" tokens.txt

# ---- Mode B: streaming Zipformer EN int8 (chunk-16-left-128) --------------------------------
V=epoch-99-avg-1-chunk-16-left-128
fetch_repo csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  sherpa-onnx-streaming-zipformer-en-2023-06-26 \
  "encoder-$V.int8.onnx" "decoder-$V.int8.onnx" "joiner-$V.int8.onnx" tokens.txt

# ---- Mode C: whisper-tiny (LID + multilingual) ---------------------------------------------
fetch_repo csukuangfj/sherpa-onnx-whisper-tiny \
  sherpa-onnx-whisper-tiny \
  tiny-encoder.int8.onnx tiny-decoder.int8.onnx tiny-tokens.txt

# ---- Mode C: Parakeet-TDT-0.6b-v3 int8 (accurate 25-lang recognizer) ------------------------
fetch_repo csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 \
  sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 \
  encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt

echo "get-models OK -> $MODELS_DIR  (Mode A v2, Mode B zipformer, Mode C whisper-tiny + v3; INT8; nothing committed)"
