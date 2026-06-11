#!/usr/bin/env bash
# Clean-env Zipformer benchmark (GeniePod stopped). CPU path = the deployable candidates.
# Runs zip70 (68MB) + zip28 (27MB) over the pinned LibriSpeech-300 -> WER + RTF + peak mem.
set -uo pipefail
cd "$HOME/edge-stt/work"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$HOME/edge-stt/ort/onnxruntime-linux-aarch64-gpu-1.18.1/lib

echo "### baseline (pre-run) ###"; free -m | awk 'NR==1||/Mem:/'

run() {
  local name=$1 dir=$2
  mkdir -p "results/$name"
  echo "=== $name START $(date +%T)  model=$dir ==="
  STT_ENGINE=sherpa MODE_A_MODEL_DIR="$HOME/edge-stt/models/$dir" MODE_A_MODEL_TYPE=transducer \
    ./transcribe --mode A --input data/manifest.jsonl \
      --out "results/$name/hyps.jsonl" --report "results/$name/report.json" --device cpu
  echo "=== $name DONE $(date +%T) ==="
}

run zip70 sherpa-onnx-zipformer-en-2023-06-26
run zip28 sherpa-onnx-zipformer-small-en-2023-06-26
echo "ALL_DONE $(date +%T)"
