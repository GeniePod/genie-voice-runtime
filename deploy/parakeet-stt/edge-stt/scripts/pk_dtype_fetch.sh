#!/usr/bin/env bash
# Fetch the f32-derived 110m GGUF quant variants for the GPU dtype sweep. Resumable (curl -C -).
set -uo pipefail
cd "$HOME/parakeet-models"
BASE="https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main"
for V in q4_k q5_k f16; do
  F="tdt_ctc-110m-${V}.gguf"
  if [ -s "$F" ]; then echo "have $F ($(du -h $F|cut -f1))"; continue; fi
  echo "=== fetching $F  ($(date +%T)) ==="
  curl -L -C - --retry 5 --retry-delay 3 -o "$F" "$BASE/$F" 2>&1 | tail -1
  echo "  done $F -> $(du -h $F 2>/dev/null | cut -f1)  ($(date +%T))"
done
echo "FETCH_DONE $(date +%T)"
ls -la "$HOME"/parakeet-models/tdt_ctc-110m-*.gguf
