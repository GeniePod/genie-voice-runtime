#!/usr/bin/env bash
# scripts/build-parakeet.sh — build parakeet.cpp (ggml CUDA backend) from source on the Jetson.
#
# This is the DEPLOYABLE STT engine and the submission headline: on the pinned LibriSpeech-300 it
# reaches WER 1.836 % @ 60x (tdt) or 2.383 % @ 107x (ctc), +450 MB, ~126 ms/call — see
# results/parakeet/RESULTS.md. The ggml CUDA backend is ~4-14x faster than ONNX Runtime on the same
# 25 W Orin (sherpa-onnx topped out at 25x; the 0.6 B on ORT was compute-bound at 7.4x).
#
# WHY FROM SOURCE: the only reproducibility concern with parakeet.cpp was the *prebuilt* binaries
# (CUDA-13/Ubuntu-24.04 mismatch, undocumented sm_87). A clean native build pinned to v0.1.1 + sm_87
# removes that — the bench/transcribe subcommands the harness uses are upstream (no patch).
#
# Toolchain (asserted by scripts/preflight.sh): CUDA 12.6 / nvcc, sm_87. Native cmake — NOT the
# build-aarch64 cross script. Runtime needs LD_LIBRARY_PATH=/usr/local/cuda/lib64.
set -euo pipefail

SRC="${PARAKEET_SRC:-$HOME/parakeet.cpp}"
TAG="${PARAKEET_TAG:-v0.1.1}"
JOBS="${JOBS:-4}"   # 8 GB unified RAM is tight; bound parallelism

if [ ! -d "$SRC/.git" ]; then
  echo "[build-parakeet] cloning $TAG"
  git clone --recursive --branch "$TAG" https://github.com/mudler/parakeet.cpp "$SRC"
fi
cd "$SRC"
git submodule update --init --recursive   # ggml backend

command -v nvcc >/dev/null 2>&1 || export PATH=/usr/local/cuda/bin:$PATH
echo "[build-parakeet] $(git describe --tags 2>/dev/null || git rev-parse --short HEAD) | nvcc=$(command -v nvcc) | -j$JOBS"

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DPARAKEET_GGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DPARAKEET_BUILD_CLI=ON \
  -DPARAKEET_BUILD_TESTS=OFF
cmake --build build -j"$JOBS" --target parakeet-cli

BIN="$SRC/build/examples/cli/parakeet-cli"
echo "[build-parakeet] done -> $BIN"
[ -x "$BIN" ] && echo "[build-parakeet] OK ($("$BIN" 2>&1 | head -1 || true))"
echo "[build-parakeet] runtime: export LD_LIBRARY_PATH=/usr/local/cuda/lib64"
