#!/usr/bin/env bash
# Build the sherpa-onnx GPU binary NATIVELY on the Jetson (the spine for Modes B/C + Mode-A fallback).
#
# NOTE: do NOT use sherpa-onnx's build-aarch64-linux-gnu.sh here — that is a CROSS-compile script
# (needs aarch64-linux-gnu-gcc + a toolchain file). On the Jetson we are already aarch64, so we
# drive cmake directly with the native gcc/g++.
#
# Pinned: sherpa-onnx v1.13.2. GPU onnxruntime: 1.18.1 (the version sherpa-onnx documents for
# Jetson Orin Nano Super / JetPack 6.2 / CUDA 12.6 / cuDNN 9). The Jetson's link to the GitHub
# release CDN that hosts the ~49 MB ORT-GPU tarball is SLOW/flaky, and cmake's FetchContent
# restarts from zero on any stall. So we PRE-STAGE the tarball once (resumable curl) and point
# sherpa-onnx's SHERPA_ONNXRUNTIME_{INCLUDE,LIB}_DIR env vars at the extracted tree — the build
# then skips the FetchContent download entirely. Stage with:
#   curl -L -C - --retry 20 -o ~/Downloads/onnxruntime-linux-aarch64-gpu-cuda12-1.18.1.tar.bz2 \
#     https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.18.1/onnxruntime-linux-aarch64-gpu-cuda12-1.18.1.tar.bz2
set -euo pipefail

SHERPA_SRC="${SHERPA_SRC:-$HOME/edge-stt/sherpa-onnx}"
ORT_VERSION="${ORT_VERSION:-1.18.1}"
ENABLE_GPU="${ENABLE_GPU:-ON}"
JOBS="${JOBS:-4}"   # bound parallelism: 8GB unified RAM is tight on this shared box

# --- Pre-staged onnxruntime-gpu (skip the slow FetchContent download if present) ---
# NOTE: the release tarball is named '...-gpu-cuda12-<ver>.tar.bz2' but its INTERNAL
# top dir is '...-gpu-<ver>' (no 'cuda12'), so we glob for the extracted dir by its
# header rather than hard-coding the name.
ORT_NAME="onnxruntime-linux-aarch64-gpu-cuda12-${ORT_VERSION}"
ORT_TARBALL="${ORT_TARBALL:-$HOME/Downloads/${ORT_NAME}.tar.bz2}"
ORT_ROOT="${ORT_ROOT:-$HOME/edge-stt/ort}"
_find_ort() {
  local d
  for d in "$ORT_ROOT"/onnxruntime-linux-aarch64-gpu*"${ORT_VERSION}"; do
    [ -f "$d/include/onnxruntime_cxx_api.h" ] && { echo "$d"; return 0; }
  done
  return 1
}
ORT_DIR="${ORT_DIR:-$(_find_ort || true)}"
if [ -z "$ORT_DIR" ] && [ -f "$ORT_TARBALL" ]; then
  echo "[build-sherpa] extracting pre-staged ORT tarball from $ORT_TARBALL"
  mkdir -p "$ORT_ROOT"
  tar xjf "$ORT_TARBALL" -C "$ORT_ROOT"
  ORT_DIR="$(_find_ort || true)"
fi
if [ -n "$ORT_DIR" ] && [ -d "$ORT_DIR/include" ] && [ -d "$ORT_DIR/lib" ]; then
  export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$ORT_DIR/include"
  export SHERPA_ONNXRUNTIME_LIB_DIR="$ORT_DIR/lib"
  echo "[build-sherpa] using pre-staged ORT at $ORT_DIR (FetchContent download skipped)"
else
  echo "[build-sherpa] WARN: no pre-staged ORT under $ORT_ROOT or $ORT_TARBALL — cmake will download v$ORT_VERSION (slow on this link)"
fi

cd "$SHERPA_SRC"
echo "[build-sherpa] $(git describe --tags 2>/dev/null || git rev-parse --short HEAD) | GPU=$ENABLE_GPU | ORT=$ORT_VERSION | -j$JOBS"

mkdir -p build && cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DSHERPA_ONNX_ENABLE_GPU="$ENABLE_GPU" \
  -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION="$ORT_VERSION" \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  ..
make -j"$JOBS" sherpa-onnx-offline sherpa-onnx

echo "[build-sherpa] done -> $SHERPA_SRC/build/bin/"
ls -lh "$SHERPA_SRC/build/bin/" | grep -E 'sherpa-onnx(-offline)?$' || ls "$SHERPA_SRC/build/bin/"
