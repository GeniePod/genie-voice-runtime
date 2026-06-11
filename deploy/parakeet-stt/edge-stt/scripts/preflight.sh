#!/usr/bin/env bash
# scripts/preflight.sh — the SINGLE preflight contract (asserts the toolchain the runtime needs).
#
# M0 (this file): HARD-asserts the load-bearing runtime toolchain (CUDA 12.6 / cuDNN 9.3) and
# RECORDS L4T + TensorRT. `transcribe --selfcheck` shells this. M3-8 EXTENDS this same file
# (do not fork a second one).
#
# ENGINE DECISION (2026-06-09, ROADMAP §Decision amendment + work/docs/TRADE-OFFS.md): all three
# modes ship on sherpa-onnx + onnxruntime-gpu (CUDA Execution Provider). The CUDA EP needs
# CUDA + cuDNN, NOT TensorRT — so TensorRT is now a WARN-ONLY record (the deferred Mode-A
# contingency toolchain), not a hard gate. A sherpa-only reproduction must NOT fail for lack of
# the exact TRT version.
#
# ADAPTATION (shared device): the original plan pinned an EXACT L4T 36.4.3 clean-room string and
# would hard-fail otherwise. This device is a shared box at L4T 36.4.7 used as-is, so we assert
# the R36 major release + the load-bearing CUDA/cuDNN versions and only WARN on L4T point-release
# drift. MAXN/clock asserts are intentionally NOT enforced (locking clocks would disturb other
# workloads); RTF results carry the power-mode caveat.
set -euo pipefail

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
warn() { echo "PREFLIGHT WARN: $*" >&2; }

# --- CUDA 12.6 (HARD — onnxruntime-gpu CUDA EP) ---
if command -v nvcc >/dev/null 2>&1; then NVCC=nvcc; else NVCC=/usr/local/cuda/bin/nvcc; fi
"$NVCC" --version 2>/dev/null | grep -q 'release 12.6' || fail "CUDA != 12.6"

# --- cuDNN 9.3 (HARD — onnxruntime-gpu CUDA EP) ---
# Match the version field independent of dpkg column spacing (e.g. 'libcudnn9-cuda-12  9.3.0.75-1').
dpkg -l 2>/dev/null | grep -E '^ii +libcudnn9' | grep -Eq ' 9\.3\.' || fail "cuDNN != 9.3"

# --- TensorRT 10.3 (WARN-ONLY — deferred Mode-A contingency, not required by the sherpa runtime) ---
if dpkg -l 2>/dev/null | grep -E '^ii +libnvinfer' | grep -Eq ' 10\.3\.'; then
  TRT_NOTE="TensorRT 10.3 present (deferred Mode-A contingency)"
else
  TRT_NOTE="TensorRT 10.3 NOT found — fine: sherpa-onnx runtime does not need it"
  warn "$TRT_NOTE"
fi

# --- L4T: record, warn-only on point-release drift (adaptation) ---
L4T="$(grep -oE 'R[0-9]+ \(release\), REVISION: [0-9.]+' /etc/nv_tegra_release 2>/dev/null || true)"
[ -n "$L4T" ] || fail "not a Jetson / /etc/nv_tegra_release missing"
echo "$L4T" | grep -q 'R36 (release)' || fail "unexpected L4T major (need R36): $L4T"
case "$L4T" in
  *"REVISION: 4.3"*) ;;
  *) warn "L4T is '$L4T' (plan referenced 36.4.3); proceeding as-is on shared device" ;;
esac

echo "preflight OK: $L4T / CUDA 12.6 / cuDNN 9.3 (runtime toolchain matches; device used as-is) | $TRT_NOTE"
