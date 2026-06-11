#!/usr/bin/env bash
# CLI wrapper over the one memory accountant. Samples total RAM + RSS + CUDA around a command
# (or an already-running --pid). On the Jetson it uses tegrastats; elsewhere /proc/meminfo.
#   scripts/mem-sample.sh -- <command...>
#   scripts/mem-sample.sh --pid <pid> --duration 5 --json
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" -m src.common.memory_accountant "$@"
