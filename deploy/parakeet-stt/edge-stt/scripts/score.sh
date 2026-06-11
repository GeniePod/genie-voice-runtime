#!/usr/bin/env bash
# Thin WER-join wrapper over the one reporter. Recomputes WER over an EXISTING hyps.jsonl
# without re-running inference. Same report.json schema as `transcribe --report`.
#   scripts/score.sh <hyps.jsonl> <manifest.jsonl> [--report report.json]
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" -m src.common.wer_rtf_report "$@"
