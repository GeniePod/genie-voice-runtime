#!/usr/bin/env bash
# GPU dtype sweep for tdt_ctc-110m on the Orin (clean box). Each config runs through the OFFICIAL
# harness (transcribe -> scorer + memory accountant) so WER / RTFx / peak-mem are apples-to-apples
# with the submission's other numbers. Upstream only benched dtypes on CPU; this is the GPU truth.
set -uo pipefail
cd "$HOME/edge-stt/work"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
mkdir -p results/dtype
MODELS="$HOME/parakeet-models"

summ() { python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
m=d.get('mem',{})
print('  WER=%5s%%  RTFx=%6s  peak_total=%sMB  proc_rss=%sMB'%(
  d.get('wer_pct'), d.get('rtfx'), m.get('peak_total_mb'), m.get('proc_rss_mb')))
" "$1" 2>/dev/null || echo "  (no report)"; }

for DT in q8_0 q4_k q5_k f16; do
  G="$MODELS/tdt_ctc-110m-${DT}.gguf"
  if [ ! -s "$G" ]; then echo "### $DT: MISSING $G — skip"; continue; fi
  SZ=$(du -h "$G" | cut -f1)
  for DEC in ctc tdt; do
    echo "=== dtype=$DT ($SZ)  decoder=$DEC  ($(date +%T)) ==="
    PK_MODEL="$G" PK_DECODER="$DEC" \
      ./transcribe --mode A --input data/manifest.jsonl \
        --out "results/dtype/${DT}_${DEC}.jsonl" \
        --report "results/dtype/${DT}_${DEC}.json" --device cuda 2>/dev/null
    summ "results/dtype/${DT}_${DEC}.json"
  done
done
echo "DTYPE_BENCH_DONE $(date +%T)"
