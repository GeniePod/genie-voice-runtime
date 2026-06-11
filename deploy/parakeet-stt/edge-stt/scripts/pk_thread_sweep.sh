#!/usr/bin/env bash
# parakeet.cpp Jetson thread sweep — the kDefaultThreads=8 in ggml_graph.cpp was tuned on a
# 20-core dual-CCD HOST CPU, not this 6-core Orin. Sweep --threads on the GPU path to find the
# Jetson optimum. Clean env (GeniePod stopped). Full LibriSpeech-300, ctc decoder (the fast path).
set -uo pipefail
cd "$HOME/edge-stt/work"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
BIN="$HOME/parakeet.cpp/build/examples/cli/parakeet-cli"
MODEL="$HOME/parakeet-models/tdt_ctc-110m-q8_0.gguf"

# plain wav-path-per-line manifest, resolved to the Jetson's wav dir
python3 -c "import json;[print(json.loads(l)['wav']) for l in open('data/manifest.jsonl')]" \
  | sed "s#.*/wav/#$HOME/edge-stt/work/data/wav/#" > /tmp/pk_wavs.txt
echo "manifest: $(wc -l < /tmp/pk_wavs.txt) wavs   ($(date +%T))"

rtfx() { python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
# find the per-file list (objects carrying proc_ms)
def files(o):
    if isinstance(o,list) and o and isinstance(o[0],dict) and 'proc_ms' in o[0]: return o
    if isinstance(o,dict):
        for v in o.values():
            r=files(v)
            if r: return r
    return None
fs=files(d) or []
a=sum(x['audio_sec'] for x in fs); p=sum(x['proc_ms'] for x in fs)/1000.0
print('  RTFx=%.1f  audio=%.0fs proc=%.2fs  load_ms=%s  n=%d'%(a/p if p else 0,a,p,d.get('load_ms'),len(fs)))
" "$1"; }

for DEC in ctc tdt; do
  for T in 3 4 5 6 8; do
    echo "=== decoder=$DEC threads=$T  ($(date +%T)) ==="
    "$BIN" bench --model "$MODEL" --manifest /tmp/pk_wavs.txt --decoder "$DEC" \
       --threads "$T" --json "/tmp/pk_${DEC}_t${T}.json" 2>/tmp/pk_${DEC}_t${T}.err
    rtfx "/tmp/pk_${DEC}_t${T}.json" 2>/dev/null || echo "  (parse fail; err: $(tail -1 /tmp/pk_${DEC}_t${T}.err))"
  done
done
echo "SWEEP_DONE $(date +%T)"
