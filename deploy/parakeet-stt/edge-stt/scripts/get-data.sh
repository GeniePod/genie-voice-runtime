#!/usr/bin/env bash
# The ONE committed data materializer. From an empty data/ it re-downloads LibriSpeech and
# writes data/wav/<id>.wav + data/manifest.jsonl (first-300 test-clean, sorted by id). The
# clean-room flow runs this before ./transcribe so the wavs are present from scratch. No audio
# payloads are committed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

# eval set: first-300 test-clean -> data/wav/<id>.wav + data/manifest.jsonl + data/manifest.sha256
"$PY" scripts/prepare_data.py --out data --n 300

# verify the eval manifest checksum
( cd data && sha256sum -c manifest.sha256 )

# record total audio seconds (RTFx denominator) into data/README.md
TOTAL_SEC="$("$PY" -c "import json;print(round(sum(json.loads(l)['duration_sec'] for l in open('data/manifest.jsonl')),1))")"
N="$(wc -l < data/manifest.jsonl)"
cat > data/README.md <<EOF
# Eval data (regenerated; not committed)

- Source: LibriSpeech \`test-clean\` (https://www.openslr.org/resources/12/test-clean.tar.gz, ~346 MB)
- Subset: first-$N by utterance id (ASSUMPTIONS.md FND-SUBSET)
- Utterances: $N
- Total audio: ${TOTAL_SEC} s

Regenerate with: \`bash scripts/get-data.sh\`. Manifests + checksums are committed; wav payloads are not.
EOF

echo "get-data OK: $N eval wavs materialized under data/wav/  (total ${TOTAL_SEC}s)"
