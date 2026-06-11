#!/usr/bin/env python3
"""
Mode A PRIMARY engine — parakeet.cpp (ggml CUDA backend).

This is the deployable headline (results/parakeet/RESULTS.md): on LibriSpeech-300, tdt_ctc-110m
gives WER 1.808 % @ 61x (decoder=tdt, q5_k) or 2.383 % @ 107x (decoder=ctc, q8_0), +450 MB,
~126 ms/call — the fastest path measured (the ggml CUDA backend is ~4-14x ONNX Runtime on the same
25 W Orin). The per-decoder GGUF default comes from the on-GPU dtype sweep (results/dtype/).

It drives `parakeet-cli bench` over a wav-per-line manifest: the model loads ONCE (load_ms, excluded
from RTF) and each file reports audio_sec + proc_ms (inference-only) + text. Built by
scripts/build-parakeet.sh; runtime needs LD_LIBRARY_PATH=/usr/local/cuda/lib64.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

_HOME = Path(os.path.expanduser("~"))
DEFAULTS = {
    "binary": os.environ.get("PARAKEET_CLI", str(_HOME / "parakeet.cpp/build/examples/cli/parakeet-cli")),
    "model": os.environ.get("PK_MODEL"),               # None -> pick by decoder (see _DECODER_GGUF)
    "decoder": os.environ.get("PK_DECODER", "tdt"),    # tdt = 1.81%/61x · ctc = 2.38%/107x (both <3%)
    "threads": os.environ.get("PK_THREADS", "8"),
    "cuda_lib": os.environ.get("CUDA_LIB_DIR", "/usr/local/cuda/lib64"),
}

# Per-decoder default GGUF (GPU dtype sweep, results/dtype/RESULTS.md). On the Orin GPU, RTFx is
# dtype-INVARIANT (kernel-launch/latency-bound at batch=1), so quantization is a MEMORY lever, not a
# speed one — which lets the accuracy (tdt) path drop to q5_k for free:
#   tdt -> q5_k : 1.808% WER @ 61x @ 137 MB  (matches q8_0's 1.836% at 23% smaller, identical speed)
#   ctc -> q8_0 : 2.383% WER @ 107x @ 177 MB (the speed default; f16 is 0.01pt better but +86 MB)
# Override either with PK_MODEL=<path/to.gguf>.
_DECODER_GGUF = {"tdt": "tdt_ctc-110m-q5_k.gguf", "ctc": "tdt_ctc-110m-q8_0.gguf"}


def _resolve_model(cfg: dict) -> str:
    """PK_MODEL / explicit cfg override wins; else the per-decoder default GGUF above."""
    if cfg.get("model"):
        return cfg["model"]
    name = _DECODER_GGUF.get(cfg["decoder"], "tdt_ctc-110m-q8_0.gguf")
    return str(_HOME / "parakeet-models" / name)


def transcribe(items: list[dict], *, device: str = "cuda", cfg: dict | None = None):
    """Transcribe `items` via parakeet-cli bench. Returns (rows, timing).
    infer_sec per row = parakeet's per-file proc_ms (inference-only; model-load excluded)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    cfg["model"] = _resolve_model(cfg)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in (cfg["cuda_lib"], env.get("LD_LIBRARY_PATH", "")) if p)

    mf = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for it in items:
        mf.write(str(it["wav"]) + "\n")
    mf.close()
    jout = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    cmd = [cfg["binary"], "bench", "--model", cfg["model"], "--manifest", mf.name,
           "--decoder", cfg["decoder"], "--threads", str(cfg["threads"]), "--json", jout]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"parakeet-cli bench failed ({proc.returncode}):\n{proc.stderr[-2000:]}")

    b = json.load(open(jout))
    by_base = {os.path.basename(f["path"]): f for f in b["files"]}
    rows = []
    for it in items:
        f = by_base.get(os.path.basename(str(it["wav"])))
        if f is None:
            raise RuntimeError(f"parakeet: no result for {it['id']}")
        rows.append({"id": it["id"], "text": f["text"],
                     "audio_sec": round(float(f["audio_sec"]), 3),
                     "infer_sec": round(float(f["proc_ms"]) / 1000.0, 6)})
    os.unlink(mf.name)
    os.unlink(jout)
    # precision label from the GGUF name (tdt_ctc-110m-<dtype>.gguf -> <dtype>), so the report
    # reflects the actual quant (q5_k/q8_0/f16/...), not a hardcoded one.
    precision = Path(cfg["model"]).stem.split("-")[-1]
    return rows, {"infer_total_sec": round(sum(r["infer_sec"] for r in rows), 6),
                  "wall_sec": round(wall, 6), "load_ms": b.get("load_ms"),
                  "decoder": cfg["decoder"], "engine": "parakeet.cpp",
                  "model": Path(cfg["model"]).name, "precision": precision}
