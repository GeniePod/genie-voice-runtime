#!/usr/bin/env python3
"""
Mode B — true streaming via the `sherpa-onnx` (online) streaming Zipformer INT8 binary.

Per the 2026-06-09 engine decision, Mode B is sherpa-onnx's cache-aware streaming Zipformer
(chunk-16-left-128, INT8). The recognizer is cache-aware (batch=1, real streaming with
state carry); for offline manifest scoring we feed each clip and collect its final hypothesis.
The model is small (~73 MB) so it co-resides with GeniePod without the Mode-A OOM risk —
no chunk/split needed.

Honesty (carried from the ROADMAP): low-latency streaming WER on test-clean is ~3.3-3.9% and
MAY breach the 3% gate — disclosed per look-ahead, never hidden. The look-ahead is fixed by the
model's chunk config (chunk-16-left-128); STREAM_LOOKAHEAD is recorded for the report.

Runtime: like Mode A, the binary links onnxruntime as a shared lib -> LD_LIBRARY_PATH must hold
the ORT lib dir + CUDA libs.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

_HOME = Path(os.path.expanduser("~"))
DEFAULTS = {
    "binary": os.environ.get("SHERPA_ONLINE_BIN", str(_HOME / "edge-stt/sherpa-onnx/build/bin/sherpa-onnx")),
    "model_dir": os.environ.get(
        "MODE_B_MODEL_DIR", str(_HOME / "edge-stt/models/sherpa-onnx-streaming-zipformer-en-2023-06-26")),
    "ort_lib": os.environ.get("ORT_LIB_DIR", str(_HOME / "edge-stt/ort/onnxruntime-linux-aarch64-gpu-1.18.1/lib")),
    "cuda_lib": os.environ.get("CUDA_LIB_DIR", "/usr/local/cuda/lib64"),
    "num_threads": os.environ.get("MODE_B_THREADS", "4"),
    "lookahead": os.environ.get("STREAM_LOOKAHEAD", "chunk-16-left-128"),
}
_ELAPSED_RE = re.compile(r"Elapsed seconds:\s*([0-9.]+)")


def _env(ort_lib: str, cuda_lib: str) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in (ort_lib, cuda_lib, env.get("LD_LIBRARY_PATH", "")) if p)
    return env


def _find(md: str, prefix: str) -> str:
    g = sorted(glob.glob(f"{md}/{prefix}*.int8.onnx"))
    if not g:
        raise RuntimeError(f"Mode B: no {prefix}*.int8.onnx under {md}")
    return g[0]


def _parse_texts(stdout: str, n: int) -> list[str]:
    """sherpa-onnx (online) prints, per wav, the wav path then a JSON line with 'text'."""
    texts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"text"' in line:
            try:
                texts.append((json.loads(line).get("text") or "").strip())
            except json.JSONDecodeError:
                pass
    if len(texts) != n:
        raise RuntimeError(f"Mode B: parsed {len(texts)} result(s) but expected {n}")
    return texts


def _run_chunk(items: list[dict], cfg: dict, md: str, provider: str):
    cmd = [
        cfg["binary"],
        f"--encoder={_find(md, 'encoder')}",
        f"--decoder={_find(md, 'decoder')}",
        f"--joiner={_find(md, 'joiner')}",
        f"--tokens={md}/tokens.txt",
        f"--provider={provider}",
        f"--num-threads={cfg['num_threads']}",
    ] + [str(it["wav"]) for it in items]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=_env(cfg["ort_lib"], cfg["cuda_lib"]), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"sherpa-onnx (online) failed ({proc.returncode}):\n{proc.stderr[-1500:]}")
    # The streaming binary writes BOTH the result JSON and "Elapsed seconds" to STDERR
    # (unlike sherpa-onnx-offline, which puts the JSON on stdout) — parse the combined streams.
    combined = proc.stdout + "\n" + proc.stderr
    texts = _parse_texts(combined, len(items))
    # "Elapsed seconds" is printed PER WAV (not one total) -> SUM them.
    elapsed = _ELAPSED_RE.findall(combined)
    infer = sum(float(x) for x in elapsed) if elapsed else wall
    return texts, infer, wall


def transcribe(items: list[dict], *, device: str = "cuda", cfg: dict | None = None,
               chunk_size: int | None = None):
    """Stream-decode `items` via sherpa-onnx (online), in CHUNKS so a single invocation does not
    accumulate 300 wavs' state and get OOM-killed under GeniePod CPU/RAM contention on the shared
    box. The streaming model is tiny so reload-per-chunk is cheap. Returns (rows, timing)."""
    import sys
    cfg = {**DEFAULTS, **(cfg or {})}
    md = cfg["model_dir"]
    provider = "cuda" if device == "cuda" else "cpu"
    n = chunk_size or int(os.environ.get("MODE_B_CHUNK", "50"))
    rows: list[dict] = []
    infer_total = wall_total = 0.0
    chunks = [items[i:i + n] for i in range(0, len(items), n)]
    for ci, chunk in enumerate(chunks, 1):
        texts, infer, wall = _run_chunk(chunk, cfg, md, provider)
        infer_total += infer
        wall_total += wall
        durs = [float(it.get("duration_sec") or 0.0) for it in chunk]
        tot = sum(durs) or 1.0
        for it, text, d in zip(chunk, texts, durs):
            rows.append({"id": it["id"], "text": text, "audio_sec": round(d, 3),
                         "infer_sec": round(infer * (d / tot), 6)})
        print(f"  [mode-b] chunk {ci}/{len(chunks)} ({len(chunk)}) infer={infer:.1f}s",
              file=sys.stderr, flush=True)
    return rows, {"infer_total_sec": round(infer_total, 6), "wall_sec": round(wall_total, 6),
                  "lookahead": cfg["lookahead"], "chunks": len(chunks)}
