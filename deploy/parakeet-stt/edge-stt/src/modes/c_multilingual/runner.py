#!/usr/bin/env python3
"""
Mode C — multilingual + dynamic, via sherpa-onnx-offline with a Whisper model.

Per the 2026-06-09 engine decision, Mode C is sherpa-onnx multi-recognizer with a
Whisper-tiny LID router. Whisper natively does BOTH language identification (the JSON
"lang" field is the router's decision) and multilingual transcription, so whisper-tiny
serves as the LID + a multilingual recognizer in one. (The ROADMAP's accurate production
recognizer is Parakeet-TDT-0.6b-v3, 25-lang; it is DEFERRED here because HuggingFace
rate-limited the VM mid-stage — v3 drops in behind the same router when staged. The
TTL/LRU one-resident residency manager is M2 production polish.)

Whisper-tiny is small (~104 MB) so it co-resides with GeniePod without the Mode-A OOM.
Runtime: LD_LIBRARY_PATH must hold the ORT lib dir + CUDA libs (same as A/B).
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
    "binary": os.environ.get("SHERPA_OFFLINE_BIN", str(_HOME / "edge-stt/sherpa-onnx/build/bin/sherpa-onnx-offline")),
    "model_dir": os.environ.get("MODE_C_MODEL_DIR", str(_HOME / "edge-stt/models/sherpa-onnx-whisper-tiny")),
    "ort_lib": os.environ.get("ORT_LIB_DIR", str(_HOME / "edge-stt/ort/onnxruntime-linux-aarch64-gpu-1.18.1/lib")),
    "cuda_lib": os.environ.get("CUDA_LIB_DIR", "/usr/local/cuda/lib64"),
    "num_threads": os.environ.get("MODE_C_THREADS", "4"),
}
_ELAPSED_RE = re.compile(r"Elapsed seconds:\s*([0-9.]+)")


def _env(ort_lib: str, cuda_lib: str) -> dict:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(p for p in (ort_lib, cuda_lib, env.get("LD_LIBRARY_PATH", "")) if p)
    return env


def _find(md: str, suffix: str) -> str:
    g = sorted(glob.glob(f"{md}/*{suffix}"))
    if not g:
        raise RuntimeError(f"Mode C: no *{suffix} under {md}")
    return g[0]


def _parse_results(combined: str, n: int) -> list[tuple[str, str]]:
    """Return [(text, detected_lang)] per wav. Whisper's result JSON carries the LID in 'lang'."""
    out: list[tuple[str, str]] = []
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith("{") and '"text"' in line:
            try:
                j = json.loads(line)
                out.append(((j.get("text") or "").strip(), (j.get("lang") or "").strip()))
            except json.JSONDecodeError:
                pass
    if len(out) != n:
        raise RuntimeError(f"Mode C: parsed {len(out)} result(s) but expected {n}")
    return out


def _run_chunk(items, cfg, md, provider, language):
    cmd = [
        cfg["binary"],
        f"--whisper-encoder={_find(md, 'encoder.int8.onnx')}",
        f"--whisper-decoder={_find(md, 'decoder.int8.onnx')}",
        f"--tokens={_find(md, 'tokens.txt')}",
        f"--whisper-language={language}",   # "" => auto-detect (LID router decision)
        "--whisper-task=transcribe",
        f"--provider={provider}",
        f"--num-threads={cfg['num_threads']}",
    ] + [str(it["wav"]) for it in items]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=_env(cfg["ort_lib"], cfg["cuda_lib"]), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"sherpa-onnx-offline (whisper) failed ({proc.returncode}):\n{proc.stderr[-1500:]}")
    combined = proc.stdout + "\n" + proc.stderr
    results = _parse_results(combined, len(items))
    elapsed = _ELAPSED_RE.findall(combined)
    infer = float(elapsed[-1]) if elapsed else wall
    return results, infer, wall


def transcribe(items, *, device: str = "cuda", cfg: dict | None = None,
               language: str = "", chunk_size: int | None = None):
    """Transcribe `items` with whisper-tiny (LID auto-detect when language=""), in chunks.
    Returns (rows, timing); rows carry the detected language for LID-accuracy reporting."""
    import sys
    cfg = {**DEFAULTS, **(cfg or {})}
    md = cfg["model_dir"]
    provider = "cuda" if device == "cuda" else "cpu"
    n = chunk_size or int(os.environ.get("MODE_C_CHUNK", "20"))
    rows: list[dict] = []
    infer_total = wall_total = 0.0
    chunks = [items[i:i + n] for i in range(0, len(items), n)]
    for ci, chunk in enumerate(chunks, 1):
        results, infer, wall = _run_chunk(chunk, cfg, md, provider, language)
        infer_total += infer
        wall_total += wall
        durs = [float(it.get("duration_sec") or 0.0) for it in chunk]
        tot = sum(durs) or 1.0
        for it, (text, lang), d in zip(chunk, results, durs):
            rows.append({"id": it["id"], "text": text, "lang": lang,
                         "audio_sec": round(d, 3), "infer_sec": round(infer * (d / tot), 6)})
        print(f"  [mode-c] chunk {ci}/{len(chunks)} ({len(chunk)}) infer={infer:.1f}s "
              f"langs={set(r['lang'] for r in rows[-len(chunk):])}", file=sys.stderr, flush=True)
    return rows, {"infer_total_sec": round(infer_total, 6), "wall_sec": round(wall_total, 6),
                  "chunks": len(chunks), "recognizer": "whisper-tiny", "lid": "whisper-auto"}
