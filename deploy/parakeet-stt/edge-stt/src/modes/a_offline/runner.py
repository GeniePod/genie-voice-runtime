#!/usr/bin/env python3
"""
Mode A — offline transcription via the `sherpa-onnx-offline` GPU binary.

Per the 2026-06-09 engine decision (ROADMAP §Decision amendment + TRADE-OFFS.md:
sherpa-onnx for all three modes, TensorRT deferred), Mode A loads
Parakeet-TDT-0.6b-v2 INT8 once and decodes the whole item list in a single
`sherpa-onnx-offline` invocation. The model-load cost is paid once and is
EXCLUDED from the RTF number (we use sherpa's own "Elapsed seconds", which is
inference-only, not the wall time that includes the ~15 s recognizer build).

Runtime requirement (see memory: sherpa-spine-and-model-staging): the binary
links onnxruntime as a shared lib, so LD_LIBRARY_PATH must contain the ORT lib
dir (which also holds libonnxruntime_providers_cuda.so) and the CUDA libs.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

_HOME = Path(os.path.expanduser("~"))
import glob as _glob

# On-device defaults (overridable via env so the same code runs in M2 packaging).
DEFAULTS = {
    "binary": os.environ.get(
        "SHERPA_OFFLINE_BIN", str(_HOME / "edge-stt/sherpa-onnx/build/bin/sherpa-onnx-offline")),
    "model_dir": os.environ.get(
        "MODE_A_MODEL_DIR", str(_HOME / "edge-stt/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")),
    # nemo_transducer for Parakeet-v2/v3; "transducer" (or "") for a plain Zipformer transducer.
    "model_type": os.environ.get("MODE_A_MODEL_TYPE", "nemo_transducer"),
    "ort_lib": os.environ.get(
        "ORT_LIB_DIR", str(_HOME / "edge-stt/ort/onnxruntime-linux-aarch64-gpu-1.18.1/lib")),
    "cuda_lib": os.environ.get("CUDA_LIB_DIR", "/usr/local/cuda/lib64"),
    "num_threads": os.environ.get("MODE_A_THREADS", "4"),
}

_ELAPSED_RE = re.compile(r"Elapsed seconds:\s*([0-9.]+)")


def _find(md: str, prefix: str) -> str:
    """Glob encoder/decoder/joiner so the runner handles both 'encoder.int8.onnx' (Parakeet) and
    'encoder-epoch-99-avg-1.int8.onnx' (Zipformer)."""
    g = sorted(_glob.glob(f"{md}/{prefix}*.int8.onnx"))
    if not g:
        raise RuntimeError(f"Mode A: no {prefix}*.int8.onnx under {md}")
    return g[0]


def _env(ort_lib: str, cuda_lib: str) -> dict:
    env = os.environ.copy()
    parts = [p for p in (ort_lib, cuda_lib, env.get("LD_LIBRARY_PATH", "")) if p]
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def _parse_texts(stdout: str, n: int) -> list[str]:
    """Pull the per-wav result JSON lines (in wav-argument order) from sherpa's stdout."""
    texts: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and '"text"' in line:
            try:
                texts.append((json.loads(line).get("text") or "").strip())
            except json.JSONDecodeError:
                pass
    if len(texts) != n:
        raise RuntimeError(f"Mode A: parsed {len(texts)} result(s) but expected {n} "
                           f"(sherpa output format changed?)")
    return texts


def _run_chunk(items: list[dict], cfg: dict, device: str, retries: int):
    """Decode one chunk in a single sherpa-onnx-offline call, retrying on transient
    GPU contention (CUBLAS_STATUS_ALLOC_FAILED → SIGABRT on the SHARED Jetson, where
    GeniePod's resident LLM intermittently spikes unified memory). Returns (rows, infer_sec, wall_sec)."""
    md = cfg["model_dir"]
    provider = "cuda" if device == "cuda" else "cpu"
    cmd = [
        cfg["binary"],
        f"--encoder={_find(md, 'encoder')}",
        f"--decoder={_find(md, 'decoder')}",
        f"--joiner={_find(md, 'joiner')}",
        f"--tokens={md}/tokens.txt",
        f"--model-type={cfg['model_type']}",
        f"--provider={provider}",
        f"--num-threads={cfg['num_threads']}",
    ] + [str(it["wav"]) for it in items]

    last_err = ""
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=_env(cfg["ort_lib"], cfg["cuda_lib"]),
                              capture_output=True, text=True)
        wall = time.perf_counter() - t0
        if proc.returncode == 0:
            try:
                texts = _parse_texts(proc.stdout, len(items))
            except RuntimeError as e:
                last_err = str(e)
            else:
                elapsed = _ELAPSED_RE.findall(proc.stdout + "\n" + proc.stderr)
                infer = float(elapsed[-1]) if elapsed else wall
                durs = [float(it.get("duration_sec") or 0.0) for it in items]
                tot = sum(durs) or 1.0
                rows = [{
                    "id": it["id"], "text": text,
                    "audio_sec": round(d, 3),
                    "infer_sec": round(infer * (d / tot), 6),
                } for it, text, d in zip(items, texts, durs)]
                return rows, infer, wall
        else:
            tail = (proc.stderr or "")[-300:].replace("\n", " ")
            last_err = f"rc={proc.returncode} {tail}"
        if attempt < retries:
            print(f"    [mode-a] chunk failed ({last_err[:120]}); retry {attempt + 1}/{retries}",
                  flush=True)
            time.sleep(4)
    raise RuntimeError(f"sherpa-onnx-offline chunk failed after {retries + 1} attempts: {last_err}")


def _decode_resilient(chunk: list[dict], cfg: dict, device: str, retries: int, stats: dict):
    """Decode a chunk; on a hard OOM/abort (ORT BFC arena exhaustion on the 8 GB box) SPLIT it
    and recurse, down to singles. A single that still can't fit yields an empty-text row (a real
    'did-not-fit' data point) instead of aborting the whole 300-clip run. Accumulates infer/wall
    in `stats`. Returns the chunk's hyp rows in input order."""
    try:
        cr, infer, wall = _run_chunk(chunk, cfg, device, retries)
        stats["infer"] += infer
        stats["wall"] += wall
        stats["ok"] += len(cr)
        return cr
    except RuntimeError as e:
        if len(chunk) > 1:
            mid = len(chunk) // 2
            return (_decode_resilient(chunk[:mid], cfg, device, retries, stats)
                    + _decode_resilient(chunk[mid:], cfg, device, retries, stats))
        it = chunk[0]
        stats["failed"] += 1
        print(f"    [mode-a] FAILED to fit utt {it['id']} "
              f"(dur={it.get('duration_sec')}s): {str(e)[-100:]}", flush=True)
        return [{"id": it["id"], "text": "", "audio_sec": round(float(it.get("duration_sec") or 0.0), 3),
                 "infer_sec": 0.0, "oom": True}]


def transcribe(items: list[dict], *, device: str = "cuda", cfg: dict | None = None,
               chunk_size: int | None = None, retries: int = 2):
    """
    Decode `items` (rows with id/wav/duration_sec) via sherpa-onnx-offline, in CHUNKS. Items are
    sorted SHORTEST-first so short clips batch cheaply; a chunk that overflows the ORT CUDA BFC
    arena (the 8 GB Orin Nano's binding constraint) is split and retried down to singles, and a
    single that still won't fit is recorded as a non-fatal OOM row — so the full run always
    completes and we learn exactly which clips don't fit. Model reload cost is EXCLUDED from RTF
    (we sum sherpa's per-chunk "Elapsed seconds"). chunk_size defaults to env MODE_A_CHUNK (8).

    Returns (rows, timing): rows in ORIGINAL input order -> [{id, text, audio_sec, infer_sec[, oom]}];
    timing -> {infer_total_sec, wall_sec, chunks, oom}. Aggregate RTFx = total_audio /
    sum(infer_sec) is the model-load-excluded on-device inference RTF.
    """
    import sys
    cfg = {**DEFAULTS, **(cfg or {})}
    n = chunk_size or int(os.environ.get("MODE_A_CHUNK", "8"))
    order = {id(it): i for i, it in enumerate(items)}
    by_len = sorted(items, key=lambda it: float(it.get("duration_sec") or 0.0))
    stats = {"infer": 0.0, "wall": 0.0, "ok": 0, "failed": 0}
    by_id: dict[str, dict] = {}
    chunks = [by_len[i:i + n] for i in range(0, len(by_len), n)]
    for ci, chunk in enumerate(chunks, 1):
        for r in _decode_resilient(chunk, cfg, device, retries, stats):
            by_id[r["id"]] = r
        print(f"  [mode-a] chunk {ci}/{len(chunks)} done  "
              f"ok={stats['ok']} oom={stats['failed']} infer={stats['infer']:.1f}s",
              file=sys.stderr, flush=True)
    rows = [by_id[it["id"]] for it in items]  # restore original input order
    return rows, {"infer_total_sec": round(stats["infer"], 6), "wall_sec": round(stats["wall"], 6),
                  "chunks": len(chunks), "oom": stats["failed"]}
