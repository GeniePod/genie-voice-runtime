#!/usr/bin/env python3
"""
The single grader-facing entrypoint implementing the FROZEN CONTRACT (docs/CONTRACT.md).

  ./transcribe --mode {A|B|C} --input <dir|manifest.jsonl> --out hyps.jsonl
               [--report report.json] [--limit N] [--device cuda]
               [--replay {sequential|batched}] [--selfcheck]

Modes are STUBS at M0 (echo the manifest reference, no model loaded) so the interface shape
is exercised end-to-end and M1's first real numbers flow through this same harness. Real
engines land in src/modes/{a_offline,b_streaming,c_multilingual}/ in M2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.common import audio

AUDIO_EXTS = (".wav", ".flac")


# --------------------------------------------------------------------------- input loading
def _resolve_wav(row: dict, base: Path) -> str:
    """Resolve a manifest row's wav path PORTABLY. The committed manifest stores absolute
    paths from wherever it was generated; on a fresh clone / the Jetson those won't exist,
    so fall back to <manifest_dir>/wav/<id>.wav (then basename) before giving up."""
    w = row.get("wav")
    if w and Path(w).exists():
        return w
    cand = base / "wav" / f"{row.get('id', '')}.wav"
    if cand.exists():
        return str(cand)
    if w:
        cand2 = base / "wav" / Path(w).name
        if cand2.exists():
            return str(cand2)
    return w or ""  # leave as-is; decode raises a clear error if truly missing


def load_items(input_path: str, limit: int | None) -> list[dict]:
    """Return manifest rows: {id, wav, ref?, duration_sec?}. Accepts a manifest.jsonl or a dir."""
    p = Path(input_path)
    items: list[dict] = []
    if p.is_file() and p.suffix == ".jsonl":
        base = p.resolve().parent
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                row["wav"] = _resolve_wav(row, base)
                items.append(row)
    elif p.is_dir():
        for wav in sorted(p.iterdir()):
            if wav.suffix.lower() in AUDIO_EXTS:
                items.append({"id": wav.stem, "wav": str(wav.resolve()), "ref": None})
    else:
        sys.exit(f"--input must be a .jsonl manifest or a directory of wav/flac: {input_path}")
    if limit is not None:
        items = items[:limit]
    if not items:
        sys.exit(f"no utterances found in {input_path}")
    return items


# --------------------------------------------------------------------------- the one core fn
def transcribe_core(item: dict, mode: str, device: str) -> dict:
    """
    The ONE internal code path all modes route through (CLI-only — no HTTP/WS twin). Returns a
    hyps.jsonl row {id, text, audio_sec, infer_sec}. M0 STUB: echoes the reference (or empty for a
    bare wav-dir input, which then emits hyps with no WER).
    """
    runner = _MODE_STUBS[mode]
    audio_sec = item.get("duration_sec") or audio.duration_sec(item["wav"]) or 0.0
    t0 = time.perf_counter()
    text = runner(item, device)
    infer_sec = time.perf_counter() - t0
    return {
        "id": item["id"],
        "text": text,
        "audio_sec": round(float(audio_sec), 3),
        "infer_sec": round(infer_sec, 6),
    }


def _stub(item: dict, device: str) -> str:
    """Placeholder until real engines land: echo the reference so the WER-join path is exercised."""
    return item.get("ref") or ""


# mode -> runner. All point at the stub for now; M2 swaps in real per-mode engines.
_MODE_STUBS = {"A": _stub, "B": _stub, "C": _stub}


# --------------------------------------------------------------------------- driver
def run(args: argparse.Namespace) -> None:
    items = load_items(args.input, args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    precision, engine = "stub", "stub"

    # sample memory through the one accountant around the whole inference loop
    from src.common.memory_accountant import MemorySampler
    if args.mode == "A":
        # Mode A. PRIMARY engine = parakeet.cpp (ggml CUDA) — the deployable headline
        # (60-107x RTF, 1.8-2.4% WER, +450MB; results/parakeet/RESULTS.md). Set STT_ENGINE=sherpa
        # to use the sherpa-onnx alternative (configurable model via MODE_A_MODEL_DIR). Per-file
        # inference time only; model-load excluded from RTF.
        stt_engine = os.environ.get("STT_ENGINE", "parakeet").lower()
        if stt_engine == "parakeet":
            from src.modes.a_offline import parakeet_runner as mode_a
            precision = "q5_k/q8_0"  # per-decoder default; overwritten from timing below
            print(f"[transcribe] mode=A device={args.device} n={len(items)}  "
                  f"(parakeet.cpp ggml-CUDA, decoder={os.environ.get('PK_DECODER', 'tdt')})", file=sys.stderr)
        else:
            from src.modes.a_offline import runner as mode_a
            precision = "int8"
            print(f"[transcribe] mode=A device={args.device} replay={args.replay} "
                  f"n={len(items)}  (sherpa-onnx-offline INT8)", file=sys.stderr)
        with MemorySampler(pid=os.getpid()) as sampler:
            rows, timing = mode_a.transcribe(items, device=args.device)
        engine = timing.get("engine", "sherpa-onnx-offline")
        precision = timing.get("precision", precision)  # actual GGUF dtype (parakeet) or int8 (sherpa)
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[transcribe] inference {timing['infer_total_sec']}s "
              f"(wall {timing['wall_sec']}s; load {timing.get('load_ms')}ms)", file=sys.stderr)
    elif args.mode == "B":
        # Mode B is REAL (sherpa-onnx online streaming Zipformer INT8). One streaming pass
        # over the batch; small model, co-resides with GeniePod.
        from src.modes.b_streaming import runner as mode_b
        precision, engine = "int8", "sherpa-onnx-online(streaming-zipformer)"
        print(f"[transcribe] mode=B device={args.device} n={len(items)}  "
              f"(sherpa-onnx streaming Zipformer INT8 — cache-aware, batch=1)", file=sys.stderr)
        with MemorySampler(pid=os.getpid()) as sampler:
            rows, timing = mode_b.transcribe(items, device=args.device)
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"[transcribe] inference {timing['infer_total_sec']}s lookahead={timing.get('lookahead')}",
              file=sys.stderr)
    elif args.mode == "C":
        # Mode C — multilingual + dynamic (whisper-tiny LID + recognizer; v3 deferred).
        from src.modes.c_multilingual import runner as mode_c
        precision, engine = "int8", "sherpa-onnx-offline(whisper-tiny)"
        lang = os.environ.get("MODE_C_LANG", "")  # "" = auto-detect (LID)
        print(f"[transcribe] mode=C device={args.device} n={len(items)}  "
              f"(whisper-tiny LID+multilingual, lang={lang or 'auto'})", file=sys.stderr)
        with MemorySampler(pid=os.getpid()) as sampler:
            rows, timing = mode_c.transcribe(items, device=args.device, language=lang)
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        langs = {}
        for r in rows:
            langs[r.get("lang", "")] = langs.get(r.get("lang", ""), 0) + 1
        print(f"[transcribe] inference {timing['infer_total_sec']}s detected-langs={langs}", file=sys.stderr)
    else:
        print(f"[transcribe] mode={args.mode} device={args.device} replay={args.replay} "
              f"n={len(items)}  (M0 STUB: echoing references, no model loaded)", file=sys.stderr)
        with MemorySampler(pid=os.getpid()) as sampler:
            with out.open("w") as f:
                for item in items:
                    row = transcribe_core(item, args.mode, args.device)
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
    mem = {k: sampler.summary().get(k) for k in
           ("idle_total_mb", "peak_total_mb", "proc_rss_mb", "cuda_used_mb")}
    print(f"[transcribe] wrote {len(rows)} hyps -> {out}", file=sys.stderr)

    if args.report:
        # delegate to the one reporter so transcribe --report and score.sh emit identical schema
        from src.common import wer_rtf_report
        rep = wer_rtf_report.build_report(
            hyps=rows, manifest_path=args.input if args.input.endswith(".jsonl") else None,
            mode=args.mode, replay=args.replay, device=args.device, limit=args.limit, mem=mem,
            precision=precision, engine=engine,
        )
        Path(args.report).write_text(json.dumps(rep, indent=2))
        print(f"[transcribe] wrote report -> {args.report}", file=sys.stderr)
        print(f"[transcribe] WER={rep.get('wer_pct')}%  RTFx={rep.get('rtfx')}  "
              f"peak_total_mb={rep.get('mem', {}).get('peak_total_mb')}", file=sys.stderr)


def selfcheck() -> int:
    """`--selfcheck`: shell scripts/preflight.sh (full impl in M3-8). No-op-OK if absent."""
    repo = Path(__file__).resolve().parents[2]
    pf = repo / "scripts" / "preflight.sh"
    if not pf.exists():
        print(f"[selfcheck] {pf} not present yet (M0-T8 delivers it)", file=sys.stderr)
        return 0
    return subprocess.run(["bash", str(pf)]).returncode


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="transcribe", description="Edge-STT grader entrypoint (frozen CLI).")
    ap.add_argument("--mode", choices=["A", "B", "C"])
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--replay", default="sequential", choices=["sequential", "batched"])
    ap.add_argument("--selfcheck", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.selfcheck:
        sys.exit(selfcheck())
    missing = [f for f in ("mode", "input", "out") if getattr(args, f) is None]
    if missing:
        sys.exit(f"missing required args: {', '.join('--' + m for m in missing)}")
    run(args)


if __name__ == "__main__":
    main()
