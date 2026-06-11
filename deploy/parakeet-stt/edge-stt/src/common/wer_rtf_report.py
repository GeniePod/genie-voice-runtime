#!/usr/bin/env python3
"""
The ONE WER/RTF reporter. Joins hyps.jsonl to the manifest by id, computes Whisper-normalized
WER + RTFx, pulls memory from the accountant, and emits the frozen report.json schema
(docs/CONTRACT.md §4).

Two entry points, identical output schema:
  * transcribe --report          -> calls build_report() with a live memory sample
  * scripts/score.sh a.jsonl m.jsonl --report r.json
                                 -> recomputes WER over an existing hyps.jsonl (no inference)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ---- Whisper-style normalizer (prefer the real one; documented fallback otherwise) ----
try:
    from whisper_normalizer.english import EnglishTextNormalizer
    _wn = EnglishTextNormalizer()

    def normalize(t: str) -> str:
        return _wn(t)

    _NORMALIZER = "whisper"
except Exception:  # pragma: no cover - fallback path
    _punc = re.compile(r"[^\w\s]")

    def normalize(t: str) -> str:
        return re.sub(r"\s+", " ", _punc.sub(" ", t.lower())).strip()

    _NORMALIZER = "fallback-lower-punct"


_EMPTY_MEM = {"idle_total_mb": None, "peak_total_mb": None, "proc_rss_mb": None, "cuda_used_mb": None}


def _wer(refs: list[str], hyps: list[str]) -> float | None:
    if not refs:
        return None
    try:
        import jiwer
        return jiwer.wer(refs, hyps)
    except ImportError:
        # minimal Levenshtein-over-words fallback so the harness still produces a number
        total_err = total_words = 0
        for r, h in zip(refs, hyps):
            rw, hw = r.split(), h.split()
            total_err += _edit_distance(rw, hw)
            total_words += len(rw)
        return total_err / total_words if total_words else None


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def build_report(hyps: list[dict], manifest_path: str | None = None, *, mode: str | None = None,
                 replay: str = "sequential", device: str = "cuda", limit: int | None = None,
                 mem: dict | None = None, precision: str = "stub", engine: str = "stub") -> dict:
    """Assemble the frozen report.json from hyps rows (+ manifest refs for WER)."""
    refs_by_id: dict[str, str] = {}
    if manifest_path:
        for line in Path(manifest_path).read_text().splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("ref") is not None:
                    refs_by_id[row["id"]] = row["ref"]

    refs, norm_hyps = [], []
    for h in hyps:
        if h["id"] in refs_by_id:
            refs.append(normalize(refs_by_id[h["id"]]))
            norm_hyps.append(normalize(h.get("text", "")))

    total_audio = sum(float(h.get("audio_sec") or 0.0) for h in hyps)
    wall = sum(float(h.get("infer_sec") or 0.0) for h in hyps)
    rtfx = (total_audio / wall) if wall else None
    wer = _wer(refs, norm_hyps)

    return {
        "mode": mode,
        "n": len(hyps),
        "total_audio_sec": round(total_audio, 3),
        "wall_sec": round(wall, 6),
        "rtfx": round(rtfx, 3) if rtfx else None,
        "rtf": round(1 / rtfx, 6) if rtfx else None,
        "wer_pct": round(wer * 100, 3) if wer is not None else None,
        "n_scored": len(refs),
        "normalizer": _NORMALIZER,
        "mem": mem or dict(_EMPTY_MEM),
        "replay": replay,
        "precision": precision,
        "engine": engine,
        "device": device,
        "limit": limit,
    }


def _load_hyps(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="WER/RTF reporter (score.sh wraps this).")
    ap.add_argument("hyps", help="hyps.jsonl")
    ap.add_argument("manifest", nargs="?", help="manifest.jsonl (for WER refs)")
    ap.add_argument("--report", help="write report.json here (else stdout)")
    ap.add_argument("--mode")
    ap.add_argument("--replay", default="sequential")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    rep = build_report(_load_hyps(args.hyps), args.manifest, mode=args.mode,
                       replay=args.replay, device=args.device)
    text = json.dumps(rep, indent=2)
    if args.report:
        Path(args.report).write_text(text)
        print(f"[score] WER={rep['wer_pct']}%  RTFx={rep['rtfx']}  n_scored={rep['n_scored']} -> {args.report}")
    else:
        print(text)


if __name__ == "__main__":
    main()
