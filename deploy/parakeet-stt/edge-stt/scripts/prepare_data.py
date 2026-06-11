#!/usr/bin/env python3
"""
Prepare a deterministic 300-sample LibriSpeech test-clean subset for the parakeet.cpp spike.

- Downloads LibriSpeech test-clean (if absent) and extracts it.
- Builds the full (utt_id, flac, reference) list, sorts by utt_id, takes the first N (default 300).
  NOTE: "first-300 sorted by id" is our documented ASSUMPTIONS.md default for the unspecified
  "default subset of 300 samples" in the brief — confirm/pin this with the assessor (FND-5).
- Transcodes each chosen .flac -> 16 kHz mono PCM WAV (LibriSpeech is already 16 kHz mono).
- Writes manifest.jsonl: {id, wav, ref, duration_sec} and manifest.sha256.

Usage:
  python3 prepare_data.py --out data --n 300
"""
import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"  # ~346 MB


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[skip] {dest} already present")
        return
    print(f"[download] {url} -> {dest} (~346 MB)")

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r  {pct}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, hook)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--split", default=None,
                    help="LibriSpeech split subdir under LibriSpeech/ (default: derived from --url)")
    args = ap.parse_args()

    import soundfile as sf  # libsndfile: reads flac, writes wav

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # derive the split (e.g. 'test-clean', 'dev-clean') from the URL so --url actually works
    split = args.split or Path(args.url).name.replace(".tar.gz", "")
    tar = out / f"{split}.tar.gz"
    download(args.url, tar)

    root = out / "LibriSpeech" / split
    if not root.exists():
        print(f"[extract] {tar.name}")
        with tarfile.open(tar) as t:
            t.extractall(out)

    # gather (id, flac_path, reference) from the per-chapter *.trans.txt files
    items = []
    for trans in root.rglob("*.trans.txt"):
        chapter_dir = trans.parent
        for line in trans.read_text().splitlines():
            uid, _, text = line.partition(" ")
            flac = chapter_dir / f"{uid}.flac"
            if flac.exists():
                items.append((uid, flac, text.strip()))
    items.sort(key=lambda x: x[0])
    chosen = items[: args.n]
    print(f"[select] {len(chosen)} / {len(items)} utterances (sorted by id, first {args.n})")

    wavdir = out / "wav"
    wavdir.mkdir(exist_ok=True)
    manifest = out / "manifest.jsonl"
    total_dur = 0.0
    with manifest.open("w") as mf:
        for uid, flac, ref in chosen:
            audio, sr = sf.read(str(flac), dtype="int16")
            if sr != 16000:  # LibriSpeech is already 16 kHz; resample only if needed
                import numpy as np
                import soxr
                f = soxr.resample(audio.astype("float32") / 32768.0, sr, 16000)
                audio = (f * 32768.0).clip(-32768, 32767).astype("int16")
                sr = 16000
            wav = wavdir / f"{uid}.wav"
            sf.write(str(wav), audio, sr, subtype="PCM_16")
            dur = len(audio) / sr
            total_dur += dur
            mf.write(json.dumps({
                "id": uid,
                "wav": str(wav.resolve()),
                "ref": ref,
                "duration_sec": round(dur, 3),
            }) + "\n")

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (out / "manifest.sha256").write_text(f"{digest}  manifest.jsonl\n")
    print(f"[done] {len(chosen)} wavs, total audio {total_dur/60:.1f} min ({total_dur:.0f} s)")
    print(f"       manifest: {manifest}")
    print(f"       sha256:   {digest}")


if __name__ == "__main__":
    main()
