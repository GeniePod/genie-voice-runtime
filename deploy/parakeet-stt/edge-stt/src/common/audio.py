#!/usr/bin/env python3
"""
Audio decode front-end (the harness owns decode; RTF is measured over inference only).

Decode contract (docs/CONTRACT.md §6): libsndfile (soundfile) -> 16 kHz mono PCM16,
resample with soxr only if sr != 16000. Mirrors spike/prepare_data.py.
"""
from __future__ import annotations

import wave
from pathlib import Path

TARGET_SR = 16000


def duration_sec(path: str | Path) -> float | None:
    """Audio duration in seconds. soundfile if present, else the stdlib wave reader (wav only)."""
    path = str(path)
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def decode_16k_mono(path: str | Path):
    """Return (int16 mono ndarray @ 16 kHz, sr). Requires soundfile (+ soxr if resampling)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="int16")
    if audio.ndim > 1:  # downmix to mono
        audio = (audio.astype("int32").mean(axis=1)).astype("int16")
    if sr != TARGET_SR:
        import soxr
        f = soxr.resample(audio.astype("float32") / 32768.0, sr, TARGET_SR)
        audio = (f * 32768.0).clip(-32768, 32767).astype("int16")
        sr = TARGET_SR
    return audio, sr
