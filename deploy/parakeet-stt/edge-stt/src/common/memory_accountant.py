#!/usr/bin/env python3
"""
The ONE memory accountant. Every downstream measurement reads memory through this module —
no ad-hoc `free` scrapes. It samples three readings together for a target process:

  * total system RAM   — Jetson unified memory. `tegrastats` path on the Orin
                         (parses `RAM used/total MB`), `/proc/meminfo` fallback elsewhere.
  * process RSS        — /proc/<pid>/status VmRSS for the workload PID (and its children).
  * CUDA device memory — cudaMemGetInfo(free,total) via a tiny ctypes shim. On Tegra unified
                         memory this tracks system RAM, but we record it separately.

An idle baseline is captured BEFORE the workload so `delta = peak - idle` is meaningful
(mirrors spike/measure_mem.sh, which samples idle after `sleep 1`).

Usage:
  # sample around a command, print the summary + delta
  python3 -m src.common.memory_accountant -- sherpa-onnx-offline --provider=cuda ... a.wav
  # sample an already-running pid for a fixed duration
  python3 -m src.common.memory_accountant --pid 12345 --duration 5 --json
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

_TEGRA_RE = re.compile(r"RAM (\d+)/(\d+)MB")


# --------------------------------------------------------------------------- total RAM
class _TotalRamReader:
    """Latest (used_mb, total_mb) of unified/system RAM. Prefers tegrastats on Jetson."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.backend = "tegrastats" if shutil.which("tegrastats") else "meminfo"
        self._latest: tuple[int, int] | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self.backend == "tegrastats":
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._thread = threading.Thread(target=self._pump_tegrastats, daemon=True)
            self._thread.start()

    def _pump_tegrastats(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = _TEGRA_RE.search(line)
            if m:
                self._latest = (int(m.group(1)), int(m.group(2)))

    def read(self) -> tuple[int, int] | None:
        if self.backend == "tegrastats":
            return self._latest
        # /proc/meminfo: used = MemTotal - MemAvailable (works on Jetson unified + x86)
        total = avail = None
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) // 1024
                if total is not None and avail is not None:
                    break
        except OSError:
            return None
        if total is None or avail is None:
            return None
        return (total - avail, total)

    def stop(self) -> None:
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# --------------------------------------------------------------------------- process RSS
def _proc_rss_mb(pid: int, include_children: bool = True) -> float | None:
    """VmRSS of pid (+ direct descendants) in MB. None if the pid is gone."""
    pids = [pid]
    if include_children:
        pids += _descendants(pid)
    total_kb = 0
    found = False
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
                    found = True
                    break
        except OSError:
            continue
    return total_kb / 1024 if found else None


def _descendants(pid: int) -> list[int]:
    """Best-effort recursive child PIDs via /proc/<pid>/task/*/children (Linux)."""
    out: list[int] = []
    stack = [pid]
    while stack:
        cur = stack.pop()
        try:
            for task in Path(f"/proc/{cur}/task").iterdir():
                kids = (task / "children").read_text().split()
                for k in kids:
                    ki = int(k)
                    out.append(ki)
                    stack.append(ki)
        except OSError:
            continue
    return out


# --------------------------------------------------------------------------- CUDA memory
class _Cuda:
    """cudaMemGetInfo via ctypes. Disabled gracefully where libcudart is absent."""

    def __init__(self):
        self.lib = None
        for name in ("cudart", "libcudart.so", "libcudart.so.12"):
            path = ctypes.util.find_library("cudart") if name == "cudart" else name
            if not path:
                continue
            try:
                self.lib = ctypes.CDLL(path)
                break
            except OSError:
                continue

    def used_total_mb(self) -> tuple[float, float] | None:
        if self.lib is None:
            return None
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        rc = self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if rc != 0 or total.value == 0:
            return None
        used = (total.value - free.value) / (1024 * 1024)
        return (used, total.value / (1024 * 1024))


# --------------------------------------------------------------------------- sampler
class MemorySampler:
    """
    with MemorySampler(pid, interval_ms=100) as s:
        ... run workload ...
    rec = s.summary()  # idle_total_mb, peak_total_mb, proc_rss_mb, cuda_used_mb, samples_n
    """

    def __init__(self, pid: int | None = None, interval_ms: int = 100):
        self.pid = pid
        self.interval_ms = interval_ms
        self._ram = _TotalRamReader(interval_ms)
        self._cuda = _Cuda()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.total_used: list[int] = []
        self.proc_rss: list[float] = []
        self.cuda_used: list[float] = []
        self.cuda_total_mb: float | None = None

    def __enter__(self) -> "MemorySampler":
        self._ram.start()
        # settle so tegrastats emits its first line; capture an idle baseline first
        time.sleep(max(self.interval_ms, 200) / 1000.0)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            rec = self._ram.read()
            if rec is not None:
                self.total_used.append(rec[0])
            if self.pid is not None:
                rss = _proc_rss_mb(self.pid)
                if rss is not None:
                    self.proc_rss.append(rss)
            cu = self._cuda.used_total_mb()
            if cu is not None:
                self.cuda_used.append(cu[0])
                self.cuda_total_mb = cu[1]
            self._stop.wait(self.interval_ms / 1000.0)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._ram.stop()

    def summary(self) -> dict:
        idle = self.total_used[0] if self.total_used else None
        peak = max(self.total_used) if self.total_used else None
        return {
            "idle_total_mb": idle,
            "peak_total_mb": peak,
            "delta_total_mb": (peak - idle) if (idle is not None and peak is not None) else None,
            "proc_rss_mb": round(max(self.proc_rss), 1) if self.proc_rss else None,
            "cuda_used_mb": round(max(self.cuda_used), 1) if self.cuda_used else None,
            "cuda_total_mb": round(self.cuda_total_mb, 1) if self.cuda_total_mb else None,
            "ram_backend": self._ram.backend,
            "samples_n": len(self.total_used),
        }


def sample_command(cmd: list[str], interval_ms: int = 100) -> dict:
    """Run cmd, sampling memory around it. Returns summary + the child return code."""
    proc = subprocess.Popen(cmd)
    with MemorySampler(pid=proc.pid, interval_ms=interval_ms) as s:
        rc = proc.wait()
    out = s.summary()
    out["returncode"] = rc
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="One memory accountant: total RAM + RSS + CUDA.")
    ap.add_argument("--pid", type=int, help="sample an already-running pid")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds to sample a --pid")
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command> to run + measure")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if cmd:
        rec = sample_command(cmd, args.interval_ms)
    elif args.pid is not None:
        with MemorySampler(pid=args.pid, interval_ms=args.interval_ms) as s:
            time.sleep(args.duration)
        rec = s.summary()
    else:
        # no target: just measure the system floor for `duration`
        with MemorySampler(pid=None, interval_ms=args.interval_ms) as s:
            time.sleep(args.duration)
        rec = s.summary()

    if args.json:
        print(json.dumps(rec))
    else:
        for k, v in rec.items():
            print(f"{k:>16}: {v}")


if __name__ == "__main__":
    _main()
