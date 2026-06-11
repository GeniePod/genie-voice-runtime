#!/home/aihpc/wakeword-venv/bin/python3
"""GeniePod wake-word listener (LyraT/APE, "Hey Jarvis"). LISTENING is emitted
instantly; the model loads in a background thread while the main loop keeps the
process responsive (reading the mic), so genie-core's short spawn window is met."""
import sys, os, signal, time, threading, subprocess
import glob as _glob

# genie-core spawns this script with HOME unset, which causes Python to drop
# ~/.local from sys.path. Hardcode the user-site path so openwakeword /
# onnxruntime / tflite_runtime are always importable regardless of environment.
for _sp in ["/home/aihpc/.local/lib/python3.10/site-packages"] + \
           _glob.glob("/home/aihpc/.local/lib/python3*/site-packages") + \
           _glob.glob("/home/aihpc/wakeword-venv/lib/python3*/site-packages"):
    if os.path.isdir(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)

def _log(m):
    try:
        open("/home/aihpc/wakeword.log", "a").write(
            f"{time.strftime('%H:%M:%S')} pid={os.getpid()} {m}\n"
        )
    except Exception:
        pass

def _sig(n, f):
    _log(f"SIGNAL {n}")
    sys.exit(0)

for _s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(_s, _sig)

_log("START")
import numpy as np                                       # fast import (~0.3 s)
print("LISTENING", flush=True); _log("LISTENING_SENT")  # instant readiness signal

DEV    = os.environ.get("GENIE_WAKE_DEV",       "plughw:APE,0")
SR     = int(os.environ.get("GENIE_WAKE_SR",    "24000"))
CH     = int(os.environ.get("GENIE_WAKE_CH",    "2"))
THR    = float(os.environ.get("GENIE_WAKE_THRESHOLD", "0.35"))
GAIN   = float(os.environ.get("GENIE_WAKE_GAIN",      "1.5"))
PHRASE = os.environ.get("GENIE_WAKE_MODEL", "hey_jarvis")

# LyraT captures at 24 kHz stereo; OpenWakeWord expects 16 kHz mono @ 1280 samples.
FRAMES = 1920
_X16   = np.linspace(0, FRAMES - 1, 1280)
_XP    = np.arange(FRAMES)
need   = FRAMES * CH * 2

_model = [None]

def _loader():
    try:
        _log("LOADER cwd=" + os.getcwd() + " HOME=" + os.environ.get("HOME", "?"))
        from openwakeword.model import Model
        _model[0] = Model(wakeword_models=[PHRASE], inference_framework="onnx")
        _log("MODEL_LOADED")
    except BaseException as e:
        import traceback
        _log("LOADER_EXC " + traceback.format_exc().replace(chr(10), "|"))

threading.Thread(target=_loader, daemon=True).start()

def open_mic():
    return subprocess.Popen(
        ["arecord", "-q", "-D", DEV, "-f", "S16_LE",
         "-r", str(SR), "-c", str(CH), "-t", "raw"],
        stdout=subprocess.PIPE,
    )

def close_mic(p):
    try:
        p.terminate(); p.wait(timeout=1)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass

ar = open_mic()
_log("MIC_OPENED")

try:
    while True:
        cd = 0
        while True:
            buf = ar.stdout.read(need)
            if not buf or len(buf) < need:
                close_mic(ar); ar = open_mic(); continue
            m = _model[0]
            if m is None:
                continue  # discard audio until model ready
            a = np.frombuffer(buf, dtype=np.int16).reshape(-1, CH).mean(axis=1)
            if GAIN != 1.0:
                a = np.clip(a * GAIN, -32768, 32767)
            a16   = np.interp(_X16, _XP, a).astype(np.int16)
            score = float(m.predict(a16).get(PHRASE, 0.0))
            if cd > 0:
                cd -= 1; continue
            if score > THR:
                close_mic(ar)
                _log(f"WAKE {score:.3f}")
                print(f"WAKE {score:.3f}", flush=True)
                m.reset()
                break
        line = sys.stdin.readline()
        _log(f"stdin {line!r}")
        if not line or line.strip() == "QUIT":
            _log("BREAK"); break
        ar = open_mic()
except SystemExit:
    raise
except BaseException:
    import traceback
    _log("EXC " + traceback.format_exc().replace(chr(10), "|"))
finally:
    _log("EXIT"); close_mic(ar)
