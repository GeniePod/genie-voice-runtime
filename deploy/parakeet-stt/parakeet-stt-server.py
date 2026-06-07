#!/usr/bin/env python3
"""Resident parakeet STT server: drop-in for whisper.cpp's POST /inference on :8178.
Spawns `parakeet-cli serve` once (model stays loaded) and pipes each request to it."""
import http.server, socketserver, subprocess, tempfile, os, json, cgi, threading, sys, time, glob
BIN=os.path.expanduser("~/parakeet.cpp/build/examples/cli/parakeet-cli")
MODEL=os.environ.get("PK_MODEL", os.path.expanduser("~/parakeet-models/tdt_ctc-110m-q8_0.gguf"))
DECODER=os.environ.get("PK_DECODER","tdt"); PORT=int(os.environ.get("PK_PORT","8178"))
ENV=dict(os.environ); ENV["PATH"]="/usr/local/cuda/bin:"+ENV.get("PATH",""); ENV["LD_LIBRARY_PATH"]="/usr/local/cuda/lib64:"+ENV.get("LD_LIBRARY_PATH","")
_lock=threading.Lock()
proc=subprocess.Popen([BIN,"serve","--model",MODEL,"--decoder",DECODER],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1,env=ENV)
ready=threading.Event()
def _drain():
    for ln in proc.stderr:
        if "[serve] ready" in ln: ready.set()
threading.Thread(target=_drain,daemon=True).start()
ready.wait(timeout=120)
sys.stderr.write(f"[shim] serve ready={ready.is_set()} pid={proc.pid} alive={proc.poll() is None}\n"); sys.stderr.flush()
def _xc(wav_path):
    with _lock:
        proc.stdin.write(wav_path+"\n"); proc.stdin.flush()
        return (proc.stdout.readline() or "").strip()
try:
    w=sorted(glob.glob(os.path.expanduser("~/parakeet-eval/wav/*.wav")))[0]; _xc(w); sys.stderr.write("[shim] warmed\n"); sys.stderr.flush()
except Exception as e: sys.stderr.write(f"[shim] warmup skip: {e}\n")
def transcribe(b):
    with tempfile.TemporaryDirectory() as d:
        raw=os.path.join(d,"in.wav"); open(raw,"wb").write(b); w16=os.path.join(d,"16k.wav")
        r=subprocess.run(["sox",raw,"-r","16000","-c","1","-b","16",w16],capture_output=True)
        return _xc(w16 if (r.returncode==0 and os.path.exists(w16)) else raw)
class H(http.server.BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"
    def _j(self,o,c=200):
        bb=json.dumps(o).encode(); self.send_response(c); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(bb))); self.end_headers(); self.wfile.write(bb)
    def do_GET(self): self._j({"status":"ok","engine":"parakeet-resident","model":os.path.basename(MODEL),"alive":proc.poll() is None})
    def do_POST(self):
        if self.path.split("?")[0] not in ("/inference","/v1/audio/transcriptions"): self._j({"error":"not found"},404); return
        fs=cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={"REQUEST_METHOD":"POST","CONTENT_TYPE":self.headers.get("Content-Type",""),"CONTENT_LENGTH":self.headers.get("Content-Length","0")})
        data=None
        for k in ("file","audio_file","audio"):
            if k in fs and getattr(fs[k],"file",None): data=fs[k].file.read(); break
        if not data: self._j({"error":"no file field"},400); return
        try: self._j({"text":transcribe(data)})
        except Exception as e: self._j({"error":str(e)},500)
    def log_message(self,*a): pass
socketserver.TCPServer.allow_reuse_address=True
with socketserver.ThreadingTCPServer(("127.0.0.1",PORT),H) as s:
    print(f"parakeet-resident-shim :{PORT} model={os.path.basename(MODEL)}",flush=True); s.serve_forever()
