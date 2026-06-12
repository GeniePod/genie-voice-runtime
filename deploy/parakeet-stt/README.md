# parakeet.cpp STT — deploy artifacts

Resident parakeet.cpp STT that drop-in replaces whisper-server on
`127.0.0.1:8178` (same `POST /inference` → `{"text": ...}` contract, so
genie-core needs no change). See [`../../docs/stt-parakeet-evaluation.md`](../../docs/stt-parakeet-evaluation.md)
for the whisper-vs-parakeet comparison and rationale.

For a deeper benchmark of the same engine — full **LibriSpeech-300** WER/RTF/memory, a GPU **dtype
sweep** (which `q*_k` to use), and a 3-mode (offline / streaming / multilingual) harness behind one
CLI — see [`edge-stt/`](edge-stt/) (the Edge-AI STT take-home built around this engine).

Contents:
- `serve-mode.patch` — adds a resident `serve` subcommand to parakeet.cpp's CLI
  (load model once, transcribe wav paths read from stdin). Against `v0.1.1`.
- `parakeet-stt-server.py` — HTTP shim: spawns `parakeet-cli serve` once and
  serves `POST /inference` on `:8178`; resamples input to 16 kHz mono via `sox`.
- `genie-parakeet.service` — systemd unit (resident, `Restart=always`, on boot).

## Build (on the Jetson)

```sh
# 1. parakeet.cpp v0.1.1, native CUDA build (sm_87)
git clone --recursive --branch v0.1.1 https://github.com/mudler/parakeet.cpp ~/parakeet.cpp
cd ~/parakeet.cpp
git apply /path/to/serve-mode.patch          # adds the `serve` subcommand
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DPARAKEET_GGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=87 -DPARAKEET_BUILD_CLI=ON -DPARAKEET_BUILD_TESTS=OFF
cmake --build build -j4 --target parakeet-cli

# 2. model (lean, near-lossless q8_0)
mkdir -p ~/parakeet-models
curl -fL -o ~/parakeet-models/tdt_ctc-110m-q8_0.gguf \
  https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/tdt_ctc-110m-q8_0.gguf
```

## Install (swap whisper → parakeet)

```sh
sudo install -m644 genie-parakeet.service /etc/systemd/system/
install -m755 parakeet-stt-server.py ~/parakeet-stt-server.py
sudo systemctl daemon-reload
sudo systemctl disable --now genie-whisper      # free GPU + :8178
sudo systemctl enable  --now genie-parakeet     # parakeet now answers :8178
curl -s http://127.0.0.1:8178/                  # {"engine":"parakeet-resident", ...}
```

## Revert (back to whisper)

```sh
sudo systemctl disable --now genie-parakeet
sudo systemctl enable  --now genie-whisper
```

## Tunables

- `PK_MODEL` (service env) — swap the model, e.g. `tdt-0.6b-v2-q8_0.gguf` for more
  accuracy at higher footprint.
- `PK_DECODER` — `tdt` (default) or `ctc`.
- Runtime needs `LD_LIBRARY_PATH=/usr/local/cuda/lib64` (set in the unit).
