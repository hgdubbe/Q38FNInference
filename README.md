# Q38FNInference

A Windows launcher for running **Qwen3.8-Flash-Next** (`Qwen/Qwen3.8-Flash-Next`,
architecture id `qwen4exp` in llama.cpp) locally via a CUDA-accelerated
`llama-server`, with GGUF support, model discovery/download from Hugging
Face, and a small local web control panel.

It is a launcher, not a reimplementation of an inference server: model
loading, the OpenAI-compatible API (streaming and non-streaming), and the
testing web UI are all `llama-server`'s own, built with CUDA from
`ggml-org/llama.cpp`. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
why, and for what upstream already supports for this specific architecture.

## What's here

| Path | What |
|---|---|
| `cmd/q38fninference` | Launcher entrypoint (Go): starts the control panel, opens a browser |
| `internal/gguf` | Dependency-free GGUF header/metadata/tensor-size reader |
| `internal/tuning` | qwen4exp-aware GPU/CPU offload planner (`--n-cpu-moe`, `-ngl`, context, etc.) |
| `internal/gpu` | NVIDIA VRAM detection (`nvidia-smi`) |
| `internal/hf` | Hugging Face Hub client: list files, resumable download with progress |
| `internal/models` | Discovers already-downloaded GGUF files (HF cache + configured dirs) |
| `internal/server` | Manages the `llama-server` child process (start/stop/log tail) |
| `internal/httpapi` | The launcher's own local JSON API + embedded web control panel |
| `internal/appconfig` | Persisted settings |
| `internal/downloadmgr` | Tracks in-flight Hugging Face downloads |
| `patches/` | The one (additive, log-only) patch this project applies to llama.cpp |
| `.github/workflows/build-windows-cuda.yml` | CI: builds llama.cpp+CUDA and this launcher, packages a release zip |
| `scripts/build-windows-cuda.ps1` | Same build, runnable locally on Windows |

## Building

### The launcher itself

Requires Go 1.24+.

```sh
go build -o q38fninference.exe ./cmd/q38fninference   # cross-compiles fine from any OS: GOOS=windows GOARCH=amd64
go test ./...
```

### llama.cpp with CUDA (Windows)

Either let CI do it (`.github/workflows/build-windows-cuda.yml`, manually
triggered or weekly), or run locally from a "Developer PowerShell for VS
2022" prompt with CMake, Ninja, and the CUDA Toolkit (12.4+) installed:

```powershell
.\scripts\build-windows-cuda.ps1
```

This clones `ggml-org/llama.cpp`, applies `patches/0001-qwen4exp-hybrid-layer-banner.patch`,
builds `llama-server.exe` (CPU/dynamic-backend base) and `ggml-cuda.dll`
separately — mirroring upstream's own release process, where CUDA ships as a
dynamically-loaded backend DLL rather than being statically linked — copies
the CUDA runtime DLLs, and builds the launcher, all into `.\dist`.

Both the CI workflow and the script land everything (`q38fninference.exe`,
`llama-server.exe`, `ggml-cuda.dll`, `ggml*.dll`, `cudart64_*.dll`, ...) in
one directory: the launcher looks for `llama-server.exe` next to itself (or
in a `llama\` subfolder, or on `PATH`, or wherever you point it in Settings).

## Running

1. Run `q38fninference.exe`. It opens a browser to a local control panel
   (`http://127.0.0.1:8787` by default).
2. **Models tab**: either pick an already-downloaded model (auto-discovered
   from your Hugging Face cache and this app's own models folder), or search/
   download GGUF files straight from a Hugging Face repo (e.g.
   `Qwen/Qwen3.8-Flash-Next`), with resumable progress.
3. **Run tab**: select a model to get a computed offload plan (see
   `internal/tuning`) — how many MoE-expert layers stay on CPU RAM vs. GPU,
   context size, flash-attention, KV cache quantization — tweak it if you
   want, then **Start server**.
4. Once running, the panel links straight to `llama-server`'s own web UI
   (for interactive testing) and its OpenAI-compatible API
   (`http://127.0.0.1:<port>/v1/...`, streaming and non-streaming, same as
   any other OpenAI-compatible client).

GGUF split files (`model-00001-of-0000N.gguf`, ...) are handled
transparently: point at any shard and the rest are found automatically, both
for local discovery and for llama-server's own loader.

## Status / limitations

Built and unit-tested (`go test -race ./...` passes) without access to real
Qwen3.8-Flash-Next weights or a CUDA GPU — huggingface.co was unreachable and
no GPU was available in the environment this was built in. The llama.cpp
build itself is upstream's own CI-tested process; this project's own code
(GGUF parsing, tuning math, HTTP/download plumbing, process management) is
unit-tested but has not been run end-to-end against the real model. See
`docs/ARCHITECTURE.md` and `TODO.md` for specifics and follow-ups.
