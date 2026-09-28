# Q38FNInference

A Windows app for running **Qwen3.8-Flash-Next** (`Qwen/Qwen3.8-Flash-Next`,
llama.cpp architecture `qwen4exp`) locally on NVIDIA GPUs, built on a CUDA
build of `ggml-org/llama.cpp`'s `llama-server`.

- Opens a local web control panel automatically when started.
- Finds GGUF models you already have (Hugging Face cache, its own models
  folder, extra folders) and downloads new ones from Hugging Face, with
  resume and split-shard handling.
- Computes a GPU/CPU offload plan from the model's real tensor sizes, across
  one or several GPUs (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
- Common model settings: context, KV cache type, sampling (temperature,
  top-p/k, min-p, penalties, seed, max tokens), reasoning, threads, batch
  sizes, API key, extra llama-server arguments.
- System prompt override, applied to every chat request (default-only or
  force-replace).
- Chat tab for quick testing, llama.cpp's own web UI, and an OpenAI-compatible
  API at `http://127.0.0.1:8080/v1` (streaming and non-streaming; the
  Anthropic-style `/v1/messages` works too).

## Download

Get `Q38FNInference-windows-x64-cuda12.zip` from the
[Releases](https://github.com/hgdubbe/Q38FNInference/releases) page, extract
it, and run `q38fninference.exe`. It needs an NVIDIA driver that supports
CUDA 12.4 (R550 or newer). Running the exe again while it's open just
reopens the control panel; **Quit** (top right) stops the model and exits.

Settings and `launcher.log` live in `%APPDATA%\Q38FNInference`; downloaded
models go to `%APPDATA%\Q38FNInference\models`.

## Using it

1. **Models**: pick a local GGUF, or search Hugging Face / list a repo's GGUF
   files and download one (split models download all shards).
2. **Run**: tick the GPUs to use, check the launch plan (per-GPU memory,
   layers, which MoE experts stay in CPU RAM), optionally edit the
   llama-server arguments, then **Start**. The status pill turns green when
   the model is loaded.
3. **Chat**, **Open llama.cpp web UI**, or point any OpenAI client at
   `http://127.0.0.1:8080/v1`.
4. **Settings**: system prompt, model/sampling options, API port and bind
   address (127.0.0.1 or LAN), llama-server path, Hugging Face token.

## Building

The launcher is Go 1.24+ with no cgo, so it cross-compiles from any OS:

```sh
go test ./...
GOOS=windows GOARCH=amd64 go build -ldflags "-H windowsgui" -o q38fninference.exe ./cmd/q38fninference
```

llama.cpp with CUDA:

- CI: `.github/workflows/build-windows-cuda.yml` builds llama.cpp (pinned
  commit, with `patches/` applied) and the launcher, and publishes a release
  when a `v*` tag is pushed. It can also be run manually from the Actions tab.
- Locally on Windows: `.\scripts\build-windows-cuda.ps1` (Visual Studio with
  C++ and Clang tools, CMake, Ninja, CUDA Toolkit 12.4+, Go) produces the same
  folder in `dist\Q38FNInference`.

## Layout

| Path | What |
|---|---|
| `cmd/q38fninference` | Entry point: control panel, API port, browser, single instance |
| `internal/tuning` | Offload planner (single and multi-GPU) |
| `internal/gguf` | GGUF metadata and tensor-size reader |
| `internal/proxy` | API reverse proxy that applies the system prompt |
| `internal/httpapi` | Control panel API and embedded web UI |
| `internal/server` | llama-server child process |
| `internal/hf`, `internal/downloadmgr` | Hugging Face listing and resumable downloads |
| `internal/models` | Local GGUF discovery |
| `internal/gpu` | NVIDIA GPU detection (`nvidia-smi`) |
| `internal/appconfig` | Persisted settings |
| `patches/` | The patch applied to llama.cpp |

## Status

Unit tests and an end-to-end run against a stand-in llama-server pass, but
this has not yet been run against real Qwen3.8-Flash-Next weights on real
CUDA hardware. See `docs/ARCHITECTURE.md` and `TODO.md`.
