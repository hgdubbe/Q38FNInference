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
  sizes, API key, extra llama-server arguments; as defaults, or per model.
- Reasoning level (off/low/medium/high), translated per model from its
  chat template: on/off switches, effort levels, or a thinking-token budget.
- System prompt override, applied to every chat request (default-only or
  force-replace), with optional presets.
- One-click **Open chat** (llama.cpp's own chat UI), and an OpenAI-compatible
  API at `http://127.0.0.1:8080/v1` (streaming and non-streaming; the
  Anthropic-style `/v1/messages` works too).
- On-demand mode: every local model is offered through the API and loaded
  when a request names it (one at a time, optional idle unload), with
  endpoints to list, load and unload models.

## Download

Get `Q38FNInference-windows-x64-cuda12.zip` from the
[Releases](https://github.com/hgdubbe/Q38FNInference/releases) page, extract
it, and run `q38fninference.exe`. It needs an NVIDIA driver that supports
CUDA 12.4 (R550 or newer). The launcher lives in the system tray:
left-click the icon to open the control panel, right-click for the model's
status, the chat UI and **Quit** (stops the model and exits; the panel has
a Quit button too). Running the exe again while it's open just reopens the
panel. Closing the panel leaves it running unless "exit when this panel is
closed" is ticked in Settings.

Settings and `launcher.log` live in `%APPDATA%\Q38FNInference`; downloaded
models go to `%APPDATA%\Q38FNInference\models`.

## Using it

1. **Models**: search Hugging Face or open a repository, and download a
   GGUF. Each file shows its quantization and whether it fits your GPU,
   GPU + RAM, or is too large; split models download all parts.
2. **Run**: pick the model, untick any GPU you don't want used, check how it
   will be loaded (memory per GPU, what stays in RAM), then **Start**. The
   status turns green when the model is ready; the sidebar shows it on every
   page.
3. **Open chat** for llama.cpp's chat window, or point any OpenAI or
   Anthropic client at `http://127.0.0.1:8080/v1` (the Copy button next to
   the address copies it).
4. **Settings**: system prompt (with presets), thinking level and sampling,
   performance options (folded away by default), API port, network access
   and key. Changes show a save bar until saved. "Settings for" switches
   between the defaults and one model's own settings (also reachable from
   the Run page's "Its settings" link).
5. **Profiling**: per model, "Record expert usage" counts how evenly the
   model uses its MoE experts, per profile; "Keep the most-used experts in
   GPU memory" then fills the GPUs with the busiest experts of a chosen
   profile instead of whole layers. Auto-profiling records a profile for
   each chosen use case (coding, roleplay, storytelling, ...) by prompting
   the model itself (see docs/ARCHITECTURE.md).

## API

Everything goes to `http://127.0.0.1:8080` (port and bind address are in
Settings). It is llama-server's own API, passed through the launcher's proxy
(which only adds the system prompt), so every llama-server endpoint works:
OpenAI-style `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
Anthropic-style `/v1/messages`, `/tokenize`, `/props`, `/health`, and so on,
streaming or not.

In **on-demand mode** (Run tab) llama-server runs as a router:

| Request | What it does |
|---|---|
| `GET /v1/models`, `GET /models` | all available models and their status (`loaded`, `loading`, `unloaded`, `sleeping`) |
| any completion with `"model": "<id>"` | loads that model if needed (unloading the current one) and answers |
| `POST /models/load` `{"model": "<id>"}` | load a model ahead of time |
| `POST /models/unload` `{"model": "<id>"}` | unload it and free the VRAM |
| `GET /models/sse` | live model status events |
| `POST /models` `{"model": "<hf-repo>:<quant>"}`, `DELETE /models?model=<id>` | download a model into llama.cpp's cache / delete it |

Model ids are the file names without shard suffix or `.gguf` (e.g.
`RVN-Qwen3.8-Flash-Next-IQ4_XS`). Each model loads with its own offload plan
and its settings (its own, or the defaults). The list is rescanned every 30 seconds, so
new downloads (from the Models tab, another tool, or `POST /models`) appear
without a restart. With "Unload after idle" set, a model that gets no
requests for that long is unloaded and reloads on the next request.

## Building

The launcher is Go 1.24+ with no cgo, so it cross-compiles from any OS:

```sh
go test ./...
GOOS=windows GOARCH=amd64 go build -ldflags "-H windowsgui" -o q38fninference.exe ./cmd/q38fninference
```

llama.cpp with CUDA:

- CI: `.github/workflows/build-windows-cuda.yml` builds llama.cpp (pinned
  commit, with `patches/` applied) and the launcher, and publishes a release
  when a `v*` tag is pushed, or when a pushed commit message contains
  `[release]` (it then tags that commit `v` + the contents of `VERSION`).
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
| `patches/` | The patches applied to llama.cpp (startup info for qwen4exp; performance-core threads on Windows; qwen4exp indexer skipped while it can't be sparse; fused, lower-memory qwen4exp indexer; no pinned weight copy without op offload; opt-in QSA block selection; qwen4exp quantization fix; faster CPU MoE matmul on repacked weights) |

## Status

Unit tests and an end-to-end run against a stand-in llama-server pass, but
this has not yet been run against real Qwen3.8-Flash-Next weights on real
CUDA hardware. See `docs/ARCHITECTURE.md` and `TODO.md`.
