# Architecture notes

## What Qwen3.8-Flash-Next actually is

Per `ggml-org/llama.cpp`'s own conversion script (`conversion/qwen4exp.py`,
commit `6c7a87f` as read on 2026-09-28):

> Qwen3.8-Flash-Next. Shares the Qwen3.5 gated delta net and interleaved
> mrope, and adds three things: hyper-connections in place of every layer
> norm, QSA sparse attention on the full attention layers, and PLE n-gram
> hash embeddings on a single layer.

Concretely (from the model's own README, huggingface.co unreachable from
the sandbox that wrote this, but GitHub reachable):

- 125B dense params + 51B params in the PLE n-gram hash embedding table,
  ~6B active per token (MoE).
- Hybrid attention: most layers are Gated DeltaNet (GDN) — a linear/recurrent
  attention variant with fixed-size state, no growing KV cache — with every
  4th layer (`full_attention_interval`, configurable) a full "QSA" attention
  layer using an indexer for sparse top-k token selection.
- "Hyper-connections": a gated-residual scheme with multiple parallel
  residual streams (`hc_count` in the config) instead of a single stream.
- One layer carries a PLE (n-gram hash) embedding table — most of that 51B
  embedding-param count lives here, on a single layer.
- 262144 native context length.

llama.cpp identifies it as architecture id `qwen4exp` (`LLM_ARCH_QWEN4EXP`).

## What upstream llama.cpp already does for this model

As of the commit this project targets, `ggml-org/llama.cpp` **already has
native, CUDA-accelerated support** for this architecture:

- `src/models/qwen4exp.cpp` — the graph builder (hyper-connections, indexer/
  QSA sparse attention, PLE hash embeddings).
- `ggml/src/ggml-cuda/gated_delta_net.cu` — a templated, `__launch_bounds__`-
  tuned CUDA kernel for the GDN recurrence, plus a fused state-cache-copy path
  in `ggml-cuda.cu`.
- Metal, Vulkan, and Hexagon (Qualcomm NPU) backends also have gated-delta-net
  kernels.
- `--n-cpu-moe` / `-ncmoe` in `tools/server` already implements "keep the
  first N layers' MoE expert weights on CPU RAM" — exactly the mechanism a
  176B-total/6B-active MoE model like this one needs on consumer GPUs.

**We did not hand-write or hand-tune CUDA kernels for this model.** Two
things ruled that out:

1. `src/models/qwen4exp.cpp` carries an explicit upstream comment:
   `// [TAG_QWEN4_REIMPLEMENT] TODO: this graph implementation is pending
   complete reimplementation - do not use it as a reference`. Patching
   around code the maintainers themselves call provisional, without the
   ability to compile against real weights on real CUDA hardware (this
   sandbox has neither), would be guessing, not engineering.
2. The gated-delta-net CUDA kernel is already a carefully tuned templated
   kernel (`__launch_bounds__` computed from warp size, per-dtype
   specializations). Modifying it blind is far more likely to introduce a
   subtle correctness or performance regression than to improve on work
   the ggml maintainers already tuned with real hardware in the loop.

## Where this project's "architecture-aware" work actually is

Two places:

1. **`patches/0001-qwen4exp-hybrid-layer-banner.patch`** — a small, additive,
   log-only patch to `src/llama-model.cpp`. It prints the qwen4exp hybrid
   layer plan at startup (GDN vs. QSA layer counts, hyper-connection count,
   indexer top-k, PLE layer presence) the same way llama.cpp already does
   for DeepSeek/Qwen2MoE/etc.-specific fields. Zero behavior change — it
   can't break inference, only make the existing startup log more legible
   for this architecture. This was checked with `git apply --check` against
   upstream at the commit noted in the patch header.

2. **`internal/tuning`**: the offload planner. It reproduces llama.cpp's
   own placement rules from `load_tensors` in `src/llama-model.cpp` so the
   plan is exact:
   - `token_embd` (the input layer) always stays on CPU.
   - There are `n_layer + 1` offloadable slots (every block plus the output
     layer), and `-ngl k` puts the *last* k of them on GPUs.
   - `--tensor-split` assigns those slots to devices in order by cumulative
     proportion, so passing slot counts (e.g. `2,5`) places exactly that many
     consecutive slots on each GPU.
   - `--n-cpu-moe N` pins the expert tensors of blocks `0..N-1`
     (`ffn_{up,down,gate,gate_up}_exps`, the same regex llama.cpp uses) to
     CPU RAM wherever the rest of the block lives.

   Each slot costs its weights (real per-tensor sizes, from `sizeof(block_*)`
   compiled out of `ggml/src/ggml-common.h`) plus its share of the context
   cache, from GGUF metadata: KV and indexer keys for full (QSA) attention
   blocks, conv and delta-rule state for gated-delta-net blocks, per parallel
   slot. Each GPU's budget is its free memory minus the CUDA context, compute
   buffers (larger on the first GPU, which holds the logits) and a safety
   margin.

   The planner then picks the smallest `--n-cpu-moe` for which all slots pack
   into the selected GPUs in order (greedy filling is optimal for an ordered
   contiguous split). If even with every expert in RAM the dense weights
   don't fit, it offloads as many trailing slots as fit, with all experts
   pinned, so GPU-resident blocks never drag their experts onto the GPU.
   Several GPUs pool their memory, which is what lets a 176B-total/6B-active
   model keep far more experts on GPU than any single card could.

   GPUs are selected with `CUDA_VISIBLE_DEVICES`, and llama-server runs with
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` so its device numbering matches
   `nvidia-smi`'s. The plan is passed with `--fit off`, since llama.cpp's
   `--fit` would otherwise try to re-plan.

   We didn't build this into llama.cpp's own `--fit` auto-planner
   (`common/fit.cpp`) because that estimator currently has **zero
   awareness** of this architecture's extra memory consumers — no
   `ssm_d_state`/recurrent-cache accounting, no PLE hash-table accounting
   (confirmed by grepping `common/fit.cpp` for `is_recr`, `ssm_`, `ple_`:
   no hits). Teaching a ~1000-line generic bin-packing algorithm we can't
   compile-and-test against real hardware about a brand-new, still-WIP
   architecture is exactly the kind of change this project chose not to
   guess at. See `TODO.md`.

## API, web UI and system prompt

`llama-server` already provides the OpenAI-compatible API
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, ...; streaming and
non-streaming), an Anthropic-style `/v1/messages`, and a web UI. The launcher
doesn't reimplement any of that. It starts llama-server on a private loopback
port and puts a reverse proxy (`internal/proxy`) on the public API port
(8080 by default), because llama-server has no system-prompt option. For
`/v1/chat/completions`, `/chat/completions` and `/v1/messages`, the proxy
either adds the configured system prompt when a request has none
("default") or replaces the request's system/developer messages
("override"). Everything else, including SSE streams, passes through
unchanged. llama-server's web UI is also served through the proxy, so the
prompt applies there too.

The control panel (`internal/httpapi`, port 8787) hosts model management,
the offload plan, settings, and a small chat tab that talks to the same
proxy.

The release exe is linked as a Windows GUI program (`-H windowsgui`): no
console window, child processes are started hidden, logs go to
`launcher.log`, and starting a second copy just reopens the running
instance's panel.

## Known limitations / open questions

- Nothing here has been run against real Qwen3.8-Flash-Next weights or real
  CUDA hardware — huggingface.co was unreachable and no GPU was available in
  the environment this was built in. The GGUF parser, tuning heuristic, and
  HTTP/download plumbing all have unit tests that build and pass in that
  environment; the llama.cpp build pipeline and CUDA kernels are upstream's
  own, exercised by their CI, not this project's.
- The cache estimates are deliberately high (QSA KV compression ratios are
  ignored, recurrent state is counted at f32) and the compute buffer sizes
  are fixed guesses. Expect the plan to leave some VRAM unused rather than
  run out.
- Multi-GPU uses `--split-mode layer` only; `row`/`tensor` modes are not
  planned for.
- The qwen4exp graph implementation is explicitly WIP upstream; re-check
  `patches/0001-qwen4exp-hybrid-layer-banner.patch` still applies (and that
  the hparam field names it reads haven't moved) before every build.
