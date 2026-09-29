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

## On-demand mode (llama-server router)

In on-demand mode the launcher starts llama-server without a model, in
llama.cpp's router mode, with `--models-max 1` and a generated
`router-presets.ini` (in the config directory). Each INI section is one
model: its id, its GGUF path and the same arguments single-model mode would
use (offload plan, context, cache type, sampling, reasoning). Plans are
computed once per model against the GPU memory measured when the router
starts, before anything is loaded, which is what each model gets when only
one is loaded at a time. The router loads a model on the first request that
names it, evicting the previous one, and exposes load/unload/list endpoints.

A background rescan (every 30 s, or on demand) rebuilds the INI from the
local model folders plus any model the router lists from its own cache
(e.g. downloaded through `POST /models`), which are given a plan under the
router's own id; when the file changes the launcher calls
`GET /models?reload=1`. The INI format has no quoting: values containing
`;` or `#` can't be written, and such settings are skipped and reported.
`--api-key` and `--sleep-idle-seconds` are passed to the router, which the
model instances inherit.

## Performance work (backend source review)

What the llama.cpp source review turned up, and what was done about it.
Nothing here could be measured on an NVIDIA GPU (none was available);
everything that could be run was run on a CPU build of the same commit.

Changed:

- **Hybrid CPUs on Windows** (`patches/0002-windows-hybrid-cpu-threads.patch`).
  llama.cpp picks the math thread count from the performance cores only on
  Linux; on Windows it counted every physical core, E-cores included.
  Every graph step waits for its slowest thread, so on Intel 12th-14th gen
  the E-cores held back the CPU-resident expert matmuls. The patch applies
  the same rule on Windows (cores with the highest `EfficiencyClass`); on
  non-hybrid CPUs nothing changes.
- **Op-offload staging VRAM.** When weights stay in RAM, llama.cpp copies a
  RAM-resident layer's whole weight tensor to the first GPU for each prompt
  micro-batch of at least `GGML_OP_OFFLOAD_MIN_BATCH` tokens
  (`ggml-backend.cpp`, cause "1.off"), and reserves compute-buffer space
  for that at load. The planner now keeps the largest RAM-resident block
  free on the main GPU (previously a flat 1 GiB, which a large MoE layer
  exceeds: a load-time out-of-memory), and drops op offload
  (`--no-op-offload`) when a small GPU would lose whole layers to it.
- **Micro-batch 2048 with RAM-resident weights.** Each of those copies moves
  every expert of the layer over PCIe; a 4x larger micro-batch spreads that
  fixed cost over 4x more prompt tokens. The compute-buffer estimate grows
  accordingly.
- **MoE-aware offload threshold.** llama.cpp uses 32 tokens for dense and
  MoE ops alike, but for a sparse MoE the copy moves all experts while the
  CPU only reads the ones a batch routes to. The launcher sets
  `GGML_OP_OFFLOAD_MIN_BATCH=128` (Settings → Model), roughly where a PCIe
  4.0 x16 copy of a large MoE layer and a desktop CPU's prompt throughput
  break even.
- **QSA indexer skipped while it can't be sparse**
  (`patches/0003-qwen4exp-qsa-dense-bypass.patch`). On every full-attention
  layer, llama.cpp's qwen4exp graph ran the whole indexer each token: a
  64-head query projection, re-pooling and scoring every block of the
  indexer key cache, and a top-k over every cell. Top-k keeps
  `min(n_kv, top_k + ratio - 1)` cells, so until the cache outgrows the
  budget it selects every cell and the result is exactly dense attention.
  The patch then skips all of it and only stores the indexer key for later;
  once the cache is larger the graph is rebuilt with the sparse path. It also
  drops a copy of the whole pooled key cache per layer in the sparse path
  (the block mean now reads the strided slices directly). Logits are
  bit-identical to the unpatched build, dense, sparse, and across the switch;
  on the random-weight test model token generation got 1.55x faster at an
  empty context and 3.5x at 4096 tokens. On the real model the share is
  smaller (the experts dominate), but the query projection alone is 12
  layers x n_embd x 8192 weights read per token for nothing.
- **RAM check.** The plan warns when the weights kept in RAM exceed physical
  memory (they are memory-mapped and would be re-read from disk per token).
- **Benchmark** (Run tab): a fixed ~1500-token prompt plus exactly 128
  generated tokens, straight against llama-server, reporting its own
  timings, so each of the above can be checked and tuned on real hardware.

Looked at and left alone:

- **The 51B-parameter PLE n-gram table** is already an input-layer tensor in
  llama.cpp (kept in RAM, lazily read through mmap), and the planner already
  counted it as such.
- **CPU matmul kernels for the RAM-resident experts.** With a CUDA build the
  CPU-side weights live in pinned host memory, so llama.cpp's repacked CPU
  layouts are never used; IQ4_XS experts go through its newer tiled matmul
  (batches of 8+ rows per expert) or the AVX2 dot product, and token
  generation from RAM is bandwidth-bound either way. (In a CPU-only build
  the repack path has a real weakness: its MoE matmul runs one token at a
  time and never uses its batched kernel, making repacked Q4_K experts ~35%
  slower than unrepacked ones at prompt batches on an AVX-512 CPU. Not this
  app's path, so not patched.)
- **CUDA kernels** (gated delta net, MMQ, flash attention) are already
  specialised per architecture; changing them without a GPU to verify
  correctness and speed would be guesswork.
- **Other graph work.** The hyper-connection mixers already use fused ops
  upstream; the remaining small elementwise ops are a few microseconds each
  on a GPU next to milliseconds of expert streaming.

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
