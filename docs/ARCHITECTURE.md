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
the offload plan and settings; chatting happens in llama.cpp's own web UI
(Open chat), which goes through the same proxy.

The panel follows established usability guidance: the model's state is
always visible (sidebar and page header, NN/g "visibility of system
status"); advanced options are one disclosure level down, never deeper
(progressive disclosure); every empty list says what's missing and offers
the next step (NN/g empty-state guidelines); system messages are toasts,
and only destructive or overwriting actions ask for confirmation; download
choices show whether a file fits the detected GPUs and RAM, as LM Studio
does.

Visually it follows a "kinetic typography" system, all tokens in the `:root`
block of `web/style.css`: Space Grotesk (bundled in `web/fonts/`, SIL Open
Font License, so the panel works offline), uppercase display type at poster
scale (page titles `clamp(2rem, 5.5vw, 5.5rem)`, the model state as the
headline), one acid-yellow accent on near-black, flat 2px geometry with no
radius or shadows, rows and tiles that flood with the accent on hover, and
two marquees (a live status ticker and the local-model list) built in
`app.js` from CSS keyframes. Deliberate departures for a tool rather than a
poster: typed values keep their case (paths, keys, arguments are
case-sensitive), placeholders stay legible because they carry real defaults,
and errors and Stop use a red, since the single accent can't also mean
danger. Text contrast is at least 7:1, focus is always visible, and every
animation stops under prefers-reduced-motion.

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

Split models (`-0000k-of-0000N.gguf`) become one section pointing at part 1;
llama.cpp loads the rest. A model with missing parts is left out and named in
the scan notes, and a file reached through two overlapping model folders is
counted once. Model ids strip only the shard suffix and `.gguf`, so dots in
names (`Qwen3.8`) survive. Single-model mode passes the same id as `--alias`
(unless one is set), so `/v1/models` reports it instead of the part-1 path.

Per-model settings (`model_overrides` in config.json, keyed by that id)
replace the shared `model` settings for that model, both when it is planned
and launched alone and in its router section. Settings the router can only
take once for all models (the API key, and the environment variables for
the GPU prompt offload and block attention) always come from the defaults
in on-demand mode.

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
- **QSA indexer memory** (`patches/0004-qwen4exp-qsa-fused-indexer.patch`).
  Once the sparse path runs, qwen4exp scored every block with a plain
  matmul: a `[blocks x 64 heads x tokens]` f32 tensor, then a ReLU copy of
  it, then a sum over the heads. At 32k context and micro-batch 2048 each
  is 4 GiB, and they set the compute buffer (VRAM, on every GPU with a
  full-attention layer): 8.5 GiB on the random-weight test model, whose
  KV cache is 32 MiB. The patch uses ggml's fused lightning-indexer op
  (written for DeepSeek's indexer, CUDA kernel for head size 128 with 32/64
  heads), which never materialises it, and adds the attention mask to the
  transposed block scores directly instead of copying both. Compute buffer
  8549 -> 825 MiB (ubatch 2048) and 2143 -> 207 MiB (ubatch 512); logits
  identical; CPU prompt processing +12%, generation +6% at 4096 tokens.
  What remains is ~13 bytes per cell per micro-batch token (the block
  scores expanded to cells, the selection mask).
- **Planner: QSA compute and indexer cache.** The planner now reserves that
  remaining 13 B x context x micro-batch on each GPU when the context can
  outgrow the indexer budget, and raises the micro-batch for RAM-resident
  experts only as far as it doesn't cost more than one extra block's
  experts. It also counted the indexer cache as 64 heads x 128 per token;
  llama.cpp stores one 128-wide key per cell, so the old plan
  overestimated it 64x (several GiB at 64k context) and pushed experts to
  RAM for nothing.
- **Prompt cache capped to spare RAM.** llama-server keeps up to 8 GiB of
  idle-slot state in host RAM by default (`--cache-ram 8192`). With the
  experts memory-mapped from a model that already fills RAM, that cache
  evicts expert pages, which are then re-read from disk per token. The
  launcher passes `--cache-ram` with what the RAM-resident weights leave
  (minus 4 GiB for the OS and buffers), or leaves the default when it fits.
- **No pinned copy without op offload**
  (`patches/0005-no-pinned-copy-without-op-offload.patch`, any model). With a
  GPU present, llama.cpp stores every CPU-resident weight in a pinned CUDA
  host buffer: a full, non-pageable copy that replaces the memory map. Its
  only purpose is faster copies to the GPU for large batches (op offload),
  yet it is made with `--no-op-offload` too. The patch makes
  `--no-op-offload` imply `--no-host`, so those weights stay memory-mapped
  (and repackable for the CPU kernels). The planner already switches op
  offload off on GPUs too small for its staging area.
- **CPU MoE matmul on repacked weights**
  (`patches/0008-cpu-repack-moe-gemm-prefer-tiled.patch`, any model). The
  repacked-weight path for `MUL_MAT_ID` ran its one-token kernel once per
  token for every expert and never its 4-token kernel, re-reading each
  expert's weights per token. It now gathers each expert's rows in fours
  for the batched kernel. MoE matmul at 127 tokens: Q4_0 15.5 -> 12.2 ms,
  IQ4_NL 17.3 -> 12.4 ms, at 32 tokens Q4_0 4.6 -> 3.4 ms; results match
  the unrepacked path. Affects CPU-only builds and weights kept out of the
  pinned buffer (above).
- **Q4_K/Q2_K left to the tiled kernel on x86** (same patch). llama.cpp's
  newer tiled K-quant matmul can't read repacked weights, and for these two
  types the repacked kernel is the slower one: at a 512-token micro-batch
  0.54x (dense) and 0.45x (MoE) of the tiled speed for Q4_K, 0.6-0.7x for
  Q2_K, and equal at 1 token. x86 builds now leave them unrepacked, which
  also keeps them memory-mapped instead of copied. Set
  `GGML_CPU_TILED_MM=0` to get the old behaviour. The patch also fixes the
  tiled kernel's src1 type/layout check, which was unreachable.
- **qwen4exp models quantized by llama.cpp crashed**
  (`patches/0007-qwen4exp-quantize-ple-norms.patch`). `llama-quantize`
  quantized the PLE norm weights (2-D `[n_embd, hc]`, so its norm rule
  missed them) and the graph then aborted in an elementwise multiply. They
  now stay f32.
- **Opt-in QSA block selection** (`patches/0006-qwen4exp-qsa-block-select.patch`,
  Settings -> Model, off by default, `LLAMA_QWEN4EXP_QSA_BLOCKS=1`). Top-k
  over the block scores instead of every cell, then a block mask expanded
  to cells. Compute buffer 825 -> 569 MiB at 32k / ubatch 2048; logits
  within ~1e-4 of the per-cell path on the test model, same greedy tokens;
  CPU speed within noise. It keeps whole blocks, where the reference keeps
  `top_k + ratio - 1` cells (possibly part of one more block, which part
  depending on the backend's tie-breaking). CUDA's sparse flash attention
  keeps at most `n_kv_max` visible cells per row; the bound passed allows
  one block more than a contiguous context needs.
- **Plans can't go stale or silently overflow VRAM.** Three fixes after a
  report of generation dropping from ~10 to ~4 t/s:
  - A plan made while a model is running adds that model's own GPU memory
    back (from llama-server's logged buffer sizes); before, opening the
    panel with a model loaded planned everything onto the CPU, and Start
    then launched that plan.
  - Start re-plans with the memory free at that moment unless the command
    line was edited by hand.
  - After each load the launcher compares llama-server's per-GPU buffers
    with what the GPU had free. On Windows an overshoot doesn't fail, it
    spills into shared system memory at a fraction of the speed, so the
    overshoot is saved per model and GPU (`vram_corrections` in the
    config), the model is restarted once with the corrected plan, and later
    plans keep that memory free. Those buffer lines are only logged from
    llama-server's log level 4, so the launcher passes `--log-verbosity 4`
    (at the default level the check had nothing to read).
  - The reverse: the per-GPU reserves (CUDA context, compute buffers, a
    safety margin of 512 MiB + 2% of the card) are estimates made before
    anything loads and left 2 GiB of an 8 GiB card and 4 GiB of a 16 GiB
    one unused on a real system. After a load whose plan keeps experts in
    RAM, the launcher reads each GPU's free memory (nvidia-smi); where more
    than 768 MiB is free, all but 512 MiB of it becomes a negative
    correction for that model and GPU, and the model is restarted once with
    the fuller plan. An overflow later drops a negative correction before
    adding its own. Every plan that leaves weights in RAM lists each GPU's
    budget (free − context − compute − staging − sparse attention − safety
    ± learned).
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
- **Recurrent-state rollback slots and context checkpoints** only cost
  memory with speculative decoding (slots) or every 8192 tokens
  (checkpoints of the small recurrent state); nothing to change.
- **Only the used experts are copied** for op offload already (the
  scheduler reads the routing ids and copies the used experts' ranges), and
  every weight input starts its own split, so that applies to gate, up and
  down alike.
- **Caching pooled indexer keys** (only the newest QSA block changes per
  token) would save re-pooling the whole cache each step; it needs a new
  cache tensor, and pooling is ~1/16 of the scoring cost at ratio 4.
- **Other graph work.** The hyper-connection mixers already use fused ops
  upstream; the remaining small elementwise ops are a few microseconds each
  on a GPU next to milliseconds of expert streaming.

## Expert usage recording

llama.cpp keeps each layer's experts in one tensor, so a layer's experts
are either all in VRAM or all in RAM. Whether splitting them (the most-used
experts on the GPU) would pay off depends on how unevenly the router picks
them, which nothing in the GGUF says. `patches/0009-moe-expert-usage-stats.patch`
measures it: with `LLAMA_EXPERT_STATS=<file>` set, each MoE layer's top-k
selection is kept as a graph output, read back after every batch, and
counted per layer and expert, separately for prompt batches and one-token
generation. A background thread rewrites `<file>` as JSON once a second
while the counts change (and at exit). Output is unchanged; the cost is one
sync per batch and the lost CUDA top-k fusion, only while enabled.

The launcher sets it with the "Record expert usage" model setting (one-model
mode; in router mode every model would share one file): each launch writes
`expert-stats/<model id>/live/run-<time>.json` in the config directory,
plus a `.meta` file with how many blocks kept their experts in RAM under
that launch's plan. Profiles (named on the Run page, the active one in
`expert-stats/<model id>/active`) keep workloads apart, since coding and
chat may route differently. Switching profile takes effect at once: a
`.seg` file next to the live run records which profile each stretch of it
belongs to and the counts when the stretch began, and once the model has
stopped the run is split into one finished run per stretch, in
`expert-stats/<model id>/<profile>/`. The latest session's routing is compared
with every profile (per-layer cosine similarity of the expert counts,
averaged) and the closest one is suggested; a later hot-expert placement can
use the same match to pick its profile. The Run page sums a profile's runs
and shows, for the
RAM-resident layers, the share of routed tokens that went to each layer's
busiest 10/25/50% of experts, next to the even-spread baseline. If the
busiest quarter takes at least half the tokens, hot-expert placement is
worth building.

## Hot experts

`patches/0010-moe-hot-experts.patch` (`LLAMA_MOE_HOT=<file>`, one line per
block: `<block>: <expert> <expert> ...`). After loading, llama.cpp copies the
listed experts of each block whose expert tensors are in host memory into a
new buffer on that block's GPU (plus their per-expert scales, and a small
table mapping expert id to hot slot). `build_moe_ffn` then splits the
routing: the GPU runs the hot copies with the non-hot choices' weights set
to 0 (and a dummy slot), the CPU runs the full tensors with the hot choices'
ids set to -1, which the CPU `MUL_MAT_ID` (generic and repacked) now skips,
writing zero rows; the two results are added. Per-expert scales are looked
up with the real ids on the CPU side. Only for batches below the op-offload
threshold (`GGML_OP_OFFLOAD_MIN_BATCH`), since larger ones are copied to the
GPU whole anyway; not for expert biases or LoRA. On the CPU build, with the
split forced onto the CPU (`LLAMA_MOE_HOT_TEST=1`), greedy output is
identical to the unsplit graph (2 and 16 experts, one/some/all hot, prompts
split and unsplit). The CUDA side uses only existing ops (get_rows, cast,
mul, sub, mul_mat_id on a smaller tensor) and is untested on hardware.

Planner: with "Keep the most-used experts in GPU memory" and at least 1,000
recorded tokens in the chosen profile, a plan that would keep experts in RAM
instead keeps all of them there (`--n-cpu-moe` = block count) and fills each
GPU's remaining budget with the most-used experts of the blocks on it, by
recorded hits per byte (`pickHotExperts`). Since a block's hot experts
live on the block's GPU, the ordered fill (first GPU first) would put every
block on the first GPU once only their non-expert parts are left, and the
other GPUs would hold nothing; with two or three GPUs the planner instead
tries the contiguous splits and keeps the one whose hot experts cover the
most recorded traffic (`bestHotSplit`). Whole blocks are a special case
of that choice, so for skewed routing it covers more traffic in the same
memory; the plan reports both shares. The hot list goes from the plan's
command line (`--q38-moe-hot <file>`, removed before launch) into the
environment. Hot experts come from their own profile choice, kept apart
from the recording profile: a fixed profile only changes when it is also
recorded into (routing doesn't depend on placement, so recording with hot
experts on is still unbiased), and "auto" picks at each start the profile
the latest session resembled most, else the recording profile if it has
enough data, else the one with the most. Changing it needs a restart: the
copies are made at load.

## Profiling tab and auto-profiling

Everything about expert usage lives on the Profiling tab and belongs to one
model: its recording and hot-expert switches (`settings.json` in its
expert-stats folder; before, these were model settings, which are still
read while a model has no such file), its profiles, and the choices of
recording and hot-expert profile. The page asks one question per card:
"Hot experts: Off / Auto / <profile>" (switch and profile in one menu,
with a sentence saying what the next start does and a restart button when
the running load, whose profile the launcher remembers from the hot-expert
file's `# profile` line, differs), creating profiles automatically,
"Record: Off / into <profile> / into a new profile", and a list of profiles
with how much work their busiest quarter of experts does.

Auto-profiling (`autoprofile.go`) creates one profile per chosen use case
(assistant, coding, agentic coding, roleplay, storytelling, writing,
research, math, translation, summaries). It stops whatever runs, loads the
model with recording on and hot experts off (`tuneWith(..., hot=false)`,
and `launch` records while a run is active), and keeps the API proxy
detached so outside requests can't mix in. For each use case it switches
the live recording profile (waiting 1.5 s first so the previous counts are
written out) and sends that use case's prompts, with its system prompt,
straight to llama-server until the chosen number of generated tokens. Then
it stops the model, which files each stretch into its profile, restores the
recording profile and starts the model that was running before again (in
one-model mode; on-demand mode is left stopped with a note). Its runs are
marked `auto` in their `.meta` files: they fill profiles but are never "the
latest session" that profiles are matched against, so auto's choice keeps
following real use. Cancelling keeps what was recorded so far.

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
