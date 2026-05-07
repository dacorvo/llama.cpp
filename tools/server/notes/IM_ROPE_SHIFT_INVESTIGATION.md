# Investigation plan: K-shift compatibility with IM-RoPE

## Why this exists

`feat/cache-reuse-symmetric` extends `--cache-reuse` so the splice search
walks both `head_c` (cache) and `head_p` (recipient) on miss, letting it
discover chunks that recur at non-aligned positions in donor and
recipient. The patch covers the agent-trace recurrence pattern that the
legacy single-pointer scan cannot reach.

Tested end-to-end against the goose splice manifest using Llama-3.2-1B,
the patch fires on real workload data and applies positive-shift splices
correctly — but **not** against the actual production target,
`unsloth/Qwen3.6-35B-A3B`. That model's architecture (`qwen35moe`) uses
**IM-RoPE** (Interleaved Multi-axial RoPE), and llama.cpp's
`llama_kv_cache::get_can_shift()` returns false for any model with
`hparams.n_pos_per_embd() > 1`:

```cpp
// src/llama-kv-cache.cpp
bool llama_kv_cache::get_can_shift() const {
    if (model.arch == LLM_ARCH_STEP35) return false;
    if (hparams.n_pos_per_embd() > 1) return false;   // IMROPE / MROPE
    return true;
}

// src/llama-hparams.cpp
uint32_t llama_hparams::n_pos_per_embd() const {
    return rope_type == LLAMA_ROPE_TYPE_MROPE ||
           rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
}

// src/llama-model.cpp — IM-RoPE assignment
case LLM_ARCH_QWEN3VL:
case LLM_ARCH_QWEN3VLMOE:
case LLM_ARCH_QWEN35:
case LLM_ARCH_QWEN35MOE:
    return LLAMA_ROPE_TYPE_IMROPE;
```

The blocker is *generic* — any cache-reuse use case (prefix or symmetric)
is gated off for these architectures, not just our extension.

This document is the investigation plan to determine whether IM-RoPE
shift is **structurally feasible** for text-only Qwen3.5/3.6 inputs, and
if so, what the implementation cost looks like.

## Goal

Either (a) a concrete patch design with bounded scope that flips
`get_can_shift()` to true for IMROPE under defined conditions, or (b) a
documented reason it's structurally infeasible.

## Phase 1 — Read the existing code (1-2 h)

Goal: understand the IMROPE forward pass and the K-shift code path well
enough to identify *exactly* which line assumes scalar position.

**Files to read:**

1. **IM-RoPE rotation.** Where rotation is applied during regular forward
   prefill / decode:
   - `src/llama-graph.cpp` — grep `LLAMA_ROPE_TYPE_IMROPE` /
     `imrope` / `n_rot` / `rope_type`. Identify which `ggml_rope_*`
     variant is invoked and what shape its position argument has.
   - `ggml/include/ggml.h` and `ggml/src/ggml-cuda/rope.cu`:
     - `ggml_rope_multi(...)` and `GGML_OP_ROPE_MULTI` — the multi-axis
       variant probably takes a `[n_tokens, n_pos_per_embd]` position
       tensor instead of `[n_tokens]`.
   - CPU fallback: `ggml/src/ggml-cpu/ops.cpp` (or wherever the CPU op
     handler lives) for the same op.
2. **Position storage in the KV cache.**
   - `src/llama-kv-cells.h` — does each cell hold a single
     `llama_pos pos` field or a 4-tuple? If a single value, the multi-axis
     positions must be reconstructed at attention time (likely from token
     index + an axis encoding stored elsewhere).
   - `src/llama-kv-cache.cpp::cells.pos_set` / `pos_add` — what they take.
3. **The K-shift code path** (the thing `get_can_shift` gates):
   - Find the function that applies `pos_add` to all cells and triggers
     the K rotation. Look for `kv_self_update` or `apply_shift` or
     similar in `src/llama-kv-cache.cpp` and `src/llama-context.cpp`.
   - Identify the exact `ggml_rope_*` call that does the K rotation,
     and the position argument it builds.

**Output of phase 1:** a 50–100 line writeup at the bottom of this doc
under `## Findings — Phase 1`, naming:

- The exact ggml op used for IMROPE forward (e.g. `ggml_rope_multi`).
- The exact line in the shift path that builds a scalar position
  argument and would need to become multi-axis-aware.
- Whether the cache stores 1 or 4 positions per cell.

## Phase 2 — Per-axis analysis (2-3 h)

Goal: determine, for **text-only** Qwen3.5/3.6, what the 4 IMROPE axes
encode and whether they're linearly decomposable under shift.

The IM-RoPE family was designed for vision-language models — the 4 axes
encode (T, H, W, ?) for image tokens, where T is sequence position, H/W
are 2D image position. For text-only inputs (no image), we expect three
of the axes to be constant (typically zero) and one to track token index.
**That hypothesis needs to be verified, not assumed.**

**Concrete probe:** instrument `llama_batch_allocr::init` in
`src/llama-batch.cpp` to dump the per-token 4-axis position tuple for a
text-only Qwen3.5 / 3.6 request. If the dump shows `(t, 0, 0, 0)` for
every token where `t` matches token index, the axes are decomposable and
shift is per-axis-independent on the temporal axis.

If instead the dump shows two or more axes varying coherently, or any
axis that varies non-monotonically with token index, IMROPE shift on
text inputs is *not* a simple temporal-axis shift — and the
investigation may end there.

**Implementation of the probe:** ~10-20 lines in `llama-batch.cpp`
behind an env var (`LLAMA_DUMP_MROPE_POS=1`) so it doesn't pollute
normal logs. Run it once with a small Qwen3.5/3.6 GGUF (e.g. the
`qwen3.6-35b-a3b` Q4_K_M we already have) on a 50-token prompt, capture
the dump, document.

**Output of phase 2:** explicit answer to "are positions linearly
decomposable for text-only IM-RoPE?" If yes, proceed to phase 3.

## Phase 3 — Math design (only if phase 2 says yes) (2-4 h)

Design the K-shift for IMROPE-with-text:

```text
For each axis i in {0..3}:
    if axis i is constant for the cell being shifted:
        no rotation needed (the K already has axis i baked in correctly)
    else if axis i is the temporal axis:
        rotate K by delta * theta_axis_i  (standard scalar shift)
    else:
        ERROR — axis varies but not as expected; refuse shift
```

Concretely this means:

- The K-shift code path must know which axis is the temporal one. For
  text-only, this is presumably axis 0.
- The rotation has to operate per axis. For most ggml backends the IMROPE
  forward already does this (`ggml_rope_multi`); the shift just needs to
  call the same op but with a delta-vector that's `(delta, 0, 0, 0)`
  instead of a scalar `delta`.

**What to design:**

1. New `ggml` op variant or argument: `ggml_rope_multi_shift` (or extend
   `ggml_rope_multi` to accept a "delta-vector" mode).
2. Backend implementations: CUDA (the prod target), CPU (testing fallback).
   Optionally Metal/ROCm/SYCL/Vulkan but not blockers.
3. Plumbing in `llama_kv_cache::apply_shift` (or wherever) to call the
   new op when `n_pos_per_embd > 1` instead of returning false.

**Output of phase 3:** patch sketch with:

- Files touched.
- LoC estimate per file.
- A clear go/no-go decision based on whether scope fits in roughly
  200 LoC across backends.

## Phase 4 — Verification design (1-2 h)

Before writing any shift code, design how we verify it produces correct
outputs. Two-prong test, mirroring the existing `tests/server/unit/test_cache_reuse.py`:

- **Mechanical, in-process (offline)**: prefill prompt P1 with positions
  `[0..N)`. Apply shift `+delta`. Compare resulting K tensor to: prefill
  P2 with positions `[delta..delta+N)` from scratch. Bit-exact at fp32,
  numerically close at fp16/Q4. Lives as a llama.cpp unit test under
  `tests/test-kv-shift-imrope.cpp` (or extend an existing test).
- **End-to-end (server)**: re-run the symmetric cache-reuse smoke test
  (`tools/server/tests/unit/test_cache_reuse.py`) on a Qwen3.5/3.6 GGUF.
  Verify `cache_n` reflects the splice, the response is sensible,
  position metadata in the KV cache stays consistent.

**Output of phase 4:** a test spec that can be implemented before any
shift code (TDD-style) and used as the gate for landing the patch.

## Phase 5 — Go/no-go decision

Concrete criteria:

- **Go** if all hold:
  - Phase 2 confirms text-only IMROPE positions are decomposable (one
    temporal axis varies, others constant per cell).
  - Phase 3 patch scope ≤ ~200 LoC across CUDA + CPU.
  - Phase 4 mechanical test design has a clean pass/fail signal.
- **No-go** if any:
  - Phase 2 reveals nonlinear axis coupling.
  - Phase 3 scope is multi-backend invasive surgery (> ~500 LoC).
  - There's an architectural reason text-only IMROPE positions can't be
    treated as decomposable (unlikely but possible if some pre-RoPE
    layer expects a specific 4-tuple structure).

If **go**: phase 4 tests land first, then phase 3 patch, then re-run the
agentcap manifest test against Qwen3.6.

If **no-go**: write up the blocker as a separate llama.cpp issue
(referenced from this branch's commit message) and pivot the reagent
work toward request-semantic admission instead of KV-cell splice. The
cache-reuse-symmetric patch still lands as-is, useful for non-IMROPE
models.

## Time budget

| phase | estimate |
|---|---|
| 1. Read code | 1-2 h |
| 2. Per-axis analysis (incl. instrumentation patch) | 2-3 h |
| 3. Math design (if go) | 2-4 h |
| 4. Test design | 1-2 h |
| **Investigation before any shift implementation** | **6-11 h** |

If go, actual implementation is probably another 1-2 days on top.

## Findings

### Phase 1 — code reading

**Forward pass (IMROPE rotation).**

- The op is `ggml_rope_multi` (`ggml/include/ggml.h:1858`,
  `ggml/src/ggml.c:4198`). Used for IMROPE on Qwen3.5/3.6 in
  `src/models/qwen35.cpp:250` and `src/models/qwen35moe.cpp:263`.
- IMROPE packs 4 axes into one head as `[ttyxttyxttyxttyx00]` over the
  first `n_rot` dims (`ggml/include/ggml.h:1847`). `sections[4]` from the
  GGUF (`LLM_KV_ROPE_DIMENSION_SECTIONS`) selects how many dim-pairs each
  axis gets; sum = `n_rot / 2`.
- The position tensor `b` for `ggml_rope_multi` is a 1D `I32` of length
  `n_tokens * n_pos_per_embd` (`src/llama-graph.cpp:1798`). Layout is
  axis-major: 4 contiguous blocks of `n_tokens` ints, one per axis.
- `llm_graph_input_pos::set_input` (`src/llama-graph.cpp:105-124`)
  fills text-only inputs as `pos_data[k*n_tokens + i] = ubatch->pos[i]`
  for `k ∈ {0,1,2}` and `pos_data[3*n_tokens + i] = 0`. So all three
  "t-like" axes carry the token index, and the 4th ("w") is zero. This
  is the structure phase 2 needs to confirm at runtime.

**Position storage in the KV cache.**

- Each cell stores a single scalar `llama_pos pos`
  (`src/llama-kv-cells.h:464`) plus a `llama_kv_cell_ext{x, y}` for 2D
  image positions (`src/llama-kv-cells.h:13-28, 467`). No 4-tuple
  anywhere — the IMROPE 4th axis is implicit (text=0).
- `shift` is also a single scalar per cell
  (`src/llama-kv-cells.h:484`). The current data structure can only
  represent a uniform delta across whichever axes are "temporal";
  per-axis differential shifts have no place to live.
- `pos_add` / `pos_set` / `get_shift` all take/return a scalar
  `llama_pos`, no axis index.

**K-shift code path.**

- `llama_kv_cache::update` (`src/llama-kv-cache.cpp:777-810`) gates on
  `get_can_shift()`, then calls `build_graph_shift`.
- `get_can_shift()` (`src/llama-kv-cache.cpp:1090-1099`) returns false
  for `n_pos_per_embd() > 1` — this is the blocker.
- `build_graph_shift` (`src/llama-kv-cache.cpp:1800-1843`) creates
  `inp->k_shift = ggml_new_tensor_1d(I32, kv_size * n_stream)` —
  **one scalar per cell, no axis dim** — and per layer calls
  `build_rope_shift`.
- `set_input_k_shift`
  (`src/llama-kv-cache.cpp:1404-1416`) writes `cells.get_shift(i)`
  (scalar) into that tensor.
- `build_rope_shift` (`src/llama-kv-cache.cpp:1721-1771`) **already
  has an IMROPE/MROPE branch** at line 1739-1745:

  ```cpp
  const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE ||
                           hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                              ? LLAMA_ROPE_TYPE_NEOX
                              : hparams.rope_type;
  ```

  It then calls `ggml_rope_ext_inplace(ctx, K, k_shift, factors, n_rot,
  NEOX, ...)` (line 1765). Since `get_can_shift()` returns false first,
  **this branch is dead code today.**

**The exact line that assumes scalar position.**

`src/llama-kv-cache.cpp:1806`:
```cpp
inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
```
One I32 per cell. To support per-axis shift this needs to become
`[n_pos_per_embd, get_size()*n_stream]` (or equivalent), and
`set_input_k_shift` and the rope op call need to change accordingly.

**Why the existing IMROPE workaround is wrong for text-only Qwen3.5/3.6.**

Forward IMROPE rotates each dim-pair `(2i, 2i+1)` by
`axis_pos(i) * theta_i` where `axis_pos(i)` is the position on the axis
selected by section index for that pair (one of t,t,t,w in the
`[ttyx]` interleave). For text-only with `(t,t,t,0)`, three quarters of
the dim-pairs rotate by `t*theta`, one quarter by `0`.

The NEOX workaround computes
`ggml_rope_ext_inplace(K, scalar_delta, ...)` which rotates **every**
dim-pair in `[0..n_rot)` by `delta * theta_i` — including the dim-pairs
that the forward pass left at zero rotation. After this shift those
dims would be off by `delta * theta_i`. So the workaround composes
correctly when *every* axis is the same temporal one (uniform
`(t,t,t,t)` over text), but Qwen3.5/3.6 forward sets the w-axis to 0,
so the workaround over-rotates the w-axis dim-pairs.

Concretely: a correct K-shift for text-only IMROPE needs to apply
`ggml_rope_multi` with a per-axis delta vector `(δ, δ, δ, 0)` (matching
the `(t,t,t,0)` layout that prefill used), not a scalar `δ`. That is
the structural change phase 3 would design.

**Summary for phase 2.**

- Cache stores 1 scalar pos per cell (+ optional 2D `ext.{x,y}`); 4th
  IMROPE axis is implicit.
- Forward op = `ggml_rope_multi` with axis-major position tensor of
  shape `[4, n_tokens]`.
- The line that assumes scalar shift is the I32 1D tensor allocation in
  `build_graph_shift` plus the matching scalar fill in
  `set_input_k_shift`. Everything downstream of that is parameterised
  in `n_rot` and rope_type and could in principle accept a multi-axis
  position arg if `ggml_rope_multi` is wired in.
- The phase 2 probe (dump per-token 4-axis positions for a text-only
  Qwen3.5/3.6 prompt) should confirm `(t,t,t,0)`. If so, phase 3 is
  "swap the scalar k_shift tensor for a `[4, kv_size*n_stream]` one
  and call `ggml_rope_multi` instead of `ggml_rope_ext_inplace` in the
  IMROPE branch of `build_rope_shift`."

### Phase 2 — per-axis analysis

**Static analysis (sufficient on its own).**

`llm_graph_input_pos::set_input` (`src/llama-graph.cpp:105-124`)
unconditionally writes the per-axis layout for any ubatch with text
tokens and `n_pos_per_embd == 4`:

```cpp
pos_data[             i] = ubatch->pos[i];
pos_data[  n_tokens + i] = ubatch->pos[i];
pos_data[2*n_tokens + i] = ubatch->pos[i];
pos_data[3*n_tokens + i] = 0;
```

There is no other code path that writes axes for the text branch. The
4-tuple `(t, t, t, 0)` is hardcoded.

**Empirical confirmation (probe).**

Probe added to the same `set_input` behind `LLAMA_DUMP_MROPE_POS=1`
(diff in working tree, ~30 lines, single-spot). Run on
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` with prompt "The capital of France is"
(`-n 4`, `-ngl 999`):

```
prefill (n_tokens=11):
  axis 0: 0 1 2 3 4 5 6 7 8 9 10
  axis 1: 0 1 2 3 4 5 6 7 8 9 10
  axis 2: 0 1 2 3 4 5 6 7 8 9 10
  axis 3: 0 0 0 0 0 0 0 0 0 0 0
  invariant: (t,t,t,0) holds for all 11 tokens

decode batch (n_tokens=4): axis 0/1/2 = 11..14, axis 3 = 0,0,0,0
decode 1 token:            axis 0/1/2 = 15,    axis 3 = 0
decode 1 token:            axis 0/1/2 = 16,    axis 3 = 0
decode 1 token:            axis 0/1/2 = 17,    axis 3 = 0
```

Plus two warm-up ubatches (`n_tokens=2`) with the same layout. Every
ubatch — prefill, multi-token decode, single-token decode — satisfies
`(t, t, t, 0)`.

**Answer.**

Yes — text-only IM-RoPE positions are linearly decomposable. Three
axes carry the token index identically; the 4th is constant zero. A
shift `δ` on the temporal axis is therefore a delta-vector
`(δ, δ, δ, 0)` applied to all four axes, and this composes correctly
with `ggml_rope_multi`'s axis-major position layout (because the
forward op rotates each dim-pair by `axis_pos[axis(pair)] * theta_i` —
so adding `δ` to the three t-axes shifts those dim-pairs by `δ*theta_i`,
and adding `0` to the w-axis leaves its dim-pairs untouched, which is
exactly what we want).

**Out-of-scope (multimodal).** For ubatches that contain image tokens,
`set_input` falls through to the `else` branch and uses `ubatch->pos`
verbatim — meaning callers (mtmd) are expected to have packed a real
4-axis position into the batch. That is not on the cache-reuse path
(images aren't currently cache-reused) and is explicitly out of scope
for this investigation; the K-shift design below assumes the
text-only `(t,t,t,0)` invariant and refuses shift if any cell's
forward-time positions deviated from it.

**Phase 2 verdict: GO** for phase 3. Probe is left in the working tree
for reproducibility; one Edit to `src/llama-graph.cpp` reverts it.

### Phase 3 — math design

**Math.**

Forward (per dim-pair `(2i, 2i+1)`) for IMROPE-with-text rotates by
angle `axis_pos(i) * theta_i`, where `axis_pos(i) ∈ {t, t, t, 0}`
selected by the sector index. After prefill, K's stored value is
`R(axis_pos(i) * theta_i) · K_raw`.

To shift cell from `t` to `t + δ` we want `R((axis_pos(i) + Δ_i) *
theta_i) · K_raw` where `Δ_i = δ` for the three "t" axes and `Δ_i = 0`
for the w-axis. Since 2D rotations compose additively
(`R(a) · R(b) = R(a+b)`), applying a *second* `ggml_rope_multi` with
position vector `(δ, δ, δ, 0)` exactly produces this — without
re-deriving K_raw and without affecting the w-axis dim-pairs.

**Backend confirmation.**

`ggml/src/ggml-cuda/rope.cu:231-251` — IMROPE branch reads
`pos[i2 + ne02 * k]` for `k ∈ {0,1,2,3}`, axis-major flat.
`pos[i2]` is "t", `pos[i2 + ne02]` is "h", `pos[i2 + 2*ne02]` is "w",
`pos[i2 + 3*ne02]` is the 4th. For `i2 = cell_index` and `ne02 =
kv_size*n_stream`, a shift tensor of length `4 * kv_size*n_stream`
filled axis-major with `(δ, δ, δ, 0)` per cell drives the correct
rotation. No new ggml op needed.

**Patch sketch.**

All changes in `src/llama-kv-cache.cpp` — no backend changes.

1. `get_can_shift()` (`~12 LoC`): allow IMROPE/MROPE under text-only
   conditions:

   ```cpp
   bool llama_kv_cache::get_can_shift() const {
       if (model.arch == LLM_ARCH_STEP35) return false;
       if (hparams.n_pos_per_embd() > 1) {
           const auto rt = hparams.rope_type;
           if (rt != LLAMA_ROPE_TYPE_MROPE && rt != LLAMA_ROPE_TYPE_IMROPE) {
               return false;
           }
           // Refuse if any used cell has a 2D spatial position (image cell).
           // ext.{x,y} default to 0 for text cells (set in pos_set, never written
           // unless ext_set is called by mtmd image insertion).
           for (uint32_t s = 0; s < n_stream; ++s) {
               const auto & cells = v_cells[s];
               for (uint32_t i = 0; i < cells.size(); ++i) {
                   if (cells.is_empty(i)) continue;
                   const auto & ext = cells.ext_get(i);
                   if (ext.x != 0 || ext.y != 0) return false;
               }
           }
       }
       return true;
   }
   ```

2. `build_graph_shift()` (`~3 LoC`): allocate the shift tensor at
   length `kv_size * n_stream * n_pos_per_embd`:

   ```cpp
   const int n_axes = (int) hparams.n_pos_per_embd();
   inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
                                     (int64_t) get_size() * n_stream * n_axes);
   ```
   (For `n_axes==1` this is identical to today's allocation.)

3. `set_input_k_shift()` (`~15 LoC`): fill axis-major
   `(δ, δ, δ, 0)`-style:

   ```cpp
   const int n_axes = (int) hparams.n_pos_per_embd();
   const uint32_t kvz = v_cells[0].size();
   for (int a = 0; a < n_axes; ++a) {
       for (uint32_t s = 0; s < n_stream; ++s) {
           const auto & cells = v_cells[s];
           for (uint32_t i = 0; i < kvz; ++i) {
               llama_pos val = 0;
               if (!cells.is_empty(i)) {
                   val = (a < 3) ? cells.get_shift(i) : 0;
               }
               data[(size_t) a * kvz * n_stream + s * kvz + i] = val;
           }
       }
   }
   ```

4. `build_rope_shift()` (`~25 LoC`): replace the existing
   "MROPE→NEOX" workaround with a real IMROPE/MROPE branch using
   `ggml_rope_multi[_inplace]`:

   ```cpp
   const auto n_pos_per_embd = hparams.n_pos_per_embd();

   if (n_pos_per_embd > 1) {
       int sections[GGML_MROPE_SECTIONS];
       std::copy(hparams.rope_sections.begin(),
                 hparams.rope_sections.begin() + 4, sections);
       if (ggml_is_quantized(cur->type)) {
           tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);
           tmp = ggml_mul_mat_aux(ctx, tmp, rot);
           tmp = ggml_rope_multi(ctx, tmp, shift, factors, n_rot, sections,
                                 hparams.rope_type, n_ctx_orig,
                                 freq_base, freq_scale,
                                 yarn_ext_factor, yarn_attn_factor,
                                 yarn_beta_fast, yarn_beta_slow);
           tmp = ggml_mul_mat_aux(ctx, tmp, rot);
           tmp = ggml_cpy(ctx, tmp, cur);
       } else {
           tmp = ggml_rope_multi_inplace(ctx, cur, shift, factors, n_rot, sections,
                                         hparams.rope_type, n_ctx_orig,
                                         freq_base, freq_scale,
                                         yarn_ext_factor, yarn_attn_factor,
                                         yarn_beta_fast, yarn_beta_slow);
       }
       return tmp;
   }
   // existing scalar path unchanged below…
   ```

   The current MROPE→NEOX swap (`src/llama-kv-cache.cpp:1739-1745`)
   is incorrect for text-only IMROPE (it over-rotates the w-axis
   dim-pairs by `δ*theta_i` when their forward rotation was 0) and is
   also dead code today behind `get_can_shift`. Remove it.

**Files touched.**

- `src/llama-kv-cache.cpp` — only file. Total ~55 LoC across the four
  functions above.

**No backend touchpoints.** `ggml_rope_multi[_inplace]` is the same op
that IMROPE forward already uses on every supported backend (CUDA,
CPU, Metal, etc.). The op accepts the same shape we'd build for the
shift tensor (axis-major flat, length `n_tokens × n_axes`).

**Phase 3 verdict: GO.** Patch fits well under the 200 LoC budget
single-file, single-backend (well, zero-backend).

### Phase 4 — verification

The patch only adds a third path through one well-isolated function, so
the verification surface is small. Two prongs, both already have
analogous tests in-tree.

**Prong 1 — mechanical math test (CPU-only, CI-runnable).**

Extend `tests/test-rope.cpp` with a new test mode dedicated to the
text-only IMROPE shift pattern (`(t,t,t,0)` + `(δ,δ,δ,0) ≡
(t+δ,t+δ,t+δ,0)`).

The existing `test-rope.cpp` (modes `m=2..4`, lines `141-258`) already
proves the additive-composition property of `ggml_rope_multi` under
arbitrary axis values, so an IMROPE-shift regression in `ggml_rope_multi`
itself would fail that test. The new mode is belt-and-braces — it
locks down the *specific* `(t,t,t,0)/(δ,δ,δ,0)` shape the K-shift
relies on, and serves as documentation for what the patch needs to be
correct.

Sketch (~40 LoC alongside the existing modes):

```cpp
// IMROPE text-only K-shift mathematical correctness:
// rope_multi(rope_multi(x, p_init), p_shift) ≈ rope_multi(x, p_full)
// where p_init = (t,t,t,0), p_shift = (δ,δ,δ,0), p_full = (t+δ,t+δ,t+δ,0)
{
    const int64_t n_rot = 128;
    const int64_t ne[4] = { 2*n_rot, 32, 73, 1 };
    const int n_past_0 = 100;
    const int delta    = -67;       // shift; can be negative
    const int sections[4] = {16, 24, 24, 0};   // text-only IMROPE
    auto * x = get_random_tensor_f32(ctx0, 4, ne, -1.0f, 1.0f);

    auto * p_init  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]*4);
    auto * p_shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]*4);
    auto * p_full  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ne[2]*4);

    for (int i = 0; i < ne[2]; ++i) {
        const int t = n_past_0 + i;
        for (int j = 0; j < 3; ++j) {
            ((int32_t *) p_init->data )[i + ne[2]*j] = t;
            ((int32_t *) p_shift->data)[i + ne[2]*j] = delta;
            ((int32_t *) p_full->data )[i + ne[2]*j] = t + delta;
        }
        ((int32_t *) p_init->data )[i + ne[2]*3] = 0;
        ((int32_t *) p_shift->data)[i + ne[2]*3] = 0;
        ((int32_t *) p_full->data )[i + ne[2]*3] = 0;
    }

    auto * r0 = ggml_rope_multi(ctx0, x,  p_init,  nullptr, n_rot, sections,
                                GGML_ROPE_TYPE_IMROPE, 32768, 1000000, 1, 0, 1, 32, 1);
    auto * r1 = ggml_rope_multi(ctx0, r0, p_shift, nullptr, n_rot, sections,
                                GGML_ROPE_TYPE_IMROPE, 32768, 1000000, 1, 0, 1, 32, 1);
    auto * r2 = ggml_rope_multi(ctx0, x,  p_full,  nullptr, n_rot, sections,
                                GGML_ROPE_TYPE_IMROPE, 32768, 1000000, 1, 0, 1, 32, 1);
    /* compute graph, then assert rel_err(r1, r2) < 1e-4 (same threshold as
       the existing modes). At fp32 the result is ~bit-exact. */
}
```

Pass criterion: `diff/sum < 1e-4` (matches existing test-rope.cpp
threshold). At fp32 the result is bit-exact; this only fails if a
backend regresses `ggml_rope_multi` itself.

**Prong 2 — end-to-end server smoke test.**

Extend `tools/server/tests/unit/test_cache_reuse.py` with a
parametrized variant that runs the existing
`test_cache_reuse_symmetric_chunk_in_middle` against an IMROPE model
when a local Qwen3.5/3.6 GGUF is available.

Today on Qwen3.5/3.6 the path is silently degraded:
`server-context.cpp:2404-2410` sets `can_cache_reuse =
llama_memory_can_shift(...) && !has_mtmd`, and `get_can_shift` returns
false for `n_pos_per_embd > 1`, so `n_cache_reuse` is logged-and-zeroed
with no splice attempted. The test today therefore can't be run against
Qwen3.5/3.6 — that's the regression this patch is fixing.

Sketch:

```python
import os, pytest

QWEN35_GGUF = os.environ.get("LLAMA_TEST_QWEN35_GGUF")
slow = pytest.mark.skipif(
    not QWEN35_GGUF or not os.path.exists(QWEN35_GGUF),
    reason="LLAMA_TEST_QWEN35_GGUF not set or missing",
)

@slow
def test_cache_reuse_symmetric_imrope():
    """End-to-end: confirm symmetric cache-reuse splices on an IMROPE
    model after the get_can_shift fix lands."""
    server.model_alias = "qwen35"
    server.model_file  = QWEN35_GGUF
    # keep n_ctx tight so loading is cheap; one slot, kv_unified
    server.n_ctx = 2048
    server.n_slots = 1
    server.kv_unified = True
    server.start()
    donor_tokens, recipient_tokens, n_shared = _build_donor_recipient_token_lists()
    _post_tokens(donor_tokens, 1)
    res = _post_tokens(recipient_tokens, 4)
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n >= n_shared - 2, (
        f"expected ~{n_shared} reused tokens, got cache_n={cache_n} — "
        f"K-shift may have been refused (check server log for "
        f"\"cache reuse is not supported\")"
    )
```

Pass criterion: same as the existing tinyllama test —
`cache_n ≥ n_shared - 2`. Today this would fail because the splice is
silently skipped (cache_n ≈ 0).

A negative test (image cells → `get_can_shift` returns false) is
harder to wire without mtmd plumbing in pytest; deferred as a
follow-up. The text-only invariant is sufficient for the cache-reuse
use case (mtmd is already excluded upstream at `server-context.cpp:870-873`).

**Phase 4 verdict: GO.** Both prongs have direct analogues in-tree;
sketches above are ready to drop in once the patch lands. Combined
~80 LoC of test code.

### Phase 5 — decision

**Verdict: GO.** All three go-criteria from the plan hold:

| criterion | status |
|---|---|
| Phase 2: text-only IMROPE positions decomposable | ✓ confirmed `(t,t,t,0)` by static read + runtime probe on Qwen3.6-35B-A3B |
| Phase 3: patch ≤ ~200 LoC across CUDA + CPU | ✓ ~55 LoC, single file (`src/llama-kv-cache.cpp`), zero backend changes |
| Phase 4: mechanical test has clean pass/fail signal | ✓ extends an existing additive-composition test pattern in `tests/test-rope.cpp` |

**Why so cheap.** Most of the heavy lifting (`ggml_rope_multi`) is
already in the codebase and tested — IMROPE forward uses it on every
backend Qwen3.5/3.6 supports. The K-shift just needs to *call* it
instead of falling through to a NEOX scalar shift. The failure mode of
the current (dead) MROPE→NEOX workaround at
`src/llama-kv-cache.cpp:1739-1745` is also informative: it shows
someone thought the shift was "easy" but didn't account for the
text-only `(t,t,t,0)` invariant — the workaround would over-rotate
the w-axis dim-pairs.

**Order of work to land.**

1. Land the mechanical test (phase 4 prong 1) **first**, on master, to
   establish a baseline. Should pass today (it tests
   `ggml_rope_multi`, which already works).
2. Apply the patch (phase 3) on this branch.
3. Wire the e2e test (phase 4 prong 2) and run against
   `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` locally.
4. Re-run the agentcap manifest test against Qwen3.6 to confirm the
   symmetric cache-reuse path actually fires on real workload data.
5. Open the upstream issue / PR. Disclose AI assistance per
   `AGENTS.md`.

**Hygiene before any PR.**

- Revert the `LLAMA_DUMP_MROPE_POS` probe in `src/llama-graph.cpp` —
  it was a phase-2 throwaway (~30 lines). Single Edit reverts.
- The probe-confirmed `(t,t,t,0)` invariant becomes a comment near
  `get_can_shift` and `set_input_k_shift` so the next reader knows why
  the IMROPE branch sets axis-3 to 0.
- Keep this investigation doc in the branch; reference it from the
  PR description as design context (not a substitute for a real PR
  description).

**Out-of-scope follow-ups.**

- Image cells (mtmd): `get_can_shift` refuses if any cell has
  `ext.{x,y} != 0`. A future patch could allow image-aware K-shift,
  but it would need cells to store a per-cell axis tuple
  (`llama_kv_cell_ext` extended to 4 entries) and the shift to track
  per-axis deltas. Not needed for the cache-reuse use case (mtmd
  excluded upstream).
- Step35 (`LLM_ARCH_STEP35`): independent blocker (per-layer RoPE
  dims). Not addressed here.

**Time estimate to land vs. the plan's 1-2 days post-investigation.**

Investigation took ~2h (vs. 6-11h budgeted). Implementation
estimate: half a day for the patch + mechanical test + e2e wiring,
plus 2-3 hours for review iteration. Single-file scope, single-arch
verification (CUDA via the existing 35B GGUF) keeps it tight.

## Findings — implementation run

After the writeup above I went ahead and applied the patch in the
working tree to validate it end-to-end. Two things changed the
picture.

### What the implementation confirmed

- **Mechanical test passes.** A new test mode in `tests/test-rope.cpp`
  exercising the `(t,t,t,0)` + `(δ,δ,δ,0) ≡ (t+δ,t+δ,t+δ,0)` IMROPE
  composition produces `rel_err ≈ 0` at fp32. The math sketched in
  phase 3 is correct.
- **Patch builds clean and the K-shift fires on Qwen3.6.** With the
  patch, `get_can_shift` returns true for text-only IMROPE,
  `n_cache_reuse > 0` is no longer silently zeroed at startup, and
  the symmetric splice plumbs through `seq_cp` → `seq_add` →
  `seq_cp` without aborting. `cache_n=28` (full SHARED span) on the
  agent-style donor/recipient probe.

### What the implementation revealed that the plan missed

- **Phase 3 patch was incomplete.** The cells layer needed two more
  changes the writeup didn't anticipate:

  - `llama_kv_cache::seq_add` had a hard
    `GGML_ASSERT(n_pos_per_embd() == 1)` (`src/llama-kv-cache.cpp:517`,
    pre-patch). seq_add is the entry point for cache-reuse splice
    (and ctx_shift). The assert fires before K-shift even gets a
    chance — so just fixing `get_can_shift` and `build_rope_shift`
    leaves the cache-reuse path dead.
  - `llama_kv_cells::pos_add` does not touch `ext.{x,y}`. After
    seq_add the pos metadata says `t+δ` but ext still says `t`,
    breaking the text-only invariant `ext.{x,y} == pos[i]` that
    the next `get_can_shift` call (or the next `seq_add` validation)
    depends on. Needs an opt-in `shift_ext` flag passed from
    `seq_add` for the multi-axis path.

  Both extensions land in the same file as the rest of the patch.
  Updated patch size: ~85 LoC (was estimated at ~55).

- **Wrong gate condition in the writeup.** Phase 3 said image cells
  are "`ext.{x,y} != 0`". They aren't. For text-only Qwen3.6 the
  forward pass writes `(t,t,t,0)` per token, and
  `set_input` (`src/llama-kv-cache.cpp:1046-1054`) populates `ext.x =
  ubatch.pos[i + 2*n_tokens] = t`, `ext.y = ubatch.pos[i + n_tokens]
  = t` — so text cells routinely have non-zero ext. The correct
  invariant is `ext.x == ext.y == pos[i]`; image cells differ
  because their `(h, w)` aren't equal to the token's temporal index.
  Fixed in the working-tree patch.

### What the implementation revealed about the *enclosing* feature

The investigation set out to enable cache-reuse-symmetric for
Qwen3.6. The patch enables the path, but the e2e correctness
check (cold prefill vs spliced prefill, temp=0, identical recipient
prompt) shows the spliced run produces a *different* continuation:

```
baseline: ' words unique to recipient\n\n```python\n# Create a list ...'
spliced : '\nThe following table lists the recipient prefix words ...'
```

Following that thread:

- **Even legacy single-shift cache-reuse on tinyllama (scalar RoPE,
  non-hybrid) diverges from cold prefill** after ~11 chars. So the
  divergence is *not* introduced by the IMROPE patch — it is a
  pre-existing characteristic of llama.cpp's cache-reuse-with-shift.
  The existing `tests/server/unit/test_cache_reuse.py` only asserts
  `cache_n ≥ n_shared - 2`; it does not gate on output equivalence.
  So this is an unmeasured property of the master branch, not a
  regression.
- **Qwen3.6 divergence is dramatically worse than tinyllama's**, and
  the reason isn't (only) IMROPE precision. `LLM_ARCH_QWEN35` and
  `LLM_ARCH_QWEN35MOE` are listed in `llm_arch_is_hybrid`
  (`src/llama-arch.cpp:847-865`). The hybrid memory routes seq_add
  to *both* the attention KV cache *and* the recurrent (gated delta
  net) state. `llama_memory_recurrent::seq_add`
  (`src/llama-memory-recurrent.cpp:283-311`) only updates `cell.pos
  += shift` — the recurrent state itself encodes the donor's token
  sequence and isn't recomputable from a position relabel.
  `llama_memory_recurrent::get_can_shift` returns true with comment
  "shifting the pos is trivial for recurrent models" — that is
  correct for ctx_shift (sequence preserved, only positions
  relabeled) but not for cache-reuse splice (sequence reordered;
  the recurrent state was rolled forward from the donor's history,
  and continuing decode from it on the recipient's path produces
  garbage).

So the original goal — symmetric cache-reuse on Qwen3.6 — has *two*
blockers, and the investigation only solved one:

| layer | blocker | status |
|---|---|---|
| attention cache (K) | get_can_shift gates IMROPE/MROPE off; build_rope_shift's NEOX workaround is wrong | **solved** by this patch |
| recurrent cache (gated delta net) | `seq_add` only relabels pos; the recurrent state does not survive a sequence reorder | **out of scope** — separate redesign |

### Revised verdict

- **K-shift patch is mathematically correct** (mechanical test) and
  removes one of the two blockers for cache-reuse on multi-axis-RoPE
  models. It is sufficient for *non-hybrid* IMROPE/MROPE
  architectures (any pure-attention transformer that uses
  `LLAMA_ROPE_TYPE_IMROPE` or `LLAMA_ROPE_TYPE_MROPE`).
- **For hybrid models (Qwen3.5/3.6)**, landing this patch alone
  enables a code path that produces semantically degraded output
  for cache-reuse-with-shift. That is arguably *worse* than the
  pre-patch behavior of silently dropping `n_cache_reuse` with a
  warning. Two ways to handle:
  1. Land the K-shift patch AND tighten `llama_memory_hybrid::get_can_shift`
     to refuse shift when any cell has a recurrent counterpart with
     populated state. Closes the door on hybrid+cache-reuse until the
     recurrent-state issue is solved separately.
  2. Land the K-shift patch as-is and accept "best effort" cache-reuse
     on hybrid (degraded but not broken). Probably fine for ctx_shift
     (state-survives-relabel approximation) but bad for cache-reuse
     (reorder).

  Recommendation: option 1, with a follow-up issue documenting the
  recurrent-state problem.
- **Pre-existing cache-reuse divergence on tinyllama** is a separate
  finding worth documenting on its own (probably a llama.cpp issue);
  not introduced by this branch and out of scope for this
  investigation.

### Files in the working tree (not committed)

- `src/llama-kv-cache.cpp` — patch: get_can_shift,
  build_graph_shift, set_input_k_shift, build_rope_shift,
  seq_add (~85 LoC).
- `src/llama-kv-cells.h` — `pos_add` gains `shift_ext` flag and
  resets ext on cell drop (~5 LoC).
- `tests/test-rope.cpp` — IMROPE text-shift mechanical test (~75 LoC).
- `tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md` — this file.

The Phase-2 probe in `src/llama-graph.cpp` has been reverted.
The debug `LLAMA_DBG_CAN_SHIFT` printfs added during e2e debug have
been reverted. No `git add`/`commit` has been run; the branch is in
the same logical state as `feat/cache-reuse-symmetric` plus the
working-tree changes above.
