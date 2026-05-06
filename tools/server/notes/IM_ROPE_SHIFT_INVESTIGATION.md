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

*To be filled in as phase 1 runs.*

### Phase 2 — per-axis analysis

*To be filled in.*

### Phase 3 — math design

*To be filled in (only if phase 2 says yes).*

### Phase 4 — verification

*To be filled in.*

### Phase 5 — decision

*To be filled in.*
