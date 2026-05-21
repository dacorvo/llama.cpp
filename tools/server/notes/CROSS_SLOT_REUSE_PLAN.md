# Cross-slot cache reuse — implementation plan

Sequel to the symmetric cache-reuse work (commit `f68517789`) and the
post-eviction gate (commit `99fc73039`). The goal is to extend the
cache-reuse mechanism from "one prior context per slot" to a pool of
prior contexts addressable across slots — so that recurring chunks
appearing in *any* slot's history can be spliced into a new request's
prefill.

## Motivation

The current cache-reuse path is single-slot:

- `slot.prompt.tokens` retains one prior request per slot.
- CP (`get_common_prefix`) and the symmetric splice scan both read
  this single buffer.
- The scheduler does pick the slot with the highest LCP fraction
  against the new request (server-context.cpp:1143), but once a slot
  is selected, *only that slot's* prior content contributes to
  reuse — runner-up slots' content is ignored.

In real agent workloads (cf. agentcap traces), the recurring chunks
(tool outputs, file reads, command results) are scattered across
many independent sessions. Within a single session of a single slot,
typical cross-turn recurrence is dominated by the conversation's own
history — which is already covered by the existing path. The unique
value of cross-slot reuse is the cross-session case: chunk X appears
in slot A's history and recipient request lands on slot B.

## What's already in place

- **Symmetric scan** (`f68517789`): independent `head_c` and `head_p`
  pointers — handles non-prefix recurrence within one slot.
- **Pos-min gate** (`99fc73039`): drops candidates whose donor cells
  have been evicted. Required for cross-slot, since OTHER slots'
  KV may have evicted older cells.
- **Inter-seq splice machinery**: `seq_cp(src_seq, dst_seq, p0, p1)`
  + temp-seq-255 trampoline already supports moving cells between
  seq ids. The current cache-reuse path uses it intra-slot (slot.id
  → splice_temp → slot.id with position rephase); the same primitive
  serves inter-slot copies (other_slot.id → splice_temp → this_slot.id).
- **Scheduler LCP scoring** (server-context.cpp:1143): already
  computes `LCP / new_request.size` per slot. The selection picks the
  best; runner-up scores are discarded.

## Step plan

Each step is one commit, demoable in isolation, gated behind a flag
until validated.

### Step 1 — Plumb scheduler LCP into `update_slots`

The scheduler computes `sim_best` and `LCP(slot.prompt.tokens,
task.tokens)` per slot, picks the highest, then drops the data.
`update_slots` re-walks `get_common_prefix` to derive `n_past`.

- Add `int lcp_at_select; float sim_at_select;` fields to
  `server_slot`, populated at slot selection time.
- In `update_slots`, use `lcp_at_select` as the initial `n_past`
  instead of recomputing.

**Net behavior change:** none. **Validation:** rung-2.5 replay
produces identical splice events and timings.

**Effort:** ~30 lines, half a day.

### Step 2 — Cross-slot CP

After own-slot CP determines `n_past`, walk other idle slots and find
the one with the highest `LCP(S.prompt.tokens, task.tokens)`. If any
slot's LCP exceeds `n_past`:

- `seq_cp(S.id, this.id, 0, lcp_S)` — copy matching prefix cells from
  S's seq into this slot's seq. The match is position-aligned (both
  at position 0 onward), so no RoPE rephase needed.
- Advance `n_past` to `lcp_S`.

Gate behind `sim_at_select < CROSS_SLOT_THRESHOLD` (default 0.9):
the continuation case has near-perfect own-slot LCP, never benefits
from a cross-slot scan.

**Validation:** contrive a replay where two slots share a system
prompt + tools header; route a request to the slot that *doesn't*
have the recipient's session; confirm `n_past` jumps to include the
shared prefix.

**Effort:** ~80 lines, one day.

### Step 3 — Cross-slot symmetric scan (IN PROGRESS — deep-copy primitive)

**Status:** depends on a new `seq_cp_deep` memory-module primitive.
Scaffolding (header + metadata-only stub) is WIP on the asymmetric
branch; the K/V graph-side data move is the remaining work. The
destructive intra-slot path stays as-is — cross-slot is a separate
code path on top of the deep-copy primitive.

#### Background — why the obvious path doesn't work

The intent was: after own-slot symmetric scan, also scan **other
slots'** `prompt.tokens` for chunks matching the input past CP, and
for each match `seq_cp` the donor's cells into the splice temp
trampoline as today's intra-slot path does. A prototype was built
and reverted.

What stopped the naive approach from being safely shippable:

Under `--kv-unified`, all seq ids share a single underlying cell
pool. Same-stream `seq_cp(src, dst, p0, p1)` is **tag-additive**,
not data-copy — it just adds `dst` to the cells' seq bitmask. To
get the cells to the recipient's position the splice has to
positionally shift them via `seq_add(splice_temp, ..., +base)`. The
shift would corrupt the donor's view (donor still has its tag on
cells now at packed positions ~16M+, so the donor's `seq_pos_max`
jumps into the packed range), so the splice path is forced to
`seq_rm(donor.id, ...)` before the shift. After that, the donor's
seq has a **hole** at `[head_c, head_c + n)`.

`pos_min` doesn't catch the hole — pos_min is the *minimum*
position with cells, and the donor still has cells at `[0, head_c)`
and `[head_c+n, end)`, so `pos_min = 0`. The standard CP path
doesn't validate cell liveness either, only token equality. So the
donor's next request — typically a continuation of the donor's own
session — would compute `n_past = M > head_c` from token match,
treat all cells `[0, M)` as live, then crash on attention when it
hits the hole.

CP is the most valuable cache primitive for TTFT. Breaking it on
the donor every time a sibling slot does a cross-slot splice is
unshippable.

#### Path forward — `seq_cp_deep`

Add a `seq_cp_deep(src, dst, p0, p1, dst_offset)` primitive to
`llama-kv-cache.cpp` for the unified case. It allocates fresh
cells, copies K and V tensors from source to destination cells,
RoPE-rephases K to the destination positions via the existing
`cell.shift` mechanism, and tags the new cells with the
destination seq id. **The source cells stay untouched**, so the
donor's CP keeps working. This is the same primitive rung 4
(persistent disk-backed pool) will need to hydrate cells back
into GPU KV at boot, so the investment compounds.

The alternative — running with non-unified KV so each slot is its
own stream and cross-stream `seq_cp` (llama-kv-cache.cpp:462+)
deep-copies on its own — was rejected: the splice-temp trampoline
relies on seq id 255 being addressable, which only works in
unified mode. Reworking that plus paying extra memory for
per-stream cell pools is a worse trade.

#### Sub-steps

- **3a — metadata scaffolding** (WIP, uncommitted): adds
  `cell_copy_info { src_idxs, dst_idxs }` to `llama-kv-cache.h`,
  declares `seq_cp_deep` on both `llama_memory_i` (with abort
  default) and `llama_kv_cache`, implements the cell allocation +
  `pos_set` / `ext_set` / `seq_add` / `pos_add(shift)` metadata
  path, and enqueues (src_idx, dst_idx) pairs into `cc_info`.
  Compiles. Not callable for real use yet — data not moved.

- **3b — graph-side data move**: build a one-shot graph using
  `ggml_get_rows` + `ggml_set_rows` per layer for K and for V
  (V layout depends on `v_trans`). Run it in `update()` before
  the K-shift pass so the freshly-copied K cells get rephased
  by the existing shift mechanism. Wire `cc_info` into graph
  build the same way `defrag_info` already is. Add ISWA wrapper
  delegating to the underlying caches. Expose
  `llama_memory_seq_cp_deep` as a public API on
  `llama_memory_i`.

- **3c — cross-slot splice integration**: separate code path
  from today's intra-slot destructive splice. For cross-slot
  donors, use `seq_cp_deep(donor.id, this.id, hc, hc+n,
  base - hc)` directly — no splice-temp trampoline, no
  `seq_rm` on donor. Gate same way as step 2:
  `sim_at_select < CROSS_SLOT_SIM_THRESHOLD`.

**Validation:** rung-2.5 multi-slot replay — confirm cross-slot
splice events fire, donor's subsequent same-session continuation
still hits its own CP at full length (no hole).

**Effort:** 3a done (~150 lines). 3b is the bulk (~300-500 lines
in the memory module). 3c is small (~50 lines) once 3b lands.
Total ~1 week wall, gated behind a flag until validated.

### Step 4 — Rolling-hash index for the slot pool

The symmetric scan today is O(R × C × K) worst case (R = recipient
size, C = cache size, K = `n_cache_reuse` threshold). Acceptable for
single-slot single-turn (~10-50ms per request). Linearly worse with
multi-slot: O(R × C × N × K). At N = 10 and full slots, the inner
scan dominates the cache-reuse path.

Replace the inner `for (hc = head_c_min; hc < tokens.size(); hc++)`
with a hash-index lookup:

- Per-server `unordered_map<uint64_t, vector<{slot_id, position}>>`
  keyed by token-k-gram hash (k = 16 tokens, smaller than
  `n_cache_reuse`).
- Maintained at `prompt.tokens.push_back` (insert hash at new tail
  position) and at `prompt_clear` / slot reset (drop hashes).
- For each `head_p`, lookup `hash(input[head_p .. head_p+k])` →
  candidate list of `(slot_id, hc)`. Match extension stays O(K) per
  candidate.

The offline `categorize_matches.py` tool already uses the same hash
algorithm — sharing a primitive across offline manifest analysis and
online serving keeps the splice-candidate predictions consistent
between the two.

**Validation:** rung-2.5 single-slot replay — timing of the candidate
scan portion drops by ~10x. Then with N = 10 slots and full contexts,
total scan stays sub-100ms per request.

**Effort:** ~250 lines + invalidation tests, 2-3 days.

### Step 5 — Validation harness

Extend the replay harness (`trace_analysis/replay_multiturn.py` in
the consumer repo) to a multi-slot configuration:

- Warm N slots with N distinct captured donor sessions.
- Send a target recipient session, force it to a slot whose own
  prior content has minimal overlap (low sim_best, triggers slow
  path).
- Confirm cross-slot splice events fire, measure cell-transit volume,
  verify recipient continuation correctness.

Demoable result: table comparing "all-donors-in-one-slot" (rung 2)
vs "donors-spread-across-N-slots" (rung 3). Expectation: rung 3
fires cross-session splices that rung 2 cannot, because rung 2's
single-slot eviction at long contexts wipes earlier donor content
before the recipient arrives.

**Effort:** one day.

### Step 6 — Flags + docs

- `--cross-slot-scan` — gate the slow-path scan (default off until
  validated, then default on).
- `--cross-slot-sim-threshold <float>` — `sim_at_select` below which
  the slow path engages. Default 0.9.
- README / docs update: deployment guidance for sizing N slots × ctx
  for a target hit-rate.

**Effort:** half a day.

## Total

| step | scope                              | lines   | wall   | status      |
|------|------------------------------------|---------|--------|-------------|
| 1    | plumb scheduler LCP                | ~30     | 0.5d   | done        |
| 2    | cross-slot CP                      | ~80     | 1.0d   | done        |
| 3a   | seq_cp_deep metadata scaffolding   | ~150    | 0.5d   | done        |
| 3b   | seq_cp_deep K/V graph data move    | ~150    | 1d     | done        |
| 3c   | cross-slot splice integration      | ~150    | 0.5d   | done        |
| 4    | rolling-hash index                 | ~225    | 1d     | done        |
| 5    | validation harness                 | ~420    | 0.5d   | done        |
| 6    | flags + docs                       | small   | 0.5d   | in progress |
| R4.1 | rung 4 phase 1: D2H snapshot       | ~95     | 0.25d  | done        |
| R4.2 | rung 4 phase 2: device eviction    | ~20     | 0.1d   | done        |
| R4.3 | rung 4 phase 3: host kgram index   | ~100    | 0.5d   | done        |
| R4.4 | rung 4 phase 4: H2D hydration      | ~115    | 0.5d   | done        |
| R4.5 | rung 4 phase 5: validation rerun   | small   | 0.25d  | done        |
|      | **total**                          |         |        |             |

## Rung 3 results (on Gemma-4-E4B-it, real captured agentcap traces)

Validated on rung-3 multi-slot harness (`trace_analysis/replay_multislot.py`),
2 captured (donor, recipient) pairs from a transformers-coding session:

| metric                                | cold     | rung-3 (cross-slot on) |
|---------------------------------------|----------|------------------------|
| recipient 0 prefill (chunk 69922 chars) | 20148 ms | **1271 ms (15.9x)**    |
| recipient 1 prefill (chunk 41840 chars) | 36321 ms | **8178 ms (4.4x)**     |
| cross-slot splices fired              | n/a      | 2/2 recipients         |

Hash-index scan-time A/B (live-slot scan only, large donor):

| scan mode | task        | scan time |
|-----------|-------------|-----------|
| linear    | task 26 (donor 1 vs donor 0's 50K toks) | 3321 ms |
| hashed    | same task   | **29 ms (113x)** |

## Rung 3 limitation: unified-KV n_kv tax

A diagnosed structural cost (not specific to cross-slot splice): in
unified KV mode, attention reads up to `n_kv = used_max_p1(pool)` cells
per query token. With multiple slots' content in the pool, `n_kv`
includes foreign-seq cells that get mask-zeroed but still cost compute.
For donor warming on a non-empty pool, this manifests as ~1.8x per-batch
slowdown for the duration of the prefill. Splice recipients pay
negligible tax because they only run a handful of batches before
falling out of prefill via the splice.

Rung 4 addresses this directly: by parking idle slots' K/V in host RAM
and evicting their device cells, the device pool stays small.

## Rung 4 (done)

CPU-tier cold storage for released slots' K/V. Enabled via
``--cross-slot-cpu-tier`` (CLI) or ``LLAMA_CROSS_SLOT_CPU_TIER=1`` (env).

Mechanism:
- On slot release: ``llama_state_seq_get_data_ext`` dumps the slot's
  seq state to a host buffer (~56 KB/token for E4B, all 26 layers ×
  K + V at f16). Then ``prompt_clear`` evicts the device cells.
- On cross-slot scan: ``server_kgram_index`` covers both live slots
  AND host snapshots, distinguished by an ``is_host`` flag on each
  candidate entry.
- On host scan match: ``llama_state_seq_set_data_ext`` hydrates the
  donor's full seq state into ``splice_temp`` at original positions,
  ``seq_rm`` trims to the matched chunk, ``seq_add`` shifts to packed
  positions, and the existing cursor handler lands the cells at
  ``head_p`` exactly like the intra-slot/live-cross-slot paths.

Measured on the same rung-3 harness, with ``--cross-slot-cpu-tier``:

| metric                       | rung-3 only       | rung-4 (cpu tier) |
|------------------------------|-------------------|--------------------|
| donor 1 warm (74K tokens)    | 65294 ms (n_kv tax)| **37223 ms (cold-shape)** |
| recipient 0 splice firing    | live slot 0       | host snapshot 0    |
| recipient 0 prefill          | 1271 ms           | 3378 ms            |
| host hydration time (50K)    | n/a               | 257 ms (2.8 GiB)   |

Tradeoff: rung-4 trades device-resident pool tax (per-batch attention
overhead) for one-shot PCIe hydration (per splice). Wins decisively in
deployments where total session count × session size exceeds the device
pool capacity. Neutral when everything fits on device (rung-3 alone is
faster).

### Rung 4 MVP limitations (future work)

- **One host splice per request.** ``splice_temp`` (seq id 255) is
  cleared by ``state_seq_set_data_ext`` on every hydration. A second
  host splice in the same request would lose the first one. Multi-host
  needs a small pool of temp seq ids, with the cursor handler keyed by
  splice index.
- **First-match selection.** The current scan commits the first host
  candidate that crosses ``n_cache_reuse``; the globally-largest host
  match might be at a later ``head_p``. Refining the selection (rank
  all host candidates, pick the largest) is independent of the
  multi-host fix.
- **Full-snapshot hydration.** We load the donor's entire seq state
  into ``splice_temp`` and then trim. A specialised partial-load API
  on ``llama_state_seq_set_data`` (load only positions [p0, p1)) would
  cut the hydration cost proportionally.

## Out of scope (rung 5+)

- Persistent disk-backed cell store. The native serialisation already
  works (``llama_state_seq_save_file`` / ``_load_file``) — the missing
  pieces are: eviction policy (host RAM → disk under memory pressure),
  index of disk-resident snapshots (extend ``kgram_index`` further),
  and warm-up at boot.
- Content-deduplicated chunk pool. Same hash index, but storing each
  chunk once instead of per-snapshot. Requires reference counting and
  invalidation on session edits.
