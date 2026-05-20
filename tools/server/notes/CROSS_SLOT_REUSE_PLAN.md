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

### Step 3 — Cross-slot symmetric scan (BLOCKED on deep-copy primitive)

**Status:** prototyped, reverted. Blocked on a missing memory-module
primitive.

The intent was: after own-slot symmetric scan, also scan **other
slots'** `prompt.tokens` for chunks matching the input past CP, and
for each match `seq_cp` the donor's cells into the splice temp
trampoline as today's intra-slot path does.

What stops it from being safely shippable:

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

**Unblockers:**

1. Add a `seq_cp_deep(src, dst, p0, p1, dst_offset)` primitive to
   `llama-kv-cache.cpp` for the unified case. This would allocate
   fresh cells, copy K and V tensors from the source, RoPE-rephase
   K to the destination positions, and tag the new cells with the
   destination seq id. The source cells stay untouched and the
   donor's CP keeps working. ~300-500 lines in the memory module
   plus the splice-path integration.

2. Or run with non-unified KV so each slot is its own stream and
   the cross-stream `seq_cp` (llama-kv-cache.cpp:462+) deep-copies
   on its own. But the splice-temp trampoline relies on seq id 255
   being addressable, which only works in unified mode — we'd have
   to rework that, plus pay extra memory for per-stream cell pools.

Option 1 is the right foundation: it's the same primitive rung 4
(persistent disk-backed pool) will need to hydrate cells back into
GPU KV at boot. Step 3 reopens once that lands.

**What ships from this step in the meantime:** nothing. The
revert keeps cross-slot CP (step 2, non-destructive) as the only
cross-slot mechanism. Step 4 (hash index) can still proceed — its
benefit applies to the own-slot symmetric scan that already
exists, plus to step-2-style cross-slot CP, plus to whatever
unblocked step 3 looks like.

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

| step | scope                              | lines | wall  |
|------|------------------------------------|-------|-------|
| 1    | plumb scheduler LCP                | ~30   | 0.5d  |
| 2    | cross-slot CP                      | ~80   | 1.0d  |
| 3    | cross-slot symmetric scan          | ~150  | 2.0d  |
| 4    | rolling-hash index                 | ~250  | 2-3d  |
| 5    | validation harness                 | small | 1.0d  |
| 6    | flags + docs                       | small | 0.5d  |
|      | **total**                          |       | ~1.5w |

## Out of scope (rung 4)

- Persistent disk-backed cell store + hydration at boot.
- Content-deduplicated chunk pool (instead of per-slot whole-context
  buffers) — same hash index, but with cells stored once and
  referenced by multiple "virtual slots."

These build on rung 3's pool primitives, but require separate design
work: serialization format for K/V tensors, model-version compat,
disk → GPU hydration pipeline.
