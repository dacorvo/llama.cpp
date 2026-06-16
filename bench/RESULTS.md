# Disk prefix-cache benchmark — results

Hardware: 1× A10G (24 GB), local NVMe ext4. llama-server on branch `feat/kv-disk-cache`.
Metric: client-side TTFT, first request after restart (RAM empty → disk serves).
Workload: canonical pi/transformers captures (`dacorvo/transformers-coding-session-captures`).

## Recurring first-turn prefix per agent (gemma-4-26B captures)

| agent    | shared prefix | avg prompt | above checkpoint floor (256)? |
|----------|--------------:|-----------:|-------------------------------|
| goose    | ~122 tok      | ~462 tok   | **no** — sub-floor            |
| pi       | ~1,723 tok    | ~1,753 tok | yes                           |
| opencode | ~3,545 tok    | ~3,578 tok | yes                           |
| hermes   | ~5,220 tok    | ~5,264 tok | yes                           |

The cache reuses the recurring prefix; benefit scales with its length.

## TTFT: cold (full prefill) vs warm (disk restore), 10 distinct tasks, 10/10 hits

| model / agent              | cold (med) | warm (med, page-cache) | warm (cold-disk) | speedup        | entry size |
|----------------------------|-----------:|-----------------------:|-----------------:|----------------|-----------:|
| gemma-4-26B (SWA) · pi      | 908 ms     | 426 ms (2.1×)          | 522 ms (1.7×)    | 1.7–2.1×       | 433 MiB    |
| qwen3.6-35B (recurrent) · pi| 845 ms     | 268 ms (3.2×)          | 383 ms (2.2×)    | 2.2–3.2×       | 157 MiB    |
| gemma-4-26B (SWA) · hermes  | 1,960 ms   | 465 ms (4.2×)          | —                | 4.2× (≤9.5×)   | ~1 GiB     |

## Costs (measured)

- **Capture penalty: ~0 ms** on TTFT (device→host snapshot is fast; the disk write is async).
- **Cold-disk read** (OS cache dropped): +96 ms (gemma 433 MiB) / +115 ms (qwen 157 MiB)
  on local NVMe. A network-mounted cache dir would add more — untested, and the main
  risk for the large SWA entries.
- **Correctness**: restored output byte-identical to cold (greedy); `compat_id`
  segregates models in a shared dir (qwen never restores gemma state).

## Verdict

Worth shipping for agents with a substantial recurring prefix — which is the popular
ones (pi/opencode/hermes). The win is real and **grows with prefix length** (prefill
cost grows ~linearly with tokens; restore I/O grows far slower): pi ~2–3×, hermes ~4×
(up to 9.5×).

### Small-prompt regression (goose sub-floor) — measured

Small goose first-turn prompts (~200–650 tok), gemma-26B, exact recurrence (each
restores its own entry, 10/10 hits):

| | cold | warm | speedup |
|---|------|------|---------|
| median | 271 ms | 304 ms | **0.89× (regression)** |

The warm path has a **~300 ms fixed floor** (lookup + ~100–200 MiB `set_data` upload +
first decode); cold prefill scales with prompt size. **Break-even ≈ 500 tok**: below it
restore is slower than prefill, so the cache makes TTFT **10–30% worse**. The
negative-value-restore guard (planned Step-4 cost model: skip restore when est. cost >
prefill saved) is **not implemented**, so short prompts regress rather than fail safe.
This is a second floor, distinct from the 256-tok checkpoint floor.

Honest bounds:
- **Add a min-reuse guard before shipping** — without it, short prompts regress (above).
- Value materializes **only when RAM is cold** — post-restart / cold-start / autoscale /
  model-swap. A warm long-lived server already gets this from the RAM cache (~106 ms
  slot reuse); the disk cache adds nothing in steady state. Payoff ∝ restart frequency.
- **Hard floor**: a shared prefix shorter than the checkpoint step (256 tok) yields
  **zero** reuse on SWA/recurrent (`do_reset` → full recompute). goose (~122 tok) is
  below it — inert there. (goose is low-usage / being removed, so this is a corner.)
- Tested on **local NVMe**; confirm on the real (possibly network) cache mount before
  shipping, since restore I/O is the variable that can shrink the win for big entries.
