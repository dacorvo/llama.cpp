# Disk prefix-cache benchmark

TTFT benchmark for the `--prefix-cache-path` feature (branch `feat/kv-disk-cache`).
Measures the only thing the disk cache is for: **time-to-first-token on the first
request after a server restart**, when the in-RAM prompt cache is empty and only the
disk cache can serve a recurring prefix.

## Method (honest by construction)

- **Headline metric = client-side TTFT** (request sent → first streamed token),
  because it includes the whole cost chain: index lookup + disk read + device upload
  + reprefill of the novel tail. Server `prompt_ms` is prefill-only and hides the
  restore I/O, so it is not used as the verdict.
- **Realistic config**: RAM cache on (default) + disk cache on. The disk lookup is
  wired through the RAM-cache path, so `--cache-ram 0` disables it — do NOT use it to
  "isolate" the disk cache (it silently turns the feature off).
- **Cold baseline** = `cache_prompt:false` → true full prefill, no reuse anywhere.
- **Warm** = restart the server before each measured request so RAM+slot are empty
  and the request must fall back to the disk cache (confirmed via `/metrics`
  `prefix_cache_hit_total`).
- **Workload** = real agent first-turn requests pulled from the canonical captures
  `dacorvo/transformers-coding-session-captures` (one parquet per agent×model);
  first-turn = messages with no `assistant`/`tool` role. These share the recurring
  agent system/instruction prefix, which is what the cache reuses.

## Files

- `bench.py MODEL ALIAS [reqs_dir]` — populate → cold baseline → warm (restart per
  request); prints per-request cold/warm TTFT + hit + speedup.
- `colddisk.py MODEL ALIAS` — warm restore from page-cache vs a truly cold disk
  (drops OS page cache via `sudo drop_caches`); isolates restore I/O cost.
- `cost.py` — capture penalty (first_turn true vs false, same cold prefill).
- `ttft.py` / `warm.py` — earlier single-server probes (kept for reference).
- `reqs2/` (pi), `reqs_hermes/`, `reqs_goose/` — extracted first-turn requests,
  10 distinct tasks each, sharing the dominant system prefix.

## Reproduce

Paths are derived from the script location (run from anywhere); the first arg is the
model `.gguf`. Works on CUDA (`CUDA_VISIBLE_DEVICES=0`) or Apple Metal (`-ngl 99`);
`colddisk.py` drops the page cache via `sudo purge` on macOS, `drop_caches` on Linux.

```bash
# gemma-4 E4B (Apple Metal) — point -m at your cached gguf
python3 bench/bench.py "$(echo ~/.cache/huggingface/hub/models--ggml-org--gemma-4-E4B-it-GGUF/snapshots/*/gemma-4-E4B-it-Q4_K_M.gguf)" gemma bench/reqs2

# original A10G (CUDA) runs
python3 bench/bench.py models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf gemma bench/reqs_hermes
python3 bench/bench.py models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf  qwen  bench/reqs2
```
