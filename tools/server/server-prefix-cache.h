#pragma once

#include "llama.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// On-disk format for a server prefix-cache entry ("ggpc"). Server-owned framing of
// already-serialized state buffers (main state via state_seq_get_data_ext(flags=0),
// each checkpoint via PARTIAL_ONLY). The layout is built for lazy reads:
//
//   [ header: magic | version | compat_id | n_tokens | tokens
//             n_ckpt | per ckpt: pos_min, pos_max, n_tokens, n_data ]
//   [ main_size | main state blob ]
//   [ checkpoint blobs, back to back, in table order ]
//
// `peek` reads only the header (cheap index scan, no blobs); the main blob is
// handed to the caller as a mapped span without materializing it in anonymous
// memory; checkpoint blobs are read individually on demand.

// Compatibility key written into / checked on every entry: the model file identity
// (common fs_file_id) folded with the cache config that affects KV state layout.
// Two contexts with the same id can exchange a persisted entry; a different id means
// it's incompatible. Built on fs_file_id so different model files never collide,
// which makes it safe as the index's candidate selector.
uint64_t prefix_cache_compat_id(
    const std::string & model_path,
            ggml_type   type_k,
            ggml_type   type_v,
                 bool   swa_full,
                 bool   kv_unified);

// One checkpoint to persist: position metadata + PARTIAL_ONLY blob (write side).
struct prefix_cache_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
    int64_t   n_tokens = 0;
    std::vector<uint8_t> data;
};

// A complete entry as captured at end of prefill (write side).
struct prefix_cache_entry {
    uint64_t compat_id = 0;
    std::vector<llama_token> tokens;
    std::vector<uint8_t> main;    // main state blob (flags = 0)
    std::vector<prefix_cache_checkpoint> checkpoints;
};

// Checkpoint metadata as read back from a file header (read side).
struct prefix_cache_ckpt_meta {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;
    int64_t   n_tokens = 0;
    uint64_t  size     = 0;   // blob size in bytes
    uint64_t  offset   = 0;   // absolute file offset of the blob (computed on peek)
};

// Everything the index and the lazy restore need, read without touching any blob.
struct prefix_cache_header {
    uint64_t compat_id = 0;
    std::vector<llama_token> tokens;
    std::vector<prefix_cache_ckpt_meta> checkpoints;
    uint64_t main_offset = 0;
    uint64_t main_size   = 0;
};

// An indexed entry: a file path plus its header, kept resident for lookup.
struct prefix_cache_index_entry {
    std::string         path;
    prefix_cache_header header;
};

// Write `entry` to `path` (truncates). Returns bytes written, 0 on I/O failure.
// entry.compat_id must be non-zero (asserted) - a caller without a usable compat
// id should not be capturing at all.
size_t prefix_cache_file_save(const std::string & path, const prefix_cache_entry & entry);

// Header-only read. Validates magic/version and that the file is at least as large
// as the layout it declares (rejects truncated files at scan time). Returns false
// on any mismatch (out is reset).
bool prefix_cache_file_peek(const std::string & path, prefix_cache_header & out);

// Hand the main blob to `fn` as one contiguous span (mmap-backed where available;
// the span is only valid for the duration of the call). Returns false on I/O
// failure, otherwise fn's result.
bool prefix_cache_file_read_main(
            const std::string & path,
    const prefix_cache_header & header,
    const std::function<bool(const uint8_t * data, size_t size)> & fn);

// Read checkpoint blob `i` into `dst`, which must hold at least
// header.checkpoints[i].size bytes (the caller knows the size from `peek`).
bool prefix_cache_file_read_ckpt(
            const std::string & path,
    const prefix_cache_header & header,
                       size_t   i,
                      uint8_t * dst);

// Runtime owner of the on-disk prefix cache: writes captured snapshots to `dir`
// from a background thread, so decoding never blocks on disk. Entries are
// content-addressed, so a recurring prefix overwrites its own file rather than
// duplicating. compat_id must be non-zero.
class server_prefix_cache {
public:
    server_prefix_cache(std::string dir, uint64_t compat_id, size_t size_limit);
    ~server_prefix_cache();

    server_prefix_cache(const server_prefix_cache &)             = delete;
    server_prefix_cache & operator=(const server_prefix_cache &) = delete;

    // Queue an entry for an asynchronous write; compat_id is stamped here.
    void async_save(prefix_cache_entry && entry);

    // Longest-common-prefix match for `prompt`, or nullptr. Skips entries over
    // `max_tokens` (won't fit) and any with no shared prefix.
    const prefix_cache_index_entry * lookup(const std::vector<llama_token> & prompt,
                                             int32_t max_tokens) const;

    void record_hit(const std::string & path); // counts a hit and touches the file's mtime
    void record_miss() { ++n_miss_; }

    uint64_t n_hit()     const { return n_hit_; }
    uint64_t n_miss()    const { return n_miss_; }
    uint64_t n_capture() const { return n_capture_; }
    uint64_t n_evict()   const { return n_evict_; }

private:
    void writer_loop();
    void build_index();
    void enforce_limits(); // filesystem-only: prune .tmp, stale, and over-cap entries
    std::string path_for(const prefix_cache_entry & entry) const;

    const std::string dir_;
    const uint64_t    compat_id_;
    const size_t      size_limit_; // 0 = no limit

    // built once at construction; read-only afterwards (no live captures added)
    std::vector<prefix_cache_index_entry> index_;

    std::mutex                     mtx_;
    std::condition_variable        cv_;
    std::queue<prefix_cache_entry> queue_;
    bool                           stop_ = false;
    std::thread                    worker_;

    std::atomic<uint64_t> n_hit_{0};
    std::atomic<uint64_t> n_miss_{0};
    std::atomic<uint64_t> n_capture_{0};
    std::atomic<uint64_t> n_evict_{0};
};
