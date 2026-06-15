#include "server-prefix-cache.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#ifndef _WIN32
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#endif

namespace {

constexpr uint32_t PREFIX_CACHE_MAGIC   = 0x67677063; // "ggpc"
constexpr uint32_t PREFIX_CACHE_VERSION = 1;

// sanity caps to avoid huge allocations when scanning a corrupt file
constexpr uint32_t PREFIX_CACHE_MAX_TOKENS = 16u*1024*1024;
constexpr uint32_t PREFIX_CACHE_MAX_CKPTS  = 4096;

int common_prefix_len(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return (int) i;
}

} // namespace

uint64_t prefix_cache_compat_id(
        const std::string & model_path,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   swa_full,
                     bool   kv_unified) {
    const uint64_t file_id = fs_file_id(model_path);
    if (file_id == 0) {
        return 0; // no file identity -> no usable compat id
    }

    // fold the layout-affecting cache config into the file identity
    const uint32_t tk = (uint32_t) type_k;
    const uint32_t tv = (uint32_t) type_v;
    const uint8_t  sf = swa_full   ? 1 : 0;
    const uint8_t  ku = kv_unified ? 1 : 0;

    uint64_t h = file_id;
    h = common_fnv1a(&tk, sizeof(tk), h);
    h = common_fnv1a(&tv, sizeof(tv), h);
    h = common_fnv1a(&sf, sizeof(sf), h);
    h = common_fnv1a(&ku, sizeof(ku), h);

    return h;
}

size_t prefix_cache_file_save(const std::string & path, const prefix_cache_entry & entry) {
    GGML_ASSERT(entry.compat_id != 0);

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return 0; // streams do not throw by default - an unwritable path lands here
    }

    // header
    f.write(reinterpret_cast<const char *>(&PREFIX_CACHE_MAGIC),   sizeof(PREFIX_CACHE_MAGIC));
    f.write(reinterpret_cast<const char *>(&PREFIX_CACHE_VERSION), sizeof(PREFIX_CACHE_VERSION));
    f.write(reinterpret_cast<const char *>(&entry.compat_id),      sizeof(entry.compat_id));

    const uint32_t n_tokens = (uint32_t) entry.tokens.size();
    f.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
    f.write(reinterpret_cast<const char *>(entry.tokens.data()), (std::streamsize)(n_tokens * sizeof(llama_token)));

    const uint32_t n_ckpt = (uint32_t) entry.checkpoints.size();
    f.write(reinterpret_cast<const char *>(&n_ckpt), sizeof(n_ckpt));
    for (const auto & c : entry.checkpoints) {
        f.write(reinterpret_cast<const char *>(&c.pos_min),  sizeof(c.pos_min));
        f.write(reinterpret_cast<const char *>(&c.pos_max),  sizeof(c.pos_max));
        f.write(reinterpret_cast<const char *>(&c.n_tokens), sizeof(c.n_tokens));
        const uint64_t n_data = c.data.size();
        f.write(reinterpret_cast<const char *>(&n_data), sizeof(n_data));
    }

    // main blob
    const uint64_t main_size = entry.main.size();
    f.write(reinterpret_cast<const char *>(&main_size), sizeof(main_size));
    f.write(reinterpret_cast<const char *>(entry.main.data()), (std::streamsize) main_size);

    // checkpoint blobs, in table order
    for (const auto & c : entry.checkpoints) {
        f.write(reinterpret_cast<const char *>(c.data.data()), (std::streamsize) c.data.size());
    }

    if (!f) {
        return 0; // a failed write above only sets the stream's failbit - this catches them all
    }
    return (size_t) f.tellp();
}

bool prefix_cache_file_peek(const std::string & path, prefix_cache_header & out) {
    out = {};

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    f.seekg(0, std::ios::end);
    const uint64_t file_size = (uint64_t) f.tellg();
    f.seekg(0, std::ios::beg);

    uint32_t magic = 0, version = 0;
    if (!f.read(reinterpret_cast<char *>(&magic),   sizeof(magic)) ||
        !f.read(reinterpret_cast<char *>(&version), sizeof(version))) {
        return false;
    }
    if (magic != PREFIX_CACHE_MAGIC || version != PREFIX_CACHE_VERSION) {
        return false;
    }

    if (!f.read(reinterpret_cast<char *>(&out.compat_id), sizeof(out.compat_id)) || out.compat_id == 0) {
        return false;
    }

    uint32_t n_tokens = 0;
    if (!f.read(reinterpret_cast<char *>(&n_tokens), sizeof(n_tokens)) || n_tokens > PREFIX_CACHE_MAX_TOKENS) {
        return false;
    }
    out.tokens.resize(n_tokens);
    if (n_tokens && !f.read(reinterpret_cast<char *>(out.tokens.data()), (std::streamsize)(n_tokens * sizeof(llama_token)))) {
        return false;
    }

    uint32_t n_ckpt = 0;
    if (!f.read(reinterpret_cast<char *>(&n_ckpt), sizeof(n_ckpt)) || n_ckpt > PREFIX_CACHE_MAX_CKPTS) {
        return false;
    }
    out.checkpoints.resize(n_ckpt);
    for (auto & c : out.checkpoints) {
        if (!f.read(reinterpret_cast<char *>(&c.pos_min),  sizeof(c.pos_min))  ||
            !f.read(reinterpret_cast<char *>(&c.pos_max),  sizeof(c.pos_max))  ||
            !f.read(reinterpret_cast<char *>(&c.n_tokens), sizeof(c.n_tokens)) ||
            !f.read(reinterpret_cast<char *>(&c.size),     sizeof(c.size))) {
            return false;
        }
    }

    if (!f.read(reinterpret_cast<char *>(&out.main_size), sizeof(out.main_size))) {
        return false;
    }
    out.main_offset = (uint64_t) f.tellg();

    // compute blob offsets and validate the declared layout against the file size
    uint64_t off = out.main_offset + out.main_size;
    for (auto & c : out.checkpoints) {
        c.offset = off;
        off += c.size;
    }

    if (off > file_size) {
        out = {};
        return false; // truncated
    }

    return true;
}

bool prefix_cache_file_read_main(
            const std::string & path,
    const prefix_cache_header & header,
    const std::function<bool(const uint8_t * data, size_t size)> & fn) {
    if (header.main_size == 0) {
        return fn(nullptr, 0);
    }

#ifndef _WIN32
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }

    // mmap offsets must be page-aligned
    const uint64_t page    = (uint64_t) sysconf(_SC_PAGESIZE);
    const uint64_t aligned = header.main_offset & ~(page - 1);
    const uint64_t delta   = header.main_offset - aligned;
    const size_t   len     = (size_t)(header.main_size + delta);

    void * base = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, (off_t) aligned);
    close(fd); // the mapping holds its own reference

    if (base == MAP_FAILED) {
        return false;
    }
    madvise(base, len, MADV_SEQUENTIAL);

    const bool res = fn(static_cast<const uint8_t *>(base) + delta, (size_t) header.main_size);

    munmap(base, len);

    return res;
#else
    // fallback: read the main blob into a buffer, materializing it in host memory
    // TODO: use MapViewOfFile to get the same lazy/reclaimable behavior as the mmap
    //       path (! view offsets align to dwAllocationGranularity, not page size)
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    std::vector<uint8_t> buf(header.main_size);
    f.seekg((std::streamoff) header.main_offset);
    if (!f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) header.main_size)) {
        return false;
    }
    return fn(buf.data(), buf.size());
#endif
}

bool prefix_cache_file_read_ckpt(
            const std::string & path,
    const prefix_cache_header & header,
                       size_t   i,
                      uint8_t * dst) {
    if (i >= header.checkpoints.size()) {
        return false;
    }
    const auto & c = header.checkpoints[i];

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    f.seekg((std::streamoff) c.offset);
    if (c.size && !f.read(reinterpret_cast<char *>(dst), (std::streamsize) c.size)) {
        return false;
    }

    return true;
}

// size_limit caps the directory in bytes (0 = no limit)
server_prefix_cache::server_prefix_cache(std::string dir, uint64_t compat_id, size_t size_limit) :
        dir_(std::move(dir)), compat_id_(compat_id), size_limit_(size_limit) {
    GGML_ASSERT(compat_id_ != 0);
    enforce_limits(); // prune stale / over-cap / torn files before indexing
    build_index();
    worker_ = std::thread(&server_prefix_cache::writer_loop, this);
}

server_prefix_cache::~server_prefix_cache() {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void server_prefix_cache::record_hit(const std::string & path) {
    ++n_hit_;
    // bump mtime so LRU eviction sees this entry as recently used (reads don't)
    std::error_code ec;
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec);
}

void server_prefix_cache::async_save(prefix_cache_entry && entry) {
    entry.compat_id = compat_id_;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        queue_.push(std::move(entry));
    }
    cv_.notify_one();
}

std::string server_prefix_cache::path_for(const prefix_cache_entry & entry) const {
    // content-addressed on (compat_id, tokens): identical prefixes map to the same file
    const uint64_t h = common_fnv1a(entry.tokens.data(), entry.tokens.size() * sizeof(llama_token), compat_id_);
    char name[32];
    snprintf(name, sizeof(name), "%016" PRIx64 ".ggpc", h);
    return dir_ + name;
}

void server_prefix_cache::build_index() {
    std::error_code ec;
    for (const auto & de : std::filesystem::directory_iterator(dir_, ec)) {
        if (!de.is_regular_file() || de.path().extension() != ".ggpc") {
            continue;
        }

        prefix_cache_header header;
        if (!prefix_cache_file_peek(de.path().string(), header)) {
            continue; // foreign / corrupt / truncated - skip, don't fail the scan
        }
        if (header.compat_id != compat_id_) {
            continue; // built for a different model / KV config
        }

        index_.push_back({ de.path().string(), std::move(header) });
    }

    LOG_INF("%s: indexed %zu prefix cache entr%s\n", __func__, index_.size(), index_.size() == 1 ? "y" : "ies");
}

void server_prefix_cache::enforce_limits() {
    namespace fs = std::filesystem;
    std::error_code ec;

    struct file_info {
        std::string        path;
        uintmax_t          size;
        fs::file_time_type mtime;
    };

    std::vector<file_info> entries;
    uintmax_t total = 0;

    // span the whole dir by mtime (global last-hit), so a superseded model's cold
    // entries are reclaimed for the current one, not just our own compat_id
    for (const auto & de : fs::directory_iterator(dir_, ec)) {
        const auto & path = de.path();
        const auto   ext  = path.extension();

        if (ext == ".tmp") {
            fs::remove(path, ec); // leftover from an interrupted write
            continue;
        }
        if (ext != ".ggpc" || !de.is_regular_file()) {
            continue;
        }

        const uintmax_t size = fs::file_size(path, ec);
        if (ec) {
            continue;
        }
        const fs::file_time_type mtime = fs::last_write_time(path, ec);
        if (ec) {
            continue;
        }

        entries.push_back({ path.string(), size, mtime });
        total += size;
    }

    // size cap: evict the least recently hit (oldest mtime) until under the limit
    if (size_limit_ > 0 && total > size_limit_) {
        std::sort(entries.begin(), entries.end(),
            [](const file_info & a, const file_info & b) { return a.mtime < b.mtime; });
        for (const auto & f : entries) {
            if (total <= size_limit_) {
                break;
            }
            fs::remove(f.path, ec);
            total -= f.size;
            ++n_evict_;
        }
    }
}

const prefix_cache_index_entry * server_prefix_cache::lookup(
        const std::vector<llama_token> & prompt, int32_t max_tokens) const {
    const prefix_cache_index_entry * best = nullptr;
    int best_lcp = 0;

    for (const auto & e : index_) {
        if ((int32_t) e.header.tokens.size() > max_tokens) {
            continue; // would not fit the slot's context
        }
        const int lcp = common_prefix_len(e.header.tokens, prompt);
        if (lcp > best_lcp) {
            best_lcp = lcp;
            best     = &e;
        }
    }

    return best;
}

void server_prefix_cache::writer_loop() {
    while (true) {
        prefix_cache_entry entry;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                return; // reached only once stop_ is set and the queue has drained
            }
            entry = std::move(queue_.front());
            queue_.pop();
        }

        const std::string path = path_for(entry);
        const std::string tmp  = path + ".tmp";

        // .tmp + rename so a startup scan never sees a torn file
        if (prefix_cache_file_save(tmp, entry) == 0) {
            LOG_WRN("%s: failed to write prefix cache entry %s\n", __func__, tmp.c_str());
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            continue;
        }

        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            LOG_WRN("%s: failed to rename %s -> %s (%s)\n", __func__, tmp.c_str(), path.c_str(), ec.message().c_str());
            std::filesystem::remove(tmp, ec);
            continue;
        }

        ++n_capture_;
        enforce_limits(); // keep the dir within the cap after adding an entry
    }
}
