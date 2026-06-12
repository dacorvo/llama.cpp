#include "server-prefix-cache.h"

#include "common.h"

#include <cstring>
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
