// tests for tools/server/server-prefix-cache.{h,cpp} — the server prefix-cache
// on-disk framing ("ggpc"): save / header-only peek / lazy main + checkpoint reads

#include "server-prefix-cache.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, msg)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            printf("FAIL(%d): %s\n", __LINE__, msg);      \
            n_fail++;                                     \
        }                                                 \
    } while (0)

int main() {
    const auto dir = std::filesystem::temp_directory_path();

    const std::string path_model = (dir / "test-pcf-model.bin").string();
    const std::string path_entry = (dir / "test-pcf-entry.ggpc").string();

    // a fake model file to anchor the compat id
    {
        std::ofstream f(path_model, std::ios::binary);
        f << "fake-model-weights";
    }

    // compat id: deterministic, sensitive to file and to each config field
    const uint64_t cid = prefix_cache_compat_id(path_model, GGML_TYPE_F16, GGML_TYPE_F16, false, true);
    CHECK(cid != 0, "compat_id is 0 for a valid file");
    CHECK(cid == prefix_cache_compat_id(path_model, GGML_TYPE_F16, GGML_TYPE_F16, false, true),
          "compat_id is not deterministic");
    CHECK(cid != prefix_cache_compat_id(path_model, GGML_TYPE_Q8_0, GGML_TYPE_F16, false, true), "type_k not folded in");
    CHECK(cid != prefix_cache_compat_id(path_model, GGML_TYPE_F16, GGML_TYPE_Q8_0, false, true), "type_v not folded in");
    CHECK(cid != prefix_cache_compat_id(path_model, GGML_TYPE_F16, GGML_TYPE_F16, true,  true), "swa_full not folded in");
    CHECK(cid != prefix_cache_compat_id(path_model, GGML_TYPE_F16, GGML_TYPE_F16, false, false), "kv_unified not folded in");
    CHECK(prefix_cache_compat_id((dir / "test-pcf-missing.bin").string(), GGML_TYPE_F16, GGML_TYPE_F16, false, true) == 0,
          "compat_id of a missing file should be 0");

    // an entry with two checkpoints of distinct sizes/patterns
    prefix_cache_entry entry;
    entry.compat_id = cid;
    entry.tokens = {1, 22, 333, 4444, 55555, 6};
    entry.main.resize(4096 + 13); // odd size: exercises non-page-aligned blob offsets
    for (size_t i = 0; i < entry.main.size(); ++i) {
        entry.main[i] = (uint8_t)(i*7 + 3);
    }

    prefix_cache_checkpoint c0;
    c0.pos_min = 0; c0.pos_max = 2; c0.n_tokens = 3;
    c0.data = {9, 8, 7, 6, 5};

    prefix_cache_checkpoint c1;
    c1.pos_min = 0; c1.pos_max = 5; c1.n_tokens = 6;
    c1.data.resize(1000);
    for (size_t i = 0; i < c1.data.size(); ++i) {
        c1.data[i] = (uint8_t)(i ^ 0x5A);
    }

    entry.checkpoints = {c0, c1};

    // save
    const size_t n_written = prefix_cache_file_save(path_entry, entry);
    CHECK(n_written > 0, "save failed");

    // peek: header only, table identical, offsets consistent
    prefix_cache_header hdr;
    CHECK(prefix_cache_file_peek(path_entry, hdr), "peek failed");
    CHECK(hdr.compat_id == cid,        "peek compat_id mismatch");
    CHECK(hdr.tokens == entry.tokens,  "peek tokens mismatch");
    CHECK(hdr.main_size == entry.main.size(), "peek main_size mismatch");
    CHECK(hdr.checkpoints.size() == 2, "peek checkpoint count mismatch");
    if (hdr.checkpoints.size() == 2) {
        for (int i = 0; i < 2; ++i) {
            const auto & m = hdr.checkpoints[i];
            const auto & e = entry.checkpoints[i];
            CHECK(m.pos_min  == e.pos_min,     "ckpt meta pos_min mismatch");
            CHECK(m.pos_max  == e.pos_max,     "ckpt meta pos_max mismatch");
            CHECK(m.n_tokens == e.n_tokens,    "ckpt meta n_tokens mismatch");
            CHECK(m.size     == e.data.size(), "ckpt meta size mismatch");
        }
        CHECK(hdr.checkpoints[0].offset == hdr.main_offset + hdr.main_size, "ckpt 0 offset");
        CHECK(hdr.checkpoints[1].offset == hdr.checkpoints[0].offset + hdr.checkpoints[0].size, "ckpt 1 offset");
    }

    // a caller filters candidates by comparing compat ids
    const uint64_t cid_other = prefix_cache_compat_id(path_model, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, true);
    CHECK(hdr.compat_id != cid_other, "different config must not match the stored compat_id");

    // lazy main read: span identical to what was written
    {
        bool same = false;
        const bool ok = prefix_cache_file_read_main(path_entry, hdr,
            [&](const uint8_t * data, size_t size) {
                same = size == entry.main.size() && memcmp(data, entry.main.data(), size) == 0;
                return true;
            });
        CHECK(ok,   "read_main failed");
        CHECK(same, "main blob mismatch");
    }

    // lazy checkpoint reads - the caller supplies any buffer sized from the header
    {
        std::vector<uint8_t> blob;
        blob.resize(hdr.checkpoints[0].size);
        CHECK(prefix_cache_file_read_ckpt(path_entry, hdr, 0, blob.data()) && blob == c0.data, "ckpt 0 blob mismatch");
        blob.resize(hdr.checkpoints[1].size);
        CHECK(prefix_cache_file_read_ckpt(path_entry, hdr, 1, blob.data()) && blob == c1.data, "ckpt 1 blob mismatch");
        CHECK(!prefix_cache_file_read_ckpt(path_entry, hdr, 2, blob.data()), "out-of-range ckpt index should fail");
    }

    // entry without checkpoints round-trips
    {
        prefix_cache_entry e0;
        e0.compat_id = cid;
        e0.tokens = {7, 8};
        e0.main = {1, 2, 3};
        const std::string p0 = (dir / "test-pcf-zero.ggpc").string();
        CHECK(prefix_cache_file_save(p0, e0) > 0, "save (0 ckpt) failed");
        prefix_cache_header h0;
        CHECK(prefix_cache_file_peek(p0, h0), "peek (0 ckpt) failed");
        CHECK(h0.checkpoints.empty() && h0.tokens == e0.tokens && h0.main_size == 3, "0-ckpt header mismatch");
        bool same = false;
        prefix_cache_file_read_main(p0, h0, [&](const uint8_t * d, size_t s) {
            same = s == 3 && memcmp(d, e0.main.data(), 3) == 0;
            return true;
        });
        CHECK(same, "0-ckpt main mismatch");
        std::filesystem::remove(p0);
    }

    // truncated file is rejected at scan time
    {
        std::filesystem::resize_file(path_entry, n_written - 5);
        prefix_cache_header ht;
        CHECK(!prefix_cache_file_peek(path_entry, ht), "peek should reject a truncated file");
    }

    // garbage file is rejected
    {
        const std::string pg = (dir / "test-pcf-garbage.ggpc").string();
        {
            std::ofstream f(pg, std::ios::binary);
            f << "this is not a ggpc file";
        }
        prefix_cache_header hg;
        CHECK(!prefix_cache_file_peek(pg, hg), "peek should reject a non-ggpc file");
        std::filesystem::remove(pg);
    }

    // runtime cache: async_save drains on shutdown, then build_index + lookup find the entries
    {
        const std::filesystem::path cdir = dir / "test-pc-runtime";
        std::filesystem::remove_all(cdir);
        std::filesystem::create_directory(cdir);
        const std::string cpath = cdir.string() + "/";

        auto count_ggpc = [&]() {
            int n = 0;
            for (const auto & de : std::filesystem::directory_iterator(cdir)) {
                if (de.path().extension() == ".ggpc") { n++; }
            }
            return n;
        };

        {
            server_prefix_cache cache(cpath, cid, 0);
            for (int i = 0; i < 3; ++i) {
                prefix_cache_entry e;
                e.tokens = { 1, 2, 3, 100 + i }; // shared {1,2,3} prefix, distinct tail
                e.main   = { (uint8_t) i };
                cache.async_save(std::move(e));
            }
        } // destructor drains the writer queue and joins

        CHECK(count_ggpc() == 3, "async_save drained 3 entries on shutdown");

        {
            server_prefix_cache cache(cpath, cid, 0); // reopen: index rebuilt from disk

            const auto * e_exact = cache.lookup({ 1, 2, 3, 101 }, 1 << 20);
            CHECK(e_exact && e_exact->header.tokens == std::vector<llama_token>({ 1, 2, 3, 101 }),
                  "lookup returns the longest-common-prefix match");

            const auto * e_part = cache.lookup({ 1, 2, 3, 999 }, 1 << 20);
            CHECK(e_part && e_part->header.tokens.size() >= 3 &&
                  e_part->header.tokens[0] == 1 && e_part->header.tokens[1] == 2 && e_part->header.tokens[2] == 3,
                  "lookup falls back to a shorter shared prefix");

            CHECK(cache.lookup({ 7, 8, 9 },      1 << 20) == nullptr, "lookup: no shared prefix returns null");
            CHECK(cache.lookup({ 1, 2, 3, 101 }, 3)       == nullptr, "lookup: entries over the token cap are skipped");
        }

        std::filesystem::remove_all(cdir);
    }

    // enforce_limits: the size cap evicts oldest, .tmp leftovers are cleaned
    {
        const std::filesystem::path cdir = dir / "test-pc-evict";
        std::filesystem::remove_all(cdir);
        std::filesystem::create_directory(cdir);
        const std::string cpath = cdir.string() + "/";

        auto count_ggpc = [&]() {
            int n = 0;
            for (const auto & de : std::filesystem::directory_iterator(cdir)) {
                if (de.path().extension() == ".ggpc") { n++; }
            }
            return n;
        };

        {
            server_prefix_cache cache(cpath, cid, 0); // no limit
            for (int i = 0; i < 3; ++i) {
                prefix_cache_entry e;
                e.tokens = { i };
                e.main.assign(2000, (uint8_t) i); // ~2 KiB each
                cache.async_save(std::move(e));
            }
        }
        CHECK(count_ggpc() == 3, "3 entries written under no limit");

        { std::ofstream f(cpath + "stray.ggpc.tmp", std::ios::binary); f << "torn"; }

        { server_prefix_cache cache(cpath, cid, 3000); } // cap fits ~1 entry; ctor prunes

        CHECK(!std::filesystem::exists(cpath + "stray.ggpc.tmp"), "enforce_limits cleans .tmp leftovers");
        CHECK(count_ggpc() == 1, "size cap evicts down to the limit");

        std::filesystem::remove_all(cdir);
    }

    std::filesystem::remove(path_entry);
    std::filesystem::remove(path_model);

    if (n_fail == 0) {
        printf("all server-prefix-cache tests passed\n");
        return 0;
    }
    printf("%d server-prefix-cache test(s) FAILED\n", n_fail);
    return 1;
}
