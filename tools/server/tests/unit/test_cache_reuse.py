import pytest
from utils import *

server = ServerPreset.tinyllama2()


# Chunk content used for the splice. Tokenization of the actual prompts
# is done at runtime via /tokenize so the donor and recipient share the
# *same token id sequence* for the chunk regardless of the surrounding
# text — without that, BPE left-context can split the boundary and break
# the byte-exact match the cache-reuse loop looks for.
SHARED_TEXT = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
)
DONOR_PREFIX_TEXT     = "Donor side preamble that is unique to the donor"
RECIPIENT_PREFIX_TEXT = "Different recipient prefix words unique to recipient"
DONOR_SUFFIX_TEXT     = "Donor trailing content not seen by recipient"
RECIPIENT_SUFFIX_TEXT = "Recipient trailing content"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 1024
    server.n_slots = 1     # one slot — donor and recipient land in the same KV
    server.n_predict = 4
    server.n_cache_reuse = 8
    server.temperature = 0.0
    # Don't auto-save+clear idle slots between requests, otherwise the
    # donor's prompt gets evicted before the recipient can reuse it.
    server.no_cache_idle_slots = True
    server.kv_unified = True


def _tokenize(text: str) -> list[int]:
    res = server.make_request(
        "POST", "/tokenize",
        data={"content": text, "add_special": False},
    )
    assert res.status_code == 200
    return res.body["tokens"]


def _post_tokens(tokens: list[int], n_predict: int) -> dict:
    return server.make_request(
        "POST",
        "/completion",
        data={
            "prompt": tokens,
            "n_predict": n_predict,
            "temperature": 0.0,
            "cache_prompt": True,
        },
    )


def _build_donor_recipient_token_lists():
    """Tokenize prefix / shared / suffix pieces independently and
    concatenate. This guarantees the SHARED token id sequence is
    identical in donor and recipient regardless of how BPE would have
    handled the boundary in a single-shot tokenization. That isolation
    is what the test depends on — the cache-reuse loop matches token id
    sequences, so we want full control over which positions agree."""
    bos = _tokenize("")  # add_special=False → empty list, but we'll let server prepend BOS
    shared    = _tokenize(SHARED_TEXT)
    donor_pre = _tokenize(DONOR_PREFIX_TEXT)
    rec_pre   = _tokenize(RECIPIENT_PREFIX_TEXT)
    donor_suf = _tokenize(DONOR_SUFFIX_TEXT)
    rec_suf   = _tokenize(RECIPIENT_SUFFIX_TEXT)
    donor_tokens     = donor_pre + shared + donor_suf
    recipient_tokens = rec_pre + shared + rec_suf
    return donor_tokens, recipient_tokens, len(shared)


def test_cache_reuse_legacy_suffix():
    """Legacy case (issue #5793): recipient's content past CP is a
    contiguous *suffix* of the donor's content past CP. The pre-existing
    --cache-reuse loop walks ``head_c`` forward through the donor's
    cache until it finds the suffix and splices it down. The symmetric
    extension must continue to find this match — regression-guard for
    the original semantics."""
    server.start()
    shared = _tokenize(SHARED_TEXT)
    donor_pre = _tokenize(DONOR_PREFIX_TEXT)
    donor_tokens     = donor_pre + shared
    recipient_tokens = shared
    _post_tokens(donor_tokens, 1)
    res = _post_tokens(recipient_tokens, 4)
    assert res.status_code == 200
    cache_n  = res.body["timings"]["cache_n"]
    prompt_n = res.body["timings"]["prompt_n"]
    # Recipient is exactly the SHARED tokens. After CP=0 (or 1 if BOS
    # added on both sides), --cache-reuse should find the SHARED span
    # at the donor's higher positions and splice it down.
    assert cache_n >= len(shared) - 2, (
        f"expected ~{len(shared)} tokens reused via suffix splice, "
        f"got cache_n={cache_n} prompt_n={prompt_n}"
    )


def test_cache_reuse_symmetric_chunk_in_middle():
    """Symmetric case (the agent-trace pattern): the same chunk recurs
    in both donor and recipient surrounded by *different* content on
    both sides. The legacy algorithm cannot find this match: ``head_p``
    is pinned on the recipient's divergent prefix, which never appears
    in the donor's cache; ``head_c`` walks the donor and exits empty.
    The symmetric extension advances ``head_p`` past the divergent
    prefix and finds the chunk in cache. The splice is staged via a
    temp sequence when the position shift is positive so overlapping
    source and destination ranges don't collide."""
    server.start()
    donor_tokens, recipient_tokens, n_shared = _build_donor_recipient_token_lists()
    _post_tokens(donor_tokens, 1)
    res = _post_tokens(recipient_tokens, 4)
    assert res.status_code == 200
    cache_n  = res.body["timings"]["cache_n"]
    prompt_n = res.body["timings"]["prompt_n"]
    # Without the symmetric extension this would be ~0 (recipient's
    # first token differs from donor's, so CP scan + legacy cache-reuse
    # find nothing). With the extension, the SHARED span gets spliced.
    assert cache_n >= n_shared - 2, (
        f"expected ~{n_shared} tokens reused via symmetric splice, "
        f"got cache_n={cache_n} prompt_n={prompt_n}"
    )


def test_cache_reuse_disabled_no_splice():
    """Sanity guard: with --cache-reuse 0 the new code path is gated
    off, just like the original feature. Only the byte-stable prefix is
    reused (essentially nothing here, since donor and recipient diverge
    at the very first non-BOS token)."""
    server.n_cache_reuse = 0
    server.start()
    donor_tokens, recipient_tokens, _ = _build_donor_recipient_token_lists()
    _post_tokens(donor_tokens, 1)
    res = _post_tokens(recipient_tokens, 4)
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    # With cache-reuse off, only the byte-stable prefix between donor
    # and recipient is shared. Both start with BOS but diverge
    # immediately after.
    assert cache_n <= 2, (
        f"expected cache_n <= 2 with cache-reuse disabled, got {cache_n}"
    )
