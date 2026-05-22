import os
import pytest
from utils import *

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


MERGED_MODEL_PATH = "/tmp/stories15M_MOE-shakespeare-merged.gguf"


def test_no_corruption_cross_model_via_lora_merge():
    # Build the second model by merging the test LoRA into the base via
    #     build/bin/llama-export-lora -m <base> --lora-scaled <lora>:1.0 -o <merged>
    if not os.path.exists(MERGED_MODEL_PATH):
        pytest.skip(f"merged model not built at {MERGED_MODEL_PATH}")
    global server

    # Save under base
    server = ServerPreset.stories15m_moe()
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 0,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "cross_merged.bin"})
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
    })
    base_tokens = res.body["content"]
    server.stop()

    # Fresh decode under merged — the ground truth for the new model
    server = ServerPreset.stories15m_moe()
    server.offline = False
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = MERGED_MODEL_PATH
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
    })
    merged_fresh = res.body["content"]
    server.stop()

    # Sanity: the two models must produce different greedy output
    assert base_tokens != merged_fresh

    # Restore base-saved state into the merged-model context and continue
    server = ServerPreset.stories15m_moe()
    server.offline = False
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = MERGED_MODEL_PATH
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.start()
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "cross_merged.bin"})
    if res.status_code < 400:
        # restore accepted — continuation must match fresh under merged
        res = server.make_request("POST", "/completion", data={
            "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
        })
        assert res.body["content"] == merged_fresh


def test_no_corruption_runtime_lora_toggle():
    global server
    lora = download_file("https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf")

    # Save with LoRA scale 0.0
    server = ServerPreset.stories15m_moe()
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.lora_files = [lora]
    server.start()
    res = server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 0.0}])
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 0,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "lora_toggle.bin"})
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
    })
    base_tokens = res.body["content"]
    server.stop()

    # Fresh decode with LoRA scale 1.0 — what the user expects after toggling
    server = ServerPreset.stories15m_moe()
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.lora_files = [lora]
    server.start()
    res = server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 1.0}])
    assert res.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
    })
    lora_tokens = res.body["content"]
    server.stop()

    # Sanity: the LoRA must actually change generation
    assert base_tokens != lora_tokens

    # Restore the (scale=0.0) saved state under scale=1.0 and continue
    server = ServerPreset.stories15m_moe()
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.lora_files = [lora]
    server.start()
    res = server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": 1.0}])
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "lora_toggle.bin"})
    if res.status_code < 400:
        res = server.make_request("POST", "/completion", data={
            "prompt": "Look in thy glass", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
        })
        assert res.body["content"] == lora_tokens


def test_no_corruption_rope_freq_drift():
    global server

    # Save with --rope-freq-base 10000
    server.rope_freq_base = 10000.0
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?", "id_slot": 0, "cache_prompt": True, "n_predict": 0,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "rope_drift.bin"})
    assert res.status_code == 200
    server.stop()

    # Fresh decode under --rope-freq-base 500000
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.rope_freq_base = 500000.0
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
    })
    fresh_tokens = res.body["content"]
    server.stop()

    # Restore the saved state under the shifted rope freq and continue
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.seed = 42
    server.rope_freq_base = 500000.0
    server.start()
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "rope_drift.bin"})
    if res.status_code < 400:
        res = server.make_request("POST", "/completion", data={
            "prompt": "What is the capital of France?", "id_slot": 0, "cache_prompt": True, "n_predict": 128,
        })
        assert res.body["content"] == fresh_tokens
