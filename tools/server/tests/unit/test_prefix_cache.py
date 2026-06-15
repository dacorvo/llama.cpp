import glob
import os
import shutil

import pytest
from utils import *

PREFIX_CACHE_DIR = "./tmp/prefix-cache"


@pytest.fixture(autouse=True)
def prefix_cache_dir():
    shutil.rmtree(PREFIX_CACHE_DIR, ignore_errors=True)
    os.makedirs(PREFIX_CACHE_DIR, exist_ok=True)
    yield
    shutil.rmtree(PREFIX_CACHE_DIR, ignore_errors=True)


def make_server() -> ServerProcess:
    server = ServerPreset.tinyllama2()
    server.prefix_cache_path = PREFIX_CACHE_DIR
    server.server_metrics = True
    server.temperature = 0.0
    return server


def test_prefix_cache_capture_and_restore_across_restart():
    prompt = "The quick brown fox jumps over the lazy dog while the sun is shining"

    # cold server: the first turn is processed from scratch and captured to disk
    server = make_server()
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 4})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0  # nothing reused on a cold run
    server.stop()  # graceful shutdown drains the writer -> entry lands on disk

    assert len(glob.glob(os.path.join(PREFIX_CACHE_DIR, "*.ggpc"))) == 1

    # fresh server (empty RAM cache): the same prompt is restored from the disk cache
    server = make_server()
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 4})
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0  # prefix restored from disk on a cold start

    metrics = server.make_request("GET", "/metrics").body
    assert "llamacpp:prefix_cache_hit_total 1" in metrics      # the restore is a hit
    assert "llamacpp:prefix_cache_capture_total 0" in metrics  # a restored prefix is not re-captured
