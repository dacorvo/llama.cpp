import json, time, glob, subprocess, urllib.request, sys, os, shutil, signal

PORT=8090
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR=os.path.join(ROOT,"bench","pc")+os.sep
RDIR=sys.argv[3] if len(sys.argv)>3 else os.path.join(ROOT,"bench","reqs2")
REQS=[json.load(open(p)) for p in sorted(glob.glob(RDIR+"/r*.json"))]

def kill():
    subprocess.run(["pkill","-f","build/bin/llama-server"],stderr=subprocess.DEVNULL)
    for _ in range(60):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=1); time.sleep(0.5)
        except Exception: return

def launch(model, alias):
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    subprocess.Popen(["./build/bin/llama-server","-ngl","99","--ctx-size","12288","--metrics",
        "-np","1","--host","127.0.0.1","--port",str(PORT),"--alias",alias,
        "--prefix-cache-path",DIR,"-m",model],
        cwd=ROOT, stdout=open("/tmp/srv_bench.log","w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(200):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2); return
        except Exception: time.sleep(1)
    raise RuntimeError("server did not come up")

def send(req, cache_prompt, stream=True):
    r=dict(req); r["stream"]=stream; r["cache_prompt"]=cache_prompt; r["max_tokens"]=8
    body=json.dumps(r).encode()
    rq=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    t0=time.perf_counter()
    if not stream:
        urllib.request.urlopen(rq,timeout=300).read(); return (time.perf_counter()-t0)*1000
    for raw in urllib.request.urlopen(rq,timeout=300):
        line=raw.decode("utf-8","ignore").strip()
        if line.startswith("data:") and line[5:].strip() and "[DONE]" not in line:
            try:
                d=json.loads(line[5:]); de=d.get("choices",[{}])[0].get("delta",{})
                if de.get("content") or de.get("reasoning_content"): return (time.perf_counter()-t0)*1000
            except: pass
    return None

def metric(n):
    try:
        for l in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics",timeout=10).read().decode().splitlines():
            if n in l and not l.startswith("#"): return int(float(l.split()[-1]))
    except: return -1
    return 0

model, alias = sys.argv[1], sys.argv[2]
print(f"### MODEL={alias}", flush=True)

# 1. populate (fresh disk)
shutil.rmtree(DIR, ignore_errors=True); os.makedirs(DIR, exist_ok=True)
kill(); launch(model, alias)
for o in REQS: send(o["req"], cache_prompt=True)
time.sleep(2)
print(f"populate: captured={metric('prefix_cache_capture_total')} entries; files={len(os.listdir(DIR))}", flush=True)
kill()

# 2. cold baseline: full prefill, no reuse (cache_prompt=False)
launch(model, alias)
cold=[send(o["req"], cache_prompt=False) for o in REQS]
kill()

# 3. warm: restart before each request so RAM+slot empty -> disk serves
warm=[]; hits=[]
for o in REQS:
    launch(model, alias)
    h0=metric("prefix_cache_hit_total")
    t=send(o["req"], cache_prompt=True)
    warm.append(t); hits.append(metric("prefix_cache_hit_total")-h0)
    kill()

print(f"{'task':10} {'cold_ms':>8} {'warm_ms':>8} {'hit':>4} {'speedup':>8}", flush=True)
for o,c,w,h in zip(REQS,cold,warm,hits):
    sp=f"{c/w:.1f}x" if (c and w) else "?"
    print(f"{o['task']:10} {c:8.0f} {w:8.0f} {h:>4} {sp:>8}", flush=True)
import statistics
cg=[c for c in cold if c]; wg=[w for w in warm if w]
print(f"\nmedian cold={statistics.median(cg):.0f}ms  warm={statistics.median(wg):.0f}ms  hits={sum(hits)}/{len(hits)}", flush=True)
