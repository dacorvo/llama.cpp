import json, time, glob, sys, urllib.request
PORT=8090
def ttft(req):
    body=json.dumps(req).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    t0=time.perf_counter(); first=None; resp=urllib.request.urlopen(r,timeout=300)
    for raw in resp:
        line=raw.decode("utf-8","ignore").strip()
        if line.startswith("data:") and line[5:].strip() and "[DONE]" not in line:
            try:
                d=json.loads(line[5:])
                delta=d.get("choices",[{}])[0].get("delta",{})
                if delta.get("content") or delta.get("reasoning_content"):
                    first=time.perf_counter(); break
            except: pass
    return (first-t0)*1000 if first else None
def metric(name):
    try:
        for l in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics",timeout=10).read().decode().splitlines():
            if name in l and not l.startswith("#"): return l.split()[-1]
    except: return "?"
    return "0"
reqs=sorted(glob.glob("/home/ubuntu/llama.cpp/bench/reqs2/r*.json"))
label=sys.argv[1]
print(f"# {label}: hit0={metric('prefix_cache_hit_total')} miss0={metric('prefix_cache_miss_total')}")
for p in reqs:
    o=json.load(open(p)); req=o["req"]; req["stream"]=True; req["cache_prompt"]=(sys.argv[2]=="reuse") if len(sys.argv)>2 else True
    ms=ttft(req)
    print(f"{label}\t{o['task']}\t{ms:.0f}" if ms else f"{label}\t{o['task']}\tNONE")
    time.sleep(0.3)
print(f"# {label}: hit={metric('prefix_cache_hit_total')} miss={metric('prefix_cache_miss_total')} cap={metric('prefix_cache_capture_total')}")
