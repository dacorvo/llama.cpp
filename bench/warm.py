import json, time, glob, sys, urllib.request
PORT=8090
SPACER={"messages":[{"role":"user","content":"Say OK. "+("filler unique "*400)}],
        "max_tokens":1,"temperature":0,"stream":False,"first_turn":False,"cache_prompt":True}
def post(req):
    body=json.dumps(req).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    return urllib.request.urlopen(r,timeout=300)
def ttft(req):
    t0=time.perf_counter()
    for raw in post(req):
        line=raw.decode("utf-8","ignore").strip()
        if line.startswith("data:") and line[5:].strip() and "[DONE]" not in line:
            try:
                d=json.loads(line[5:]); de=d.get("choices",[{}])[0].get("delta",{})
                if de.get("content") or de.get("reasoning_content"):
                    return (time.perf_counter()-t0)*1000
            except: pass
    return None
def m(n):
    try:
        for l in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics",timeout=10).read().decode().splitlines():
            if n in l and not l.startswith("#"): return int(float(l.split()[-1]))
    except: return -1
    return 0
mode=sys.argv[1] if len(sys.argv)>1 else "WARM"
for p in sorted(glob.glob("/home/ubuntu/llama.cpp/bench/reqs2/r*.json")):
    o=json.load(open(p)); req=o["req"]; req["stream"]=True; req["cache_prompt"]=True
    if mode=="WARM":
        post(SPACER).read()   # evict slot KV so the next request must fall back to the disk cache
    h0=m("prefix_cache_hit_total")
    ms=ttft(req)
    hit=m("prefix_cache_hit_total")-h0
    print(f"{mode}\t{o['task']}\t{ms:.0f}\thit={hit}" if ms else f"{mode}\t{o['task']}\tNONE")
    time.sleep(0.2)
