import json,time,glob,subprocess,urllib.request,os,shutil,sys
PORT=8090; DIR="/home/ubuntu/llama.cpp/bench/pc/"
MODEL=sys.argv[1]; ALIAS=sys.argv[2]
REQ=json.load(open(sorted(glob.glob("/home/ubuntu/llama.cpp/bench/reqs2/r*.json"))[0]))["req"]
def kill():
    subprocess.run(["pkill","-f","build/bin/llama-server"],stderr=subprocess.DEVNULL)
    for _ in range(60):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=1); time.sleep(0.5)
        except: return
def launch():
    subprocess.Popen(["./build/bin/llama-server","-ngl","99","--ctx-size","8192","--metrics","-np","1",
      "--host","127.0.0.1","--port",str(PORT),"--alias",ALIAS,"--prefix-cache-path",DIR,"-m",MODEL],
      cwd="/home/ubuntu/llama.cpp",stdout=open("/tmp/srv_cd.log","w"),stderr=subprocess.STDOUT,
      env=dict(os.environ,CUDA_VISIBLE_DEVICES="0"))
    for _ in range(300):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2); return
        except: time.sleep(1)
def send():
    r=dict(REQ); r["stream"]=True; r["cache_prompt"]=True; r["max_tokens"]=8
    body=json.dumps(r).encode(); rq=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    t0=time.perf_counter()
    for raw in urllib.request.urlopen(rq,timeout=300):
        l=raw.decode("utf-8","ignore").strip()
        if l.startswith("data:") and l[5:].strip() and "[DONE]" not in l:
            try:
                de=json.loads(l[5:]).get("choices",[{}])[0].get("delta",{})
                if de.get("content") or de.get("reasoning_content"): return (time.perf_counter()-t0)*1000
            except: pass
def hits():
    for l in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics",timeout=10).read().decode().splitlines():
        if "prefix_cache_hit_total" in l and not l.startswith("#"): return int(float(l.split()[-1]))
    return 0
# populate one entry
shutil.rmtree(DIR,ignore_errors=True); os.makedirs(DIR); kill(); launch(); send(); time.sleep(2); kill()
sz=os.path.getsize(DIR+os.listdir(DIR)[0])//(1024*1024)
# WARM (page cache hot): launch + first request
launch(); h0=hits(); tw=send(); hw=hits()-h0; kill()
# COLD disk: launch, drop OS page cache, first request -> blob read from disk
launch()
subprocess.run(["sudo","sh","-c","echo 3 > /proc/sys/vm/drop_caches"])
h0=hits(); tc=send(); hc=hits()-h0; kill()
print(f"{ALIAS}: entry={sz}MiB  warm(pagecache)={tw:.0f}ms hit={hw}  cold-disk(dropped)={tc:.0f}ms hit={hc}")
