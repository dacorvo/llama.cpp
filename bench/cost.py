import json,time,glob,subprocess,urllib.request,os,sys
PORT=8090
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR=os.path.join(ROOT,"bench","pc")+os.sep
MODEL=sys.argv[1]; ALIAS=sys.argv[2] if len(sys.argv)>2 else "gemma"
REQ=json.load(open(sorted(glob.glob(os.path.join(ROOT,"bench","reqs2","r*.json")))[0]))["req"]
def kill():
    subprocess.run(["pkill","-f","build/bin/llama-server"],stderr=subprocess.DEVNULL)
    for _ in range(60):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=1); time.sleep(0.5)
        except: return
def launch():
    subprocess.Popen(["./build/bin/llama-server","-ngl","99","--ctx-size","8192","--metrics","-np","1",
      "--host","127.0.0.1","--port",str(PORT),"--alias",ALIAS,"--prefix-cache-path",DIR,
      "-m",MODEL],
      cwd=ROOT,stdout=open("/tmp/srv_cost.log","w"),stderr=subprocess.STDOUT,
      env=dict(os.environ,CUDA_VISIBLE_DEVICES="0"))
    for _ in range(200):
        try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",timeout=2); return
        except: time.sleep(1)
def send(cache_prompt, first_turn):
    r=dict(REQ); r["stream"]=True; r["cache_prompt"]=cache_prompt; r["first_turn"]=first_turn; r["max_tokens"]=8
    body=json.dumps(r).encode(); rq=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",body,{"Content-Type":"application/json"})
    t0=time.perf_counter()
    for raw in urllib.request.urlopen(rq,timeout=300):
        l=raw.decode("utf-8","ignore").strip()
        if l.startswith("data:") and l[5:].strip() and "[DONE]" not in l:
            try:
                de=json.loads(l[5:]).get("choices",[{}])[0].get("delta",{})
                if de.get("content") or de.get("reasoning_content"): return (time.perf_counter()-t0)*1000
            except: pass
# capture penalty: fresh empty disk, warm up, then first_turn F vs T (both cold full prefill)
import shutil; shutil.rmtree(DIR,ignore_errors=True); os.makedirs(DIR)
kill(); launch()
send(False, False)  # warmup
nocap=[send(False, False) for _ in range(3)]
cap=[send(False, True) for _ in range(3)]   # triggers snapshot
import statistics
print(f"capture penalty: no-capture TTFT={statistics.median(nocap):.0f}ms  with-capture={statistics.median(cap):.0f}ms  delta={statistics.median(cap)-statistics.median(nocap):.0f}ms")
kill()
