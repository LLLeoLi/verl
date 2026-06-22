import json, re, glob
exec(open('/tmp/mine.py').read().split('if __name__')[0])
def short(s,n=600): s=s.strip(); return s if len(s)<=n else s[:n]+"\n...[truncated]"

def recovery_diverse(path, want=6, max_code=520):
    d=json.load(open(path)); out=[]; seen=set()
    for ep in d.values():
        if ep.get("reward",0)<=0: continue
        fam=ep.get("task_name")
        if fam in seen: continue
        seq=[]
        for calls,tools in iter_turns(ep.get("messages") or []):
            for ci,(nm,pr) in enumerate(calls):
                if nm in ("programmatic_tool_call","execute_python"):
                    resp=tools[ci] if ci<len(tools) else ""
                    seq.append((nm,pr.get("code",""),is_err(resp),resp))
        for i,(nm,code,err,emsg) in enumerate(seq):
            if err and len(code)<max_code:
                for j in range(i+1,len(seq)):
                    nm2,code2,err2,_=seq[j]
                    if nm2==nm and not err2 and 0<len(code2)<max_code and code2.strip()!=code.strip():
                        # capture concise error type line
                        et=re.search(r"^(\w*(?:Error|Exception)): .*", emsg, re.M)
                        out.append((fam,ep.get("reward"),nm,code,et.group(0) if et else emsg.strip().split(chr(10))[-1][:120],code2))
                        seen.add(fam); break
            if fam in seen: break
        if len(out)>=want: break
    return out

print("#### DIVERSE recovery examples (step33) ####")
for fam,rw,tool,bad,err,good in recovery_diverse("actor0_step33.json",want=6):
    print(f"\n@@@ {fam} | reward={rw}")
    print(f"[err type] {err}")
    print(f"[before]\n{short(bad,360)}")
    print(f"[after]\n{short(good,360)}")
