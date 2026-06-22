import json, re, glob, multiprocessing as mp
from collections import Counter
exec(open('/tmp/mine.py').read().split('if __name__')[0])

ETYPE=re.compile(r"\b(\w+Error|\w+Exception)\b\s*:", re.M)
def err_type(resp):
    if "[Terminal Error]" in resp: return "TerminalError"
    if "Command timed out" in resp: return "Timeout"
    m=ETYPE.findall(resp)
    return m[-1] if m else ("OtherErr" if is_err(resp) else None)

NAMEERR_BARE=re.compile(r"NameError: name '")  # bare env-tool call pitfall

def f(path):
    d=json.load(open(path))
    etc=Counter(); err_total=0
    first_call=0; first_call_err=0
    streaks=[]; py=0; ptc=0
    ptc_loop_agg=0  # loop over tool calls then aggregate
    eps_any_defensive=0; eps=0
    for ep in d.values():
        eps+=1
        seq=[]
        for calls,tools in iter_turns(ep.get("messages") or []):
            for ci,(nm,pr) in enumerate(calls):
                if nm in ("programmatic_tool_call","execute_python"):
                    resp=tools[ci] if ci<len(tools) else ""
                    e=is_err(resp); code=pr.get("code","")
                    seq.append((nm,code,e,resp))
                    if nm=="programmatic_tool_call": ptc+=1
                    else: py+=1
        if not seq: continue
        # first code call
        first_call+=1
        if seq[0][2]: first_call_err+=1
        # error types + streaks
        cur=0
        ep_defensive=False
        for nm,code,e,resp in seq:
            if e:
                err_total+=1
                t=err_type(resp); etc[t]+=1
                cur+=1
            else:
                if cur>0: streaks.append(cur); cur=0
            if ("try:" in code and "except" in code) or ".get(" in code or "isinstance(" in code:
                ep_defensive=True
            if re.search(r"\bfor\b.*tools\[", code, re.S) and re.search(r"\bif\b", code):
                ptc_loop_agg+=1
        if cur>0: streaks.append(cur)
        if ep_defensive: eps_any_defensive+=1
    step=int(re.search(r"step(\d+)",path).group(1))
    return (step, eps, err_total, dict(etc), first_call, first_call_err,
            (sum(streaks)/len(streaks) if streaks else 0), py, ptc, ptc_loop_agg, eps_any_defensive)

if __name__=="__main__":
    files=sorted(glob.glob("actor0_step*.json"),key=lambda p:int(re.search(r"step(\d+)",p).group(1)))
    with mp.Pool(8) as pool: res=sorted(pool.map(f,files))
    def row(step,eps,et,etc,fc,fce,ms,py,ptc,pla,defn):
        tot=max(et,1)
        ne=etc.get("NameError",0); ke=etc.get("KeyError",0); te=etc.get("TypeError",0)
        ae=etc.get("AttributeError",0); ie=etc.get("IndexError",0); se=etc.get("SyntaxError",0)
        return (f"{step:>4} | 首调错={100*fce/max(fc,1):5.1f}% | 错连击均长={ms:4.2f} | "
                f"PTC占比={100*ptc/max(ptc+py,1):5.1f}% | 任意防御式episode={100*defn/max(eps,1):4.1f}% || "
                f"NameErr={100*ne/tot:4.1f}% KeyErr={100*ke/tot:4.1f}% TypeErr={100*te/tot:4.1f}% "
                f"AttrErr={100*ae/tot:4.1f}% IdxErr={100*ie/tot:4.1f}% SynErr={100*se/tot:4.1f}%")
    print("step | first-call err | mean err-streak | PTC share | episodes w/ any defensive pattern || error-type mix (% of all code errors)")
    for r in res:
        print(row(*r))
