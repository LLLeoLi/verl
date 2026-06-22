import json, re, glob, multiprocessing as mp
from collections import Counter
exec(open('/tmp/mine.py').read().split('if __name__')[0])

# Structure-probing primitives
P = {
 "type()":      re.compile(r"\btype\s*\("),
 ".keys()":     re.compile(r"\.keys\s*\("),
 "isinstance":  re.compile(r"\bisinstance\s*\("),
 "dir()":       re.compile(r"\bdir\s*\("),
 "len()":       re.compile(r"\blen\s*\("),
 "pandas(.columns/.dtypes/.shape/.head)": re.compile(r"\.(columns|dtypes|shape|head)\b"),
}
ANYPROBE = re.compile(r"\btype\s*\(|\.keys\s*\(|\bisinstance\s*\(|\bdir\s*\(|\.(columns|dtypes|shape|head)\b")

def f(path):
    d=json.load(open(path))
    code_calls=0; probe_calls=0
    eps=0; eps_with_probe=0
    proactive=0; reactive_after_err=0; probe_total=0
    prim=Counter()
    first_probe_idx=[]; ncalls_when_probe=[]
    reward_probe=[]; reward_noprobe=[]
    for ep in d.values():
        eps+=1
        seq=[]
        for calls,tools in iter_turns(ep.get("messages") or []):
            for ci,(nm,pr) in enumerate(calls):
                if nm in ("programmatic_tool_call","execute_python"):
                    resp=tools[ci] if ci<len(tools) else ""
                    seq.append((pr.get("code",""), is_err(resp)))
        if not seq: continue
        had_probe=False; seen_err=False; fp=None
        for idx,(code,err) in enumerate(seq):
            code_calls+=1
            isprobe=bool(ANYPROBE.search(code))
            if isprobe:
                probe_calls+=1; probe_total+=1; had_probe=True
                if fp is None: fp=idx
                for k,rx in P.items():
                    if rx.search(code): prim[k]+=1
                if seen_err: reactive_after_err+=1
                else: proactive+=1
            if err: seen_err=True
        if had_probe:
            eps_with_probe+=1; first_probe_idx.append(fp)
        rw=ep.get("reward",0.0)
        (reward_probe if had_probe else reward_noprobe).append(rw)
    step=int(re.search(r"step(\d+)",path).group(1))
    return (step, code_calls, probe_calls, eps, eps_with_probe, proactive, reactive_after_err,
            probe_total, dict(prim),
            (sum(first_probe_idx)/len(first_probe_idx) if first_probe_idx else 0),
            (sum(reward_probe)/len(reward_probe) if reward_probe else 0),
            (sum(reward_noprobe)/len(reward_noprobe) if reward_noprobe else 0),
            len(reward_probe), len(reward_noprobe))

if __name__=="__main__":
    files=sorted(glob.glob("actor0_step*.json"),key=lambda p:int(re.search(r"step(\d+)",p).group(1)))
    with mp.Pool(8) as pool: res=sorted(pool.map(f,files))
    print("step | probe调用占比 | 含探查episode占比 | 探查中主动% | 探查中出错后被动% | 首次探查平均代码调用序号 | reward(探查ep) vs reward(无探查ep)")
    for (step,cc,pc,eps,ewp,pro,rea,pt,prim,fpi,rp,rnp,nrp,nrnp) in res:
        print(f"{step:>4} | {100*pc/max(cc,1):5.1f}% | {100*ewp/max(eps,1):5.1f}% | "
              f"{100*pro/max(pt,1):5.1f}% | {100*rea/max(pt,1):5.1f}% | {fpi:5.2f} | "
              f"{rp:.3f} vs {rnp:.3f}")
    print("\n=== 探查原语构成（占含探查的代码调用，step0 vs step33）===")
    s0=res[0][8]; s33=res[-1][8]; pc0=res[0][2]; pc33=res[-1][2]
    for k in P:
        print(f"{k:<40} step0={100*s0.get(k,0)/max(pc0,1):5.1f}%   step33={100*s33.get(k,0)/max(pc33,1):5.1f}%")
