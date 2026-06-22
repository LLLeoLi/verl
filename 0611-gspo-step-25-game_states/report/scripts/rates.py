import json, re, glob, multiprocessing as mp
exec(open('/tmp/mine.py').read().split('if __name__')[0])
PROBE=re.compile(r"\btype\(|\.keys\(\)|isinstance\(|\bdir\(|list\(.*\.keys")
GETDEF=re.compile(r"\.get\(")
def f(path):
    d=json.load(open(path))
    eps_with_err=0; eps_err_then_success=0
    err_calls=0; probe_after_err=0; getdef_calls=0; total_code_calls=0
    for ep in d.values():
        msgs=ep.get("messages") or []
        seq=[]
        for calls,tools in iter_turns(msgs):
            for ci,(nm,pr) in enumerate(calls):
                if nm in ("programmatic_tool_call","execute_python"):
                    resp=tools[ci] if ci<len(tools) else ""
                    seq.append((pr.get("code",""), is_err(resp)))
        had_err=any(e for _,e in seq)
        if had_err:
            eps_with_err+=1
            if ep.get("reward",0)>0: eps_err_then_success+=1
        for idx,(code,err) in enumerate(seq):
            total_code_calls+=1
            if GETDEF.search(code): getdef_calls+=1
            if err:
                err_calls+=1
                if idx+1<len(seq) and PROBE.search(seq[idx+1][0]): probe_after_err+=1
    step=int(re.search(r"step(\d+)",path).group(1))
    return (step,eps_with_err,eps_err_then_success,err_calls,probe_after_err,getdef_calls,total_code_calls)
if __name__=="__main__":
    files=sorted(glob.glob("actor0_step*.json"),key=lambda p:int(re.search(r"step(\d+)",p).group(1)))
    with mp.Pool(8) as pool: res=pool.map(f,files)
    print("step | recover-to-success% (of eps that hit a code error) | schema-probe-after-error% | .get(default) usage%")
    for step,ewe,ets,ec,pae,gd,tc in sorted(res):
        print(f"{step:>4} | {100*ets/max(ewe,1):5.1f}%  (errd eps={ewe:4d}) | {100*pae/max(ec,1):5.1f}% | {100*gd/max(tc,1):5.1f}%")
