import json, re, sys

CALL_RE = re.compile(r"<function=([a-zA-Z_]+)>(.*?)</function>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([a-zA-Z_]+)>(.*?)</parameter>", re.DOTALL)

ERR_MARKERS = ("Traceback (most recent call last)", "[Error]", "Error executing tool",
               "[stderr]", "[Terminal Error]", "Command timed out", "SyntaxError",
               "NameError", "KeyError", "TypeError", "ValueError", "AttributeError",
               "IndexError", "not found", "No such file")

def parse_calls(content):
    out=[]
    for m in CALL_RE.finditer(content):
        name=m.group(1); body=m.group(2)
        params={p.group(1):p.group(2).strip() for p in PARAM_RE.finditer(body)}
        out.append((name, params))
    return out

def is_err(tool_content):
    t=tool_content
    return any(mk in t for mk in ERR_MARKERS)

def iter_turns(msgs):
    """Yield (assistant_call_list, [following tool responses]) aligned in order."""
    i=0
    n=len(msgs)
    while i<n:
        m=msgs[i]
        if m.get("role")=="assistant":
            calls=parse_calls(m.get("content",""))
            # collect following tool messages
            j=i+1
            tools=[]
            while j<n and msgs[j].get("role")=="tool":
                tools.append(msgs[j].get("content",""))
                j+=1
            yield calls, tools
            i=j
        else:
            i+=1

def analyze_file(path):
    d=json.load(open(path))
    stat=dict(eps=0, ptc=0, ptc_err=0, ptc_if=0, ptc_try=0, ptc_loop=0,
              py=0, py_err=0, term=0, term_err=0, recover_events=0, err_then_call=0)
    for ep in d.values():
        stat["eps"]+=1
        msgs=ep.get("messages") or []
        turns=list(iter_turns(msgs))
        for ti,(calls,tools) in enumerate(turns):
            for ci,(name,params) in enumerate(calls):
                resp = tools[ci] if ci < len(tools) else ""
                code = params.get("code","")
                if name=="programmatic_tool_call":
                    stat["ptc"]+=1
                    if is_err(resp): stat["ptc_err"]+=1
                    if re.search(r"\bif\b", code) and re.search(r"\belse\b|\belif\b", code): stat["ptc_if"]+=1
                    if "try:" in code and "except" in code: stat["ptc_try"]+=1
                    if re.search(r"\bfor\b|\bwhile\b", code): stat["ptc_loop"]+=1
                elif name=="execute_python":
                    stat["py"]+=1
                    if is_err(resp): stat["py_err"]+=1
                elif name=="terminal":
                    stat["term"]+=1
                    if is_err(resp): stat["term_err"]+=1
    return stat

if __name__=="__main__":
    for p in sys.argv[1:]:
        s=analyze_file(p)
        step=re.search(r"step(\d+)", p).group(1)
        print(f"step {step:>2}: eps={s['eps']} | PTC={s['ptc']} err={s['ptc_err']} "
              f"({100*s['ptc_err']/max(s['ptc'],1):.1f}%) if/else={s['ptc_if']} try={s['ptc_try']} loop={s['ptc_loop']} "
              f"| py={s['py']} err={100*s['py_err']/max(s['py'],1):.1f}% "
              f"| term={s['term']} err={100*s['term_err']/max(s['term'],1):.1f}%")

def run_all():
    import multiprocessing as mp, glob
    files=sorted(glob.glob("actor0_step*.json"), key=lambda p:int(re.search(r"step(\d+)",p).group(1)))
    with mp.Pool(8) as pool:
        results=pool.map(analyze_file, files)
    rows=[]
    for p,s in zip(files,results):
        step=int(re.search(r"step(\d+)",p).group(1))
        rows.append((step,s))
    rows.sort()
    print("step,ptc_calls,ptc_err%,ptc_ifelse%,ptc_try%,ptc_loop%,py_err%,term_err%")
    for step,s in rows:
        print(f"{step},{s['ptc']},{100*s['ptc_err']/max(s['ptc'],1):.1f},"
              f"{100*s['ptc_if']/max(s['ptc'],1):.1f},{100*s['ptc_try']/max(s['ptc'],1):.1f},"
              f"{100*s['ptc_loop']/max(s['ptc'],1):.1f},{100*s['py_err']/max(s['py'],1):.1f},"
              f"{100*s['term_err']/max(s['term'],1):.1f}")
