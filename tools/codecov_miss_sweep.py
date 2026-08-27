import json, urllib.request, sys, time
prev=None
for ln in open(sys.argv[1]):
    sha, subj = ln.split(None, 1)
    url=f"https://api.codecov.io/api/v2/github/jobovy/repos/galpy/totals/?sha={sha}"
    try:
        t=(json.load(urllib.request.urlopen(url, timeout=60)).get("totals") or {})
        m=t.get("misses"); h=t.get("hits")
    except Exception:
        m=h=None
    if m is None:
        print(f"  {sha[:8]}  (no codecov data)   {subj.strip()[:60]}")
    else:
        delta = "" if prev is None else f"  misses {prev:+d} -> {m}  ({m-prev:+d})"
        flag = ""
        if prev is not None and m > prev: flag = "   <-- MISS INCREASE"
        print(f"  {sha[:8]}  hits={h} misses={m}{flag}   {subj.strip()[:55]}")
        prev=m
    time.sleep(1)
