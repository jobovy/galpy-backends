"""Harvest a backend-tests run into a ledger verdict.

Deliberate choices:
  * a shard with NO artifact, or whose xml has zero testcases, is NO DATA --
    never "passed". Timed-out shards look exactly like fast ones otherwise.
  * xfail staleness needs XPASS, not "the shard was green": xfail(strict=False)
    keeps a shard green either way.
  * slow-skip staleness needs the test to have actually RUN (regen mode). A
    "skipped" result proves nothing.
"""
import os, sys, glob, collections, xml.etree.ElementTree as ET

def load_ledger(path):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        arm, nodeid = ln.split(None, 1)
        # ledger lines carry INLINE annotations, e.g.
        #   jax tests/test_orbits.py::test_x  # 300s (failed)
        # Keeping them in the nodeid makes every lookup miss and every entry
        # read as MISSING -- which looks exactly like "not stale".
        nodeid = nodeid.split("#", 1)[0].strip()
        out.append((arm, nodeid))
    return out

def harvest(junit_root):
    status, dur, nodata = {}, {}, []
    for d in sorted(glob.glob(os.path.join(junit_root, "backend-junit-*"))):
        base = os.path.basename(d)
        backend = "jax" if "-jax-" in base else "torch"
        xmls = glob.glob(os.path.join(d, "*.xml"))
        if not xmls:
            nodata.append(base); continue
        n = 0
        for x in xmls:
            try:
                tree = ET.parse(x)
            except ET.ParseError:
                nodata.append(base); continue
            for tc in tree.iter("testcase"):
                n += 1
                cls = tc.get("classname") or ""
                f = cls.split(".")[-1]
                name = tc.get("name") or ""
                st = "pass"
                if tc.find("failure") is not None or tc.find("error") is not None:
                    st = "fail"
                sk = tc.find("skipped")
                if sk is not None:
                    ty = (sk.get("type") or "").lower()
                    msg = (sk.get("message") or "").lower()
                    st = "xfail" if "xfail" in ty else "skip"
                    if "xpass" in ty or "xpass" in msg:
                        st = "xpass"
                status[(backend, f, name)] = st
                try:
                    dur[(backend, f, name)] = float(tc.get("time") or 0.0)
                except ValueError:
                    pass
        if n == 0:
            nodata.append(base)
    return status, dur, nodata

def key(arm, nodeid):
    backend = arm.split("-")[0]
    fpath, tname = nodeid.split("::", 1)
    return (backend, os.path.basename(fpath).replace(".py", ""), tname)

if __name__ == "__main__":
    # ARMS is which arms this RUN actually covers ("jax,torch" for an eager
    # dispatch, "jax-jit,torch-jit" for jit=true). Without it a torch-jit ledger
    # entry gets compared against a plain-torch RESULT and reads as stale -- the
    # -jit-vs-eager false-stale mode. Entries outside ARMS are NOT COVERED, not
    # verified.
    root, xf, sl, arms = sys.argv[1], sys.argv[2], sys.argv[3], set(sys.argv[4].split(","))
    status, dur, nodata = harvest(root)
    print(f"harvested {len(status)} results; NO-DATA shards: {len(nodata)}")
    for b in nodata[:8]:
        print(f"    no data: {b}")
    xfail_all, slow_all = load_ledger(xf), load_ledger(sl)
    xfail = [(a, n) for a, n in xfail_all if a in arms]
    slow = [(a, n) for a, n in slow_all if a in arms]
    print(f"\narms covered by this run: {sorted(arms)}")
    print(f"  xfail  {len(xfail)}/{len(xfail_all)} covered, "
          f"{len(xfail_all) - len(xfail)} NOT COVERED (other arms)")
    print(f"  slow   {len(slow)}/{len(slow_all)} covered, "
          f"{len(slow_all) - len(slow)} NOT COVERED (other arms)")
    def lookup(a, n):
        k = key(a, n)
        if k in status:
            return status[k]
        # bare ledger name vs parametrized junit name
        hits = {v for (b, f, nm), v in status.items()
                if (b, f) == k[:2] and nm.startswith(k[2] + "[")}
        if not hits:
            return "MISSING"
        for pref in ("fail", "xfail", "skip", "xpass", "pass"):
            if pref in hits:
                return pref
        return hits.pop()

    stale_x = [(a, n) for a, n in xfail if lookup(a, n) == "xpass"]
    print(f"\nxfail entries: {len(xfail)}   STALE (xpassed): {len(stale_x)}")
    for a, n in stale_x:
        print(f"    {a} {n}")
    from collections import Counter
    print("\n  slow-skip entry statuses:", dict(Counter(lookup(a, n) for a, n in slow)))
    ran = [(a, n, dur.get(key(a, n), -1.0)) for a, n in slow
           if lookup(a, n) in ("pass", "xpass")]
    print(f"\nslow-skip entries: {len(slow)}   actually RAN and PASSED: {len(ran)}")
    for a, n, t in sorted(ran, key=lambda r: r[2]):
        print(f"    {t:8.1f}s  {a} {n}")
    unled = {k for k, v in status.items() if v == "fail"} - {key(a, n) for a, n in xfail_all + slow_all}
    print(f"\nunledgered failures: {len(unled)}")
    for k in list(unled)[:10]:
        print(f"    {k}")
