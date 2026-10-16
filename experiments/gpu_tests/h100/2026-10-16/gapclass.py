"""gapclass.py <junit.xml> [...]: counts + one classified line per failing test (file::test | class | detail | last error line)."""
import sys, re, collections, xml.etree.ElementTree as ET
KNOWN_NP = ("adiabaticcontraction", "coerce", "linalg", "paramgrad")
KNOWN_TESTS = {"test_backend_eddingtondf": "test_plain_backend_potential_differentiated_denspot",
               "test_backend_movingobject": "test_object_orbit_and_kernel_grad_vs_fd",
               "test_backend_orbit_accessors": "test_backend_orbit_read_under_a_numpy_pin"}
def classify(f, name, msg, txt):
    base = re.sub(r"\[.*", "", name); E = [l[1:].strip() for l in txt.splitlines() if l.startswith("E ") and l[1:].strip()]
    last = E[-1] if E else msg.splitlines()[0] if msg else "?"
    if "'ndarray' object has no attribute" in txt and ("TorchFunctionMode" in txt or "fake tensor" in txt or "_backend_interp_prep" in txt):
        return "known: dynamo traces numpy under set_default_device (upstream)", last
    if "can't convert cuda" in txt:
        if any(f == f"test_backend_{k}" for k in KNOWN_NP) or KNOWN_TESTS.get(f) == base:
            return "known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up)", last
        fr = [x for x in re.findall(r"^(\S+\.py):(\d+): in (\S+)", txt, re.M) if x[0].startswith(("galpy/", "tests/"))]
        return "REAL GPU defect: numpy conversion of a CUDA tensor" + (f" ({fr[-1][0]}:{fr[-1][1]})" if fr else ""), last
    if re.search(r"Timeout|timed out|Timeout >", txt): return "infrastructure: timeout", last
    if re.search(r"out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED", txt): return "infrastructure: out of memory", last
    if re.search(r"ModuleNotFoundError|No module named|ImportError", txt): return "infrastructure: missing package", last
    if re.search(r"Expected all tensors to be on the same device|device mismatch|different from other tensors", txt):
        return "REAL GPU defect: device mismatch", last
    if re.search(r"InductorError|triton|Triton|CompilationError", txt): return "REAL GPU defect: inductor/Triton compile error", last
    if "test_analytic_actions_" in name and ("Not equal to tolerance" in txt or "AssertionError" in txt):
        return "known: last-digit miss in test_analytic_actions_* (fixed on follow-up)", last
    m = re.search(r"Not equal to tolerance rtol=([0-9.e+-]+), atol=([0-9.e+-]+)", txt)
    if m:
        ma = re.search(r"Max absolute difference[^:]*: ([0-9.e+-]+)", txt); mr = re.search(r"Max relative difference[^:]*: ([0-9.e+-]+)", txt)
        detail = f"max abs {ma.group(1) if ma else '?'}, max rel {mr.group(1) if mr else '?'} vs rtol {m.group(1)}, atol {m.group(2)}"
        rel = float(mr.group(1)) if mr and mr.group(1) not in ("inf", "nan") else None
        return ("tolerance miss" if rel is not None and rel < 1e-6 else "CHECK: numeric mismatch (possibly wrong value)") + f" ({detail})", last
    return "UNCLASSIFIED", last
def main(paths):
  for path in paths:
      c = collections.Counter(); rows = []
      root = ET.parse(path).getroot()
      for tc in root.iter("testcase"):
          e = tc.find("failure"); e = e if e is not None else tc.find("error")
          if e is None: c["skipped" if tc.find("skipped") is not None else "passed"] += 1; continue
          c["failed"] += 1; f = tc.get("classname").split(".")[-1]
          cl, last = classify(f, tc.get("name"), e.get("message") or "", e.text or "")
          rows.append((f"tests/{f}.py::{tc.get('name')}", cl, last[:200]))
      print(f"### {path.split('/')[-1]}: {dict(c)}")
      for k, n in collections.Counter(r[1].split(' (')[0] for r in rows).most_common(): print(f"   {n:4d}  {k}")
      for r in rows: print("  |", " | ".join(r))

if __name__ == '__main__':
    main(sys.argv[1:])
