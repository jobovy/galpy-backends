"""compare_fullsuite.py <baseline.xml[.gz]> <new.xml[.gz]> [out.tsv]

Compare two full CUDA test-suite JUnit files (tools: logs/h100/*/jobs/fullsuite.sbatch)
by nodeid. A device regression is a test that passed in the baseline and fails now;
a new test that fails is listed separately. Prints counts and the regressions, and
writes every changed test (status, baseline status, one-line error) to out.tsv."""

import collections
import gzip
import sys
import xml.etree.ElementTree as ET


def load(path):
    fh = gzip.open(path) if path.endswith(".gz") else open(path, "rb")
    out = {}
    for tc in ET.parse(fh).getroot().iter("testcase"):
        nodeid = tc.get("classname").replace(".", "/") + ".py::" + tc.get("name")
        status, msg = "passed", ""
        for tag in ("failure", "error", "skipped"):
            e = tc.find(tag)
            if e is not None:
                status = "failed" if tag in ("failure", "error") else "skipped"
                msg = ((e.get("message") or "").splitlines() or [""])[0][:200]
                break
        out[nodeid] = (status, msg)
    return out


def main():
    base, new = load(sys.argv[1]), load(sys.argv[2])
    groups = collections.defaultdict(list)
    for k, (s, msg) in new.items():
        b = base.get(k, ("absent", ""))[0]
        if b == s:
            continue
        if s == "failed":
            label = "REGRESSED" if b == "passed" else ("NEW-FAILING" if b == "absent" else "SKIP->FAIL")
        elif s == "skipped":
            label = "PASS->SKIP" if b == "passed" else ("NEW-SKIPPED" if b == "absent" else "FAIL->SKIP")
        else:
            label = "FIXED" if b == "failed" else ("NEW-PASSING" if b == "absent" else "SKIP->PASS")
        groups[label].append((k, b, s, msg))
    gone = [k for k in base if k not in new]
    for k in gone:
        groups["GONE"].append((k, base[k][0], "absent", ""))
    tot = lambda d, s: sum(v[0] == s for v in d.values())
    for name, d in (("baseline", base), ("new", new)):
        print(f"{name}: {len(d)} tests, {tot(d, 'passed')} passed, "
              f"{tot(d, 'failed')} failed, {tot(d, 'skipped')} skipped")
    for label in sorted(groups):
        print(f"{label:12s} {len(groups[label])}")
    for label in ("REGRESSED", "SKIP->FAIL", "NEW-FAILING", "PASS->SKIP"):
        if groups[label]:
            print(f"== {label}")
            c = collections.Counter(m[:90] for *_, m in groups[label])
            for m, n in c.most_common(25):
                print(f"{n:5d} {m}")
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as fh:
            for label in sorted(groups):
                for k, b, s, msg in groups[label]:
                    fh.write("\t".join((label, k, b, s, msg)) + "\n")


if __name__ == "__main__":
    main()
