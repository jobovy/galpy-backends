"""Collect the test suite at every ledger-touching commit and expand the ledger
onto it: each ledger entry counts as the number of collected test ids it matches
(a whole file, a class, or a parametrized function listed without parameters
matches all of its tests). Splitting one entry into its parametrizations then
leaves the count unchanged, and whole-file defers count at their real size.

    GALPY_REPO=/home/bovy/Repos/galpy GALPY_REF=origin/feat/backends \\
        python measure/suite_from_git.py [--workers 4]

Writes data/suite.json: per commit, the existing suite's size (tests/ minus the
test_backend*.py modules), the backend-module suite's size, and per mode
(jax, torch, jax-jit, torch-jit) the number of existing-suite tests the ledger
(xfail + slow-skip) covers, plus the permanent skips (backend_skip.txt).

"effective": per mode, the tests the snapshot's OWN tests/conftest.py takes out
of the run -- the union of its xfail-ledger, slow-skip and permanent-skip sets,
loaded with that conftest's loaders and matched with its _matches. Under --jit
these loaders merge in eager rows ("<backend>-jit" inherits "<backend>"), and
which lists inherit changed over time, so the "ledgered"/"skipped" jit counts
above (the "-jit" rows alone) are only a lower bound on what --jit excludes:

  * before #1210 (2026-07-27): no inheritance, "-jit" rows only;
  * #1210 .. #1279 (2026-08-09): all three lists inherit the eager rows;
  * from #1279: xfail + permanent skip inherit, slow-skip does not.

The rule is not hard-coded: "jit_rule" records what the snapshot's loaders did
when probed with synthetic eager/jit lists.
"""

import argparse
import re
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = os.environ.get("GALPY_REPO", "/home/bovy/Repos/galpy")
REF = os.environ.get("GALPY_REF", "origin/feat/backends")
LEDGERS = ["tests/backend_xfail.txt", "tests/backend_slow_skip.txt"]
SKIP = "tests/backend_skip.txt"
MODES = ("jax", "torch", "jax-jit", "torch-jit")
OUT = Path(__file__).resolve().parents[1] / "data" / "suite.json"


def _git(*args, cwd=REPO):
    return subprocess.run(
        ["git", "-C", cwd, *args], capture_output=True, text=True
    ).stdout


def _entries(sha, path):
    out = {m: [] for m in MODES}
    for line in _git("show", f"{sha}:{path}").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(None, 1)
        if len(p) == 2 and p[0] in MODES:
            out[p[0]].append(p[1].split()[0])
    return out


def _matches(node, entry):
    return node == entry or node.startswith(entry + "::") or node.startswith(entry + "[")


def _collect(wt, args):
    env = dict(os.environ, PYTHONPATH=wt, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "--continue-on-collection-errors", *args],
        cwd=wt, env=env, capture_output=True, text=True, timeout=1800,
    )
    ids = [l.strip() for l in r.stdout.splitlines() if "::" in l and not l.startswith(" ")]
    errors = sum(1 for l in r.stdout.splitlines() if l.startswith("ERROR"))
    return ids, errors


# Run in the snapshot worktree: load ITS conftest, probe its inheritance rule on
# synthetic lists, then expand its real lists onto the collected ids (argv[1]).
_EFFECTIVE = r"""
import importlib.util, json, os, sys, tempfile
spec = importlib.util.spec_from_file_location("g13_conftest", "tests/conftest.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)
ids = json.load(open(sys.argv[1]))
LISTS = {"xfail": ("_load_ledger", "_ledger_path"),
         "slow_skip": ("_load_slow_skip", "_slow_skip_path"),
         "skip": ("_load_backend_skip", "_backend_skip_path")}
if not hasattr(cf, "_matches") or not hasattr(cf, "_load_ledger"):
    print(json.dumps(None)); sys.exit()
out = {"jit_rule": {}, "effective": {}}
fd, probe = tempfile.mkstemp(suffix=".txt"); os.close(fd)
open(probe, "w").write("jax tests/e.py::eager\njax-jit tests/j.py::jit\n")
for name, (load, path) in LISTS.items():
    if not hasattr(cf, load):
        out["jit_rule"][name] = "absent"; continue
    real = getattr(cf, path); setattr(cf, path, lambda: probe)
    got = getattr(cf, load)("jax-jit")
    setattr(cf, path, real)
    out["jit_rule"][name] = "inherits eager" if "tests/e.py::eager" in got else "jit rows only"
os.remove(probe)
for mode in ("jax", "torch", "jax-jit", "torch-jit"):
    sets = {n: (getattr(cf, l)(mode) if hasattr(cf, l) else set()) for n, (l, _) in LISTS.items()}
    hit = {n: {i for i in ids if cf._matches(i, e)} for n, e in sets.items() if e}
    rec = {n: len(hit.get(n, ())) for n in LISTS}
    rec["union"] = len(set().union(*hit.values()))
    out["effective"][mode] = rec
print(json.dumps(out))
"""


def _effective(wt, existing):
    fd, ids = tempfile.mkstemp(suffix=".json", dir="/var/tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(existing, fh)
    try:
        env = dict(os.environ, PYTHONPATH=wt, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        r = subprocess.run([sys.executable, "-c", _EFFECTIVE, ids], cwd=wt, env=env,
                           capture_output=True, text=True, timeout=600)
    finally:
        os.remove(ids)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        print(f"effective failed in {wt}: {r.stderr[-500:]}", flush=True)
        return None


def _snapshot(sha, subject):
    """The commit to collect at: the PR's ORIGINAL head (refs/pull/N/head) when the
    subject names a PR. feat/backends is periodically rebased onto main, and only
    the rebased TIP is guaranteed consistent -- an intermediate rewritten commit
    can fail to import (e.g. 313cd254c: coords imports a helper its backend lacks).
    The PR head is what CI actually ran."""
    m = re.search(r"\(#(\d+)\)\s*$", subject)
    if m:
        ref = f"refs/g13/{m.group(1)}"
        _git("fetch", "-q", "origin", f"pull/{m.group(1)}/head:{ref}")
        got = _git("rev-parse", "--verify", "-q", ref).strip()
        if got:
            return got, f"pull/{m.group(1)}/head"
    return sha, "rewritten"


_PREV = {}


def _one(job):
    k, sha, when, subject, source = job
    wt = tempfile.mkdtemp(prefix=f"g13col{k}_", dir="/var/tmp")
    try:
        _git("worktree", "add", "-q", "--detach", wt, sha)
        existing, err1 = _collect(wt, ["tests", "--ignore-glob=tests/test_backend*.py"])
        bmods = sorted(str(p.relative_to(wt)) for p in Path(wt, "tests").glob("test_backend*.py"))
        if sha[:10] in _PREV:  # --reuse-backend: the backend modules, as last collected
            backend, err2 = [None] * _PREV[sha[:10]]["backend_modules"], 0
        else:
            backend, err2 = _collect(wt, bmods) if bmods else ([], 0)
        eff = _effective(wt, existing)
    finally:
        _git("worktree", "remove", "--force", wt)
    led = {m: [] for m in MODES}
    for path in LEDGERS:
        for m, ents in _entries(sha, path).items():
            led[m] += ents
    skip = _entries(sha, SKIP)
    rec = {
        "sha": sha[:10], "date": when[:10], "subject": subject, "source": source,
        "existing": len(existing), "backend_modules": len(backend),
        "collect_errors": err1 + err2,
        "ledgered": {}, "entries": {m: len(led[m]) for m in MODES},
        "skipped": {},
        "effective": eff and eff["effective"], "jit_rule": eff and eff["jit_rule"],
    }
    for m in MODES:
        rec["ledgered"][m] = sum(1 for n in existing if any(_matches(n, e) for e in led[m]))
        rec["skipped"][m] = sum(1 for n in existing if any(_matches(n, e) for e in skip[m]))
    print(f"{rec['date']} {rec['sha']} existing {rec['existing']:5d} backend {rec['backend_modules']:5d} "
          f"ledgered torch {rec['ledgered']['torch']:5d} jax {rec['ledgered']['jax']:5d} "
          f"(entries {rec['entries']['torch']}/{rec['entries']['jax']}) errors {rec['collect_errors']} [{source}] "
          f"jit-union torch {eff and eff['effective']['torch-jit']['union']} jax {eff and eff['effective']['jax-jit']['union']}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse-backend", action="store_true",
                    help="take backend_modules from the existing suite.json (same sha)")
    ap.add_argument("--limit", type=int, default=None, help="first N snapshots only (testing)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--incremental", action="store_true",
                    help="reuse the records in --out (same PR head, or same subject for a "
                    "rewritten commit) and collect only the snapshots it lacks")
    ap.add_argument("--tip", action="store_true",
                    help="end with REF's tip (the suite at the time of writing), also when "
                    "the tip did not touch a ledger")
    args = ap.parse_args()
    if args.reuse_backend and OUT.exists():
        _PREV.update({r["sha"]: r for r in json.loads(OUT.read_text())})

    def key(source, subject):  # a PR head is stable across rebases; else the subject
        return source if source.startswith("pull/") else subject

    old = {}
    if args.incremental and Path(args.out).exists():
        old = {key(r["source"], r["subject"]): r for r in json.loads(Path(args.out).read_text())}
    shas = _git("log", REF, "--reverse", "--format=%H\t%aI\t%s", "--", *LEDGERS, SKIP).splitlines()
    if args.tip:
        tip = _git("log", "-1", "--format=%H\t%aI\t%s", REF).strip()
        if tip.split("\t")[0] != shas[-1].split("\t")[0]:
            shas.append(tip)
    jobs = []
    for k, line in enumerate(shas):  # fetch serially (ref locks), collect in parallel
        sha, when, subject = line.split("\t", 2)
        if k == len(shas) - 1 and args.tip:
            # the tip itself, not its PR's head: a PR branched before the
            # previous merges lacks their tests
            jobs.append((k, sha, when, subject, "tip"))
            continue
        m = re.search(r"\(#(\d+)\)\s*$", subject)
        if m and f"pull/{m.group(1)}/head" in old:  # reused: no fetch
            jobs.append((k, sha, when, subject, f"pull/{m.group(1)}/head"))
            continue
        snap, source = _snapshot(sha, subject)
        jobs.append((k, snap, when, subject, source))
    jobs = jobs[: args.limit]
    todo = [j for j in jobs if key(j[4], j[3]) not in old]
    print(f"{len(jobs)} snapshots, {len(jobs) - len(todo)} reused, collecting {len(todo)}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        got = {key(j[4], j[3]): r for j, r in zip(todo, ex.map(_one, todo))}
    recs = [old.get(key(j[4], j[3])) or got[key(j[4], j[3])] for j in jobs]
    Path(args.out).write_text(json.dumps(recs, indent=1) + "\n")
    print(f"wrote {len(recs)} commits -> {args.out}")


if __name__ == "__main__":
    main()
