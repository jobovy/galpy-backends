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


def _one(job):
    k, sha, when, subject, source = job
    wt = tempfile.mkdtemp(prefix=f"g13col{k}_", dir="/var/tmp")
    try:
        _git("worktree", "add", "-q", "--detach", wt, sha)
        existing, err1 = _collect(wt, ["tests", "--ignore-glob=tests/test_backend*.py"])
        bmods = sorted(str(p.relative_to(wt)) for p in Path(wt, "tests").glob("test_backend*.py"))
        backend, err2 = _collect(wt, bmods) if bmods else ([], 0)
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
    }
    for m in MODES:
        rec["ledgered"][m] = sum(1 for n in existing if any(_matches(n, e) for e in led[m]))
        rec["skipped"][m] = sum(1 for n in existing if any(_matches(n, e) for e in skip[m]))
    print(f"{rec['date']} {rec['sha']} existing {rec['existing']:5d} backend {rec['backend_modules']:5d} "
          f"ledgered torch {rec['ledgered']['torch']:5d} jax {rec['ledgered']['jax']:5d} "
          f"(entries {rec['entries']['torch']}/{rec['entries']['jax']}) errors {rec['collect_errors']} [{source}]", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    shas = _git("log", REF, "--reverse", "--format=%H\t%aI\t%s", "--", *LEDGERS, SKIP).splitlines()
    jobs = []
    for k, line in enumerate(shas):  # fetch serially (ref locks), collect in parallel
        sha, when, subject = line.split("\t", 2)
        snap, source = _snapshot(sha, subject)
        jobs.append((k, snap, when, subject, source))
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(_one, jobs))
    OUT.write_text(json.dumps(recs, indent=1) + "\n")
    print(f"wrote {len(recs)} commits -> {OUT}")


if __name__ == "__main__":
    main()
