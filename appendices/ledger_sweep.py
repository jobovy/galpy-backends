#!/usr/bin/env python3
"""Ledger-staleness sweep from a completed backend-tests run's own artifacts.

The xfail ledger (``tests/backend_xfail.txt``) records nodeids expected to fail
under a forced backend. Entries go stale as the migration lands, and the header
of that file is explicit about how to find out:

    Download a completed run's artifacts and take (ledgered AND passed), where
    passed = a <testcase> with no skipped/failure/error child. That covers all
    44 shards at once.

    Do NOT re-measure on a dev box instead. The backend matrix is python 3.14
    and shards test_orbit into 5 pytest-split groups; a local py3.13 unsplit run
    of the same files disagreed on exactly one entry.

This script is that procedure. It is deliberately *not* a local test runner.

Usage
-----
    python ledger_sweep.py <run-id> [--repo jobovy/galpy] [--ledger-ref REF]
    python ledger_sweep.py --dir <already-downloaded-artifact-dir>

The ledger is read from a git ref (default origin/v2), not
from a working tree -- see ledger_text().

``<run-id>`` is a backend-tests workflow run (``gh run list --workflow
backend-tests.yml``). Artifacts are named ``backend-junit-<backend>-<shard>``;
the shard files are pytest ``--junitxml`` output.

Output
------
Per ledger key (``jax``, ``torch``, ``jax-jit``, ...):

    STALE       ledgered but the run PASSED it   -> candidate for removal
    CONFIRMED   ledgered and the run failed it   -> still a real gap
    UNOBSERVED  ledgered but absent from the run -> the run has no such shard
                (e.g. every ``-jit`` entry: backend-tests has no --jit dimension)

Never report STALE for an entry that is merely UNOBSERVED -- that is the whole
reason the three buckets are separate.
"""

from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

GALPY = "/home/bovy/Repos/galpy"
DEFAULT_REF = "origin/v2"
LEDGER_PATH = "tests/backend_xfail.txt"


def ledger_text(ref, path=None):
    """The ledger's contents, read from git rather than a working tree.

    A checkout is whatever branch it happens to be on -- the main clone sits on
    an old feat/backends (857 entries) while the work was on feat/backends-develop (16;
    both are now v2).
    A worktree is worse still: another job may be mid-edit, which is exactly how
    an earlier count missed three lines. So read the ref unless told otherwise.
    """
    if path is not None:
        with open(path) as fh:
            return fh.read()
    return subprocess.run(
        ["git", "-C", GALPY, "show", f"{ref}:{LEDGER_PATH}"],
        check=True, capture_output=True, text=True,
    ).stdout


def read_ledger(text):
    """{backend_key: {nodeid, ...}} from the ledger text (comments stripped)."""
    out = collections.defaultdict(set)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0]].add(parts[1])
    return out


def download(run_id, repo, dest):
    """Fetch every backend-junit artifact of one run into ``dest``."""
    subprocess.run(
        ["gh", "run", "download", str(run_id), "--repo", repo, "--dir", dest,
         "--pattern", "backend-junit-*"],
        check=True,
    )
    return dest


def outcome(testcase):
    """pytest junit outcome. `is not None` matters: an empty <failure> is falsy."""
    if testcase.find("failure") is not None:
        return "failed"
    if testcase.find("error") is not None:
        return "error"
    skipped = testcase.find("skipped")
    if skipped is not None:
        # a ledgered test that ran and failed is recorded as skipped/pytest.xfail
        return "xfail" if skipped.get("type") == "pytest.xfail" else "skipped"
    return "passed"


def nodeid(testcase):
    """Reconstruct 'tests/test_x.py::test_y' from classname/name."""
    cls = testcase.get("classname", "")
    # 'tests.test_orbit' -> 'tests/test_orbit.py'; nested classes keep the last
    # dotted component as the class, which the ledger does not use.
    return f"{cls.replace('.', '/')}.py::{testcase.get('name')}"


def backend_of(path):
    """'backend-junit-<backend>-<shard>.xml' -> '<backend>'."""
    base = os.path.basename(path)
    if not base.startswith("backend-junit-"):
        return None
    rest = base[len("backend-junit-") :]
    for key in ("jax-jit", "torch-jit", "jax", "torch", "numpy"):
        if rest.startswith(key + "-"):
            return key
    return None


def collect(root, jit=False):
    """{backend: {nodeid: outcome}} over every junit xml under ``root``.

    ``jit=True`` re-keys an observed ``jax``/``torch`` to ``jax-jit``/``torch-jit``.
    This is NOT cosmetic: `backend-tests.yml` names its artifact
    ``backend-junit-${{ matrix.BACKEND }}-${{ SHARD_ID }}`` with **no jit
    marker**, so a `workflow_dispatch(jit=true)` run is indistinguishable from an
    eager one by filename. Without this flag the traced outcomes get compared
    against the EAGER ledger keys, and every eager entry the traced run happens
    to pass is reported STALE -- confidently, and wrongly.
    """
    seen = collections.defaultdict(dict)
    n_files = 0
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".xml"):
                continue
            path = os.path.join(dirpath, fn)
            # download-artifact puts each artifact in its own directory, so the
            # backend can be encoded in the directory name rather than the file.
            backend = backend_of(path) or backend_of(dirpath)
            if backend is None or os.path.getsize(path) == 0:
                continue
            n_files += 1
            try:
                tree = ET.parse(path)
            except ET.ParseError:  # truncated upload
                continue
            key = f"{backend}-jit" if jit and not backend.endswith("-jit") else backend
            for tc in tree.getroot().iter("testcase"):
                seen[key][nodeid(tc)] = outcome(tc)
    return seen, n_files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id", nargs="?", help="backend-tests workflow run id")
    ap.add_argument("--repo", default="jobovy/galpy")
    ap.add_argument("--ledger-ref", default=DEFAULT_REF,
                    help="git ref to read the ledger from (default: %(default)s)")
    ap.add_argument("--ledger", help="read a file instead of the ref (discouraged)")
    ap.add_argument("--dir", help="use already-downloaded artifacts instead")
    ap.add_argument("--jit", action="store_true",
                    help="the run was workflow_dispatch(jit=true); re-key jax/torch "
                         "to jax-jit/torch-jit (the artifact NAME does not say)")
    args = ap.parse_args(argv)

    if not args.run_id and not args.dir:
        ap.error("give a run id or --dir")

    ledger = read_ledger(ledger_text(args.ledger_ref, args.ledger))
    total = sum(len(v) for v in ledger.values())
    src = args.ledger or f"{args.ledger_ref}:{LEDGER_PATH}"
    print(f"ledger: {total} entries across {len(ledger)} keys ({src})")

    tmp = None
    root = args.dir
    if root is None:
        tmp = tempfile.mkdtemp(prefix="ledger_sweep_")
        root = download(args.run_id, args.repo, tmp)
    seen, n_files = collect(root, jit=args.jit)
    print(f"parsed {n_files} junit files; backends observed: {sorted(seen)}"
          + ("   [--jit: re-keyed to <backend>-jit]" if args.jit else "") + "\n")

    grand = collections.Counter()
    for key in sorted(ledger):
        obs = seen.get(key, {})
        # conftest's `_load_backend_nodeids` makes "<b>-jit" INHERIT "<b>"'s eager
        # entries, so a traced failure already covered by an eager entry is not a
        # gap. Mirror that here for coverage. But staleness is NOT symmetric: an
        # eager entry describes eager behaviour, so a traced PASS must never mark
        # it stale -- only literal "<b>-jit" entries can be pruned on traced
        # evidence. Getting this backwards would prune the eager ledger from a
        # traced run.
        base = key[: -len("-jit")] if key.endswith("-jit") else None
        inherited = ledger.get(base, set()) if base else set()
        stale, confirmed, unobserved = [], [], []
        for nid in sorted(ledger[key] | inherited):
            own = nid in ledger[key]
            if nid not in obs:
                if own:
                    unobserved.append(nid)
            elif obs[nid] == "passed":
                if own:
                    stale.append(nid)
            else:
                confirmed.append((nid, obs[nid]))
        grand["stale"] += len(stale)
        grand["confirmed"] += len(confirmed)
        grand["unobserved"] += len(unobserved)
        print(
            f"[{key}] {len(ledger[key])} ledgered: "
            f"{len(stale)} STALE, {len(confirmed)} CONFIRMED, "
            f"{len(unobserved)} UNOBSERVED"
        )
        for nid in stale:
            print(f"    STALE      {nid}")
        for nid, out in confirmed:
            print(f"    CONFIRMED  {nid}  ({out})")
        for nid in unobserved:
            print(f"    UNOBSERVED {nid}")
        print()

    print(f"TOTAL: {dict(grand)}")
    if grand["unobserved"]:
        print(
            "NOTE: UNOBSERVED is not evidence of staleness -- the run simply has "
            "no shard covering it (backend-tests has no --jit dimension, so every "
            "'-jit' entry lands here)."
        )
    if tmp:
        print(f"\nartifacts kept at {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
