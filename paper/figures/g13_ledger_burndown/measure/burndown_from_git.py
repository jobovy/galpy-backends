"""Reconstruct the backend xfail-ledger burndown time series from git history.

The ledger files (tests/backend_xfail.txt + tests/backend_slow_skip.txt) are
version-controlled on v2 (formerly feat/backends), so every commit that touches them is a
burndown data point -- no manual logging needed, and the series is exactly
reproducible from the repo. The ledger SIZE is the migration progress metric
(see plan Pillar 1 / memory ledger-milestone-burndown): it shrinks as each
subsystem (potentials, orbit, action-angle, df, ...) is ported and its tests
move xfail->pass at a milestone review.

Run at each milestone to refresh data/burndown.json:
    GALPY_REPO=/path/to/galpy GALPY_REF=origin/v2 \
        python measure/burndown_from_git.py

Output (data/burndown.json): one record per ledger-touching commit, oldest
first, with per-backend and per-file ledgered counts. make.py reads this and
renders byte-stably.
"""

import json
import os
import subprocess
from pathlib import Path

REPO = os.environ.get("GALPY_REPO", "/home/bovy/Repos/galpy")
REF = os.environ.get("GALPY_REF", "origin/v2")
LEDGERS = ["tests/backend_xfail.txt", "tests/backend_slow_skip.txt"]
# eager modes, and (from the --jit CI rows on) the suite under jit
_MODES = ("jax", "torch", "jax-jit", "torch-jit")
OUT = Path(__file__).resolve().parents[1] / "data" / "burndown.json"


def _git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True, check=True
    ).stdout


def _counts_at(sha, path):
    """Per-backend, per-file ledgered counts in <path> at commit <sha>."""
    try:
        text = _git("show", f"{sha}:{path}")
    except subprocess.CalledProcessError:
        return {}  # file did not exist at this commit
    per_file = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        backend, nodeid = parts[0], parts[1]
        if backend not in _MODES:
            continue
        fname = nodeid.split("::", 1)[0].replace("tests/", "")
        per_file.setdefault(fname, dict.fromkeys(_MODES, 0))[backend] += 1
    return per_file


def main():
    # commits touching either ledger file on REF, oldest -> newest
    shas = _git(
        "log", REF, "--reverse", "--format=%H\t%aI\t%s", "--", *LEDGERS
    ).splitlines()
    records = []
    for line in shas:
        sha, when, subject = line.split("\t", 2)
        # merge per-file counts across both ledger files (xfail + slow-skip)
        merged = {}
        for path in LEDGERS:
            for fname, c in _counts_at(sha, path).items():
                m = merged.setdefault(fname, dict.fromkeys(_MODES, 0))
                for k in _MODES:
                    m[k] += c[k]
        jax_total = sum(c["jax"] for c in merged.values())
        torch_total = sum(c["torch"] for c in merged.values())
        jit_totals = {
            f"{k.replace('-', '_')}_total": sum(c[k] for c in merged.values())
            for k in ("jax-jit", "torch-jit")
        }
        records.append(
            {
                "sha": sha[:10],
                "date": when[:10],
                "subject": subject,
                "jax_total": jax_total,
                "torch_total": torch_total,
                "total": jax_total + torch_total,
                **jit_totals,
                "per_file": merged,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(records)} data points -> {OUT}")
    for r in records:
        print(f"  {r['date']} {r['sha']}: jax {r['jax_total']:>4}  torch {r['torch_total']:>4}")


if __name__ == "__main__":
    main()
