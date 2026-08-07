#!/usr/bin/env python3
"""Rewrite ``tests/backend_slow_skip.txt`` from a real measurement.

Entries land in that file when a test is too slow to finish inside the per-test
cap under a forced backend. They go stale as the migration vectorizes the code
underneath them, so the way to prune is to lift them, run the suite in the CI
shard context, and keep only what still does not fit.

This applies the measurement, with two rules that exist because the failure mode
of getting it wrong is a red CI on the integration branch:

* **Margin.** A test that tips over the cap records FAILED, not skipped. So an
  entry is only un-deferred when it measured comfortably inside the cap --
  ``--keep-frac`` of it, 0.8 by default. Entries in the margin band stay, with
  their measured time written into the file.
* **Never silently.** Every retained entry gets its measured time appended as a
  comment, so the next reader sees a number instead of an assumption, and a
  block whose entries all pass loses its comment header too rather than being
  left dangling.

Usage
-----
    python slow_skip_prune.py --orig tests/backend_slow_skip.txt \\
        --junit '/tmp/orbslow_g*.xml' --key jax \\
        --match '^tests/test_orbits?\\.py::' [--cap 300] [--keep-frac 0.8] \\
        [--out NEW_FILE]

Without ``--out`` it prints the summary only, so you can look before writing.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET


def outcome(tc):
    """pytest junit outcome. ``is not None``: an empty <failure> is falsy."""
    if tc.find("failure") is not None:
        return "failed"
    if tc.find("error") is not None:
        return "error"
    sk = tc.find("skipped")
    if sk is not None:
        return "xfail" if sk.get("type") == "pytest.xfail" else "skipped"
    return "passed"


def nodeid(tc):
    return f"{tc.get('classname', '').replace('.', '/')}.py::{tc.get('name')}"


def measure(pattern):
    """{nodeid: (outcome, seconds)} over every junit xml matching ``pattern``."""
    seen = {}
    files = [f for f in glob.glob(pattern) if os.path.getsize(f)]
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:  # truncated upload / killed mid-write
            print(f"  WARNING: unparseable, skipped: {path}", file=sys.stderr)
            continue
        for tc in root.iter("testcase"):
            seen[nodeid(tc)] = (outcome(tc), float(tc.get("time", 0.0)))
    return seen, files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig", required=True, help="the slow-skip file to rewrite")
    ap.add_argument("--junit", required=True, help="glob of junit xml from the run")
    ap.add_argument("--key", required=True, help="backend key, e.g. jax / torch-jit")
    ap.add_argument("--match", required=True, help="regex the nodeid must match")
    ap.add_argument("--cap", type=float, default=300.0, help="per-test timeout (s)")
    ap.add_argument("--keep-frac", type=float, default=0.8,
                    help="keep entries slower than this fraction of the cap")
    ap.add_argument("--out", help="write the new file here (default: dry run)")
    args = ap.parse_args(argv)

    keep_above = args.keep_frac * args.cap
    seen, files = measure(args.junit)
    print(f"parsed {len(files)} junit files, {len(seen)} nodeids")
    print(f"cap {args.cap:g}s; un-defer only below {keep_above:g}s\n")

    want = re.compile(args.match)
    lines = open(args.orig).read().splitlines()

    def targeted(line):
        parts = line.split(None, 1)
        return len(parts) == 2 and parts[0] == args.key and want.search(parts[1])

    dropped, kept_slow, kept_margin, kept_unseen = [], [], [], []
    out, i = [], 0
    while i < len(lines):
        if not lines[i].startswith("#"):
            out.append(lines[i])
            i += 1
            continue
        c0 = i
        while i < len(lines) and lines[i].startswith("#"):
            i += 1
        comment, e0 = lines[c0:i], i
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#"):
            i += 1
        entries, new_entries = lines[e0:i], []
        for e in entries:
            if not targeted(e):
                new_entries.append(e)
                continue
            nid = e.split(None, 1)[1]
            if nid not in seen:
                kept_unseen.append(nid)
                new_entries.append(e)
                continue
            res, secs = seen[nid]
            if res == "passed" and secs < keep_above:
                dropped.append((secs, nid))
            elif res == "passed":
                kept_margin.append((secs, nid))
                new_entries.append(f"{e}  # {secs:.0f}s -- inside the cap, but within "
                                   f"{100 * (1 - args.keep_frac):.0f}% of it")
            else:
                kept_slow.append((secs, nid, res))
                new_entries.append(f"{e}  # {secs:.0f}s ({res})")
        if new_entries:
            out.extend(comment)
            out.extend(new_entries)
        elif entries:
            print(f"  block emptied, comment dropped: {comment[0][:70]}")
        else:
            out.extend(comment)

    print(f"\nUN-DEFERRED (< {keep_above:g}s): {len(dropped)}")
    for s, n in sorted(dropped, reverse=True)[:10]:
        print(f"    {s:7.1f}s  {n}")
    if len(dropped) > 10:
        print(f"    ... and {len(dropped) - 10} more")
    print(f"\nKEPT, margin band ({keep_above:g}-{args.cap:g}s): {len(kept_margin)}")
    for s, n in sorted(kept_margin, reverse=True):
        print(f"    {s:7.1f}s  {n}")
    print(f"\nKEPT, still too slow: {len(kept_slow)}")
    for s, n, r in sorted(kept_slow, reverse=True):
        print(f"    {s:7.1f}s  {n}  ({r})")
    if kept_unseen:
        print(f"\nKEPT, NOT OBSERVED in this run: {len(kept_unseen)}")
        for n in kept_unseen[:10]:
            print(f"              {n}")
        print("  (absence is not evidence of speed -- no shard covered these)")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"\nwrote {args.out}")
    else:
        print("\n(dry run; pass --out to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
