"""PR statistics for the process section, from the GitHub API (gh CLI).

    python paper/ms/stats/pr_stats.py [--until YYYY-MM-DD]

Definitions:
* a backend PR has a head branch backend/* or a base feat/backends* or
  backend/* (a PR stacked on another backend PR);
* a main PR has base main and is not a backend PR;
* a stack member is a backend PR whose base is another backend PR's head
  branch, or whose head branch is the base of another backend PR;
* size = additions + deletions of the merged backend PRs.
"""

import argparse
import collections
import json
import subprocess

import numpy

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-06-04")
ap.add_argument("--until", default=None)
args = ap.parse_args()
prs = json.loads(subprocess.run(
    ["gh", "pr", "list", "--repo", "jobovy/galpy", "--state", "all", "--limit", "5000",
     "--search", f"created:>={args.since}",
     "--json", "number,createdAt,baseRefName,headRefName,mergedAt,additions,deletions"],
    capture_output=True, text=True, check=True).stdout)
if args.until:
    prs = [p for p in prs if p["createdAt"][:10] <= args.until]


def backend(p):
    return (p["headRefName"].startswith("backend/") or p["baseRefName"].startswith("feat/backends")
            or p["baseRefName"].startswith("backend/"))


b = [p for p in prs if backend(p)]
merged = [p for p in b if p["mergedAt"]]
main = [p for p in prs if p["baseRefName"] == "main" and not backend(p)]
heads = {p["headRefName"] for p in b}
bases = collections.Counter(p["baseRefName"] for p in b)
stacked = [p for p in b if p["baseRefName"] in heads or bases[p["headRefName"]]]
size = numpy.array([p["additions"] + p["deletions"] for p in merged])
per_month = collections.Counter(p["createdAt"][:7] for p in b)
print(f"backend PRs opened {len(b)}, merged {len(merged)}; main PRs {len(main)}")
print(f"merged size median {numpy.median(size):.0f}, quartiles {numpy.percentile(size, 25):.0f}, {numpy.percentile(size, 75):.0f}")
print(f"stack members {len(stacked)}")
print("per month:", dict(sorted(per_month.items())))
