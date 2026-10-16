"""PR statistics for the process section, from the GitHub API (gh CLI).

    python paper/ms/stats/pr_stats.py [--until YYYY-MM-DD]

Every PR opened on jobovy/galpy since 2026-06-01 counts as part of the project
unless it is listed in EXCLUDED below (reviewed by hand, 2026-10-16):
* backend PRs: head backend/*, or base feat/backends*, backend/* or another
  backend sub-branch;
* the three-dimensional variational equations (base feat/variational3d);
* PRs to main that came out of the backend work (fixes to the numpy/C code,
  test and CI infrastructure for the backend matrix).
A stack member is a project PR whose base is another project PR's head branch,
or whose head branch is the base of another project PR; size = additions +
deletions of the merged project PRs.
"""

import argparse
import collections
import json
import subprocess

import numpy

EXCLUDED = {
    "bots (pre-commit.ci, dependabot, notebook-output updates)": [
        890, 1078, 1244, 1456, 1632, 915, 958, 995, 1030, 1207, 1240, 1287,
        961, 1035, 1125, 1135, 1321, 1329, 1502],
    "release, packaging and unrelated CI": [
        1036, 1037, 1041, 1042, 1044, 1409, 1470, 1472, 1582, 1583, 1590, 1696,
        1326, 955, 957, 1638],
    "unrelated features and maintenance": [
        889, 924, 971, 976, 980, 994, 1031, 1058, 1061, 1062, 1107, 1108, 1345,
        1349, 1501, 1588],
    "another galpy project (not yet landed)": [
        1468, 1527, 1553, 1560, 1561, 1010, 1519,
        1309, 1313, 1314, 1317, 1318, 1320, 1325, 1335, 1339, 1340, 1342, 1347,
        1350, 1351, 1353, 1361, 1362, 1373, 1376, 1378, 1379, 1384, 1385, 1387,
        1389, 1395, 1396, 1405, 1412, 1413, 1414, 1415, 1419, 1420, 1421, 1423,
        1426, 1427, 1429, 1430, 1434, 1510, 1515, 1516, 1517, 1522, 1523, 1524,
        1526, 1545, 1562, 1563],
}

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-06-01")
ap.add_argument("--until", default=None)
args = ap.parse_args()
prs = json.loads(subprocess.run(
    ["gh", "pr", "list", "--repo", "jobovy/galpy", "--state", "all", "--limit", "5000",
     "--search", f"created:>={args.since}",
     "--json", "number,createdAt,baseRefName,headRefName,mergedAt,additions,deletions"],
    capture_output=True, text=True, check=True).stdout)
if args.until:
    prs = [p for p in prs if p["createdAt"][:10] <= args.until]
excluded = {n: why for why, ns in EXCLUDED.items() for n in ns}
proj = [p for p in prs if p["number"] not in excluded]
merged = [p for p in proj if p["mergedAt"]]


def kind(p):
    if (p["headRefName"].startswith("backend/") or p["baseRefName"].startswith("feat/backends")
            or p["baseRefName"].startswith("backend/") or p["number"] in (1034, 1195, 1433)):
        return "backend"
    if p["baseRefName"] == "feat/variational3d" or p["number"] == 891:
        return "variational3d"
    return "main"


_INTEGRATION = {"main", "feat/backends", "feat/backends-develop", "feat/variational3d"}
heads = {p["headRefName"] for p in proj} - _INTEGRATION
bases = collections.Counter(p["baseRefName"] for p in proj)
stacked = [p for p in proj if p["headRefName"] not in _INTEGRATION
           and (p["baseRefName"] in heads or bases[p["headRefName"]])]
size = numpy.array([p["additions"] + p["deletions"] for p in merged])
print(f"PRs since {args.since}: {len(prs)}; excluded {sum(1 for p in prs if p['number'] in excluded)}"
      f" ({dict(collections.Counter(excluded[p['number']] for p in prs if p['number'] in excluded))})")
print(f"project PRs opened {len(proj)}, merged {len(merged)};",
      dict(collections.Counter(kind(p) for p in proj)), "merged:", dict(collections.Counter(kind(p) for p in merged)))
print(f"merged size median {numpy.median(size):.0f}, quartiles {numpy.percentile(size, 25):.0f}, {numpy.percentile(size, 75):.0f}")
print(f"stack members {len(stacked)}")
pm = collections.defaultdict(collections.Counter)
for p in proj:
    pm[p["createdAt"][:7]][kind(p)] += 1
print("per month:", {m: dict(c) for m, c in sorted(pm.items())})
inv = collections.Counter(p["createdAt"][:7] for p in prs if excluded.get(p["number"]) == "another galpy project (not yet landed)")
print("other-project PRs per month:", dict(sorted(inv.items())))
