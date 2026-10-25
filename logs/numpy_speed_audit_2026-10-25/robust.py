"""Per-job-normalized per-test slowdown audit (tip vs main CI durations)."""
import collections, math, re, statistics, sys
mj, pj, md, pd = sys.argv[1:5]
DUR = re.compile(r"Z (\d+\.\d+)s (call|setup|teardown)\s+(\S.*?)\s*$")
def jobs(p):
    return {l.split("\t", 1)[1].strip(): l.split("\t", 1)[0] for l in open(p) if "\t" in l}
def parse(path):
    d = collections.Counter()
    for l in open(path, errors="replace"):
        m = DUR.search(l)
        if m:
            d[m.group(3)] += float(m.group(1))
    return d
M, P = jobs(mj), jobs(pj)
per_test = collections.defaultdict(list)  # test -> [(norm ratio, a, b, scale, job)]
print(f"{'job':70s} {'tests':>5s} {'scale':>6s}  (median tip/main ratio of tests >= 0.5 s on main)")
for name in sorted(M):
    if name not in P:
        continue
    dm, dp = parse(f"{md}/{M[name]}.log"), parse(f"{pd}/{P[name]}.log")
    common = [t for t in dm if t in dp and dm[t] >= 0.5]
    if len(common) < 5:
        continue
    scale = statistics.median(dp[t] / dm[t] for t in common)
    print(f"{name[:70]:70s} {len(common):5d} {scale:6.2f}")
    for t in common:
        per_test[t].append((dp[t] / dm[t] / scale, dm[t], dp[t], scale, name))
rows = []
for t, v in per_test.items():
    g = math.exp(sum(math.log(x[0]) for x in v) / len(v))
    excess = sum(b - s * a for _, a, b, s, _ in v) / len(v)
    rows.append((g, excess, len(v), min(x[0] for x in v), statistics.median(x[1] for x in v), statistics.median(x[2] for x in v), t))
print("\nFLAGGED: normalized slowdown >= 1.4x in EVERY job that runs it, and >= 2 s per run beyond the job's scale")
for g, ex, n, mn, a, b, t in sorted(rows, key=lambda r: -r[1]):
    if mn >= 1.4 and ex >= 2.0:
        print(f"  {g:5.2f}x (min {mn:4.2f}x, {n} job{'s' if n > 1 else ''})  main {a:7.1f}s -> tip {b:7.1f}s  +{ex:6.1f}s  {t[:100]}")
print("\nborderline: >= 1.25x in every job, >= 1 s")
for g, ex, n, mn, a, b, t in sorted(rows, key=lambda r: -r[1]):
    if (1.25 <= mn < 1.4 and ex >= 1.0) or (mn >= 1.4 and 1.0 <= ex < 2.0):
        print(f"  {g:5.2f}x (min {mn:4.2f}x, {n} jobs)  main {a:7.1f}s -> tip {b:7.1f}s  +{ex:6.1f}s  {t[:100]}")
print(f"\n{len(rows)} tests compared (>= 0.5 s on main)")
