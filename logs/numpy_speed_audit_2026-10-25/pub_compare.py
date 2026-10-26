import sys, pickle, collections
m, f, b = (pickle.load(open(p, "rb")) for p in sys.argv[1:4])
keys = sorted(set(f) & set(b))
vals = [k for k in keys if f[k][1:] != b[k][1:]]
print(f"{len(keys)} public records; values feat != new: {len(vals)}")
for k in vals[:10]: print("  VALUE", k, f[k], b[k])
c = collections.Counter((m.get(k, ("?",))[0], f[k][0], b[k][0]) for k in keys if f[k][0] != b[k][0])
print("type changes feat -> new (main, feat, new): count")
for t, n in c.most_common(): print("  ", t, n)
same_as_main = sum(1 for k in keys if f[k][0] != b[k][0] and m.get(k, ("?",))[0] == b[k][0])
print("of type changes, new matches main's type:", same_as_main, "/", sum(c.values()))
