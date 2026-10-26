import sys, pickle
a = pickle.load(open(sys.argv[1], "rb")); b = pickle.load(open(sys.argv[2], "rb"))
diff = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print(f"{len(a)} vs {len(b)} records; {len(diff)} differ")
for k in diff[:40]: print("  ", k, "\n     base:", str(a.get(k))[:140], "\n     new: ", str(b.get(k))[:140])
