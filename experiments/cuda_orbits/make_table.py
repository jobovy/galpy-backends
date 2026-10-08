"""Markdown comparison table: this CUDA port vs the existing orbit-speed JSONs (hiN, nt=16)."""
import json
import os

F = "/home/bovy/Repos/galpy-jax/paper/figures"
S = f"{F}/_shared/orbit_speed/data"
G = f"{F}/g8_galax_orbit_speed/data"
HERE = os.path.dirname(os.path.abspath(__file__))


def ld(p, mode=None):
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return {r["N"]: r["wall_s"] for r in d["results"] if mode is None or r.get("mode") == mode}


cols = {
    "CUDA TITAN Xp (lazy dense)": ld(f"{HERE}/data/cuda_gpu_hiN.json", "lazy"),
    "CUDA TITAN Xp (galpy-faithful)": ld(f"{HERE}/data/cuda_gpu_hiN.json", "faithful"),
    "CUDA H100 (lazy)": ld(f"{HERE}/data/h100/cuda_gpu_hiN.json", "lazy"),
    "galpy C 1 thread [galpy_c_cpu_1cores_hiN.json]": ld(f"{S}/galpy_c_cpu_1cores_hiN.json"),
    "galpy C 16 threads [galpy_c_cpu_16cores_hiN.json]": ld(f"{S}/galpy_c_cpu_16cores_hiN.json"),
    "galpy C full node, 72 OpenMP threads [galpy_c_cpu_hiN.json]": ld(f"{S}/galpy_c_cpu_hiN.json"),
    "jax Dopri8 TITAN": ld(f"{S}/jax_gpu_hiN.json"),
    "jax Dopri8 H100": ld(f"{S}/h100/jax_gpu_hiN.json"),
    "galax TITAN": ld(f"{G}/galax_gpu_hiN.json"),
    "galax H100": ld(f"{G}/h100/galax_gpu_hiN.json"),
}
C1 = 2.22e-3  # s/orbit, galpy dop853_c 1 thread (chain-3 measurement; 16 thr 0.174 ms, 72 thr 0.102 ms)
Ns = sorted({N for c in cols.values() for N in c})
names = ["galpy C 1 thread (2.22 ms/orbit x N)"] + list(cols)
print("| N | " + " | ".join(names) + " |")
print("|---" * (len(names) + 1) + "|")
for N in Ns:
    row = [f"{C1 * N:.3g}"] + [f"{c[N]:.3g}" if N in c else "" for c in cols.values()]
    print(f"| {N:.0e} | " + " | ".join(row) + " |")
cu = cols["CUDA TITAN Xp (lazy dense)"]
print("\nspeed-up of CUDA TITAN (lazy) over:")
for name, c in [(names[0], {N: C1 * N for N in Ns})] + list(cols.items())[1:]:
    s = [f"N={N:.0e}: {c[N] / cu[N]:.3g}x" for N in Ns if N in c and N in cu]
    if s:
        print(f"  {name}: " + ", ".join(s))
