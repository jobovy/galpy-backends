"""Combine the measured wall times (figure JSONs) with the censused work per unit
(misc_costs.json, orbit_costs.json, orbit_stats.json, torch_costs.json,
c_dop853.json) into achieved FLOP rates, bandwidths and fractions of peak.
Pure arithmetic, no measurement. Prints the numbers ROOFLINE.md quotes and writes
roofline_tables.json.

  python roofline_tables.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
FIG = os.path.join(REPO, "paper", "figures")
W = 20.0  # flops per transcendental (convention; see ROOFLINE.md)

# ---- peaks (vendor specs; see ROOFLINE.md for sources) -----------------------
PEAK = {
    "xeon_core": dict(flops=16 * 2.3e9, bw=None),  # AVX2 2 FMA/cycle x 4 lanes x 2; base clock
    "xeon_node": dict(flops=36 * 16 * 2.3e9, bw=2 * 76.8e9),
    "epyc_core": dict(flops=16 * 2.4e9, bw=None),
    "titan": dict(flops=3840 * 2 * 1.582e9 / 32, bw=547.7e9),
    "h100": dict(flops=34e12, bw=3.35e12),
}


def load(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


misc = load(HERE, "misc_costs.json")
oc = load(HERE, "orbit_costs.json")
tc = load(HERE, "torch_costs.json")
cd = load(HERE, "c_dop853.json")
rows = []


def row(bench, series, dev, N, t, flop_eq, bytes_, peak, note="", bytes_hi=None, fp64=None):
    """flop_eq / bytes_ per unit; t = wall time for N units. fp64: FP64 arithmetic
    only (add/mul/div + W*transcendental; no compare/select/integer ops)."""
    fp64 = flop_eq if fp64 is None else fp64
    r = dict(fp64_gflops=fp64 * N / t / 1e9, fp64_per_unit=fp64, flop_eq_per_unit=flop_eq,
             bytes_per_unit=bytes_, bytes_hi_per_unit=bytes_hi,bench=bench, series=series, device=dev, N=N, t=t, per_unit_s=t / N,
             gflops=flop_eq * N / t / 1e9, gbs=bytes_ * N / t / 1e9 if bytes_ else None,
             gbs_hi=bytes_hi * N / t / 1e9 if bytes_hi else None, note=note)
    pk = PEAK[peak]
    r["pct_flops"] = 100 * r["gflops"] * 1e9 / pk["flops"]
    r["pct_fp64"] = 100 * r["fp64_gflops"] * 1e9 / pk["flops"]
    r["pct_bw"] = 100 * r["gbs"] * 1e9 / pk["bw"] if (pk["bw"] and r["gbs"]) else None
    r["pct_bw_hi"] = 100 * r["gbs_hi"] * 1e9 / pk["bw"] if (pk["bw"] and r["gbs_hi"]) else None
    rows.append(r)
    return r


def marginal(Ns, ts, i=-1, j=-2):
    return (ts[i] - ts[j]) / (Ns[i] - Ns[j])


# ================================================================ g2 ==========
g2 = load(FIG, "g2_throughput", "data", "g2_throughput.json")
g2h = load(FIG, "g2_throughput", "data", "h100", "g2_throughput.json")
pj = misc["g2_plummer_FR_jax"]
fe = pj["flops_xla"] + W * pj["transcendentals"]
tb = misc["g2_plummer_FR_torch_eager"]["bytes_per_point"]
N = 10**7
s, sh = g2["series"], g2h["series"]
row("g2", "numpy (1 core)", "Xeon", N, s["numpy (CPU)"][-1], fe, tb + 40, "xeon_core",
    "bytes: 8 unfused ops (144 B) + grid (40 B), est.")
row("g2", "numpy (1 core)", "EPYC", N, sh["numpy (CPU)"][-1], fe, tb + 40, "epyc_core")
row("g2", "jax jit", "TITAN", N, s["jax (GPU)"][-1], fe, 24, "titan")
row("g2", "jax jit", "H100", N, sh["jax (GPU)"][-1], fe, 24, "h100")
row("g2", "torch eager", "TITAN", N, s["torch (GPU)"][-1], fe, tb, "titan")
row("g2", "torch eager", "H100", N, sh["torch (GPU)"][-1], fe, tb, "h100")
Ns = g2["Ns"]
g2_marg = {
    "jax TITAN": marginal(Ns, s["jax (GPU)"]), "jax H100": marginal(Ns, sh["jax (GPU)"]),
    "torch TITAN": marginal(Ns, s["torch (GPU)"]), "torch H100": marginal(Ns, sh["torch (GPU)"]),
}
g2_marg = {k: dict(s_per_pt=v, GBs=(24 if "jax" in k else tb) / v / 1e9) for k, v in g2_marg.items()}

# ================================================================ orbits ======
def per_step(o, ncf, key):
    a, b = o["per_step_ncf10"][key], o["per_step_ncf15"][key]
    return a + (ncf - 10) * (b - a) / 5.0


orb = {}
for sname, files in (("Dopri8", (("_shared", "orbit_speed", "data", "jax_gpu_hiN.json"),
                                 ("_shared", "orbit_speed", "data", "h100", "jax_gpu_hiN.json"))),
                     ("Dopri5", (("g12_inbackend_orbit_gpu", "data", "jax_dopri5_gpu_hiN.json"),
                                 ("g12_inbackend_orbit_gpu", "data", "h100",
                                  "jax_dopri5_gpu_hiN.json")))):
    o = oc[sname]
    for dev, f, peak in (("TITAN", files[0], "titan"), ("H100", files[1], "h100")):
        d = load(FIG, *f)
        res = d["results"]
        Nmax = res[-1]["N"]
        t = res[-1]["wall_s"]
        st = o[str(Nmax)]
        ncf = st["n_cf_rmin"]
        fe_step = per_step(o, ncf, "flop_equiv")
        bi_step = per_step(o, ncf, "bytes_ideal")
        bc_step = per_step(o, ncf, "bytes")
        S, Sm = st["S_max"], st["S_mean"]
        fp_step = (per_step(o, ncf, "arith") + per_step(o, ncf, "div")
                   + W * per_step(o, ncf, "transcendentals"))
        r = row("orbit", f"jax {sname} (vmap-of-solve)", dev, Nmax, t, S * fe_step, S * bi_step,
                peak, f"performed work: S_max={S}, n_cf={ncf}", bytes_hi=S * bc_step,
                fp64=S * fp_step)
        useful = Sm * per_step(o, 10, "flop_equiv")
        r["useful_gflops"] = useful * Nmax / t / 1e9
        r["useful_pct_flops"] = 100 * r["useful_gflops"] * 1e9 / PEAK[peak]["flops"]
        r["lockstep_waste"] = S * fe_step / useful
        # marginal per orbit between the two largest N
        Ns_ = [x["N"] for x in res]
        ts_ = [x["wall_s"] for x in res]
        m = marginal(Ns_, ts_)
        r["marginal_s_per_orbit"] = m
        r["marginal_GBs_ideal"] = S * bi_step / m / 1e9
        r["marginal_GBs_census"] = S * bc_step / m / 1e9
        r["marginal_pct_bw_ideal"] = 100 * r["marginal_GBs_ideal"] * 1e9 / PEAK[peak]["bw"]
        r["per_lane_step_ns"] = t / Nmax / S * 1e9
        r["floor_s"] = res[0]["wall_s"]
        # host-synchronised while-loop iterations per solve at small N (floor model)
        nst = 13 if sname == "Dopri8" else 6
        r["loop_iters_per_step"] = 1 + nst * (1 + (ncf + 1) + 1) + 2
        orb[(sname, dev)] = r

# torch dopri5 (eager, one shared controller)
ts_ = tc["solve"]
t_bytes = ts_["bytes_per_orbit"]
t_rhs_fe = (ts_["flops_per_orbit_per_rhs"]
            + W * sum(v for k, v in ts_["transc_per_orbit_per_rhs"].items() if k != "igamma")
            + 2 * 200.0)  # 2 native igamma per RHS at ~200 flop-eq each (est.)
t_fe = t_rhs_fe * ts_["nfev"]
for dev, f, peak in (("TITAN", ("g12_inbackend_orbit_gpu", "data", "torch_cuda.json"), "titan"),
                     ("H100", ("g12_inbackend_orbit_gpu", "data", "h100", "torch_cuda.json"),
                      "h100")):
    d = load(FIG, *f)
    res = d["results"]
    r = row("orbit", "torch dopri5 (eager, shared controller)", dev, res[-1]["N"],
            res[-1]["wall_s"], t_fe, t_bytes, peak, f"nfev={ts_['nfev']} (N=1024 count)")
    Ns_ = [x["N"] for x in res]
    tt = [x["wall_s"] for x in res]
    m = marginal(Ns_, tt)
    r["marginal_s_per_orbit"] = m
    r["marginal_GBs"] = t_bytes / m / 1e9
    r["marginal_pct_bw"] = 100 * r["marginal_GBs"] * 1e9 / PEAK[peak]["bw"]
    r["floor_s"] = res[0]["wall_s"]
    r["kernels_per_solve"] = ts_["kernels"]
    r["floor_us_per_kernel"] = res[0]["wall_s"] / ts_["kernels"] * 1e6
    orb[("torch", dev)] = r

# C dop853_c: same arithmetic per RHS as the jax RHS at n_cf=10, + the Dopri8
# per-stage RK overhead from the census; nfev from the Python port.
o8 = oc["Dopri8"]
rhs10 = oc["rhs"]["n_cf=10 (r=1)"]["flop_equiv"]
rk_over = (o8["per_step_ncf10"]["flop_equiv"] - 13 * rhs10) / 13.0
c_nfev = cd["nfev_nt16"]["mean"]
c_fe = c_nfev * (rhs10 + rk_over)
c1 = cd["one_thread_N3000"]["per_orbit_ms"] * 1e-3
row("orbit", "C dop853_c, 1 thread (this note's check)", "Xeon", 3000, c1 * 3000, c_fe, None,
    "xeon_core", f"nfev={c_nfev:.0f}; arithmetic assumed = jax RHS")
cn = load(FIG, "_shared", "orbit_speed", "data", "galpy_c_cpu_hiN.json")["results"][-1]
row("orbit", "C dop853_c as measured (OpenMP, all 72 threads)", "Xeon node", cn["N"],
    cn["wall_s"], c_fe, None, "xeon_node")

# ================================================================ gD3 =========
d3 = load(FIG, "gD3_spherical_df_speed", "data", "gD3_spherical_df_speed.json")
d3h = load(FIG, "gD3_spherical_df_speed", "data", "h100", "gD3_spherical_df_speed.json")
for name in ("isotropicHernquistdf", "eddingtondf (NFW)", "osipkovmerrittdf", "kingdf"):
    c = misc[f"gD3_{name}"]
    fe = c["flops_xla"] + W * c["transcendentals"]
    fp = c["arith"] + c["div"] + W * c["transcendentals"]
    i8 = d3["Ns"].index(10**7)
    i8h = d3h["Ns"].index(10**8)
    sm = d3["sample"][name]
    smh = d3h["sample"][name]
    row("gD3", f"{name} jax jit", "TITAN", 10**7, sm["jax-gpu-jit"][i8], fe, c["bytes_ideal"],
        "titan", bytes_hi=c["bytes"], fp64=fp)
    row("gD3", f"{name} jax jit", "H100", 10**8, smh["jax-gpu-jit"][i8h], fe, c["bytes_ideal"],
        "h100", bytes_hi=c["bytes"], fp64=fp)
    row("gD3", f"{name} jax jit (4 cores)", "Xeon", 10**7, sm["jax-jit"][i8], fe, c["bytes_ideal"],
        "xeon_core", "% of ONE core; 4 cores used", bytes_hi=c["bytes"], fp64=fp)
    row("gD3", f"{name} numpy (1 core, scipy spline icdf)", "Xeon", 10**7, sm["numpy"][i8], fe,
        None, "xeon_core", "different algorithm: jax op count used as the work proxy", fp64=fp)

# ================================================================ gV2 =========
v2 = load(FIG, "gV2_vertical_inverse_speed", "data", "gV2_vertical_inverse_speed.json")
v2h = load(FIG, "gV2_vertical_inverse_speed", "data", "h100", "gV2_vertical_inverse_speed.json")
c = misc["gV2_xvFreqs_one_torus"]
fe = c["flops_xla"] + W * c["transcendentals"]
fp = c["arith"] + c["div"] + W * c["transcendentals"]
fe10 = c["flops_xla"] + 10 * c["transcendentals"]
ev, evh = v2["eval"], v2h["eval"]
i6 = v2["Ns"].index(10**6)
i7 = v2h["Ns"].index(10**7)
for ser, dev, t, N, peak in (
        ("jax jit", "TITAN", ev["jax (GPU, jit)"][i6], 10**6, "titan"),
        ("jax jit", "H100", evh["jax (GPU, jit)"][i7], 10**7, "h100"),
        ("torch eager", "TITAN", ev["torch (GPU)"][i6], 10**6, "titan"),
        ("torch eager", "H100", evh["torch (GPU)"][i7], 10**7, "h100"),
        ("jax jit (1 thread)", "Xeon", ev["jax (CPU, jit)"][i6], 10**6, "xeon_core"),
        ("numpy (1 thread)", "Xeon", ev["numpy (CPU)"][i6], 10**6, "xeon_core")):
    r = row("gV2", ser, dev, N, t, fe, c["bytes_ideal"], peak, bytes_hi=c["bytes"], fp64=fp)
    r["gflops_w10"] = fe10 * N / t / 1e9
    r["pct_flops_w10"] = 100 * r["gflops_w10"] * 1e9 / PEAK[peak]["flops"]
v2_marg = {"TITAN": marginal(v2["Ns"][: i6 + 1], ev["jax (GPU, jit)"][: i6 + 1]),
           "H100": marginal(v2h["Ns"][: i7 + 1], evh["jax (GPU, jit)"][: i7 + 1])}
v2_marg = {k: dict(s_per_pt=v, gflops=fe / v / 1e9, gflops_w10=fe10 / v / 1e9)
           for k, v in v2_marg.items()}

# ---------------------------------------------------------------- print -----
def f(x, p=3):
    return "-" if x is None else (f"{x:.{p}g}")


print(f"{'bench':6s} {'series':48s} {'dev':9s} {'N':>9s} {'t[s]':>9s} {'GF/s':>8s} {'%pk':>6s} "
      f"{'GB/s':>8s} {'%bw':>6s} {'GB/s_hi':>8s} {'%bw_hi':>6s} {'fp64GF':>8s} {'%fp64':>6s}")
for r in rows:
    print(f"{r['bench']:6s} {r['series'][:48]:48s} {r['device']:9s} {r['N']:9.3g} {r['t']:9.4g} "
          f"{f(r['gflops']):>8s} {f(r['pct_flops'],2):>6s} {f(r['gbs']):>8s} {f(r['pct_bw'],2):>6s} "
          f"{f(r['gbs_hi']):>8s} {f(r['pct_bw_hi'],2):>6s} {f(r['fp64_gflops']):>8s} {f(r['pct_fp64'],2):>6s}")
print("g2 marginal", json.dumps(g2_marg))
print("gV2 marginal", json.dumps(v2_marg))
for k, r in orb.items():
    print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()
              if kk not in ("bench", "series", "device", "note")})
print("C: nfev", c_nfev, "flop-eq/orbit", c_fe, "rhs10", rhs10, "rk_over/RHS", rk_over)
print("torch: flop-eq/RHS", t_rhs_fe, "per orbit", t_fe, "bytes/orbit", t_bytes)
with open(os.path.join(HERE, "roofline_tables.json"), "w") as fh:
    json.dump(dict(rows=rows, g2_marginal=g2_marg, gV2_marginal=v2_marg,
                   orbit={f"{a}|{b}": v for (a, b), v in orb.items()},
                   C=dict(nfev=c_nfev, flop_eq_per_orbit=c_fe), peaks=PEAK, W=W), fh, indent=1)
