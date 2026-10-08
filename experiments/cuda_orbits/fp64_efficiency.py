"""Achieved FP64 rate of the CUDA kernel vs the GPU's FP64 peak.

nvprof's flop_count_dp fails on this box (driver 580 + Pascal: "Internal
profiling error 4211:27") and Nsight Compute does not support sm_61, so the DP
work is counted in two ways:

  (a) ALGORITHMIC ops/orbit, measured exactly by flopcount.cpp (the same
      headers on the host with a counting double; each + - * / sqrt pow log exp
      counts 1);
  (b) DP-PIPE INSTRUCTIONS/orbit = (a) weighted by the sm_61 SASS fast-path
      length of each libdevice op (sass_ops.cu; cuobjdump): div = MUFU.RCP64H +
      7 DFMA + 1 DMUL; pow ~85, log ~30, exp ~18, sqrt ~10 DP instructions;
      add/mul/cmp 1.  Upper bound for add/mul (nvcc fuses some into DFMA).

Usage: python fp64_efficiency.py data/titan/diag/*.json

Peak: TITAN Xp 30 SM x 4 FP64 FMA/clk x 1.582 GHz = 1.90e11 DP instr/s
(= 0.380 TFLOP/s counting an FMA as 2).  H100 SXM: 132 SM x 64 x 1.98 GHz
= 1.67e13 DP instr/s (33.5 TFLOP/s non-tensor) = 88x the TITAN.
"""
import json
import sys

# flopcount.cpp output, per orbit (300 hiN orbits); exact = GSL/libm forms,
# fast = -DCO_FASTGAM -DCO_FASTPOW (flopcount_fast).  Host counts include ~3
# divisions/step of constants the GPU compiler folds (negligible).
COUNTS = {
    ("exact", "faithful"): dict(add=677945, mul=553525, div=423605, sqrt=3738, pow=28267,
                                log=10415, exp=3472, cmp=182697),
    ("exact", "lazy"): dict(add=530027, mul=419493, div=361037, sqrt=3219, pow=24117,
                            log=8858, exp=2953, cmp=155994),
    ("fast", "faithful"): dict(add=459224, mul=884371, div=46835, sqrt=17620, pow=495,
                               log=6941, exp=3471, cmp=178847),
    ("fast", "lazy"): dict(add=343979, mul=700913, div=40559, sqrt=15026, pow=495,
                           log=5904, exp=2952, cmp=152718),
}
NFEV = {"faithful": 3471.5, "lazy": 2952.7}
# sm_61 SASS fast-path DP-pipe instruction counts (sass_ops.cu): div = MUFU.RCP64H
# + 7 DFMA + 1 DMUL; sqrt ~10; pow ~85; log ~30; exp ~18; add/mul/cmp 1 (upper
# bound: some add/mul pairs fuse into one DFMA)
W = dict(add=1, mul=1, div=8, sqrt=10, pow=85, log=30, exp=18, cmp=1)
PEAK = {"TITAN Xp": 30 * 4 * 1.582e9, "H100": 132 * 64 * 1.98e9}


def per_orbit(arith, mode):
    a = COUNTS[(arith, mode)]
    return sum(v for k, v in a.items() if k != "cmp"), sum(v * W[k] for k, v in a.items())


if __name__ == "__main__":
    for arith in ("exact", "fast"):
        for mode in ("faithful", "lazy"):
            alg, dp = per_orbit(arith, mode)
            print(f"{arith:5s} {mode:8s}: {alg/1e6:.2f} M algorithmic ops/orbit, "
                  f"~{dp/1e6:.2f} M DP-pipe instr/orbit")
    for path in sys.argv[1:]:
        d = json.load(open(path))
        peak = PEAK["H100"] if "H100" in d["device"] else PEAK["TITAN Xp"]
        arith = "fast" if "fast" in d["variant"] else "exact"
        if "f32" in d["variant"] or d.get("fixed_nstep"):
            continue  # counts are for float64 adaptive runs only
        tag = d["variant"] + "".join(f"+{k}" for k in ("refill", "sort", "pinned") if d.get(k))
        for r in d["results"]:
            alg, dp = per_orbit(arith, r["mode"])
            s = r["nfev_mean"] / NFEV[r["mode"]]
            k = r["kernel_ms"] * 1e-3
            rate = r["N"] * dp * s / k
            print(f"{tag:28s} {r['mode']:8s} N={r['N']:8d}: kernel {r['kernel_ms']/r['N']*1e3:7.2f} us/orbit"
                  f"  ~{rate:.3g} DP instr/s = {100*rate/peak:5.1f}% of {d['device'].split(' (')[0]} FP64 peak"
                  f"  ({r['N']*alg*s/k/1e9:.0f} G alg. ops/s)")
