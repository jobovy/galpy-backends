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

Peak: TITAN Xp 30 SM x 4 FP64 FMA/clk x 1.582 GHz = 1.90e11 DP instr/s
(= 0.380 TFLOP/s counting an FMA as 2).  H100 SXM: 132 SM x 64 x 1.98 GHz
= 1.67e13 DP instr/s (33.5 TFLOP/s non-tensor) = 88x the TITAN.
"""
import json
import sys

# flopcount.cpp output, per orbit, lazy dense output (300 orbits of the hiN ICs)
ALG_LAZY = dict(add=529781, mul=419247, div=360299, sqrt=3219, pow=24117, log=8858,
                exp=2953, cmp=155499)
ALG_FAITHFUL = dict(add=677699, mul=553279, div=422867, sqrt=3738, pow=28267, log=10415,
                    exp=3472, cmp=182203)
NFEV = {"lazy": 2952.7, "faithful": 3471.5}
W = dict(add=1, mul=1, div=8, sqrt=10, pow=85, log=30, exp=18, cmp=1)
PEAK = {"TITAN Xp": 30 * 4 * 1.582e9, "H100": 132 * 64 * 1.98e9}


def per_orbit(mode):
    a = ALG_LAZY if mode == "lazy" else ALG_FAITHFUL
    alg = sum(v for k, v in a.items() if k != "cmp")
    dp = sum(v * W[k] for k, v in a.items())
    return alg, dp


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/cuda_gpu_hiN.json"
    d = json.load(open(path))
    peak = PEAK["H100"] if "H100" in d["device"] else PEAK["TITAN Xp"]
    for mode in ("lazy", "faithful"):
        alg, dp = per_orbit(mode)
        print(f"{mode}: {alg/1e6:.2f} M algorithmic DP ops/orbit, ~{dp/1e6:.1f} M DP-pipe "
              f"instructions/orbit ({alg/NFEV[mode]:.0f} / {dp/NFEV[mode]:.0f} per force eval incl. integrator share)")
    print(f"{d['device']}: FP64 peak {peak:.3g} DP instr/s")
    for r in d["results"]:
        alg, dp = per_orbit(r["mode"])
        # scale by this run's nfev (ICs differ slightly from the 300 counted)
        s = r["nfev_mean"] / NFEV[r["mode"]]
        k = r["kernel_ms"] * 1e-3
        rate_dp = r["N"] * dp * s / k
        rate_alg = r["N"] * alg * s / k
        print(f"  {r['mode']:8s} N={r['N']:8d}: kernel {k:8.3f}s  {r['fevals_per_s_kernel']:.3g} force evals/s  "
              f"~{rate_dp:.3g} DP instr/s = {100*rate_dp/peak:5.1f}% of peak  "
              f"(algorithmic {rate_alg/1e9:.1f} GFLOP/s)")
