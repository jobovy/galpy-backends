# Re-running the speed figures on another GPU (e.g. an H100)

The measure scripts write their machine-dependent timings into each figure's
`data/` directory, stamped with the GPU name; the plots are byte-stable from
those JSONs. To measure on another machine without touching the TITAN Xp data,
use a separate clone on a branch and fold the series in afterwards.

## Setup (once)

```bash
git clone https://github.com/jobovy/galpy-jax.git && cd galpy-jax
git checkout -b gpu-h100
git clone -b feat/backends https://github.com/jobovy/galpy.git ../galpy-fb
(cd ../galpy-fb && python setup.py build_ext --inplace)   # needs GSL
python -m venv ../venv && . ../venv/bin/activate
pip install "jax[cuda12]" torch torchode torchdiffeq diffrax equinox \
    numpy scipy matplotlib astropy array-api-compat pytest
python -c "import jax, torch; print(jax.devices(), torch.cuda.get_device_name(0))"
```

H100 is sm_90: current `jax[cuda12]` and CUDA-12 torch wheels both work (the
TITAN Xp, sm_61, needs older pins; see `paper/figures/_setup/`).

## Run (GPU otherwise idle; ~6-8 h for everything)

```bash
tools/rerun_figures_gpu.sh ../galpy-fb ../logs-h100            # g2 g5 g6 g17 shared g12 g15 bench render
# the newer figures measure their GPU series directly:
PYTHONPATH=../galpy-fb:. python paper/figures/gV2_vertical_inverse_speed/make.py --devices gpu
PYTHONPATH=../galpy-fb:. python paper/figures/gD3_spherical_df_speed/make.py --gpu
PYTHONPATH=../galpy-fb:. python -m paper.figures.gS5_streamspray_speed.make --measure gpu
PYTHONPATH=../galpy-fb:. python -m paper.figures.gS4_streamdf_speed.make --measure gpu
git add -A && git commit -m "H100 timings" && git push -u origin gpu-h100
```

Then I fold the `gpu-h100` branch's GPU series into the figures as extra lines.

## What to expect

galpy computes in float64. The TITAN Xp is a consumer card that runs float64
at 1/32 of its float32 rate (~0.38 TFLOPS), so the large-N GPU timings are
float64-bound. An H100 runs float64 at ~34 TFLOPS (non-tensor) with ~6x the
memory bandwidth: expect the large-N GPU lines to drop by one to two orders of
magnitude, and the small-N (launch/dispatch-bound) end to change little.
