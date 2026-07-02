# Sampling interpolator → backend Spline swaps (for the Track F sampling PR, task #130)

Verified 2026-07-02. These make the sampling inverse-CDF interpolators
backend-queryable. They are **inert until the sampling RNG is on the backend**
(queries currently pass `numpy.random.uniform`, so the numpy path is unchanged).
Do them together in the sampling PR (#130), NOT scattered into family PRs.

## Byte-identity verified at the interpolator level (query 1000 numpy pts):

1. **kingdf `_icmf`** (kingdf.py:84) `InterpolatedUnivariateSpline(x,y,k=1)` → `Spline1D(x,y,k=1)`
   - **BYTE-IDENTICAL** (0.0, incl. end-to-end `_sample_r` seeded). `Spline1D` already imported.
   - Exact edit: replace `interpolate.InterpolatedUnivariateSpline(...)` with `Spline1D(...)` (same 3 args, k=1).

2. **pvr** (sphericaldf.py:820 `_make_pvr_interpolator` return) `scipy.interpolate.RectBivariateSpline(x,y,z.T,kx=1,ky=3)` → wrap in `Spline2D(spl=rbs)`
   - **BYTE-IDENTICAL** (0.0, grid=False). Query at _sample_v:694 unchanged (Spline2D.__call__ numpy → rbs.ev).
   - Lives in the BASE sphericaldf (not a family). Import `Spline2D` from ..backend.interpolate.

3. **constantbeta `_coseta_icmf_interp`** (constantbetadf.py:158) `interp1d(coseta_cmf, cosetas, kind='linear'?, fill_value='extrapolate')` → NOT a spline
   - `Spline1D(k=1)` is **NOT byte-identical** to `interp1d(linear)` (~1 ULP, 2.2e-16) — spline-basis eval vs linear-interp formula.
   - REQUIRES adding an `interp1d`-linear / `numpy.interp`-style mode to `Spline1D` (byte-identical numpy `numpy.interp`, backend `xp.interp`/searchsorted) BEFORE swapping. Then constantbeta (and any interp1d site) can adopt it.
   - Note the `fill_value='extrapolate'` → Spline1D `ext=0` (finite extrapolation) — match it.

## Also in #130 (the actual backend enablement, without which the above are inert):
- `galpy.backend.random` RNG shim (numpy.random | jax.random.uniform(key) | torch.rand(device=)); numpy keeps numpy.random for byte-identity.
- Thread `key=`/`seed=`/`rng=` through sample()/_sample_r/_sample_position_angles/_sample_v/_sample_velocity_angles/_sample_eta (+ osipkovmerritt numpy.random.choice → xp.where(U<0.5,-1,1)).
- Drop the sampling-path `as_numpy` output casts (sphericaldf.py:628,696) so data stays on-device (removes ~10-12 of the 14 sphericaldf as_numpy — answers the "why so much as_numpy" review point).
- Device-anchor `_vmax_at_r` scalar limits (722-723) + `numpy.log10(r/scale)` host const (695) per gpu-device-anchoring memo; GPU-verify on TITAN Xp.
- Orbit-accepts-backend (sample():595) is a separate gating sub-task; `return_orbit=False` sidesteps.
