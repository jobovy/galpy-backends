# Track F — Pdf.4 quasiisothermaldf backend migration (#1071)

## Approach
Backend-migrate the qdf moment engine (`_vmomentdensity`), the 6 pv-projections
(pvRvT/pvTvz/pvRvz/pvR/pvT/pvz), `tilt`, `_vmomentsurfaceMCIntegrand`, and route
`surfacemass_z`'s fixed_quad to `galpy.backend.quadrature.fixed_quad`. Builds on
#1064 (already migrated `__call__` evaluation + `_rg`). Resolved-namespace
`xp = get_namespace(...)`, `promote_scalars` for scalar inputs, GL-table
promotion via `xp.asarray(t)*1.0`, `list.extend`→`xp.concatenate`, 3-D `.T`
→`xp.permute_dims(x,(2,1,0))` (torch errors on >2-D `.T` under `-W error`),
array-R `ndim`-guard with `xp.stack`. Bundles the [[ctypes-bridge-coords-forced-backend-leak]]
Staeckel u0 fix (7 wrappers wrapped in `with use("numpy", force=True)`).

## Dead-ends / gotchas
- **The "py3.14 issue" was a red herring.** User reported a py3.14-only backend
  failure. Set up a standard cpython-3.14.6 conda env (`gp314`, np 2.5.1, jax
  0.10.2) — the isolated `qdf.density(1.,0.1)` under forced jax PASSES on 3.14
  (1.7686925483 == numpy). The real failure is a **jax eager-XLA timeout** in
  the moment engine that only surfaces on the py3.14 *backend* shard (the only
  shard running `--backend=jax`). Not version-specific.
- **Root cause = migration-flips-numpy-fallthrough-to-eager-slow.** Migrating
  `_vmomentdensity` to `xp.*` makes `lz` a jax array, which activates #1064's
  `_rg` backend branch: `xp.where(indx, potential.rl(self._pot, lzsafe), spline)`.
  `xp.where` eagerly evaluates BOTH branches, so *every* quadrature node ran a
  full-array **brentq root-find** (`potential.rl`) — even though
  `indx = (lz>Lzmax)&(lz<Lzmin)` is EMPTY for a valid Lz grid (Lzmin<Lzmax; the
  numpy-array path splines everywhere too, its `out[indx]=rl` iterating
  `range(0)`). Grid-scale pv-projections thus blew past the 300 s per-test
  timeout under forced jax → pytest-timeout records FAILED (overriding xfail) →
  red shard. (torch's faster eager completed them: 32 min, 30 XPASS.)

## Results
- **`_rg` fix**: backend branch now splines everywhere (matches numpy-array
  effective behavior; `rl` branch was dead). AD-safe, no dead code, byte-identical
  numpy. density 15.6→10.3 s under jax. Removes the eager brentq.
- **ledger**: moved the 6 jax tests that still exceed 300 s (grid pv-projections
  + interpolated sampleV: pvRvT_staeckel[_diffngl], pvRvz_staeckel_diffngl,
  pvTvz_staeckel[_diffngl], sampleV_interpolate) from `backend_xfail.txt` →
  `backend_slow_skip.txt`. torch keeps them xfail (completes them).
- **coverage**: patch-missing was empty; the earlier `_rg` guard I tried
  introduced dead `xp.where`+`rl` lines (3143-3146) — the spline-only rewrite
  removes them. numpy+backend suite covers all diff lines.
- numpy path byte-identical throughout (only `is_backend_array` branches change).

## Follow-ups (deferred)
- **torch ledger prune**: 29 qdf/pv2qdf Staeckel moment torch entries XPASS on
  #1071's CI (my migration fixed them) → prune in a follow-up milestone regen.
  List saved. jax stays mostly xfail (moment values differ / still slow).
- Vectorize/jit-friendly moment engine → un-defer the 6 jax slow_skip tests.
- Pdf.4 done → next: Pdf.3 disk DFs or Pdf.5 streams.

## PR state (2026-07-05)
- #1071 head 84b1b7ff (feat/qdf-backend), CI re-running. Needs USER to set
  auto-merge. On feat/backends (post-#1068 base, disjoint from qdf).
- #1070 (Adiabatic C-native actions) rebased onto post-#1068 feat/backends
  (7143945b) to fix codecov/project (stale-base line-shift of the flagged
  actionAngleStaeckel.c 2514-2520 angle-wrap lines); auto-merge ARMED (SQUASH).
