# PR-4: differentiable make_smoothing_spline — VALIDATED recipe (2026-07-10)

User chose exact-reconstruction (A): replicate scipy's make_smoothing_spline + GCV exactly, differentiably. **POC PROVEN: 0.0 parity vs scipy** (max|diff| at query = 0.00e+00, coeffs = 0.00e+00, 3 trials n=15/20/25). Worktree `/var/tmp/track-smoother` (branch `feat/streamtrack-diff-smoother`, off feat/backends 735bec52, built).

## The key fact
scipy make_smoothing_spline solves a BANDED natural-spline system **`(X + λ·wE)·c = y`** (scipy/interpolate/_bsplines.py). X, wE depend ONLY on x, w (y-independent); λ from GCV. The solve is LINEAR in y → re-solve with backend y (λ from GCV probe on numpy y) = exact + differentiable.

## VALIDATED reconstruction (copy into streamTrack.py helpers)
```python
def _smoothing_spline_banded(x, w):  # y-independent; x ascending, w>0, n>=5
    from scipy.interpolate import BSpline
    from scipy.interpolate._bsplines import _coeff_of_divided_diff
    n = x.shape[0]; t = numpy.r_[[x[0]]*3, x, [x[-1]]*3]
    Xb = BSpline.design_matrix(x, t, 3).toarray()
    X = numpy.zeros((5, n))
    for i in range(1, 4): X[i, 2:-2] = Xb[i:i-4, 3:-3][numpy.diag_indices(n-4)]
    X[1,1]=Xb[0,0]; X[2,:2]=((x[2]+x[1]-2*x[0])*Xb[0,0], Xb[1,1]+Xb[1,2]); X[3,:2]=((x[2]-x[0])*Xb[1,1], Xb[2,2])
    X[1,-2:]=(Xb[-3,-3],(x[-1]-x[-3])*Xb[-2,-2]); X[2,-2:]=(Xb[-2,-3]+Xb[-2,-2],(2*x[-1]-x[-2]-x[-3])*Xb[-1,-1]); X[3,-2]=Xb[-1,-1]
    wE = numpy.zeros((5, n))
    wE[2:,0]=_coeff_of_divided_diff(x[:3])/w[:3]; wE[1:,1]=_coeff_of_divided_diff(x[:4])/w[:4]
    for j in range(2,n-2): wE[:,j]=(x[j+2]-x[j-2])*_coeff_of_divided_diff(x[j-2:j+3])/w[j-2:j+3]
    wE[:-1,-2]=-_coeff_of_divided_diff(x[-4:])/w[-4:]; wE[:-2,-1]=_coeff_of_divided_diff(x[-3:])/w[-3:]
    wE*=6; return t, X, wE
# lam = float(np.asarray(_compute_optimal_gcv_parameter(X, wE, y_np.reshape(-1,1), w)).ravel()[0])
# banded(5,n)->dense(n,n): A_dense[i,j] = (X+lam*wE)[2+i-j, j] for |i-j|<=2
# c = xp.linalg.solve(xp.asarray(A_dense), y_backend)              # natural coeffs, DIFFERENTIABLE in y
# natural->B-spline (linear in c, scipy eqns 2.2.10):
#   c0,c1,cm0,cm1 = c[0],c[1],c[-1],c[-2]
#   c_ = concat([ c0*(t[5]+t[4]-2t[3])+c1, c0*(t[5]-t[3])+c1, c[1:-1], cm0*(t[-4]-t[-6])+cm1, cm0*(2t[-4]-t[-5]-t[-6])+cm1 ])
# eval: B_q = BSpline.design_matrix(clip(q,t[3],t[-4]), t, 3).toarray(); fitted = xp.asarray(B_q) @ c_
```

## IMPLEMENTATION plan (streamTrack.py `_smooth_series` dual-path)
1. Compute mask/order from `as_numpy(y)` (structural, numpy); xv=numpy, sv=numpy weights, **yv=y[mask][order] (BACKEND if y backend)**.
2. GCV path (s_user is None, smoothing_factor==1.0) + backend yv → the reconstruction above. REPLICATE the yscale normalization EXACTLY: scipy path is `make_smoothing_spline(xv, yv/yscale, w=1/((sv/yscale)**2))` then `BSpline(t, c*yscale, 3)`. yscale=float(nanstd(as_numpy(yv))) (stop_gradient scalar). Since solve is linear, un-normalize by `*yscale` cancels correctly. λ from GCV on as_numpy(yv/yscale).
3. Return a `_DiffBSpline(t, c_, xp)` callable (`__call__(q)` = xp.asarray(design_matrix(clip(q),t,3).toarray()) @ c_) so caller's `spl(tp_grid)` is differentiable. effective_s = float(as_numpy(sum((resid/sv)**2))) (numpy scalar, hyperparameter).
4. DUAL PATH: numpy y → scipy make_smoothing_spline (BYTE-IDENTICAL). Also the <5-valid-bins linear fallback + s_user/smoothing_factor!=1 UnivariateSpline paths.
5. **Scoped FOLLOW-UP:** the s_user (reuse) + smoothing_factor!=1 paths use UnivariateSpline (FITPACK, target-SSR s). Differentiable version = root-find λ for target SSR then same penalized solve. Defer OR handle. Check whether default streamTrack iteration reuses s (→ needs it) or re-runs GCV each pass.
6. Once smoother differentiable: PR-4 wires it → PR-3's backend binned means flow through → backend track. Then _fit_one_pass cov_fine/eigh (in-place → xp.at[].set()) + Phase A-C (PR-5).

## TESTS + REVIEW
tests/test_backend_streamTrack.py: backend GCV smoother EXACT-parity vs scipy make_smoothing_spline (0.0 / ~1e-12); grad-of-fit-vs-FD; numpy path byte-identical. **RUN ADVERSARIAL REVIEW** [[adversarial-review-complex-migrations]]: boundary q (extrapolation), <5 bins, NaN bins, torch linalg.solve, yscale edge (std=0), exact-parity across many seeds/n.
