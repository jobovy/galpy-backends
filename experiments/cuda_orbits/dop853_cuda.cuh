// galpy's dop853 (util/leung_dop853.c), one orbit, dim = 6, state in registers.
// Line-by-line port: same initial-step estimate, same error norm and step
// control, same dense output at the requested times.  Differences:
//   * no malloc / SIGINT handling;
//   * LAZY=true skips the 3 extra dense-output evaluations (and the rcont
//     assembly) on accepted steps that contain no requested output time.
//     Those k's are overwritten before reuse, so the step sequence and the
//     outputs are bit-identical; only the cost changes (galpy's C always does
//     them).
//   * outputs are converted to cylindrical (rect_to_cyl_galpy) when saved,
//     instead of in a second pass.
#pragma once
#include "galpy_cuda.cuh"
#include "dop853_coeffs.inc"

#define DIM 6

// Force evaluation used by the integrator.  By default evalRectDeriv is inlined
// at all 17 call sites (the kernel is then ~45k SASS instructions = 700 kB, far
// larger than the instruction caches).  With CO_NOINLINE_FORCE the device code
// calls ONE out-of-line copy (co_dev_accel, defined in cuda_orbits.cu).
#if defined(CO_NOINLINE_FORCE) && defined(__CUDACC__)
struct CoAcc3 {
  real ax, ay, az;
};
__device__ __noinline__ CoAcc3 co_dev_accel(real x, real y, real z);
#endif
HD void co_deriv(const real *q, real *a, const PotSet &P) {
#if defined(CO_NOINLINE_FORCE) && defined(__CUDA_ARCH__)
  CoAcc3 r = co_dev_accel(q[0], q[1], q[2]);
  a[0] = q[3];
  a[1] = q[4];
  a[2] = q[5];
  a[3] = r.ax;
  a[4] = r.ay;
  a[5] = r.az;
#else
  evalRectDeriv(q, a, P);
#endif
}
#ifdef CO_FLOAT32
#define UROUND R_(1.2e-7)
#else
#define UROUND 2.3e-16
#endif

HD real co_sign(real x, real y) { return (y > R_(0.0)) ? fabs(x) : -fabs(x); }
HD real co_max(real a, real b) { return (a > b) ? a : b; }
HD real co_min(real a, real b) { return (a < b) ? a : b; }

HD void co_save(const real *y, real *out) {
  real v[DIM];
#pragma unroll
  for (int i = 0; i < DIM; i++) v[i] = y[i];
  rect_to_cyl_galpy(v);
#pragma unroll
  for (int i = 0; i < DIM; i++) out[i] = v[i];
}

struct DopStats {
  int nfev, naccept, nreject;
};

// Integrator state carried between steps (lives in registers once inlined).
struct DopState {
  real y0[DIM], k1[DIM];
  real h, hmax, pos_neg, facold, t_current, t_old, t_old_older;
  int reject, fin, nfev, nacc, nrej;
  real *result;
};

// galpy's dop853 prologue: save t[0], k1 = f(y0), initial step-size estimate.
// S.y0 must hold the rectangular initial condition and S.result the output block.
HD void dop853_init(DopState &S, int nt, const real *t, const PotSet &P, real rtol,
                    real atol) {
  using namespace dopc;
  real *y0 = S.y0, *k1 = S.k1;
  real &h = S.h;
  int &nfev = S.nfev;
  S.facold = R_(1.0e-4);
  S.hmax = t[nt - 1] - t[0];
  S.pos_neg = co_sign(R_(1.0), S.hmax);
  real hmax = S.hmax, pos_neg = S.pos_neg;
  real k2[DIM], k3[DIM];
  int i;
  real h1, dnf, dny, sk, der2, der12, sqr;
  nfev = 0;
  S.nacc = 0;
  S.nrej = 0;
  S.reject = 0;
  S.t_current = t[0];
  S.t_old = t[0];
  S.t_old_older = t[0];
  S.fin = 0;
  co_save(y0, S.result);
  S.result += DIM;

  co_deriv(y0, k1, P);
  nfev++;
  dnf = R_(0.0);
  dny = R_(0.0);
#pragma unroll
  for (i = 0; i < DIM; i++) {
    sk = atol + rtol * fabs(y0[i]);
    sqr = k1[i] / sk;
    dnf += sqr * sqr;
    sqr = y0[i] / sk;
    dny += sqr * sqr;
  }
  h = co_sign(co_min(sqrt(dny / dnf) * R_(0.01), fabs(hmax)), pos_neg);
#pragma unroll
  for (i = 0; i < DIM; i++) k3[i] = y0[i] + h * k1[i];
  co_deriv(k3, k2, P);
  nfev++;
  der2 = R_(0.0);
#pragma unroll
  for (i = 0; i < DIM; i++) {
    sk = atol + rtol * fabs(y0[i]);
    sqr = (k2[i] - k1[i]) / sk;
    der2 += sqr * sqr;
  }
  der2 = sqrt(der2) / h;
  der12 = co_max(fabs(der2), sqrt(dnf));
  h1 = pow(R_(0.01) / der12, R_(1.0) / R_(8.0));
  h = co_sign(co_min(R_(100.0) * fabs(h), co_min(fabs(h1), fabs(hmax))), pos_neg);

}

// One attempted step of galpy's dop853 main loop (accept or reject), including
// the dense output of every requested time inside an accepted step.
// fixed_h > 0 (diagnostic only): constant step, every step accepted.
template <bool LAZY>
HD void dop853_step(DopState &S, int nt, const real *t, const PotSet &P, real rtol,
                    real atol, real fixed_h) {
  using namespace dopc;
  const real safe = R_(0.9), beta = R_(0.0);
  const real expo1 = R_(1.0) / R_(8.0) - beta * R_(0.2);
  const real facc1 = R_(1.0) / R_(0.333), facc2 = R_(1.0) / R_(6.0);
  real *y0 = S.y0, *k1 = S.k1;
  real &h = S.h, &facold = S.facold, &t_current = S.t_current, &t_old = S.t_old,
       &t_old_older = S.t_old_older;
  const real hmax = S.hmax, pos_neg = S.pos_neg;
  int &reject = S.reject, &fin = S.fin, &nfev = S.nfev, &nacc = S.nacc,
      &nrej = S.nrej;
  real *&result = S.result;
  real yy1[DIM], k2[DIM], k3[DIM], k4[DIM], k5[DIM], k6[DIM], k7[DIM], k8[DIM],
      k9[DIM], k10[DIM];
  int i;
  real hnew, sk, sqr, err, err2, erri, deno, fac, fac11;
  if (fixed_h > R_(0.0)) h = pos_neg * fixed_h;  // diagnostic: lockstep fixed steps
  h = pos_neg * co_max(fabs(h), R_(1e3) * UROUND);
#pragma unroll
  for (i = 0; i < DIM; i++) yy1[i] = y0[i] + h * a21 * k1[i];
  co_deriv(yy1, k2, P);
#pragma unroll
  for (i = 0; i < DIM; i++) yy1[i] = y0[i] + h * (a31 * k1[i] + a32 * k2[i]);
  co_deriv(yy1, k3, P);
#pragma unroll
  for (i = 0; i < DIM; i++) yy1[i] = y0[i] + h * (a41 * k1[i] + a43 * k3[i]);
  co_deriv(yy1, k4, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a51 * k1[i] + a53 * k3[i] + a54 * k4[i]);
  co_deriv(yy1, k5, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a61 * k1[i] + a64 * k4[i] + a65 * k5[i]);
  co_deriv(yy1, k6, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a71 * k1[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
  co_deriv(yy1, k7, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a81 * k1[i] + a84 * k4[i] + a85 * k5[i] + a86 * k6[i] +
                          a87 * k7[i]);
  co_deriv(yy1, k8, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a91 * k1[i] + a94 * k4[i] + a95 * k5[i] + a96 * k6[i] +
                          a97 * k7[i] + a98 * k8[i]);
  co_deriv(yy1, k9, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a101 * k1[i] + a104 * k4[i] + a105 * k5[i] +
                          a106 * k6[i] + a107 * k7[i] + a108 * k8[i] + a109 * k9[i]);
  co_deriv(yy1, k10, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a111 * k1[i] + a114 * k4[i] + a115 * k5[i] +
                          a116 * k6[i] + a117 * k7[i] + a118 * k8[i] +
                          a119 * k9[i] + a1110 * k10[i]);
  co_deriv(yy1, k2, P);
#pragma unroll
  for (i = 0; i < DIM; i++)
    yy1[i] = y0[i] + h * (a121 * k1[i] + a124 * k4[i] + a125 * k5[i] +
                          a126 * k6[i] + a127 * k7[i] + a128 * k8[i] +
                          a129 * k9[i] + a1210 * k10[i] + a1211 * k2[i]);
  t_old_older = t_old;
  t_old = t_current;
  t_current = t_current + h;
  co_deriv(yy1, k3, P);
  nfev += 11;
#pragma unroll
  for (i = 0; i < DIM; i++) {
    k4[i] = b1 * k1[i] + b6 * k6[i] + b7 * k7[i] + b8 * k8[i] + b9 * k9[i] +
            b10 * k10[i] + b11 * k2[i] + b12 * k3[i];
    k5[i] = y0[i] + h * k4[i];
  }
  err = R_(0.0);
  err2 = R_(0.0);
#pragma unroll
  for (i = 0; i < DIM; i++) {
    sk = atol + rtol * co_max(fabs(y0[i]), fabs(k5[i]));
    erri = k4[i] - bhh1 * k1[i] - bhh2 * k9[i] - bhh3 * k3[i];
    sqr = erri / sk;
    err2 += sqr * sqr;
    erri = er1 * k1[i] + er6 * k6[i] + er7 * k7[i] + er8 * k8[i] + er9 * k9[i] +
           er10 * k10[i] + er11 * k2[i] + er12 * k3[i];
    sqr = erri / sk;
    err += sqr * sqr;
  }
  deno = err + R_(0.01) * err2;
  if (deno <= R_(0.0)) deno = R_(1.0);
  err = fabs(h) * err * sqrt(R_(1.0) / (deno * (real)DIM));
  if (fixed_h > R_(0.0)) err = R_(0.5);  // diagnostic: accept every step
  fac11 = pow(err, expo1);
  fac = fac11 / pow(facold, beta);
  fac = co_max(facc2, co_min(facc1, fac / safe));
  hnew = h / fac;

  if (err <= R_(1.0)) {  // accepted
    nacc++;
    facold = co_max(err, R_(1.0e-4));
    co_deriv(k5, k4, P);
    nfev++;
    bool need_dense = !LAZY || (pos_neg * t[fin + 1] < pos_neg * t_current);
    if (need_dense) {
      real rcont1[DIM], rcont2[DIM], rcont3[DIM], rcont4[DIM], rcont5[DIM],
          rcont6[DIM], rcont7[DIM], rcont8[DIM];
#pragma unroll
      for (i = 0; i < DIM; i++) {
        rcont1[i] = y0[i];
        real ydiff = k5[i] - y0[i];
        rcont2[i] = ydiff;
        real bspl = h * k1[i] - ydiff;
        rcont3[i] = bspl;
        rcont4[i] = ydiff - h * k4[i] - bspl;
        rcont5[i] = d41 * k1[i] + d46 * k6[i] + d47 * k7[i] + d48 * k8[i] +
                    d49 * k9[i] + d410 * k10[i] + d411 * k2[i] + d412 * k3[i];
        rcont6[i] = d51 * k1[i] + d56 * k6[i] + d57 * k7[i] + d58 * k8[i] +
                    d59 * k9[i] + d510 * k10[i] + d511 * k2[i] + d512 * k3[i];
        rcont7[i] = d61 * k1[i] + d66 * k6[i] + d67 * k7[i] + d68 * k8[i] +
                    d69 * k9[i] + d610 * k10[i] + d611 * k2[i] + d612 * k3[i];
        rcont8[i] = d71 * k1[i] + d76 * k6[i] + d77 * k7[i] + d78 * k8[i] +
                    d79 * k9[i] + d710 * k10[i] + d711 * k2[i] + d712 * k3[i];
      }
#pragma unroll
      for (i = 0; i < DIM; i++)
        yy1[i] = y0[i] + h * (a141 * k1[i] + a147 * k7[i] + a148 * k8[i] +
                              a149 * k9[i] + a1410 * k10[i] + a1411 * k2[i] +
                              a1412 * k3[i] + a1413 * k4[i]);
      co_deriv(yy1, k10, P);
#pragma unroll
      for (i = 0; i < DIM; i++)
        yy1[i] = y0[i] + h * (a151 * k1[i] + a156 * k6[i] + a157 * k7[i] +
                              a158 * k8[i] + a1511 * k2[i] + a1512 * k3[i] +
                              a1513 * k4[i] + a1514 * k10[i]);
      co_deriv(yy1, k2, P);
#pragma unroll
      for (i = 0; i < DIM; i++)
        yy1[i] = y0[i] + h * (a161 * k1[i] + a166 * k6[i] + a167 * k7[i] +
                              a168 * k8[i] + a169 * k9[i] + a1613 * k4[i] +
                              a1614 * k10[i] + a1615 * k2[i]);
      co_deriv(yy1, k3, P);
      nfev += 3;
#pragma unroll
      for (i = 0; i < DIM; i++) {
        rcont5[i] = h * (rcont5[i] + d413 * k4[i] + d414 * k10[i] + d415 * k2[i] +
                         d416 * k3[i]);
        rcont6[i] = h * (rcont6[i] + d513 * k4[i] + d514 * k10[i] + d515 * k2[i] +
                         d516 * k3[i]);
        rcont7[i] = h * (rcont7[i] + d613 * k4[i] + d614 * k10[i] + d615 * k2[i] +
                         d616 * k3[i]);
        rcont8[i] = h * (rcont8[i] + d713 * k4[i] + d714 * k10[i] + d715 * k2[i] +
                         d716 * k3[i]);
        k1[i] = k4[i];
        y0[i] = k5[i];
      }
      while ((fin < nt - 1) && (pos_neg * t[fin + 1] < pos_neg * t_current)) {
        real s = (t[fin + 1] - t_old) / h;
        real s1 = R_(1.0) - s;
#pragma unroll
        for (i = 0; i < DIM; i++)
          yy1[i] = rcont1[i] +
                   s * (rcont2[i] +
                        s1 * (rcont3[i] +
                              s * (rcont4[i] +
                                   s1 * (rcont5[i] +
                                         s * (rcont6[i] +
                                              s1 * (rcont7[i] + s * rcont8[i]))))));
        co_save(yy1, result);
        result += DIM;
        fin++;
      }
    } else {
#pragma unroll
      for (i = 0; i < DIM; i++) {
        k1[i] = k4[i];
        y0[i] = k5[i];
      }
    }
    hnew = (fabs(hnew) > fabs(hmax)) ? pos_neg * hmax : hnew;
    if (reject) hnew = pos_neg * co_min(fabs(hnew), fabs(h));
    reject = 0;
  } else {  // rejected
    nrej++;
    hnew = h / co_min(facc1, fac11 / safe);
    reject = 1;
    t_current = t_old;
    t_old = t_old_older;
  }
  h = hnew;
}

// Whole orbit.  y0in: rectangular (x,y,z,vx,vy,vz); rtol/atol: ALREADY exp()'d
// (galpy passes log(tol) and does rtol = exp(rtol) at the top of dop853).
// result: nt blocks of 6 (cylindrical), like galpy after rect_to_cyl.
template <bool LAZY>
HD void dop853_orbit(real *y0in, int nt, const real *t, const PotSet &P,
                     real rtol, real atol, real *result, DopStats *st,
                     real fixed_h = R_(0.0)) {
  DopState S;
#pragma unroll
  for (int i = 0; i < DIM; i++) S.y0[i] = y0in[i];
  S.result = result;
  dop853_init(S, nt, t, P, rtol, atol);
  while (S.fin < nt - 1) dop853_step<LAZY>(S, nt, t, P, rtol, atol, fixed_h);
  st->nfev = S.nfev;
  st->naccept = S.nacc;
  st->nreject = S.nrej;
}
