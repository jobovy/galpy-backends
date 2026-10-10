// galpy's fixed-step symplectic integrators (util/bovy_symplecticode.c):
// leapfrog (2nd order), symplec4 (4th, 3 force evaluations per step) and
// symplec6 (6th, 7 per step), one orbit per thread, state = q[3], p[3].
//
// Same drift-kick sequence and floating-point operation order as galpy's C
// (drift q += (c*dt)*p, kick p += (d*dt)*a; merged (c_last+c_1)*dt drift between
// consecutive steps inside one output interval), and the same per-orbit
// *_estimate_step (step doubling from the output spacing until the
// one-step-vs-two-half-steps difference passes rtol/atol, log-space norm), so
// the host build is bit-identical to galpy's leapfrog_c/symplec4_c/symplec6_c.
// Restructured only into a resumable per-substep state machine so that a GPU
// thread can switch orbits between substeps (refill).
#pragma once
#include "dop853_cuda.cuh"

#define SYMP_LEAPFROG 0
#define SYMP_SYMPLEC4 4
#define SYMP_SYMPLEC6 6
#define SYMP_MAX_DT_REDUCE 10000.

template <int SCH> struct SympCoef;
template <> struct SympCoef<SYMP_LEAPFROG> {
  static constexpr int m = 1;
  HD static real C(int k) { return R_(0.5); }  // dt/2 drifts; merged drift = dt
  HD static real D(int k) { return R_(1.0); }
};
template <> struct SympCoef<SYMP_SYMPLEC4> {
  static constexpr int m = 3;
  HD static real C(int k) {
    const real c[4] = {R_(0.6756035959798289), R_(-0.1756035959798288),
                       R_(-0.1756035959798288), R_(0.6756035959798289)};
    return c[k];
  }
  HD static real D(int k) {
    const real d[3] = {R_(1.3512071919596578), R_(-1.7024143839193153),
                       R_(1.3512071919596578)};
    return d[k];
  }
};
template <> struct SympCoef<SYMP_SYMPLEC6> {
  static constexpr int m = 7;
  HD static real C(int k) {
    const real c[8] = {R_(0.392256805238780),   R_(0.510043411918458),
                       R_(-0.471053385409758),  R_(0.687531682525198e-1),
                       R_(0.687531682525198e-1), R_(-0.471053385409758),
                       R_(0.510043411918458),   R_(0.392256805238780)};
    return c[k];
  }
  HD static real D(int k) {
    const real d[7] = {R_(0.784513610477560),  R_(0.235573213359357),
                       R_(-0.117767998417887e1), R_(0.131518632068391e1),
                       R_(-0.117767998417887e1), R_(0.235573213359357),
                       R_(0.784513610477560)};
    return d[k];
  }
};

// galpy's evalRectForce: rectangular acceleration (conservative potentials).
HD void symp_accel(const real *q, real *a, const PotSet &P) {
  real q6[DIM] = {q[0], q[1], q[2], R_(0.), R_(0.), R_(0.)}, a6[DIM];
  co_deriv(q6, a6, P);
  a[0] = a6[3];
  a[1] = a6[4];
  a[2] = a6[5];
}

// Two positions at once (two orbits per thread); identical arithmetic per point.
HD void symp_accel2(const real *qa, real *aa, const real *qb, real *ab, const PotSet &P) {
#if defined(CO_NOINLINE_FORCE) && defined(__CUDA_ARCH__)
  CoAcc6 r = co_dev_accel2(qa[0], qa[1], qa[2], qb[0], qb[1], qb[2]);
  aa[0] = r.ax;
  aa[1] = r.ay;
  aa[2] = r.az;
  ab[0] = r.bx;
  ab[1] = r.by;
  ab[2] = r.bz;
#else
  symp_accel(qa, aa, P);
  symp_accel(qb, ab, P);
#endif
}

// leapfrog_leapq / leapfrog_leapp, with galpy's (coef*dt) evaluated first
HD void symp_drift(const real *q, const real *p, real cdt, real *qn) {
#pragma unroll
  for (int i = 0; i < 3; i++) qn[i] = q[i] + cdt * p[i];
}
HD void symp_kick(const real *p, real ddt, const real *a, real *pn) {
#pragma unroll
  for (int i = 0; i < 3; i++) pn[i] = p[i] + ddt * a[i];
}

// galpy's *_estimate_step (all three share this structure: one step with dt vs
// two with dt/2, halving dt from the output spacing).  rtol/atol: log values.
template <int SCH>
HD real symp_estimate_step(const real *qo, const real *po, real dt, const PotSet &P,
                           real rtol, real atol, int *nfev) {
  typedef SympCoef<SCH> K;
  const int m = K::m;
  real err = R_(2.);
  real init_dt = dt;
  real q11[3], q12[3], p11[3], p12[3], qtmp[3], ptmp[3], a[3], scale[6];
  int ii;
  real max_val_q = log(fabs(qo[0]));
  for (ii = 1; ii < 3; ii++)
    if (log(fabs(qo[ii])) > max_val_q) max_val_q = log(fabs(qo[ii]));
  real max_val_p = log(fabs(po[0]));
  for (ii = 1; ii < 3; ii++)
    if (log(fabs(po[ii])) > max_val_p) max_val_p = log(fabs(po[ii]));
  real c = fmax(atol, rtol + max_val_q);
  real s = log(exp(atol - c) + exp(rtol + max_val_q - c)) + c;
  for (ii = 0; ii < 3; ii++) scale[ii] = s;
  c = fmax(atol, rtol + max_val_p);
  s = log(exp(atol - c) + exp(rtol + max_val_p - c)) + c;
  for (ii = 0; ii < 3; ii++) scale[ii + 3] = s;
  dt *= R_(2.);
  while (err > R_(1.) && init_dt / dt < SYMP_MAX_DT_REDUCE) {
    dt /= R_(2.);
    // one step with dt -> (q11, p11)
    real q[3], p[3];
    symp_drift(qo, po, K::C(0) * dt, q);
    for (int k = 0; k < 3; k++) p[k] = po[k];
    for (int k = 0; k < m; k++) {
      symp_accel(q, a, P);
      symp_kick(p, K::D(k) * dt, a, p);
      symp_drift(q, p, K::C(k + 1) * dt, q);
    }
    for (int k = 0; k < 3; k++) {
      q11[k] = q[k];
      p11[k] = p[k];
    }
    // two steps with dt/2 -> (q12, p12); merged drift between them
    symp_drift(qo, po, K::C(0) * dt / R_(2.), q);
    for (int k = 0; k < 3; k++) p[k] = po[k];
    for (int k = 0; k < m; k++) {
      symp_accel(q, a, P);
      symp_kick(p, K::D(k) * dt / R_(2.), a, p);
      real cc = (k < m - 1) ? K::C(k + 1) * dt / R_(2.) : (K::C(0) + K::C(m)) * dt / R_(2.);
      if (SCH == SYMP_LEAPFROG && k == m - 1) cc = dt / R_(2.);
      symp_drift(q, p, cc, q);
    }
    for (int k = 0; k < m; k++) {
      symp_accel(q, a, P);
      symp_kick(p, K::D(k) * dt / R_(2.), a, p);
      symp_drift(q, p, K::C(k + 1) * dt / R_(2.), q);
    }
    for (int k = 0; k < 3; k++) {
      q12[k] = q[k];
      p12[k] = p[k];
    }
    *nfev += 3 * m;
    err = R_(0.);
    for (ii = 0; ii < 3; ii++) {
      err += exp(R_(2.) * log(fabs(q11[ii] - q12[ii])) - R_(2.) * scale[ii]);
      err += exp(R_(2.) * log(fabs(p11[ii] - p12[ii])) - R_(2.) * scale[ii + 3]);
    }
    err = sqrt(err / R_(2.) / R_(3.));
  }
  (void)qtmp;
  (void)ptmp;
  return dt;
}

struct SympState {
  real q[3], p[3];
  real dt;
  long ndt;  // substeps per output interval
  int ii;    // output intervals done
  long jj;   // substeps done in the current interval
  int nfev;
  real *result;
};

// Start an orbit: y (rect x,y,z,vx,vy,vz) -> state; save t[0]; dt from
// estimate_step (galpy semantics) unless dt_fixed > 0 (common dt, integer
// divisor of the output spacing).
template <int SCH>
HD void symp_init(SympState &S, const real *y, int nt, const real *t, const PotSet &P,
                  real rtol, real atol, real dt_fixed) {
  for (int i = 0; i < 3; i++) {
    S.q[i] = y[i];
    S.p[i] = y[3 + i];
  }
  real v[DIM] = {S.q[0], S.q[1], S.q[2], S.p[0], S.p[1], S.p[2]};
  co_save(v, S.result);
  S.result += DIM;
  S.nfev = 0;
  real init_dt = t[1] - t[0];
  S.dt = (dt_fixed > R_(0.)) ? dt_fixed
                             : symp_estimate_step<SCH>(S.q, S.p, init_dt, P, rtol, atol, &S.nfev);
  S.ndt = (long)(init_dt / S.dt);
  S.ii = 0;
  S.jj = 0;
}

// One full substep (m kicks); the first substep of an output interval starts
// with the c_1 drift, the last ends with the c_m drift and saves the output.
template <int SCH>
HD void symp_substep(SympState &S, const PotSet &P) {
  typedef SympCoef<SCH> K;
  const int m = K::m;
  const real dt = S.dt;
  real a[3];
  if (S.jj == 0) symp_drift(S.q, S.p, K::C(0) * dt, S.q);
  const bool last = (S.jj == S.ndt - 1);
  for (int k = 0; k < m; k++) {
    symp_accel(S.q, a, P);
    symp_kick(S.p, K::D(k) * dt, a, S.p);
    real cdt;
    if (k < m - 1)
      cdt = K::C(k + 1) * dt;
    else if (last)
      cdt = (SCH == SYMP_LEAPFROG) ? dt / R_(2.) : K::C(m) * dt;
    else
      cdt = (SCH == SYMP_LEAPFROG) ? dt : (K::C(m) + K::C(0)) * dt;
    symp_drift(S.q, S.p, cdt, S.q);
  }
  S.nfev += m;
  if (last) {
    real v[DIM] = {S.q[0], S.q[1], S.q[2], S.p[0], S.p[1], S.p[2]};
    co_save(v, S.result);
    S.result += DIM;
    S.ii++;
    S.jj = 0;
  } else {
    S.jj++;
  }
}

template <int SCH>
HD void symp_orbit(const real *y, int nt, const real *t, const PotSet &P, real rtol,
                   real atol, real dt_fixed, real *result, int *nfev, real *dt_used) {
  SympState S;
  S.result = result;
  symp_init<SCH>(S, y, nt, t, P, rtol, atol, dt_fixed);
  while (S.ii < nt - 1) symp_substep<SCH>(S, P);
  *nfev = S.nfev;
  *dt_used = S.dt;
}

// Two orbits advanced together with the SAME dt and substep counter (common dt
// only): one paired force call per kick.  Same arithmetic per orbit as
// symp_substep.
template <int SCH>
HD void symp_substep2(SympState &A, SympState &B, const PotSet &P) {
  typedef SympCoef<SCH> K;
  const int m = K::m;
  const real dt = A.dt;
  real aa[3], ab[3];
  if (A.jj == 0) {
    symp_drift(A.q, A.p, K::C(0) * dt, A.q);
    symp_drift(B.q, B.p, K::C(0) * dt, B.q);
  }
  const bool last = (A.jj == A.ndt - 1);
  for (int k = 0; k < m; k++) {
    symp_accel2(A.q, aa, B.q, ab, P);
    symp_kick(A.p, K::D(k) * dt, aa, A.p);
    symp_kick(B.p, K::D(k) * dt, ab, B.p);
    real cdt;
    if (k < m - 1)
      cdt = K::C(k + 1) * dt;
    else if (last)
      cdt = (SCH == SYMP_LEAPFROG) ? dt / R_(2.) : K::C(m) * dt;
    else
      cdt = (SCH == SYMP_LEAPFROG) ? dt : (K::C(m) + K::C(0)) * dt;
    symp_drift(A.q, A.p, cdt, A.q);
    symp_drift(B.q, B.p, cdt, B.q);
  }
  A.nfev += m;
  B.nfev += m;
  if (last) {
    real va[DIM] = {A.q[0], A.q[1], A.q[2], A.p[0], A.p[1], A.p[2]};
    real vb[DIM] = {B.q[0], B.q[1], B.q[2], B.p[0], B.p[1], B.p[2]};
    co_save(va, A.result);
    co_save(vb, B.result);
    A.result += DIM;
    B.result += DIM;
    A.ii++;
    B.ii++;
    A.jj = 0;
    B.jj = 0;
  } else {
    A.jj++;
    B.jj++;
  }
}
