// CUDA port of the galpy C pieces needed to integrate orbits in MWPotential2014
// with galpy's dop853_c, one thread per orbit.  Everything is __host__ __device__
// so the SAME source also runs on the CPU (exactness check against galpy's C).
//
// Ported from galpy (read-only source at /var/tmp/fbp):
//   potential_c_ext/PowerSphericalPotentialwCutoff.c  (+ GSL gamma_inc_P)
//   potential_c_ext/MiyamotoNagaiPotential.c
//   potential_c_ext/NFWPotential.c
//   potential_c_ext/galpy_potentials.c   calcRforce/calczforce/calcphitorque
//   orbit_c_ext/integrateFullOrbit.c     evalRectDeriv, cyl<->rect
//   util/leung_dop853.c                  dop853 (same control, same dense output)
#pragma once
#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Precision: double (galpy) by default; -DCO_FLOAT32 builds the float variant
// used ONLY to show the FP64 penalty (galpy itself is float64 throughout).
#if defined(CO_COUNT)  // host-only op-counting build (flopcount.cpp)
typedef CountedReal real;
#define CO_DBL_EPSILON 2.2204460492503131e-16
#elif defined(CO_FLOAT32)
typedef float real;
#define CO_DBL_EPSILON 1.1920929e-07f  // FLT_EPSILON (series/CF stop criterion)
#else
typedef double real;
#define CO_DBL_EPSILON 2.2204460492503131e-16  // GSL_DBL_EPSILON
#endif
#define R_(x) ((real)(x))

// ---------------------------------------------------------------- potentials
// galpy's pot_type codes (orbit/integrateFullOrbit.py:_parse_pot)
#define POT_MIYAMOTONAGAI 5
#define POT_NFW 9
#define POT_POWERSPHWCUTOFF 15

#define CO_MAXPOT 16
#define CO_MAXARGS 512
#define CO_NINV 160  // CO_FASTGAM: table of 1/(a+n), n < CO_NINV

// Flat replacement for galpy's `struct potentialArg` array: a type code and an
// offset into one shared args table (no heap pointers, no function pointers).
struct PotSet {
  int npot;
  int type[CO_MAXPOT];
  int off[CO_MAXPOT];
  real args[CO_MAXARGS];
};

// ---- GSL gsl_sf_gamma_inc_P, the branches reachable for 0 < a < 10
// (gamma_inc_D with a < 10, gamma_inc_P_series, gamma_inc_Q_large_x,
// gamma_inc_Q_CF).  lg1 = lngamma(a+1) is precomputed on the host with GSL.
HD real gsl_gamma_inc_D(real a, real x, real lg1) {
  real lnr = a * log(x) - x - lg1;
  return exp(lnr);
}
HD real gsl_gamma_inc_P_series(real a, real x, real lg1) {
  const int nmax = 10000;
  real D = gsl_gamma_inc_D(a, x, lg1);
  real sum = R_(1.0), term = R_(1.0);
  int n;
  long nlow = (x > a) ? (long)(x - a) : 0;
  for (n = 1; n < nlow; n++) {
    term *= x / (a + n);
    sum += term;
  }
  for (; n < nmax; n++) {
    term *= x / (a + n);
    sum += term;
    if (fabs(term / sum) < CO_DBL_EPSILON) break;
  }
  return D * sum;
}
HD real gsl_gamma_inc_Q_large_x(real a, real x, real lg1) {
  const int nmax = 5000;
  real D = gsl_gamma_inc_D(a, x, lg1);
  real sum = R_(1.0), term = R_(1.0), last = R_(1.0);
  for (int n = 1; n < nmax; n++) {
    term *= (a - n) / x;
    if (fabs(term / last) > R_(1.0)) break;
    if (fabs(term / sum) < CO_DBL_EPSILON) break;
    sum += term;
    last = term;
  }
  return D * (a / x) * sum;
}
HD real gsl_gamma_inc_F_CF(real a, real x) {
  const int nmax = 5000;
  const real small = CO_DBL_EPSILON * CO_DBL_EPSILON * CO_DBL_EPSILON;
  real hn = R_(1.0), Cn = R_(1.0) / small, Dn = R_(1.0);
  for (int n = 2; n < nmax; n++) {
    real an;
    if (n & 1)
      an = R_(0.5) * (n - 1) / x;
    else
      an = (R_(0.5) * n - a) / x;
    Dn = R_(1.0) + an * Dn;
    if (fabs(Dn) < small) Dn = small;
    Cn = R_(1.0) + an / Cn;
    if (fabs(Cn) < small) Cn = small;
    Dn = R_(1.0) / Dn;
    real delta = Cn * Dn;
    hn *= delta;
    if (fabs(delta - R_(1.0)) < CO_DBL_EPSILON) break;
  }
  return hn;
}
HD real gsl_gamma_inc_P(real a, real x, real lg1) {
  // valid for 0 < a < 10 (the gamma_inc_D small-a branch); x >= 0
  if (x == R_(0.0)) return R_(0.0);
  if (x < R_(20.0) || x < R_(0.5) * a) return gsl_gamma_inc_P_series(a, x, lg1);
  // a <= x here (a < 10 < 20 <= x)
  real Q;
  if (a > R_(0.2) * x)
    Q = gsl_gamma_inc_D(a, x, lg1) * (a / x) * gsl_gamma_inc_F_CF(a, x);
  else
    Q = gsl_gamma_inc_Q_large_x(a, x, lg1);
  return R_(1.0) - Q;
}

#ifdef CO_FASTGAM
// Same series / asymptotic expansion as GSL, without the divisions (each DP
// division is ~10 dependent DP instructions on the GPU): 1/(a+n) comes from a
// per-potential table (n is warp-uniform -> constant-cache broadcast), the
// stop tests compare |term| with eps*|sum| instead of dividing, and the
// asymptotic series multiplies by 1/x.  Agrees with GSL to a few ulp (not
// bit-identical); the two loops of the GSL series are merged into one so n
// stays uniform across the warp.
HD real fast_gamma_inc_P(real a, real x, real lg1, const real *inv) {
  if (x == R_(0.0)) return R_(0.0);
  real D = exp(a * log(x) - x - lg1);
  if (x < R_(20.0) || x < R_(0.5) * a) {
    real sum = R_(1.0), term = R_(1.0);
    long nlow = (x > a) ? (long)(x - a) : 0;
    for (int n = 1; n < 10000; n++) {
      term *= (n < CO_NINV) ? x * inv[n] : x / (a + n);
      sum += term;
      if (n >= nlow && term < CO_DBL_EPSILON * sum) break;
    }
    return D * sum;
  }
  real ix = R_(1.0) / x;
  real sum = R_(1.0), term = R_(1.0), last = R_(1.0);
  for (int n = 1; n < 5000; n++) {
    term *= (a - n) * ix;
    if (fabs(term) > fabs(last)) break;
    if (fabs(term) < CO_DBL_EPSILON * fabs(sum)) break;
    sum += term;
    last = term;
  }
  return R_(1.0) - D * (a * ix) * sum;  // a <= 0.2 x here (a < 4 for x >= 20)
}
#endif

// ---- PowerSphericalPotentialwCutoff; galpy args (amp, alpha, rc, cache, cache)
// are re-packed on the host as
//   [amp, alpha, rc, c2pirc = 2.*M_PI*pow(rc,3.-alpha), gam = gsl_sf_gamma(a),
//    lg1 = gsl_sf_lngamma(a+1), a = 1.5-0.5*alpha]
// so the per-call constants are the same doubles galpy recomputes each call.
// galpy caches force(r2) between Rforce and zforce; computing it once here is
// the same value.
HD real PSwC_force_over_r(const real *args, real R, real Z) {
  real amp = args[0];
  real rc = args[2];
  real r2 = R * R + Z * Z;
  // mass(r2,alpha,rc) = 2.*M_PI*pow(rc,3.-alpha) * (gamma(a) * P(a, r2/rc/rc))
#ifdef CO_FASTGAM
  real mass = args[3] * (args[4] * fast_gamma_inc_P(args[6], r2 / rc / rc, args[5], args + 7));
#else
  real mass = args[3] * (args[4] * gsl_gamma_inc_P(args[6], r2 / rc / rc, args[5]));
#endif
#ifdef CO_FASTPOW
  return -amp * mass / (r2 * sqrt(r2));
#else
  return -amp * mass / pow(r2, R_(1.5));
#endif
}

// ---- MiyamotoNagaiPotential: args amp, a, b
HD real MN_Rforce(const real *args, real R, real z) {
  real amp = args[0], a = args[1], b = args[2];
  return -amp * R * pow(R * R + pow(a + pow(z * z + b * b, R_(0.5)), R_(2.)), -R_(1.5));
}
HD real MN_zforce(const real *args, real R, real z) {
  real amp = args[0], a = args[1], b = args[2];
  real sqrtbz = pow(b * b + z * z, R_(.5));
  real asqrtbz = a + sqrtbz;
  if (a == R_(0.))
    return amp * (-z * pow(R * R + asqrtbz * asqrtbz, -R_(1.5)));
  else
    return amp * (-z * asqrtbz / sqrtbz * pow(R * R + asqrtbz * asqrtbz, -R_(1.5)));
}

// ---- NFWPotential: args amp, a
#define NFW_SMALL_X R_(0.25)
HD real nfw_S(real t) {
  real t2 = t * t;
  real out = R_(1.) / R_(17.);
  for (int m = 7; m >= 1; m--) out = out * t2 + R_(1.) / (R_(2.) * m + R_(1.));
  return out * t2 * t;
}
HD real nfw_h(real x) {
  real t = x / (R_(2.) + x);
  return R_(2.) * (t * t / (R_(1.) + t) + nfw_S(t));
}
HD real NFW_Rforce(const real *args, real R, real Z) {
  real amp = args[0], a = args[1];
  real Rz = R * R + Z * Z;
  real sqrtRz = pow(Rz, R_(0.5));
  if (sqrtRz < NFW_SMALL_X * a)
    return -amp * R * nfw_h(sqrtRz / a) / sqrtRz / sqrtRz / sqrtRz;
  return amp * R * (R_(1.) / Rz / (a + sqrtRz) - log(R_(1.) + sqrtRz / a) / sqrtRz / Rz);
}
HD real NFW_zforce(const real *args, real R, real Z) {
  real amp = args[0], a = args[1];
  real Rz = R * R + Z * Z;
  real sqrtRz = pow(Rz, R_(0.5));
  if (sqrtRz < NFW_SMALL_X * a)
    return -amp * Z * nfw_h(sqrtRz / a) / sqrtRz / sqrtRz / sqrtRz;
  return amp * Z * (R_(1.) / Rz / (a + sqrtRz) - log(R_(1.) + sqrtRz / a) / sqrtRz / Rz);
}

// ---- calcRforce / calczforce (galpy_potentials.c): sum in list order.
// All three potentials are axisymmetric, so calcphitorque = sum of ZeroForce = 0.
HD void calcRzforce(real R, real Z, const PotSet &P, real *Rforce,
                    real *zforce) {
  real fR = R_(0.), fz = R_(0.);
  for (int ii = 0; ii < P.npot; ii++) {
    const real *args = P.args + P.off[ii];
    switch (P.type[ii]) {
    case POT_POWERSPHWCUTOFF: {
      real f = PSwC_force_over_r(args, R, Z);
      fR += f * R;
      fz += f * Z;
      break;
    }
#ifdef CO_FASTPOW
    // pow(x,0.5) -> sqrt, pow(d,-1.5) -> 1/(d sqrt d), shared subexpressions
    case POT_MIYAMOTONAGAI: {
      real amp = args[0], a = args[1], b = args[2];
      real sqrtbz = sqrt(b * b + Z * Z);
      real asqrtbz = a + sqrtbz;
      real d = R * R + asqrtbz * asqrtbz;
      real f = -amp / (d * sqrt(d));
      fR += f * R;
      fz += (a == R_(0.)) ? f * Z : f * Z * asqrtbz / sqrtbz;
      break;
    }
    case POT_NFW: {
      real amp = args[0], a = args[1];
      real Rz = R * R + Z * Z;
      real sqrtRz = sqrt(Rz);
      real f;
      if (sqrtRz < NFW_SMALL_X * a)
        f = -amp * nfw_h(sqrtRz / a) / (Rz * sqrtRz);
      else
        f = amp * (R_(1.) / (Rz * (a + sqrtRz)) - log1p(sqrtRz / a) / (sqrtRz * Rz));
      fR += f * R;
      fz += f * Z;
      break;
    }
#else
    case POT_MIYAMOTONAGAI:
      fR += MN_Rforce(args, R, Z);
      fz += MN_zforce(args, R, Z);
      break;
    case POT_NFW:
      fR += NFW_Rforce(args, R, Z);
      fz += NFW_zforce(args, R, Z);
      break;
#endif
    }
  }
  *Rforce = fR;
  *zforce = fz;
}

// ---- evalRectDeriv (integrateFullOrbit.c); phi only feeds non-axisymmetric
// potentials (none here), so acos() is dead code and the compiler drops it.
#ifndef CO_FORCE_HOOK
#define CO_FORCE_HOOK
#endif
HD void evalRectDeriv(const real *q, real *a, const PotSet &P) {
  CO_FORCE_HOOK;
  a[0] = q[3];
  a[1] = q[4];
  a[2] = q[5];
  real x = q[0], y = q[1], z = q[2];
  real R = sqrt(x * x + y * y);
  real sinphi = y / R;
  real cosphi = x / R;
  real Rforce, zforce;
  calcRzforce(R, z, P, &Rforce, &zforce);
  real phitorque = R_(0.);  // calcphitorque: ZeroForce for all three
  a[3] = cosphi * Rforce - R_(1.) / R * sinphi * phitorque;
  a[4] = sinphi * Rforce + R_(1.) / R * cosphi * phitorque;
  a[5] = zforce;
}

HD void cyl_to_rect_galpy(real *v) {
  real R = v[0], phi = v[5], cp = cos(phi), sp = sin(phi), vR = v[1], vT = v[2];
  v[0] = R * cp;
  v[1] = R * sp;
  v[2] = v[3];
  v[5] = v[4];
  v[3] = vR * cp - vT * sp;
  v[4] = vR * sp + vT * cp;
}
HD void rect_to_cyl_galpy(real *v) {
  // bovy_coords.c rect_to_cyl_galpy
  real x = v[0], y = v[1], vx = v[3], vy = v[4];
  v[3] = v[2];
  v[4] = v[5];
  v[5] = atan2(y, x);
  real cp = cos(v[5]), sp = sin(v[5]);
  v[0] = sqrt(x * x + y * y);
  v[1] = vx * cp + vy * sp;
  v[2] = -vx * sp + vy * cp;
}
