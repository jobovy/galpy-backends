// One-thread-per-orbit CUDA integration of galpy's dop853_c in MWPotential2014.
// C ABI for ctypes (see cuda_orbits.py).  Also runs the identical code on the
// host (co_integrate_host) to check the port against galpy's C bit-for-bit.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "galpy_cuda.cuh"

static PotSet h_pots;
__constant__ PotSet d_pots;

#include "dop853_cuda.cuh"
#include "symplectic_cuda.cuh"

#ifdef CO_NOINLINE_FORCE
// The single out-of-line copy of the force (reads the potential from d_pots).
__device__ __noinline__ CoAcc3 co_dev_accel(real x, real y, real z) {
  real q[DIM] = {x, y, z, R_(0.), R_(0.), R_(0.)}, a[DIM];
  evalRectDeriv(q, a, d_pots);
  return CoAcc3{a[3], a[4], a[5]};
}
__device__ __noinline__ CoAcc6 co_dev_accel2(real ax, real ay, real az, real bx, real by,
                                             real bz) {
  real qa[DIM] = {ax, ay, az, R_(0.), R_(0.), R_(0.)}, aa[DIM];
  real qb[DIM] = {bx, by, bz, R_(0.), R_(0.), R_(0.)}, ab[DIM];
  evalRectDeriv(qa, aa, d_pots);
  evalRectDeriv(qb, ab, d_pots);
  return CoAcc6{aa[3], aa[4], aa[5], ab[3], ab[4], ab[5]};
}
#endif

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t e_ = (call);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_),     \
              __FILE__, __LINE__);                                            \
      return -1;                                                              \
    }                                                                         \
  } while (0)

#ifndef CO_MINBLOCKS
#define CO_MINBLOCKS 1  // -DCO_MINBLOCKS=2 caps registers at 128/thread (more warps, more spills)
#endif

__device__ __forceinline__ void load_orbit(DopState &S, const real *yo, int ii, int nt,
                                           real *result) {
#pragma unroll
  for (int i = 0; i < DIM; i++) S.y0[i] = yo[(size_t)DIM * ii + i];
  cyl_to_rect_galpy(S.y0);
  S.result = result + (size_t)DIM * nt * ii;
}
__device__ __forceinline__ void store_stats(const DopState &S, int *stats, int ii) {
  if (stats) {
    stats[3 * ii] = S.nfev;
    stats[3 * ii + 1] = S.nacc;
    stats[3 * ii + 2] = S.nrej;
  }
}

// One thread = one orbit, start to finish (the v1 kernel).  A warp runs until
// its slowest lane is done, so lanes with fewer steps idle.
template <bool LAZY>
__global__ void __launch_bounds__(256, CO_MINBLOCKS)
    k_integrate(int N, const real *__restrict__ yo, int nt,
                const real *__restrict__ t, real rtol, real atol,
                real *__restrict__ result, int *__restrict__ stats, real fixed_h,
                const int *__restrict__ order) {
  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii >= N) return;
  if (order) ii = order[ii];
  DopState S;
  load_orbit(S, yo, ii, nt, result);
  dop853_init(S, nt, t, d_pots, rtol, atol);
  while (S.fin < nt - 1) dop853_step<LAZY>(S, nt, t, d_pots, rtol, atol, fixed_h);
  store_stats(S, stats, ii);
}

// "Refill" kernel: persistent threads in ONE flat step loop; a lane whose orbit
// is finished immediately takes the next orbit from a global counter, so the
// lanes of a warp stay busy until the work runs out (no waiting for the
// slowest orbit of the warp).  Same init/step code -> same results.
template <bool LAZY>
__global__ void __launch_bounds__(256, CO_MINBLOCKS)
    k_integrate_refill(int N, const real *__restrict__ yo, int nt,
                       const real *__restrict__ t, real rtol, real atol,
                       real *__restrict__ result, int *__restrict__ stats,
                       real fixed_h, int *__restrict__ counter,
                       const int *__restrict__ order) {
  DopState S;
  int ii = -1;
  S.fin = nt - 1;  // "no orbit yet"
  while (true) {
    if (S.fin >= nt - 1) {
      if (ii >= 0) store_stats(S, stats, ii);
      ii = atomicAdd(counter, 1);
      if (ii >= N) break;
      if (order) ii = order[ii];
      load_orbit(S, yo, ii, nt, result);
      dop853_init(S, nt, t, d_pots, rtol, atol);
    }
    dop853_step<LAZY>(S, nt, t, d_pots, rtol, atol, fixed_h);
  }
}

// The incomplete gamma of this build (GSL-exact, or CO_FASTGAM's table form).
__host__ __device__ real gammaP_any(real a, real x, real lg1) {
#ifdef CO_FASTGAM
  real inv[CO_NINV];
  for (int n = 1; n < CO_NINV; n++) inv[n] = R_(1.0) / (a + n);
  return fast_gamma_inc_P(a, x, lg1, inv);
#else
  return gsl_gamma_inc_P(a, x, lg1);
#endif
}

__global__ void k_gammaP(int n, const real *a, const real *x,
                         const real *lg1, real *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = gammaP_any(a[i], x[i], lg1[i]);
}

// ---------------------------------------------------------------------------
// Occupancy experiment: refill kernel with the first NSH stage slots (yy1,
// k2..k10, rcont1..8; 6 doubles each) in SHARED memory (structure-of-arrays,
// stride blockDim) instead of registers.  Same arithmetic.
template <int NSH>
struct SmemStages {
  real *base;  // smem + threadIdx.x
  int B;
  real loc[(18 - NSH) > 0 ? (18 - NSH) : 1][DIM];
  template <int S> __device__ __forceinline__ SPtr get() {
    if constexpr (S < NSH)
      return SPtr{base + S * DIM * B, B};
    else
      return SPtr{loc[S - NSH], 1};
  }
};

// MINB: min resident 128-thread blocks/SM requested from ptxas -> register cap
// 65536/(128*MINB) per thread (2: 255, 3: 170, 4: 128).
template <int NSH, int MINB>
__global__ void __launch_bounds__(128, MINB)
    k_integrate_smem(int N, const real *__restrict__ yo, int nt,
                     const real *__restrict__ t, real rtol, real atol,
                     real *__restrict__ result, int *__restrict__ stats,
                     int *__restrict__ counter, const int *__restrict__ order) {
  extern __shared__ real co_smem[];
  SmemStages<NSH> sp;
  sp.base = co_smem + threadIdx.x;
  sp.B = blockDim.x;
  DopState S;
  int ii = -1;
  S.fin = nt - 1;
  while (true) {
    if (S.fin >= nt - 1) {
      if (ii >= 0) store_stats(S, stats, ii);
      ii = atomicAdd(counter, 1);
      if (ii >= N) break;
      if (order) ii = order[ii];
      load_orbit(S, yo, ii, nt, result);
      dop853_init(S, nt, t, d_pots, rtol, atol);
    }
    dop853_step_sp<false>(S, nt, t, d_pots, rtol, atol, R_(0.), sp);
  }
}

// ---------------------------------------------------------------------------
// Symplectic integrators (leapfrog / symplec4 / symplec6), refill loop over
// substeps.  dt_fixed > 0: one common dt for every orbit (lockstep); else
// galpy's per-orbit *_estimate_step.
template <int SCH, int MINB>
__global__ void __launch_bounds__(128, MINB)
    k_symp(int N, const real *__restrict__ yo, int nt, const real *__restrict__ t,
           real rtol, real atol, real dt_fixed, real *__restrict__ result,
           int *__restrict__ stats, int *__restrict__ counter) {
  SympState S;
  int ii = -1;
  S.ii = nt - 1;
  while (true) {
    if (S.ii >= nt - 1) {
      if (ii >= 0 && stats) {
        stats[3 * ii] = S.nfev;
        stats[3 * ii + 1] = (int)S.ndt;
        stats[3 * ii + 2] = 0;
      }
      ii = atomicAdd(counter, 1);
      if (ii >= N) break;
      real y[DIM];
#pragma unroll
      for (int i = 0; i < DIM; i++) y[i] = yo[(size_t)DIM * ii + i];
      cyl_to_rect_galpy(y);
      S.result = result + (size_t)DIM * nt * ii;
      symp_init<SCH>(S, y, nt, t, d_pots, rtol, atol, dt_fixed);
    }
    symp_substep<SCH>(S, d_pots);
  }
}

// Largest block <= block whose dynamic shared memory (slots x DIM doubles per
// thread) fits the device's per-block opt-in limit (48 KB on sm_61, 227 KB on sm_90).
static int co_smem_block(int block, int slots) {
  int dev, lim = 48 * 1024;
  if (cudaGetDevice(&dev) == cudaSuccess)
    cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  while (block > 32 && (size_t)slots * DIM * block * sizeof(real) > (size_t)lim) block /= 2;
  return block;
}

// Two orbits per thread (common dt only): orbits 2p and 2p+1 in lockstep, one
// paired force call per kick (ILP).  An odd last orbit is duplicated (same values
// written twice by the same thread).
template <int SCH, int MINB>
__global__ void __launch_bounds__(128, MINB)
    k_symp2(int N, const real *__restrict__ yo, int nt, const real *__restrict__ t,
            real rtol, real atol, real dt_fixed, real *__restrict__ result,
            int *__restrict__ stats, int *__restrict__ counter) {
  SympState A, B;
  int pa = -1, ia = 0, ib = 0;
  A.ii = nt - 1;
  const int npair = (N + 1) / 2;
  while (true) {
    if (A.ii >= nt - 1) {
      if (pa >= 0 && stats) {
        stats[3 * ia] = A.nfev;
        stats[3 * ia + 1] = (int)A.ndt;
        stats[3 * ia + 2] = 0;
        stats[3 * ib] = B.nfev;
        stats[3 * ib + 1] = (int)B.ndt;
        stats[3 * ib + 2] = 0;
      }
      pa = atomicAdd(counter, 1);
      if (pa >= npair) break;
      ia = 2 * pa;
      ib = (2 * pa + 1 < N) ? 2 * pa + 1 : 2 * pa;
      real ya[DIM], yb[DIM];
#pragma unroll
      for (int i = 0; i < DIM; i++) {
        ya[i] = yo[(size_t)DIM * ia + i];
        yb[i] = yo[(size_t)DIM * ib + i];
      }
      cyl_to_rect_galpy(ya);
      cyl_to_rect_galpy(yb);
      A.result = result + (size_t)DIM * nt * ia;
      B.result = result + (size_t)DIM * nt * ib;
      symp_init<SCH>(A, ya, nt, t, d_pots, rtol, atol, dt_fixed);
      symp_init<SCH>(B, yb, nt, t, d_pots, rtol, atol, dt_fixed);
    }
    symp_substep2<SCH>(A, B, d_pots);
  }
}

template <class KF>
static int co_launch_cfg(KF kern, int N, int block, size_t smem, int *nb) {
  int dev, nsm, per_sm;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
  if (smem > 48 * 1024)
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, block, smem));
  if (per_sm < 1) return -3;
  int nbmax = (N + block - 1) / block;
  *nb = nsm * per_sm < nbmax ? nsm * per_sm : nbmax;
  return 0;
}

template <class KF>
static int co_info(KF kern, int block, size_t smem, int *out) {
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, kern));
  if (smem > 48 * 1024)
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, block, smem));
  int dev, maxthr;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  out[0] = fa.numRegs;
  out[1] = (int)fa.localSizeBytes;
  out[2] = per_sm;
  out[3] = per_sm * block;  // resident threads per SM
  out[4] = maxthr;
  out[5] = (int)smem;
  out[6] = block;  // block actually used
  return 0;
}

extern "C" {

// types/off/args: already re-packed by the Python side (see cuda_orbits.py).
int co_set_potential(int npot, const int *types, const int *off, int nargs,
                     const real *args) {
  if (npot > CO_MAXPOT || nargs > CO_MAXARGS) return -2;
  memset(&h_pots, 0, sizeof(PotSet));
  h_pots.npot = npot;
  for (int i = 0; i < npot; i++) {
    h_pots.type[i] = types[i];
    h_pots.off[i] = off[i];
  }
  for (int i = 0; i < nargs; i++) h_pots.args[i] = args[i];
  // CO_HOST_ONLY=1: host runs only -- never touch the GPU (no CUDA context)
  const char *ho = getenv("CO_HOST_ONLY");
  if (!(ho && ho[0] == '1')) CK(cudaMemcpyToSymbol(d_pots, &h_pots, sizeof(PotSet)));
  return 0;
}

// Same code on the CPU, one thread (correctness reference for the port).
int co_integrate_host(int N, const real *yo, int nt, const real *t,
                      double logrtol, double logatol, real *result, int *stats,
                      int lazy) {
  real rtol = (real)exp(logrtol), atol = (real)exp(logatol);  // as galpy's dop853
  for (int ii = 0; ii < N; ii++) {
    real y[DIM];
    for (int i = 0; i < DIM; i++) y[i] = yo[DIM * ii + i];
    cyl_to_rect_galpy(y);
    DopStats st;
    if (lazy)
      dop853_orbit<true>(y, nt, t, h_pots, rtol, atol, result + (size_t)DIM * nt * ii,
                         &st);
    else
      dop853_orbit<false>(y, nt, t, h_pots, rtol, atol, result + (size_t)DIM * nt * ii,
                          &st);
    if (stats) {
      stats[3 * ii] = st.nfev;
      stats[3 * ii + 1] = st.naccept;
      stats[3 * ii + 2] = st.nreject;
    }
  }
  return 0;
}

// Full GPU call: alloc, H2D of ICs + times, kernel, D2H of all outputs, free.
// timings_ms (optional, 3): [alloc+h2d, kernel, d2h].  yo/result may be pinned
// (co_host_alloc) or pageable host memory.  refill: persistent-thread kernel.
// fixed_h > 0: diagnostic constant-step mode (all lanes in lockstep).
// order (optional, N): schedule orbit order[k] k-th (results stay in input order).
int co_integrate_gpu(int N, const real *yo, int nt, const real *t,
                     double logrtol, double logatol, real *result, int *stats,
                     int lazy, int block, float *timings_ms, int refill,
                     double fixed_h, const int *order, int nsh) {
  real rtol = (real)exp(logrtol), atol = (real)exp(logatol);
  real *d_yo, *d_t, *d_res;
  int *d_stats = NULL, *d_counter = NULL, *d_order = NULL;
  cudaEvent_t e0, e1, e2, e3;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&e2));
  CK(cudaEventCreate(&e3));
  CK(cudaEventRecord(e0));
  CK(cudaMalloc(&d_yo, sizeof(real) * DIM * N));
  CK(cudaMalloc(&d_t, sizeof(real) * nt));
  CK(cudaMalloc(&d_res, sizeof(real) * DIM * nt * (size_t)N));
  CK(cudaMalloc(&d_counter, sizeof(int)));
  CK(cudaMemset(d_counter, 0, sizeof(int)));
  if (stats) CK(cudaMalloc(&d_stats, sizeof(int) * 3 * N));
  CK(cudaMemcpy(d_yo, yo, sizeof(real) * DIM * N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_t, t, sizeof(real) * nt, cudaMemcpyHostToDevice));
  if (order) {
    CK(cudaMalloc(&d_order, sizeof(int) * N));
    CK(cudaMemcpy(d_order, order, sizeof(int) * N, cudaMemcpyHostToDevice));
  }
  CK(cudaEventRecord(e1));
  real fh = (real)fixed_h;
  if (nsh > 0) {  // shared-memory stage storage (refill, faithful dense output)
    block = co_smem_block(block, nsh / 10);
    size_t smem = (size_t)(nsh / 10) * DIM * block * sizeof(real);
    int nb, rc;
#define CO_SMEM_CASE(K, M)                                                      \
  case 10 * K + M:                                                              \
    rc = co_launch_cfg(k_integrate_smem<K, M>, N, block, smem, &nb);            \
    if (rc) return rc;                                                          \
    k_integrate_smem<K, M><<<nb, block, smem>>>(N, d_yo, nt, d_t, rtol, atol,   \
                                                d_res, d_stats, d_counter, d_order); \
    break;
    if (block > 128) return -5;
    switch (nsh) {  // nsh = 10 * NSH + MINB
      CO_SMEM_CASE(10, 2)
      CO_SMEM_CASE(10, 3)
      CO_SMEM_CASE(14, 3)
      CO_SMEM_CASE(14, 4)
      CO_SMEM_CASE(18, 3)
      CO_SMEM_CASE(18, 4)
      default:
        return -4;
    }
  } else if (refill) {
    int dev, nsm, per_sm;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    if (lazy)
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_integrate_refill<true>, block, 0));
    else
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_integrate_refill<false>, block, 0));
    int nb = nsm * per_sm, nbmax = (N + block - 1) / block;
    if (nb > nbmax) nb = nbmax;
    if (lazy)
      k_integrate_refill<true><<<nb, block>>>(N, d_yo, nt, d_t, rtol, atol, d_res, d_stats, fh, d_counter, d_order);
    else
      k_integrate_refill<false><<<nb, block>>>(N, d_yo, nt, d_t, rtol, atol, d_res, d_stats, fh, d_counter, d_order);
  } else {
    int nb = (N + block - 1) / block;
    if (lazy)
      k_integrate<true><<<nb, block>>>(N, d_yo, nt, d_t, rtol, atol, d_res, d_stats, fh, d_order);
    else
      k_integrate<false><<<nb, block>>>(N, d_yo, nt, d_t, rtol, atol, d_res, d_stats, fh, d_order);
  }
  CK(cudaGetLastError());
  CK(cudaEventRecord(e2));
  CK(cudaMemcpy(result, d_res, sizeof(real) * DIM * nt * (size_t)N,
                cudaMemcpyDeviceToHost));
  if (stats)
    CK(cudaMemcpy(stats, d_stats, sizeof(int) * 3 * N, cudaMemcpyDeviceToHost));
  CK(cudaEventRecord(e3));
  CK(cudaEventSynchronize(e3));
  if (timings_ms) {
    CK(cudaEventElapsedTime(timings_ms, e0, e1));
    CK(cudaEventElapsedTime(timings_ms + 1, e1, e2));
    CK(cudaEventElapsedTime(timings_ms + 2, e2, e3));
  }
  cudaFree(d_yo);
  cudaFree(d_t);
  cudaFree(d_res);
  cudaFree(d_counter);
  if (d_order) cudaFree(d_order);
  if (d_stats) cudaFree(d_stats);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  return 0;
}

// Page-locked host memory (fast, steady D2H copies); free with co_host_free.
void *co_host_alloc(size_t bytes) {
  void *p = NULL;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) return NULL;
  return p;
}
void co_host_free(void *p) { cudaFreeHost(p); }

int co_gammaP(int n, const real *a, const real *x, const real *lg1,
              real *out, int on_gpu) {
  if (!on_gpu) {
    for (int i = 0; i < n; i++) out[i] = gammaP_any(a[i], x[i], lg1[i]);
    return 0;
  }
  real *da, *dx, *dl, *dout;
  CK(cudaMalloc(&da, sizeof(real) * n));
  CK(cudaMalloc(&dx, sizeof(real) * n));
  CK(cudaMalloc(&dl, sizeof(real) * n));
  CK(cudaMalloc(&dout, sizeof(real) * n));
  CK(cudaMemcpy(da, a, sizeof(real) * n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dx, x, sizeof(real) * n, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dl, lg1, sizeof(real) * n, cudaMemcpyHostToDevice));
  k_gammaP<<<(n + 127) / 128, 128>>>(n, da, dx, dl, dout);
  CK(cudaGetLastError());
  CK(cudaMemcpy(out, dout, sizeof(real) * n, cudaMemcpyDeviceToHost));
  cudaFree(da);
  cudaFree(dx);
  cudaFree(dl);
  cudaFree(dout);
  return 0;
}

int co_real_size(void) { return (int)sizeof(real); }

int co_device_name(char *buf, int len) {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  snprintf(buf, len, "%s (sm_%d%d, %d SMs)", p.name, p.major, p.minor,
           p.multiProcessorCount);
  return 0;
}
// Kernel properties: kind 0 = v1 (faithful), 1 = refill (faithful), 2 = smem
// (param = NSH), 3 = symplectic (param = scheme).  out[6]: registers/thread,
// local (spill/stack) bytes/thread, blocks/SM, threads/SM, max threads/SM,
// dynamic smem bytes/block, block size used (out has 7 entries).
int co_kernel_info(int kind, int param, int block, int *out) {
  size_t smem = 0;
  switch (kind) {
    case 0:
      return co_info(k_integrate<false>, block, 0, out);
    case 1:
      return co_info(k_integrate_refill<false>, block, 0, out);
    case 2:  // param = 10 * NSH + MINB
      block = co_smem_block(block, param / 10);
      smem = (size_t)(param / 10) * DIM * block * sizeof(real);
#define CO_SMEM_INFO(K, M) \
  if (param == 10 * K + M) return co_info(k_integrate_smem<K, M>, block, smem, out);
      CO_SMEM_INFO(10, 2)
      CO_SMEM_INFO(10, 3)
      CO_SMEM_INFO(14, 3)
      CO_SMEM_INFO(14, 4)
      CO_SMEM_INFO(18, 3)
      CO_SMEM_INFO(18, 4)
      return -4;
    case 3:  // param = 10 * scheme + MINB
#define CO_SYMP_INFO(K, M) \
  if (param == 10 * K + M) return co_info(k_symp<K, M>, block, 0, out);
      CO_SYMP_INFO(SYMP_LEAPFROG, 2)
      CO_SYMP_INFO(SYMP_LEAPFROG, 4)
      CO_SYMP_INFO(SYMP_LEAPFROG, 8)
      CO_SYMP_INFO(SYMP_SYMPLEC4, 2)
      CO_SYMP_INFO(SYMP_SYMPLEC4, 4)
      CO_SYMP_INFO(SYMP_SYMPLEC4, 8)
      CO_SYMP_INFO(SYMP_SYMPLEC6, 2)
      CO_SYMP_INFO(SYMP_SYMPLEC6, 4)
      CO_SYMP_INFO(SYMP_SYMPLEC6, 8)
      return -4;
    case 4:  // two orbits per thread, param = 10 * scheme + MINB
#define CO_SYMP2_INFO(K, M) \
  if (param == 10 * K + M) return co_info(k_symp2<K, M>, block, 0, out);
      CO_SYMP2_INFO(SYMP_LEAPFROG, 1)
      CO_SYMP2_INFO(SYMP_LEAPFROG, 2)
      CO_SYMP2_INFO(SYMP_SYMPLEC4, 1)
      CO_SYMP2_INFO(SYMP_SYMPLEC4, 2)
      CO_SYMP2_INFO(SYMP_SYMPLEC6, 1)
      CO_SYMP2_INFO(SYMP_SYMPLEC6, 2)
      return -4;
  }
  return -4;
}

// Symplectic, host (same code; bit-identical to galpy's leapfrog_c/symplec4_c/
// symplec6_c when dt_fixed <= 0).  stats (optional, N x 3): nfev, ndt, 0.
int co_symp_host(int N, const real *yo, int nt, const real *t, double logrtol,
                 double logatol, int scheme, double dt_fixed, real *result, int *stats) {
  real rtol = (real)logrtol, atol = (real)logatol;  // galpy: log-space tolerances
  for (int ii = 0; ii < N; ii++) {
    real y[DIM];
    for (int i = 0; i < DIM; i++) y[i] = yo[DIM * ii + i];
    cyl_to_rect_galpy(y);
    int nf;
    real dtu;
    real *res = result + (size_t)DIM * nt * ii;
    if (scheme == SYMP_LEAPFROG)
      symp_orbit<SYMP_LEAPFROG>(y, nt, t, h_pots, rtol, atol, (real)dt_fixed, res, &nf, &dtu);
    else if (scheme == SYMP_SYMPLEC4)
      symp_orbit<SYMP_SYMPLEC4>(y, nt, t, h_pots, rtol, atol, (real)dt_fixed, res, &nf, &dtu);
    else if (scheme == SYMP_SYMPLEC6)
      symp_orbit<SYMP_SYMPLEC6>(y, nt, t, h_pots, rtol, atol, (real)dt_fixed, res, &nf, &dtu);
    else
      return -4;
    if (stats) {
      stats[3 * ii] = nf;
      stats[3 * ii + 1] = (int)((t[1] - t[0]) / dtu);
      stats[3 * ii + 2] = 0;
    }
  }
  return 0;
}

int co_symp_gpu(int N, const real *yo, int nt, const real *t, double logrtol,
                double logatol, int scheme, double dt_fixed, real *result, int *stats,
                int block, float *timings_ms, int minb, int per_thread) {
  if (block > 128) return -5;
  if (per_thread == 2 && !(dt_fixed > 0.)) return -6;  // pairs need a common dt
  real rtol = (real)logrtol, atol = (real)logatol;
  real *d_yo, *d_t, *d_res;
  int *d_stats = NULL, *d_counter = NULL;
  cudaEvent_t e0, e1, e2, e3;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventCreate(&e2));
  CK(cudaEventCreate(&e3));
  CK(cudaEventRecord(e0));
  CK(cudaMalloc(&d_yo, sizeof(real) * DIM * N));
  CK(cudaMalloc(&d_t, sizeof(real) * nt));
  CK(cudaMalloc(&d_res, sizeof(real) * DIM * nt * (size_t)N));
  CK(cudaMalloc(&d_counter, sizeof(int)));
  CK(cudaMemset(d_counter, 0, sizeof(int)));
  if (stats) CK(cudaMalloc(&d_stats, sizeof(int) * 3 * N));
  CK(cudaMemcpy(d_yo, yo, sizeof(real) * DIM * N, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_t, t, sizeof(real) * nt, cudaMemcpyHostToDevice));
  CK(cudaEventRecord(e1));
  int nb, rc;
  real dtf = (real)dt_fixed;
#define CO_SYMP_CASE(K, M)                                                      \
  case 10 * K + M:                                                              \
    rc = co_launch_cfg(k_symp<K, M>, N, block, 0, &nb);                         \
    if (rc) return rc;                                                          \
    k_symp<K, M><<<nb, block>>>(N, d_yo, nt, d_t, rtol, atol, dtf, d_res, d_stats, \
                                d_counter);                                     \
    break;
#define CO_SYMP2_CASE(K, M)                                                     \
  case 1000 + 10 * K + M:                                                       \
    rc = co_launch_cfg(k_symp2<K, M>, (N + 1) / 2, block, 0, &nb);              \
    if (rc) return rc;                                                          \
    k_symp2<K, M><<<nb, block>>>(N, d_yo, nt, d_t, rtol, atol, dtf, d_res, d_stats, \
                                 d_counter);                                    \
    break;
  switch ((per_thread == 2 ? 1000 : 0) + 10 * scheme + minb) {
    CO_SYMP2_CASE(SYMP_LEAPFROG, 1)
    CO_SYMP2_CASE(SYMP_LEAPFROG, 2)
    CO_SYMP2_CASE(SYMP_SYMPLEC4, 1)
    CO_SYMP2_CASE(SYMP_SYMPLEC4, 2)
    CO_SYMP2_CASE(SYMP_SYMPLEC6, 1)
    CO_SYMP2_CASE(SYMP_SYMPLEC6, 2)
    CO_SYMP_CASE(SYMP_LEAPFROG, 2)
    CO_SYMP_CASE(SYMP_LEAPFROG, 4)
    CO_SYMP_CASE(SYMP_LEAPFROG, 8)
    CO_SYMP_CASE(SYMP_SYMPLEC4, 2)
    CO_SYMP_CASE(SYMP_SYMPLEC4, 4)
    CO_SYMP_CASE(SYMP_SYMPLEC4, 8)
    CO_SYMP_CASE(SYMP_SYMPLEC6, 2)
    CO_SYMP_CASE(SYMP_SYMPLEC6, 4)
    CO_SYMP_CASE(SYMP_SYMPLEC6, 8)
    default:
      return -4;
  }
  CK(cudaGetLastError());
  CK(cudaEventRecord(e2));
  CK(cudaMemcpy(result, d_res, sizeof(real) * DIM * nt * (size_t)N, cudaMemcpyDeviceToHost));
  if (stats) CK(cudaMemcpy(stats, d_stats, sizeof(int) * 3 * N, cudaMemcpyDeviceToHost));
  CK(cudaEventRecord(e3));
  CK(cudaEventSynchronize(e3));
  if (timings_ms) {
    CK(cudaEventElapsedTime(timings_ms, e0, e1));
    CK(cudaEventElapsedTime(timings_ms + 1, e1, e2));
    CK(cudaEventElapsedTime(timings_ms + 2, e2, e3));
  }
  cudaFree(d_yo);
  cudaFree(d_t);
  cudaFree(d_res);
  cudaFree(d_counter);
  if (d_stats) cudaFree(d_stats);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaEventDestroy(e3);
  return 0;
}
}
