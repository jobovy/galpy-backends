// One-thread-per-orbit CUDA integration of galpy's dop853_c in MWPotential2014.
// C ABI for ctypes (see cuda_orbits.py).  Also runs the identical code on the
// host (co_integrate_host) to check the port against galpy's C bit-for-bit.
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "galpy_cuda.cuh"

static PotSet h_pots;
__constant__ PotSet d_pots;

#include "dop853_cuda.cuh"

#ifdef CO_NOINLINE_FORCE
// The single out-of-line copy of the force (reads the potential from d_pots).
__device__ __noinline__ CoAcc3 co_dev_accel(real x, real y, real z) {
  real q[DIM] = {x, y, z, R_(0.), R_(0.), R_(0.)}, a[DIM];
  evalRectDeriv(q, a, d_pots);
  return CoAcc3{a[3], a[4], a[5]};
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
  CK(cudaMemcpyToSymbol(d_pots, &h_pots, sizeof(PotSet)));
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
                     double fixed_h, const int *order) {
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
  if (refill) {
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
}
