// Host-only dynamic operation count of the ported code (same headers, `real`
// replaced by a counting double).  Counts per force evaluation (evalRectDeriv,
// incl. GSL gamma_inc_P) and per orbit (integrator + forces).
//   g++ -O1 -std=c++17 flopcount.cpp -o flopcount && ./flopcount
#include <cmath>
#include <cstdio>
#include <vector>
enum { ADD, MUL, DIV, SQRT, POW, LOG, EXP, TRIG, CMP, NCAT };
static const char *catname[NCAT] = {"add/sub", "mul", "div", "sqrt", "pow", "log", "exp", "trig", "cmp"};
static long long cnt[NCAT];
static long long nforce = 0;
struct CountedReal {
  double v;
  constexpr CountedReal() : v(0) {}
  constexpr CountedReal(double x) : v(x) {}
  explicit operator long() const { return (long)v; }
  constexpr CountedReal operator-() const { return CountedReal(-v); }
  CountedReal &operator+=(CountedReal o) { cnt[ADD]++; v += o.v; return *this; }
  CountedReal &operator*=(CountedReal o) { cnt[MUL]++; v *= o.v; return *this; }
};
#define BIN(op, c)                                                              \
  inline CountedReal operator op(CountedReal a, CountedReal b) { cnt[c]++; return CountedReal(a.v op b.v); } \
  inline CountedReal operator op(CountedReal a, double b) { cnt[c]++; return CountedReal(a.v op b); } \
  inline CountedReal operator op(double a, CountedReal b) { cnt[c]++; return CountedReal(a op b.v); } \
  inline CountedReal operator op(CountedReal a, int b) { cnt[c]++; return CountedReal(a.v op b); } \
  inline CountedReal operator op(int a, CountedReal b) { cnt[c]++; return CountedReal(a op b.v); }
BIN(+, ADD) BIN(-, ADD) BIN(*, MUL) BIN(/, DIV)
#define CMPOP(op)                                                               \
  inline bool operator op(CountedReal a, CountedReal b) { cnt[CMP]++; return a.v op b.v; } \
  inline bool operator op(CountedReal a, double b) { cnt[CMP]++; return a.v op b; }
CMPOP(<) CMPOP(>) CMPOP(<=) CMPOP(==)
inline CountedReal fabs(CountedReal a) { return CountedReal(std::fabs(a.v)); }
inline CountedReal sqrt(CountedReal a) { cnt[SQRT]++; return CountedReal(std::sqrt(a.v)); }
inline CountedReal pow(CountedReal a, CountedReal b) { cnt[POW]++; return CountedReal(std::pow(a.v, b.v)); }
inline CountedReal log(CountedReal a) { cnt[LOG]++; return CountedReal(std::log(a.v)); }
inline CountedReal exp(CountedReal a) { cnt[EXP]++; return CountedReal(std::exp(a.v)); }
inline CountedReal cos(CountedReal a) { cnt[TRIG]++; return CountedReal(std::cos(a.v)); }
inline CountedReal sin(CountedReal a) { cnt[TRIG]++; return CountedReal(std::sin(a.v)); }
inline CountedReal atan2(CountedReal a, CountedReal b) { cnt[TRIG]++; return CountedReal(std::atan2(a.v, b.v)); }
#define CO_COUNT
static long long fcnt[NCAT];
struct ForceScope {  // RAII: attribute the ops inside evalRectDeriv to the force
  long long c0[NCAT];
  ForceScope() { for (int c = 0; c < NCAT; c++) c0[c] = cnt[c]; }
  ~ForceScope() { for (int c = 0; c < NCAT; c++) fcnt[c] += cnt[c] - c0[c]; nforce++; }
};
#define CO_FORCE_HOOK ForceScope _fs
#include "dop853_cuda.cuh"

int load_pots(PotSet &P) {  // flopcount_input.txt: npot, (type off)*, nargs, args*
  FILE *f = fopen("flopcount_input.txt", "r");
  if (!f) return 1;
  int nargs;
  if (fscanf(f, "%d", &P.npot) != 1) return 1;
  for (int i = 0; i < P.npot; i++)
    if (fscanf(f, "%d %d", &P.type[i], &P.off[i]) != 2) return 1;
  if (fscanf(f, "%d", &nargs) != 1) return 1;
  for (int i = 0; i < nargs; i++) {
    double v;
    if (fscanf(f, "%lf", &v) != 1) return 1;
    P.args[i] = v;
  }
  fclose(f);
  return 0;
}

int main() {
  // MWPotential2014 repacked args (from mw2014_params.json via cuda_orbits.repack)
  PotSet P;
  if (load_pots(P)) return 1;
  FILE *f = fopen("flopcount_ics.txt", "r");
  if (!f) { fprintf(stderr, "need flopcount_ics.txt\n"); return 1; }
  std::vector<double> ics;
  double v;
  while (fscanf(f, "%lf", &v) == 1) ics.push_back(v);
  fclose(f);
  int N = ics.size() / 6, nt = 16;
  std::vector<CountedReal> t(nt), res(6 * nt);
  for (int j = 0; j < nt; j++) t[j] = 56.2 * j / (nt - 1);
  for (int lazy = 0; lazy < 2; lazy++) {
    for (int c = 0; c < NCAT; c++) cnt[c] = 0;
    long long nf = 0;
    for (int ii = 0; ii < N; ii++) {
      CountedReal y[6];
      for (int i = 0; i < 6; i++) y[i] = ics[6 * ii + i];
      cyl_to_rect_galpy(y);
      DopStats st;
      CountedReal rt = std::exp(std::log(1e-8)), at = rt;
      if (lazy) dop853_orbit<true>(y, nt, t.data(), P, rt, at, res.data(), &st);
      else dop853_orbit<false>(y, nt, t.data(), P, rt, at, res.data(), &st);
      nf += st.nfev;
    }
    long long tot = 0;
    printf("%s dense output: N=%d, nfev/orbit=%.1f\n", lazy ? "lazy" : "faithful", N, (double)nf / N);
    printf("  per orbit (integrator + forces):");
    for (int c = 0; c < NCAT; c++) { printf(" %s=%.0f", catname[c], (double)cnt[c] / N); tot += cnt[c]; }
    printf("\n");
  }
  printf("per force evaluation (evalRectDeriv incl. gamma_inc_P), mean over %lld evals:", nforce);
  for (int c = 0; c < NCAT; c++) printf(" %s=%.2f", catname[c], (double)fcnt[c] / nforce);
  printf("\n");
  return 0;
}
