// The AMD transcendental instructions on a card: v_rcp_f32, v_rsq_f32, v_sqrt_f32, v_exp_f32, v_log_f32, v_sin_f32, v_cos_f32
// over all 2^32 inputs (plain ordinary kernels, the instruction forced with inline asm), as hashes (same definition as
// mufu_probe.cu: 4096-result blocks, FNV-1a 64 over the 32-bit results, a chunk is FNV-1a over its blocks' hashes), as
// a check against a double-precision reference (counts only), or as a dump of a range. Our own code.
//
//   hipcc -O2 amd_trans_probe.cpp -o amd_trans_probe
//   amd_trans_probe <op> hash [chunk_bits=26]       one line a chunk and one "all" line
//   amd_trans_probe <op> check                      counts against a reference, over all inputs
//   amd_trans_probe <op> dump <start-hex> <count>   raw results to stdout as binary
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(3); } } while (0)

enum { RCP, RSQ, SQRT, EXP, LOG, SIN, COS, NOPS };
static const char* kNames[] = {"RCP", "RSQ", "SQRT", "EXP", "LOG", "SIN", "COS"};

template <int OP> __device__ __forceinline__ uint32_t run(uint32_t b) {
  float x = __uint_as_float(b), r;
  if (OP == RCP) asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
  if (OP == RSQ) asm volatile("v_rsq_f32 %0, %1" : "=v"(r) : "v"(x));
  if (OP == SQRT) asm volatile("v_sqrt_f32 %0, %1" : "=v"(r) : "v"(x));
  if (OP == EXP) asm volatile("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
  if (OP == LOG) asm volatile("v_log_f32 %0, %1" : "=v"(r) : "v"(x));
  if (OP == SIN) asm volatile("v_sin_f32 %0, %1" : "=v"(r) : "v"(x));
  if (OP == COS) asm volatile("v_cos_f32 %0, %1" : "=v"(r) : "v"(x));
  return __float_as_uint(r);
}
static const uint64_t kB = 0xcbf29ce484222325ull, kP = 0x100000001b3ull;

template <int OP> __global__ void hash_k(uint32_t base, uint32_t nblocks, uint64_t* out) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nblocks) return;
  uint32_t s = base + (t << 12);
  uint64_t h = kB;
  for (uint32_t i = 0; i < 4096; ++i) h = (h ^ run<OP>(s + i)) * kP;
  out[t] = h;
}
template <int OP> __global__ void dump_k(uint32_t base, uint32_t n, uint32_t* out) {
  uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = run<OP>(base + t);
}
// reference in double; returns false when the result is special (compared by class only)
template <int OP> __device__ double ref(float xf) {
  double x = xf;
  if (OP == RCP) return 1.0 / x;
  if (OP == RSQ) return 1.0 / sqrt(x);
  if (OP == SQRT) return sqrt(x);
  if (OP == EXP) return exp2(x);
  if (OP == LOG) return log2(x);
  if (OP == SIN) return sinpi(2.0 * x);
  return cospi(2.0 * x);
}
// counts: [0] inputs, [1] ordinary results compared, [2] != RN(ref), [3] err > 0.5 ulp+eps ... via buckets, [8] special-class mismatch, [9] max ulp bits
__global__ void zero_k(unsigned long long* c) { c[threadIdx.x] = 0; }
template <int OP> __global__ void check_k(unsigned long long* c) {
  unsigned long long loc[12] = {0};
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 32); i += (uint64_t)gridDim.x * blockDim.x) {
    uint32_t b = (uint32_t)i;
    float x = __uint_as_float(b);
    if ((OP == SIN || OP == COS) && !(fabsf(x) <= 256.0f)) continue;   // the documented input range
    uint32_t rb = run<OP>(b);
    float r = __uint_as_float(rb);
    double d = ref<OP>(x);
    loc[0]++;
    if (isnan(d) || isinf(d) || d == 0.0 || isnan(r) || isinf(r) || fabs(d) > 3.4e38 || fabs(d) < 1.2e-38) {
      uint32_t dc = isnan(d) ? 1 : isinf(d) ? 2 : d == 0.0 ? 3 : 0;
      uint32_t rc = isnan(r) ? 1 : isinf(r) ? 2 : r == 0.0f ? 3 : 0;
      if (dc != rc || (dc >= 2 && signbit(d) != signbit(r))) loc[8]++;
      if (dc == rc && dc == 0) loc[10]++;
      continue;
    }
    loc[1]++;
    float rn = (float)d;
    if (rn != r) loc[2]++;
    // error in ulps of the float format at d
    int ex; frexp(d, &ex);
    double ulp = ldexp(1.0, ex - 24);
    double e = fabs((double)r - d) / ulp;
    if (e > 0.5000001) loc[3]++;
    if (e > 1.0) loc[4]++;
    if (e > 2.0) loc[5]++;
    if (e > 4.0) loc[6]++;
    if (e > 1.0 && false) loc[7]++;
    // truncation vs rounding indicator: result below d in magnitude
    if (fabs((double)r) < fabs(d)) loc[9]++;
    // max ulp
    unsigned long long eb = (unsigned long long)(e * 1e6);
    if (eb > loc[11]) loc[11] = eb;
  }
  for (int k = 0; k < 11; ++k) atomicAdd(&c[k], loc[k]);
  atomicMax(&c[11], loc[11]);
}

// The double forms: 2^64 inputs cannot be covered; a deterministic pseudo-random sample (splitmix64 of the index) of 2^n
// inputs, compared with the correctly rounded double result by distance in ulps (bit patterns), plus the special classes.
__device__ __forceinline__ uint64_t mix(uint64_t z) { z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
template <int OP> __device__ __forceinline__ uint64_t run64(uint64_t b) {
  double x = __longlong_as_double((long long)b), r;
  if (OP == RCP) asm volatile("v_rcp_f64 %0, %1" : "=v"(r) : "v"(x));
  if (OP == RSQ) asm volatile("v_rsq_f64 %0, %1" : "=v"(r) : "v"(x));
  if (OP == SQRT) asm volatile("v_sqrt_f64 %0, %1" : "=v"(r) : "v"(x));
  return (uint64_t)__double_as_longlong(r);
}
// counts: [0] n, [1] ordinary, [2] != RN, [3] |ulp diff|==1, [4] ==2, [5] >2, [6] max ulp diff, [7] special-class mismatch, [8] result below RN (magnitude)
template <int OP> __global__ void peek64_k(const uint64_t* in, uint64_t* out) { out[threadIdx.x] = run64<OP>(in[threadIdx.x]); }
template <int OP> __global__ void check64_k(uint64_t n, unsigned long long* c) {
  unsigned long long loc[9] = {0};
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
    uint64_t b = mix(i);
    if ((i & 3) == 0) b = (b & 0x800fffffffffffffull) | ((1023ull + (mix(i + 77) % 40) - 20) << 52);   // exponents near 1
    double x = __longlong_as_double((long long)b);
    double d = OP == RCP ? 1.0 / x : OP == SQRT ? sqrt(x) : 1.0 / sqrt(x);
    uint64_t rb = run64<OP>(b); double r = __longlong_as_double((long long)rb);
    loc[0]++;
    bool dn = isnan(d) || isinf(d) || d == 0.0 || fabs(d) < 2.3e-308, rn_ = isnan(r) || isinf(r) || r == 0.0 || fabs(r) < 2.3e-308;
    if (dn || rn_) { uint32_t dc = isnan(d) ? 1 : isinf(d) ? 2 : d == 0.0 ? 3 : 0, rc = isnan(r) ? 1 : isinf(r) ? 2 : r == 0.0 ? 3 : 0; if (dc != rc || (dc >= 2 && signbit(d) != signbit(r))) loc[7]++; continue; }
    loc[1]++;
    uint64_t db = (uint64_t)__double_as_longlong(d);
    long long df = (long long)(rb & 0x7fffffffffffffffull) - (long long)(db & 0x7fffffffffffffffull);
    unsigned long long a = df < 0 ? -df : df;
    if (a) loc[2]++; if (a == 1) loc[3]++; else if (a == 2) loc[4]++; else if (a > 2) loc[5]++;
    if (a > loc[6]) loc[6] = a;
    if (df < 0) loc[8]++;
  }
  for (int k = 0; k < 9; ++k) if (k != 6) atomicAdd(&c[k], loc[k]);
  atomicMax(&c[6], loc[6]);
}
#define DISPATCH(op, f, ...) switch (op) { case RCP: f<RCP><<<__VA_ARGS__; break; case RSQ: f<RSQ><<<__VA_ARGS__; break; case SQRT: f<SQRT><<<__VA_ARGS__; break; case EXP: f<EXP><<<__VA_ARGS__; break; case LOG: f<LOG><<<__VA_ARGS__; break; case SIN: f<SIN><<<__VA_ARGS__; break; default: f<COS><<<__VA_ARGS__; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: amd_trans_probe <op> hash|check|dump\n"); return 2; }
  int op = -1; for (int i = 0; i < NOPS; ++i) if (!strcmp(argv[1], kNames[i])) op = i;
  if (op < 0) { fprintf(stderr, "unknown op\n"); return 2; }
  std::string mode = argv[2];
  hipDeviceProp_t p; OK(hipGetDeviceProperties(&p, 0));
  fprintf(stderr, "# device %s %s\n", p.name, p.gcnArchName);
  if (mode == "hash") {
    unsigned cb = argc > 3 ? atoi(argv[3]) : 26; uint32_t nb = 1u << (cb - 12);
    uint64_t* d; OK(hipMalloc(&d, nb * 8)); std::vector<uint64_t> h(nb);
    uint64_t total = kB;
    for (uint64_t s = 0; s < (1ull << 32); s += (1ull << cb)) {
      DISPATCH(op, hash_k, (nb + 63) / 64, 64>>>((uint32_t)s, nb, d));
      OK(hipGetLastError()); OK(hipDeviceSynchronize());
      OK(hipMemcpy(h.data(), d, nb * 8, hipMemcpyDeviceToHost));
      uint64_t c = kB; for (uint32_t i = 0; i < nb; ++i) { c = (c ^ h[i]) * kP; total = (total ^ h[i]) * kP; }
      printf("%s %08x %016llx\n", kNames[op], (uint32_t)s, (unsigned long long)c); fflush(stdout);
    }
    printf("%s all %016llx\n", kNames[op], (unsigned long long)total);
  } else if (mode == "check") {
    unsigned long long* d; OK(hipMalloc(&d, 16 * 8)); unsigned long long c[16];
    zero_k<<<1, 16>>>(d);
    DISPATCH(op, check_k, 1024, 256>>>(d));
    OK(hipGetLastError()); OK(hipDeviceSynchronize()); OK(hipMemcpy(c, d, 16 * 8, hipMemcpyDeviceToHost));
    printf("%s inputs=%llu ordinary=%llu notRN=%llu gt0.5=%llu gt1=%llu gt2=%llu gt4=%llu specialMismatch=%llu below=%llu specialSameClassOrdinaryEdge=%llu maxulp=%.6f\n", kNames[op], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[8], c[9], c[10], c[11] / 1e6);
  } else if (mode == "check64") {
    if (op > SQRT) { fprintf(stderr, "RCP, RSQ or SQRT\n"); return 2; }
    unsigned long long* d; OK(hipMalloc(&d, 16 * 8)); unsigned long long c[16];
    zero_k<<<1, 16>>>(d);
    uint64_t n = 1ull << (argc > 3 ? atoi(argv[3]) : 27);
    switch (op) { case RCP: check64_k<RCP><<<1024, 256>>>(n, d); break; case RSQ: check64_k<RSQ><<<1024, 256>>>(n, d); break; default: check64_k<SQRT><<<1024, 256>>>(n, d); }
    OK(hipGetLastError()); OK(hipDeviceSynchronize()); OK(hipMemcpy(c, d, 16 * 8, hipMemcpyDeviceToHost));
    printf("%s64 n=%llu ordinary=%llu notRN=%llu ulp1=%llu ulp2=%llu gt2=%llu maxulp=%llu specialMismatch=%llu below=%llu\n", kNames[op], c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
  } else if (mode == "peek64") {   // the double forms on the hex operands given as arguments (at most 64)
    int n = argc - 3; uint64_t hin[64], hout[64]; for (int i = 0; i < n; ++i) hin[i] = strtoull(argv[3 + i], 0, 16);
    uint64_t *di, *dout; OK(hipMalloc(&di, 512)); OK(hipMalloc(&dout, 512)); OK(hipMemcpy(di, hin, 512, hipMemcpyHostToDevice));
    switch (op) { case RCP: peek64_k<RCP><<<1, n>>>(di, dout); break; case RSQ: peek64_k<RSQ><<<1, n>>>(di, dout); break; default: peek64_k<SQRT><<<1, n>>>(di, dout); }
    OK(hipGetLastError()); OK(hipDeviceSynchronize()); OK(hipMemcpy(hout, dout, 512, hipMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) printf("%016llx %016llx\n", (unsigned long long)hin[i], (unsigned long long)hout[i]);
  } else if (mode == "dump") {
    uint32_t start = strtoul(argv[3], 0, 16), n = strtoul(argv[4], 0, 0);
    uint32_t* d; OK(hipMalloc(&d, n * 4)); std::vector<uint32_t> h(n);
    DISPATCH(op, dump_k, (n + 255) / 256, 256>>>(start, n, d));
    OK(hipGetLastError()); OK(hipDeviceSynchronize()); OK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
    fwrite(h.data(), 4, n, stdout);
  }
  return 0;
}
