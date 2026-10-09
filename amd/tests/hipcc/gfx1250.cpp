// gfx1250 (CDNA 5, MI455X) instructions the executor runs its own way, each against the host (gfx1250.gfx1250,
// wave32). Their meanings are the ISA XML's descriptions, as the simulator reads them; there is no card to say
// whether the hardware agrees.
//
//   - 64-bit integer arithmetic in the vector unit (add, sub, mul, min, max, multiply-add of a 32-bit pair);
//   - v_add_max / v_add_min: a clamped add, then the other operand;
//   - v_ashr_pk_i8_i32 and v_ashr_pk_u8_i32, v_sat_pk4_i4_i8 and v_sat_pk4_u4_u8 (16-bit results);
//   - v_tanh_f32, the bfloat16 transcendentals, and the 8-bit float to half conversions;
//   - an f64 atomic add, and the prefetch the compiler writes for __builtin_prefetch.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

constexpr int kLanes = 32;

__global__ void int64_ops(const uint64_t* a, const uint64_t* b, uint64_t* o) {
  const int l = threadIdx.x;
  __builtin_prefetch(a + l);
  const uint64_t x = a[l], y = b[l];
  o[6 * l + 0] = x + y;
  o[6 * l + 1] = x - y;
  o[6 * l + 2] = x * y;
  o[6 * l + 3] = x < y ? x : y;
  o[6 * l + 4] = static_cast<int64_t>(x) > static_cast<int64_t>(y) ? x : y;
  o[6 * l + 5] = static_cast<uint64_t>(static_cast<uint32_t>(x)) * static_cast<uint32_t>(y) + b[(l + 1) % kLanes];
}
__global__ void add_minmax(const int* a, const int* b, const int* c, int* o) {
  const int l = threadIdx.x;
  int r0, r1, r2, r3;
  asm volatile("v_add_max_i32 %0, %1, %2, %3" : "=v"(r0) : "v"(a[l]), "v"(b[l]), "v"(c[l]));
  asm volatile("v_add_min_i32 %0, %1, %2, %3" : "=v"(r1) : "v"(a[l]), "v"(b[l]), "v"(c[l]));
  asm volatile("v_add_max_u32 %0, %1, %2, %3" : "=v"(r2) : "v"(a[l]), "v"(b[l]), "v"(c[l]));
  asm volatile("v_add_min_u32 %0, %1, %2, %3" : "=v"(r3) : "v"(a[l]), "v"(b[l]), "v"(c[l]));
  o[4 * l + 0] = r0;
  o[4 * l + 1] = r1;
  o[4 * l + 2] = r2;
  o[4 * l + 3] = r3;
}
__global__ void packs(const int* a, const int* b, const int* n, const unsigned* q, unsigned* o) {
  const int l = threadIdx.x;
  unsigned r0, r1, r2, r3;
  asm volatile("v_ashr_pk_i8_i32 %0, %1, %2, %3" : "=v"(r0) : "v"(a[l]), "v"(b[l]), "v"(n[l]));
  asm volatile("v_ashr_pk_u8_i32 %0, %1, %2, %3" : "=v"(r1) : "v"(a[l]), "v"(b[l]), "v"(n[l]));
  asm volatile("v_sat_pk4_i4_i8 %0, %1" : "=v"(r2) : "v"(q[l]));
  asm volatile("v_sat_pk4_u4_u8 %0, %1" : "=v"(r3) : "v"(q[l]));
  o[4 * l + 0] = r0;
  o[4 * l + 1] = r1;
  o[4 * l + 2] = r2;
  o[4 * l + 3] = r3;
}
__global__ void floats(const float* x, const unsigned* bf, const unsigned* f8, float* tanh_out, unsigned* o) {
  const int l = threadIdx.x;
  float t;
  asm volatile("v_tanh_f32 %0, %1" : "=v"(t) : "v"(x[l]));
  tanh_out[l] = t;
  unsigned r[7];
  asm volatile("v_rcp_bf16 %0, %1" : "=v"(r[0]) : "v"(bf[l]));
  asm volatile("v_sqrt_bf16 %0, %1" : "=v"(r[1]) : "v"(bf[l]));
  asm volatile("v_rsq_bf16 %0, %1" : "=v"(r[2]) : "v"(bf[l]));
  asm volatile("v_log_bf16 %0, %1" : "=v"(r[3]) : "v"(bf[l]));
  asm volatile("v_exp_bf16 %0, %1" : "=v"(r[4]) : "v"(bf[l]));
  asm volatile("v_cvt_f16_fp8 %0, %1" : "=v"(r[5]) : "v"(f8[l]));
  asm volatile("v_cvt_pk_f16_bf8 %0, %1" : "=v"(r[6]) : "v"(f8[l]));
  for (int k = 0; k < 7; ++k) o[7 * l + k] = r[k];
}
__global__ void f64_atomic(double* sum) {
  atomicAdd(sum, 0.5 * (threadIdx.x + 1));
}

static int g_checks = 0, g_failed = 0;
// A pointer to a work-item's own array and one into LDS, as a function that cannot see where they point takes them: flat
// addresses, the first made from src_flat_scratch_base with the lane's number in it.
__attribute__((noinline)) __device__ int weighted_sum(const int* p, int n) {
  int s = 0;
  for (int i = 0; i < n; ++i) s += p[i] * (i + 1);
  return s;
}
__global__ void flat_pointers(int* o) {
  int a[8];
  for (int i = 0; i < 8; ++i) a[i] = static_cast<int>(threadIdx.x) * 10 + i;
  __shared__ int lds[64];
  lds[threadIdx.x] = static_cast<int>(threadIdx.x) + 100;
  __syncthreads();
  o[2 * threadIdx.x] = weighted_sum(a, 8);
  o[2 * threadIdx.x + 1] = weighted_sum(lds + threadIdx.x, 1);
}

static void report(const char* name, int wrong, int n) {
  ++g_checks;
  if (wrong) ++g_failed;
  std::printf("%s: %d of %d wrong\n", name, wrong, n);
}
static uint32_t sat8(int64_t v, bool is_signed) {
  return static_cast<uint32_t>(is_signed ? (v < -128 ? -128 : v > 127 ? 127 : v) : (v < 0 ? 0 : v > 255 ? 255 : v)) & 0xFF;
}
static uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>(u >> 16);
}
static float bf16_value(uint32_t bits) {
  const uint32_t u = bits << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

template <class T>
static T* upload(const T* host, size_t n) {
  T* d = nullptr;
  if (hipMalloc(&d, n * sizeof(T)) != hipSuccess) return nullptr;
  hipMemcpy(d, host, n * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

int main() {
  uint64_t a64[kLanes], b64[kLanes];
  int ai[kLanes], bi[kLanes], ci[kLanes], ni[kLanes];
  unsigned q[kLanes];
  uint32_t seed = 12345;
  const auto next = [&] {
    seed = seed * 1103515245u + 12345u;
    return seed;
  };
  for (int l = 0; l < kLanes; ++l) {
    a64[l] = (uint64_t{next()} << 32 | next()) >> (l % 5 * 8);
    b64[l] = (uint64_t{next()} << 32 | next()) >> (l % 7 * 8);
    ai[l] = static_cast<int>(next());
    bi[l] = static_cast<int>(next());
    ci[l] = static_cast<int>(next()) >> (l % 4 * 8);
    ni[l] = l;
    q[l] = next();
  }
  ai[0] = INT32_MAX; bi[0] = 1; ci[0] = 5;          // the sum saturates upward
  ai[1] = INT32_MIN; bi[1] = -1; ci[1] = -5;        // and downward
  ai[2] = -1; bi[2] = 1; ci[2] = 7;                 // as unsigned, wrapping past 2^32 - 1
  a64[0] = ~0ull; b64[0] = 1;

  // 64-bit integers.
  {
    uint64_t* da = upload(a64, kLanes);
    uint64_t* db = upload(b64, kLanes);
    uint64_t* d = nullptr;
    CHECK(hipMalloc(&d, 6 * kLanes * sizeof(uint64_t)));
    int64_ops<<<1, kLanes>>>(da, db, d);
    CHECK(hipDeviceSynchronize());
    uint64_t out[6 * kLanes];
    CHECK(hipMemcpy(out, d, sizeof out, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < kLanes; ++l) {
      const uint64_t x = a64[l], y = b64[l];
      const uint64_t want[6] = {x + y, x - y, x * y, x < y ? x : y,
                                static_cast<int64_t>(x) > static_cast<int64_t>(y) ? x : y,
                                uint64_t{static_cast<uint32_t>(x)} * static_cast<uint32_t>(y) + b64[(l + 1) % kLanes]};
      for (int k = 0; k < 6; ++k) wrong += out[6 * l + k] != want[k];
    }
    report("64-bit integers", wrong, 6 * kLanes);
  }
  // Clamped adds.
  {
    int *da = upload(ai, kLanes), *db = upload(bi, kLanes), *dc = upload(ci, kLanes);
    int* d = nullptr;
    CHECK(hipMalloc(&d, 4 * kLanes * sizeof(int)));
    add_minmax<<<1, kLanes>>>(da, db, dc, d);
    CHECK(hipDeviceSynchronize());
    int out[4 * kLanes];
    CHECK(hipMemcpy(out, d, sizeof out, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < kLanes; ++l) {
      const int64_t s = std::min<int64_t>(std::max<int64_t>(int64_t{ai[l]} + bi[l], INT32_MIN), INT32_MAX);
      const uint64_t u = std::min<uint64_t>(uint64_t{static_cast<uint32_t>(ai[l])} + static_cast<uint32_t>(bi[l]), 0xFFFFFFFFu);
      const uint32_t cu = static_cast<uint32_t>(ci[l]);
      const int want[4] = {static_cast<int>(std::max<int64_t>(s, ci[l])), static_cast<int>(std::min<int64_t>(s, ci[l])),
                           static_cast<int>(std::max<uint64_t>(u, cu)), static_cast<int>(std::min<uint64_t>(u, cu))};
      for (int k = 0; k < 4; ++k) wrong += out[4 * l + k] != want[k];
    }
    report("clamped adds then max or min", wrong, 4 * kLanes);
  }
  // Packed saturating shifts and narrowing.
  {
    int *da = upload(ai, kLanes), *db = upload(bi, kLanes), *dn = upload(ni, kLanes);
    unsigned* dq = upload(q, kLanes);
    unsigned* d = nullptr;
    CHECK(hipMalloc(&d, 4 * kLanes * sizeof(unsigned)));
    packs<<<1, kLanes>>>(da, db, dn, dq, d);
    CHECK(hipDeviceSynchronize());
    unsigned out[4 * kLanes];
    CHECK(hipMemcpy(out, d, sizeof out, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < kLanes; ++l) {
      const int n = ni[l] & 31;
      unsigned want[4];
      want[0] = sat8(ai[l] >> n, true) | sat8(bi[l] >> n, true) << 8;
      want[1] = sat8(ai[l] >> n, false) | sat8(bi[l] >> n, false) << 8;
      want[2] = want[3] = 0;
      for (int k = 0; k < 4; ++k) {
        const int sb = static_cast<int8_t>(q[l] >> (8 * k)), ub = (q[l] >> (8 * k)) & 0xFF;
        want[2] |= (static_cast<unsigned>(sb < -8 ? -8 : sb > 7 ? 7 : sb) & 0xF) << (4 * k);
        want[3] |= (static_cast<unsigned>(ub > 15 ? 15 : ub) & 0xF) << (4 * k);
      }
      // (Each writes 16 bits; the other half of the register is the destination's old one.)
      for (int k = 0; k < 4; ++k) wrong += (out[4 * l + k] & 0xFFFF) != want[k];
    }
    report("packed saturating shifts and narrowing", wrong, 4 * kLanes);
  }
  // Floats: tanh, the bfloat16 functions on exact inputs, fp8 and bf8 to half.
  {
    float x[kLanes];
    unsigned bf[kLanes], f8[kLanes];
    for (int l = 0; l < kLanes; ++l) {
      x[l] = (l - 16) * 0.37f;
      const float v = 1.0f * (1 << (l % 6));   // 1, 2, 4, 8, 16, 32: exact in bfloat16
      bf[l] = bf16_bits(v);
      f8[l] = (0x30 + l) | (0x38 + l % 5) << 8;
    }
    float* dx = upload(x, kLanes);
    unsigned *dbf = upload(bf, kLanes), *df8 = upload(f8, kLanes);
    float* dt = nullptr;
    unsigned* d = nullptr;
    CHECK(hipMalloc(&dt, kLanes * sizeof(float)));
    CHECK(hipMalloc(&d, 7 * kLanes * sizeof(unsigned)));
    floats<<<1, kLanes>>>(dx, dbf, df8, dt, d);
    CHECK(hipDeviceSynchronize());
    float t[kLanes];
    unsigned out[7 * kLanes];
    CHECK(hipMemcpy(t, dt, sizeof t, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(out, d, sizeof out, hipMemcpyDeviceToHost));
    int wrong_tanh = 0, wrong_bf = 0;
    for (int l = 0; l < kLanes; ++l) {
      wrong_tanh += std::fabs(t[l] - std::tanh(x[l])) > 1e-6f;
      const float v = bf16_value(bf[l]);
      const float want[5] = {1.0f / v, std::sqrt(v), 1.0f / std::sqrt(v), std::log2(v), std::exp2(v)};
      for (int k = 0; k < 5; ++k) wrong_bf += (out[7 * l + k] & 0xFFFF) != bf16_bits(want[k]);
    }
    report("tanh", wrong_tanh, kLanes);
    report("bfloat16 reciprocal, roots, log2 and exp2", wrong_bf, 5 * kLanes);
    // fp8 (OCP E4M3) to half: the byte 0x38 is 1.0, and bf8 (E5M2)'s 0x3C is 1.0; 0x30 / 0x34 are 0.5.
    int wrong_f8 = 0;
    const auto half_of = [](unsigned bits) {
      _Float16 h;
      const uint16_t b = bits & 0xFFFF;
      std::memcpy(&h, &b, 2);
      return static_cast<float>(h);
    };
    // OCP E4M3 (bias 7, 3 mantissa bits) and E5M2 (bias 15, 2), the formats gfx12 and later use.
    const auto e4m3 = [](unsigned b) {
      const int e = (b >> 3) & 15, m = b & 7;
      return (b & 0x80 ? -1.0f : 1.0f) * (e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7));
    };
    const auto e5m2 = [](unsigned b) {
      const int e = (b >> 2) & 31, m = b & 3;
      return (b & 0x80 ? -1.0f : 1.0f) * (e == 0 ? m / 4.0f * std::ldexp(1.0f, -14) : (1.0f + m / 4.0f) * std::ldexp(1.0f, e - 15));
    };
    for (int l = 0; l < kLanes; ++l) {
      wrong_f8 += std::fabs(half_of(out[7 * l + 5]) - e4m3(f8[l] & 0xFF)) > 1e-6f;
      wrong_f8 += std::fabs(half_of(out[7 * l + 6]) - e5m2(f8[l] & 0xFF)) > 1e-6f;
      wrong_f8 += std::fabs(half_of(out[7 * l + 6] >> 16) - e5m2((f8[l] >> 8) & 0xFF)) > 1e-6f;
    }
    report("fp8 and bf8 to half", wrong_f8, 3 * kLanes);
  }
  // Flat pointers to private and to shared memory.
  {
    int* d = nullptr;
    CHECK(hipMalloc(&d, 2 * kLanes * sizeof(int)));
    flat_pointers<<<1, kLanes>>>(d);
    CHECK(hipDeviceSynchronize());
    std::vector<int> out(2 * kLanes);
    CHECK(hipMemcpy(out.data(), d, out.size() * sizeof(int), hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < kLanes; ++l) wrong += out[2 * l] != 360 * l + 168 || out[2 * l + 1] != l + 100;
    report("flat pointers to private and shared memory", wrong, kLanes);
  }
  // An f64 atomic add.
  {
    double* d = nullptr;
    CHECK(hipMalloc(&d, sizeof(double)));
    CHECK(hipMemset(d, 0, sizeof(double)));
    f64_atomic<<<1, kLanes>>>(d);
    CHECK(hipDeviceSynchronize());
    double got = 0;
    CHECK(hipMemcpy(&got, d, sizeof got, hipMemcpyDeviceToHost));
    report("f64 atomic add", got != 0.5 * kLanes * (kLanes + 1) / 2, 1);
  }
  std::printf("gfx1250: %d checks, %d failed\n", g_checks, g_failed);
  return g_failed ? 1 : 0;
}
