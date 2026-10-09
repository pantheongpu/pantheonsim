// gfx1250 (CDNA 5, MI455X) LDS atomics and exchanges that are new or newly spelled (ds1250.gfx1250, wave32), each against the
// host. The oracle is AMD's CDNA5 ISA pseudocode (section 11 and the DS instruction list): conditional subtract and
// subtract clamped at zero, packed half and bfloat16 add, masked OR, the compare-store, the two-address exchanges, the
// 32-bit conditional exchanges, the address-plus-lane-number store and load, and the backward permute that fetches from lanes
// that are off. Every lane works on a word (or pair of them) of its own: LDS starts as a known pattern, the lane's data comes
// from the global arrays, and both what the lane is handed back and what LDS ends up holding are compared.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

#define LP(T, p) ((__attribute__((address_space(3))) T*)(p))
static __device__ uint32_t lds_addr(void* p) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(LP(char, p))); }

constexpr int kWords = 128;   // LDS words in each kernel's array: 32 lanes, with room to spare for the two-address forms

// One word per lane: LDS[l] starts as init[l], the op is applied with data a[l] (and b[l]), and out gets the returned value
// (words 0-1 of the lane's slot) and the LDS word afterwards (word 2).
#define WORD_OP(NAME, ASM_RTN)                                                           \
  __global__ void NAME(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) { \
    __shared__ uint32_t lds[kWords];                                                     \
    const int l = threadIdx.x;                                                           \
    for (int i = l; i < kWords; i += 32) lds[i] = init[i];                               \
    __syncthreads();                                                                     \
    uint32_t r = 0;                                                                      \
    const uint32_t addr = lds_addr(lds + l);                                             \
    const uint32_t x = a[l], y = b[l];                                                   \
    asm volatile(ASM_RTN : "+v"(r) : "v"(addr), "v"(x), "v"(y) : "memory");             \
    asm volatile("s_wait_dscnt 0" ::: "memory");                                         \
    __syncthreads();                                                                     \
    out[3 * l] = r;                                                                      \
    out[3 * l + 2] = lds[l];                                                             \
  }
WORD_OP(k_cond_sub_rtn, "ds_cond_sub_rtn_u32 %0, %1, %2")
WORD_OP(k_sub_clamp_rtn, "ds_sub_clamp_rtn_u32 %0, %1, %2")
WORD_OP(k_pk_add_f16_rtn, "ds_pk_add_rtn_f16 %0, %1, %2")
WORD_OP(k_pk_add_bf16_rtn, "ds_pk_add_rtn_bf16 %0, %1, %2")
WORD_OP(k_mskor_rtn, "ds_mskor_rtn_b32 %0, %1, %2, %3")
// The forms with no return value.
#define WORD_OP_NORET(NAME, ASM)                                                         \
  __global__ void NAME(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) { \
    __shared__ uint32_t lds[kWords];                                                     \
    const int l = threadIdx.x;                                                           \
    for (int i = l; i < kWords; i += 32) lds[i] = init[i];                               \
    __syncthreads();                                                                     \
    const uint32_t addr = lds_addr(lds + l);                                             \
    const uint32_t x = a[l], y = b[l];                                                   \
    asm volatile(ASM : : "v"(addr), "v"(x), "v"(y) : "memory");                          \
    asm volatile("s_wait_dscnt 0" ::: "memory");                                         \
    __syncthreads();                                                                     \
    out[3 * l] = 0;                                                                      \
    out[3 * l + 2] = lds[l];                                                             \
  }
WORD_OP_NORET(k_cond_sub, "ds_cond_sub_u32 %0, %1")
WORD_OP_NORET(k_sub_clamp, "ds_sub_clamp_u32 %0, %1")
WORD_OP_NORET(k_pk_add_f16, "ds_pk_add_f16 %0, %1")
WORD_OP_NORET(k_pk_add_bf16, "ds_pk_add_bf16 %0, %1")
WORD_OP_NORET(k_mskor, "ds_mskor_b32 %0, %1, %2")

// Pairs of words: slot l is LDS words 2l and 2l + 1.
__global__ void k_pair_ops(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) {
  __shared__ uint32_t lds[kWords];
  const int l = threadIdx.x;
  for (int i = l; i < kWords; i += 32) lds[i] = init[i];
  __syncthreads();
  const uint32_t addr = lds_addr(lds + 2 * l);
  const uint64_t x = uint64_t{a[l]} | uint64_t{a[l + 32]} << 32, y = uint64_t{b[l]} | uint64_t{b[l + 32]} << 32;
  uint64_t r = 0;
  asm volatile("ds_mskor_rtn_b64 %0, %1, %2, %3" : "+v"(r) : "v"(addr), "v"(x), "v"(y) : "memory");
  asm volatile("s_wait_dscnt 0" ::: "memory");
  out[4 * l] = static_cast<uint32_t>(r);
  out[4 * l + 1] = static_cast<uint32_t>(r >> 32);
  // The compare-store, with no return value, on the same pair: compare with what the mskor left.
  const uint64_t left = uint64_t{lds[2 * l]} | uint64_t{lds[2 * l + 1]} << 32;
  const uint64_t cmp = (l & 1) ? left : ~left, newv = x ^ y;
  asm volatile("ds_cmpstore_b64 %0, %1, %2" : : "v"(addr), "v"(newv), "v"(cmp) : "memory");
  asm volatile("s_wait_dscnt 0" ::: "memory");
  __syncthreads();
  out[4 * l + 2] = lds[2 * l];
  out[4 * l + 3] = lds[2 * l + 1];
}

// The two-address exchanges: lane l swaps LDS words (2l) and (2l + 1) (offsets 2l, 2l + 1 counted in elements through the
// address, with offset0 = 0 and offset1 = 1) for its two data words.
__global__ void k_xchg2(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) {
  __shared__ uint32_t lds[kWords];
  const int l = threadIdx.x;
  for (int i = l; i < kWords; i += 32) lds[i] = init[i];
  __syncthreads();
  const uint32_t addr = lds_addr(lds + 2 * l);
  uint64_t r = 0;
  asm volatile("ds_storexchg_2addr_rtn_b32 %0, %1, %2, %3 offset0:0 offset1:1" : "+v"(r) : "v"(addr), "v"(a[l]), "v"(b[l]) : "memory");
  asm volatile("s_wait_dscnt 0" ::: "memory");
  __syncthreads();
  out[4 * l] = static_cast<uint32_t>(r);
  out[4 * l + 1] = static_cast<uint32_t>(r >> 32);
  out[4 * l + 2] = lds[2 * l];
  out[4 * l + 3] = lds[2 * l + 1];
}

// Lane l works on 64-bit elements (words 4l.. 4l + 3): the 64-bit exchange of element 0 and element 1.
__global__ void k_xchg2_64(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) {
  __shared__ uint32_t lds[kWords];
  const int l = threadIdx.x;
  for (int i = l; i < kWords; i += 32) lds[i] = init[i];
  __syncthreads();
  const uint32_t addr = lds_addr(lds + 4 * l);
  typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
  u32x4 r = {0, 0, 0, 0};
  const uint64_t x = uint64_t{a[l]} | uint64_t{a[l + 32]} << 32, y = uint64_t{b[l]} | uint64_t{b[l + 32]} << 32;
  asm volatile("ds_storexchg_2addr_rtn_b64 %0, %1, %2, %3 offset0:0 offset1:1" : "+v"(r) : "v"(addr), "v"(x), "v"(y) : "memory");
  asm volatile("s_wait_dscnt 0" ::: "memory");
  __syncthreads();
  for (int k = 0; k < 4; ++k) out[8 * l + k] = r[k];
  for (int k = 0; k < 4; ++k) out[8 * l + 4 + k] = lds[4 * l + k];
}

// The stride-64 form: offsets count 64 elements (256 bytes). Lane 0 only (the array holds 128 words, so use element
// offsets 0 and 0 -- the same element twice is not meaningful -- so use offset0:0 and offset1:0 on two different bases).
__global__ void k_xchg2_stride64(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) {
  __shared__ uint32_t lds[512];
  const int l = threadIdx.x;
  for (int i = l; i < 512; i += 32) lds[i] = init[i];
  __syncthreads();
  const uint32_t addr = lds_addr(lds + l);
  uint64_t r = 0;
  // Words l and l + 64 (a stride of 64 elements).
  asm volatile("ds_storexchg_2addr_stride64_rtn_b32 %0, %1, %2, %3 offset0:0 offset1:1" : "+v"(r) : "v"(addr), "v"(a[l]), "v"(b[l]) : "memory");
  asm volatile("s_wait_dscnt 0" ::: "memory");
  __syncthreads();
  out[4 * l] = static_cast<uint32_t>(r);
  out[4 * l + 1] = static_cast<uint32_t>(r >> 32);
  out[4 * l + 2] = lds[l];
  out[4 * l + 3] = lds[l + 64];
}

// The conditional exchanges: lane l works on words 2l and 2l + 1 (an 8-byte aligned pair).
__global__ void k_condxchg(const uint32_t* init, const uint32_t* a, const uint32_t* b, uint32_t* out) {
  __shared__ uint32_t lds[kWords];
  const int l = threadIdx.x;
  for (int i = l; i < kWords; i += 32) lds[i] = init[i];
  __syncthreads();
  const uint32_t addr = lds_addr(lds + 2 * l);
  const uint64_t data = uint64_t{a[l]} | uint64_t{b[l]} << 32;
  uint64_t r = 0;
  asm volatile("ds_condxchg32_rtn_b64 %0, %1, %2" : "+v"(r) : "v"(addr), "v"(data) : "memory");
  asm volatile("s_wait_dscnt 0" ::: "memory");
  __syncthreads();
  out[4 * l] = static_cast<uint32_t>(r);
  out[4 * l + 1] = static_cast<uint32_t>(r >> 32);
  out[4 * l + 2] = lds[2 * l];
  out[4 * l + 3] = lds[2 * l + 1];
}

// The store and load that use the lane number as part of the address (M0 supplies the base).
__global__ void k_addtid(const uint32_t* a, uint32_t* out) {
  __shared__ uint32_t lds[kWords];
  const int l = threadIdx.x;
  for (int i = l; i < kWords; i += 32) lds[i] = 0xC0DE0000u + i;
  __syncthreads();
  const uint32_t x = a[l];
  uint32_t r = 0;
  const uint32_t base = lds_addr(lds + 8);   // word 8: M0 holds it, lane l then stores at word 8 + l
  asm volatile("s_mov_b32 m0, %2\n\tds_store_addtid_b32 %1\n\ts_wait_dscnt 0\n\tds_load_addtid_b32 %0\n\ts_wait_dscnt 0"
               : "=v"(r) : "v"(x), "s"(__builtin_amdgcn_readfirstlane(base)) : "m0", "memory");
  __syncthreads();
  out[l] = r;
  for (int i = l; i < kWords; i += 32) out[32 + i] = lds[i];
}

// The backward permute that fetches from lanes that are off: lane l reads the lane that (index / 4) names, whether it is
// on or not; lanes 1 mod 3 are off and write nothing. The ordinary one gives zero for an off lane.
__global__ void k_bpermute_fi(const uint32_t* a, const uint32_t* idx, uint32_t* out) {
  const int l = threadIdx.x;
  uint32_t x = a[l], ix = idx[l];
  asm volatile("" : "+v"(x), "+v"(ix));
  uint32_t r = 0xDEADBEEFu, q = 0xDEADBEEFu;
  if (l % 3 != 1) {
    asm volatile("ds_bpermute_fi_b32 %0, %1, %2" : "+v"(r) : "v"(ix * 4), "v"(x) : "memory");
    asm volatile("ds_bpermute_b32 %0, %1, %2" : "+v"(q) : "v"(ix * 4), "v"(x) : "memory");
    asm volatile("s_wait_dscnt 0" ::: "memory");
  }
  out[l] = r;
  out[32 + l] = q;
}

static uint32_t seed = 4242;
static uint32_t rnd() {
  seed = seed * 1664525u + 1013904223u;
  return seed;
}
static float h2f(uint16_t h) {
  const int sign = h >> 15 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e == 0) return sign * std::ldexp(static_cast<float>(m), -24);
  if (e == 31) return m ? NAN : sign * INFINITY;
  return sign * std::ldexp(static_cast<float>(m + 1024), e - 25);
}
static uint16_t f2h(float f) {
  _Float16 h = static_cast<_Float16>(f);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
static float bf16f(uint16_t b) {
  const uint32_t u = uint32_t{b} << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
static uint16_t bf16_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

template <class T>
static T* up(const std::vector<T>& v) {
  T* p = nullptr;
  if (hipMalloc(&p, v.size() * sizeof(T)) != hipSuccess) return nullptr;
  if (hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) return nullptr;
  return p;
}

int main() {
  int failed = 0;
  const auto report = [&](const char* what, int wrong, int of) {
    std::printf("%s: %d of %d wrong\n", what, wrong, of);
    failed += wrong != 0;
  };
  std::vector<uint32_t> init(512), A(64), B(64);
  for (auto& v : init) v = rnd();
  for (auto& v : A) v = rnd();
  for (auto& v : B) v = rnd();
  // Small values, so that "memory at least the operand" goes both ways in the subtracts.
  std::vector<uint32_t> small_init(512), small_a(64), small_b(64);
  for (auto& v : small_init) v = rnd() % 100;
  for (auto& v : small_a) v = rnd() % 100;
  for (auto& v : small_b) v = rnd() % 100;
  uint32_t* d_init = up(init);
  uint32_t* d_a = up(A);
  uint32_t* d_b = up(B);
  uint32_t* d_si = up(small_init);
  uint32_t* d_sa = up(small_a);
  uint32_t* d_sb = up(small_b);
  uint32_t* d_out = nullptr;
  CHECK(hipMalloc(&d_out, 4096 * 4));
  std::vector<uint32_t> out(4096);
  const auto fetch = [&]() {
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(out.data(), d_out, out.size() * 4, hipMemcpyDeviceToHost));
    return 0;
  };

  // --- conditional subtract and subtract clamped at zero ---
  for (int k = 0; k < 4; ++k) {
    const bool cond = k < 2;
    if (k == 0) k_cond_sub_rtn<<<1, 32>>>(d_si, d_sa, d_sb, d_out);
    else if (k == 1) k_cond_sub<<<1, 32>>>(d_si, d_sa, d_sb, d_out);
    else if (k == 2) k_sub_clamp_rtn<<<1, 32>>>(d_si, d_sa, d_sb, d_out);
    else k_sub_clamp<<<1, 32>>>(d_si, d_sa, d_sb, d_out);
    if (fetch()) return 1;
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const uint32_t was = small_init[l], v = small_a[l];
      const uint32_t now = cond ? (was >= v ? was - v : was) : (was < v ? 0u : was - v);
      wrong += out[3 * l + 2] != now;
      if (k == 0 || k == 2) wrong += out[3 * l] != was;
    }
    report(k == 0 ? "ds_cond_sub_rtn_u32" : k == 1 ? "ds_cond_sub_u32" : k == 2 ? "ds_sub_clamp_rtn_u32" : "ds_sub_clamp_u32", wrong, 32);
  }

  // --- packed half and bfloat16 adds ---
  {
    std::vector<uint32_t> hi(512), ha(64), hb(64), bi(512), ba(64), bb(64);
    const auto one_h = [&]() { return uint32_t{f2h((static_cast<int>(rnd() % 65) - 32) / 4.0f)}; };
    const auto one_b = [&]() { return uint32_t{bf16_bits((static_cast<int>(rnd() % 65) - 32) / 4.0f)}; };
    for (auto& v : hi) v = one_h() | one_h() << 16;
    for (auto& v : ha) v = one_h() | one_h() << 16;
    for (auto& v : hb) v = one_h() | one_h() << 16;
    for (auto& v : bi) v = one_b() | one_b() << 16;
    for (auto& v : ba) v = one_b() | one_b() << 16;
    for (auto& v : bb) v = one_b() | one_b() << 16;
    for (int k = 0; k < 4; ++k) {
      const bool bf = k >= 2, rtn = k % 2 == 0;
      uint32_t* i = up(bf ? bi : hi);
      uint32_t* a = up(bf ? ba : ha);
      uint32_t* b = up(bf ? bb : hb);
      if (k == 0) k_pk_add_f16_rtn<<<1, 32>>>(i, a, b, d_out);
      else if (k == 1) k_pk_add_f16<<<1, 32>>>(i, a, b, d_out);
      else if (k == 2) k_pk_add_bf16_rtn<<<1, 32>>>(i, a, b, d_out);
      else k_pk_add_bf16<<<1, 32>>>(i, a, b, d_out);
      if (fetch()) return 1;
      int wrong = 0;
      for (int l = 0; l < 32; ++l) {
        const uint32_t was = (bf ? bi : hi)[l], v = (bf ? ba : ha)[l];
        uint32_t now = 0;
        for (int h = 0; h < 2; ++h) {
          const uint16_t x = was >> (16 * h), y = v >> (16 * h);
          now |= uint32_t{bf ? bf16_bits(bf16f(x) + bf16f(y)) : f2h(h2f(x) + h2f(y))} << (16 * h);
        }
        wrong += out[3 * l + 2] != now;
        if (rtn) wrong += out[3 * l] != was;
      }
      report(k == 0 ? "ds_pk_add_rtn_f16" : k == 1 ? "ds_pk_add_f16" : k == 2 ? "ds_pk_add_rtn_bf16" : "ds_pk_add_bf16", wrong, 32);
    }
  }

  // --- masked OR, 32-bit ---
  for (int rtn = 1; rtn >= 0; --rtn) {
    if (rtn) k_mskor_rtn<<<1, 32>>>(d_init, d_a, d_b, d_out);
    else k_mskor<<<1, 32>>>(d_init, d_a, d_b, d_out);
    if (fetch()) return 1;
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const uint32_t was = init[l], now = (was & ~A[l]) | B[l];
      wrong += out[3 * l + 2] != now;
      if (rtn) wrong += out[3 * l] != was;
    }
    report(rtn ? "ds_mskor_rtn_b32" : "ds_mskor_b32", wrong, 32);
  }

  // --- the pair operations: 64-bit masked OR with a return, then a 64-bit compare-store ---
  {
    k_pair_ops<<<1, 32>>>(d_init, d_a, d_b, d_out);
    if (fetch()) return 1;
    int w_mskor = 0, w_cmp = 0;
    for (int l = 0; l < 32; ++l) {
      const uint64_t was = uint64_t{init[2 * l]} | uint64_t{init[2 * l + 1]} << 32;
      const uint64_t x = uint64_t{A[l]} | uint64_t{A[l + 32]} << 32, y = uint64_t{B[l]} | uint64_t{B[l + 32]} << 32;
      const uint64_t left = (was & ~x) | y;
      const uint64_t cmp = (l & 1) ? left : ~left, newv = x ^ y;
      const uint64_t after = left == cmp ? newv : left;
      w_mskor += out[4 * l] != static_cast<uint32_t>(was) || out[4 * l + 1] != static_cast<uint32_t>(was >> 32);
      w_cmp += out[4 * l + 2] != static_cast<uint32_t>(after) || out[4 * l + 3] != static_cast<uint32_t>(after >> 32);
    }
    report("ds_mskor_rtn_b64", w_mskor, 32);
    report("ds_cmpstore_b64", w_cmp, 32);
  }

  // --- two-address exchanges ---
  {
    k_xchg2<<<1, 32>>>(d_init, d_a, d_b, d_out);
    if (fetch()) return 1;
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      wrong += out[4 * l] != init[2 * l] || out[4 * l + 1] != init[2 * l + 1];   // the old contents come back, in order
      wrong += out[4 * l + 2] != A[l] || out[4 * l + 3] != B[l];                  // the new ones are in place
    }
    report("ds_storexchg_2addr_rtn_b32", wrong, 32);
    k_xchg2_64<<<1, 32>>>(d_init, d_a, d_b, d_out);
    if (fetch()) return 1;
    wrong = 0;
    for (int l = 0; l < 32; ++l) {
      for (int k = 0; k < 4; ++k) wrong += out[8 * l + k] != init[4 * l + k];
      wrong += out[8 * l + 4] != A[l] || out[8 * l + 5] != A[l + 32] || out[8 * l + 6] != B[l] || out[8 * l + 7] != B[l + 32];
    }
    report("ds_storexchg_2addr_rtn_b64", wrong, 32);
    k_xchg2_stride64<<<1, 32>>>(d_init, d_a, d_b, d_out);
    if (fetch()) return 1;
    wrong = 0;
    for (int l = 0; l < 32; ++l) {
      // offset0 = 0 and offset1 = 1 elements of 64 words: words l and l + 64... a stride of 64 elements is 64 words
      wrong += out[4 * l] != init[l] || out[4 * l + 1] != init[l + 64];
      wrong += out[4 * l + 2] != A[l] || out[4 * l + 3] != B[l];
    }
    report("ds_storexchg_2addr_stride64_rtn_b32", wrong, 32);
  }

  // --- conditional exchanges ---
  {
    std::vector<uint32_t> ca(64), cb(64);
    for (int i = 0; i < 64; ++i) ca[i] = rnd(), cb[i] = rnd();
    uint32_t* a = up(ca);
    uint32_t* b = up(cb);
    k_condxchg<<<1, 32>>>(d_init, a, b, d_out);
    if (fetch()) return 1;
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      uint32_t w0 = init[2 * l], w1 = init[2 * l + 1];
      wrong += out[4 * l] != w0 || out[4 * l + 1] != w1;
      if (ca[l] >> 31) w0 = ca[l] & 0x7FFFFFFFu;
      if (cb[l] >> 31) w1 = cb[l] & 0x7FFFFFFFu;
      wrong += out[4 * l + 2] != w0 || out[4 * l + 3] != w1;
    }
    report("ds_condxchg32_rtn_b64", wrong, 32);
  }

  // --- the lane-number-addressed store and load ---
  {
    std::vector<uint32_t> data(32);
    for (auto& v : data) v = rnd();
    uint32_t* a = up(data);
    k_addtid<<<1, 32>>>(a, d_out);
    if (fetch()) return 1;
    int wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out[l] != data[l];   // stored and loaded by the same rule
    for (int i = 0; i < kWords; ++i) wrong += out[32 + i] != (i >= 8 && i < 40 ? data[i - 8] : 0xC0DE0000u + i);
    report("ds_store_addtid_b32 and ds_load_addtid_b32", wrong, 32 + kWords);
  }

  // --- the permute that fetches from lanes that are off ---
  {
    std::vector<uint32_t> data(32), index(32);
    for (int i = 0; i < 32; ++i) data[i] = rnd(), index[i] = rnd() % 32;
    uint32_t* a = up(data);
    uint32_t* x = up(index);
    k_bpermute_fi<<<1, 32>>>(a, x, d_out);
    if (fetch()) return 1;
    int w_fi = 0, w_plain = 0;
    for (int l = 0; l < 32; ++l) {
      const bool on = l % 3 != 1;
      const uint32_t src = index[l], src_on = src % 3 != 1;
      w_fi += out[l] != (on ? data[src] : 0xDEADBEEFu);
      w_plain += out[32 + l] != (on ? (src_on ? data[src] : 0u) : 0xDEADBEEFu);
    }
    report("ds_bpermute_fi_b32", w_fi, 32);
    report("ds_bpermute_b32 for comparison (an off lane gives zero)", w_plain, 32);
  }

  std::printf("ds1250: %d failed\n", failed);
  return failed != 0;
}
