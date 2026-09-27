// RDNA2 (gfx1030) instructions the executor runs its own way, each against the
// host. Built twice: rdna2.gfx1030 as HIP builds for RDNA (wave32) and
// rdna2.w64.gfx1030 with -mwavefrontsize64 (-DVGPU_W64).
//
//   - SDWA: a VOP1/VOP2/VOPC instruction reading a byte or a half of each
//     source, with or without its sign, and writing part of its destination
//     -- padding, sign-extending or keeping the rest. gfx10 has no true16, so
//     the compiler reaches a register's high half this way.
//   - A comparison's SDWA form writing a scalar register pair rather than VCC.
//   - Registers indexed through M0 (v_movrels, v_movreld), whose number
//     gfx10 has as gfx9 does and gfx11 gives to null.
//   - v_permlane16 and v_permlanex16, and DPP's row_share and row_xmask.
//   - The stack reached through a flat pointer, whose kernel sets FLAT_SCRATCH
//     first (s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO/HI)), as llama.cpp's
//     flash-attention kernels do.
#include <hip/hip_runtime.h>

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

#ifndef VGPU_W64
constexpr int kWave = 32;
typedef uint32_t Mask;
#else
constexpr int kWave = 64;
typedef uint64_t Mask;
#endif

// Each SDWA form, one to an output row.
__global__ void sdwa(const uint32_t* a, const uint32_t* b, uint32_t* out) {
  const int l = threadIdx.x;
  const uint32_t x = a[l], y = b[l];
  uint32_t r = 0x12345678u;
  // A byte of x into the high half of r, the low half kept.
  asm volatile("v_mov_b32_sdwa %0, %1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_2" : "+v"(r) : "v"(x));
  out[l] = r;
  // A byte of x plus the high half of y.
  asm volatile("v_add_nc_u32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:WORD_1"
               : "=v"(r) : "v"(x), "v"(y));
  out[kWave + l] = r;
  // x's low byte, sign-extended, plus y; the sum's low byte, sign-extended.
  asm volatile("v_add_nc_u32_sdwa %0, sext(%1), %2 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:BYTE_0 src1_sel:DWORD"
               : "=v"(r) : "v"(x), "v"(y));
  out[2 * kWave + l] = r;
  // Halves: x's high one plus y's low one, into r's high half, r's low half kept.
  r = 0xabcd1234u;
  asm volatile("v_add_f16_sdwa %0, %1, %2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:WORD_0"
               : "+v"(r) : "v"(x), "v"(y));
  out[3 * kWave + l] = r;
  // A shift of y's second byte, zero-extended, by x's low half.
  asm volatile("v_lshlrev_b32_sdwa %0, %1, %2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_1"
               : "=v"(r) : "v"(x & 7u), "v"(y));
  out[4 * kWave + l] = r;
  // A comparison of x's top byte and y's bottom one, into a scalar register.
  Mask m;
  asm volatile("v_cmp_gt_u32_sdwa %0, %1, %2 src0_sel:BYTE_3 src1_sel:BYTE_0" : "=s"(m) : "v"(x), "v"(y));
  out[5 * kWave + l] = static_cast<uint32_t>((m >> l) & 1);
}

// Registers indexed through M0 (whose number gfx10 has as gfx9 does, where
// gfx11 has null): v_movrels reads v10 + M0, v_movreld writes v10 + M0.
// (M0 is not listed as clobbered: the compiler reserves it, and nothing else
// here uses it.)
__global__ void indexed(const int* in, int k, int* out) {
  const int l = threadIdx.x;
  int r;
  int x0 = in[l], x1 = in[kWave + l], x2 = in[2 * kWave + l], x3 = in[3 * kWave + l];
  asm volatile("s_mov_b32 m0, %1\n\tv_movrels_b32 %0, v10"
               : "=v"(r) : "s"(k), "{v10}"(x0), "{v11}"(x1), "{v12}"(x2), "{v13}"(x3));
  out[l] = r;
  asm volatile("s_mov_b32 m0, %4\n\tv_movreld_b32 v10, %5"
               : "+{v10}"(x0), "+{v11}"(x1), "+{v12}"(x2), "+{v13}"(x3) : "s"(3 - k), "v"(-l));
  out[kWave + l] = x0 + x1 + x2 + x3;
}

__global__ void lanes(const int* in, int* out) {
  const int l = threadIdx.x;
  const int x = in[l];
  // Each lane of a row of 16 reads the lane its selector names: lane i of
  // the row reads lane (i * 5 + 3) % 16 (selectors 3, 8, 13, 2, 7, ...).
  out[l] = __builtin_amdgcn_permlane16(0, x, 0x61c72d83u, 0xe94fa50bu, false, false);
  // The same, from the other row of the pair.
  out[kWave + l] = __builtin_amdgcn_permlanex16(0, x, 0x61c72d83u, 0xe94fa50bu, false, false);
  // Every lane of a row reads its row's lane 5; then the lane 3 lanes away.
  out[2 * kWave + l] = __builtin_amdgcn_update_dpp(0, x, 0x155, 0xf, 0xf, false);
  out[3 * kWave + l] = __builtin_amdgcn_update_dpp(0, x, 0x163, 0xf, 0xf, false);
}

// A function that is not inlined, given a pointer into its caller's stack:
// it reaches the stack by a flat address, so the kernel sets FLAT_SCRATCH.
__device__ __attribute__((noinline)) int sum_through(const int* p, int n) {
  int s = 0;
  for (int i = 0; i < n; ++i) s += p[i] * (i + 1);
  return s;
}
__global__ void stack_by_pointer(const int* in, int* out) {
  int local[24];
  const int l = threadIdx.x;
  for (int i = 0; i < 24; ++i) local[i] = in[(l + i) % kWave] + i;
  out[l] = sum_through(local, 24 - l % 8);
}

int wrong = 0;
void report(const char* what, int bad, int of) {
  std::printf("%s: %d of %d wrong\n", what, bad, of);
  wrong += bad;
}

template <typename T>
T* upload(const T* h, size_t n) {
  T* d = nullptr;
  (void)hipMalloc(&d, n * sizeof(T));
  (void)hipMemcpy(d, h, n * sizeof(T), hipMemcpyHostToDevice);
  return d;
}
template <typename T>
void download(T* h, const T* d, size_t n) {
  (void)hipMemcpy(h, d, n * sizeof(T), hipMemcpyDeviceToHost);
}

uint16_t half_bits(_Float16 h) {
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
_Float16 half_of(uint32_t bits) {
  const uint16_t u = static_cast<uint16_t>(bits);
  _Float16 h;
  std::memcpy(&h, &u, 2);
  return h;
}

int main() {
  uint32_t seed = 11;
  const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };

  {
    uint32_t a[kWave], b[kWave], out[6 * kWave];
    for (int i = 0; i < kWave; ++i) {
      a[i] = next();
      b[i] = next();
      // Halves the sum of which is exact.
      const _Float16 hx = static_cast<_Float16>(static_cast<int>(next() % 64) - 32) / static_cast<_Float16>(4);
      const _Float16 hy = static_cast<_Float16>(static_cast<int>(next() % 64) - 32) / static_cast<_Float16>(4);
      a[i] = (a[i] & 0x0000FFFFu) | static_cast<uint32_t>(half_bits(hx)) << 16;
      b[i] = (b[i] & 0xFFFF0000u) | half_bits(hy);
    }
    uint32_t* da = upload(a, kWave);
    uint32_t* db = upload(b, kWave);
    uint32_t* dout = upload(out, 6 * kWave);
    sdwa<<<1, kWave>>>(da, db, dout);
    CHECK(hipDeviceSynchronize());
    download(out, dout, 6 * kWave);
    int bad[6] = {};
    for (int i = 0; i < kWave; ++i) {
      const uint32_t x = a[i], y = b[i];
      bad[0] += out[i] != (0x5678u | ((x >> 16) & 0xFF) << 16);
      bad[1] += out[kWave + i] != ((x >> 8) & 0xFF) + (y >> 16);
      const uint32_t s = static_cast<uint32_t>(static_cast<int8_t>(x & 0xFF)) + y;
      bad[2] += out[2 * kWave + i] != static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(s & 0xFF)));
      const _Float16 sum = half_of(x >> 16) + half_of(y);
      bad[3] += out[3 * kWave + i] != (0x1234u | static_cast<uint32_t>(half_bits(sum)) << 16);
      bad[4] += out[4 * kWave + i] != ((((y >> 8) & 0xFF) << (x & 7)) & 0xFFFF);
      bad[5] += out[5 * kWave + i] != static_cast<uint32_t>((x >> 24) > (y & 0xFF));
    }
    report("SDWA: a byte into a kept register's high half", bad[0], kWave);
    report("SDWA: a byte plus a half", bad[1], kWave);
    report("SDWA: sign-extended source and destination", bad[2], kWave);
    report("SDWA: halves, the rest of the register kept", bad[3], kWave);
    report("SDWA: a shift into a padded half", bad[4], kWave);
    report("SDWA: a comparison into a scalar register", bad[5], kWave);
  }
  {
    int in[4 * kWave], out[2 * kWave];
    for (int i = 0; i < 4 * kWave; ++i) in[i] = static_cast<int>(next() % 1000);
    int* din = upload(in, 4 * kWave);
    int* dout = upload(out, 2 * kWave);
    const int k = 2;
    indexed<<<1, kWave>>>(din, k, dout);
    CHECK(hipDeviceSynchronize());
    download(out, dout, 2 * kWave);
    int bad = 0;
    for (int i = 0; i < kWave; ++i) {
      int sum = -i;
      for (int j = 0; j < 4; ++j) sum += j == 3 - k ? 0 : in[j * kWave + i];
      bad += out[i] != in[k * kWave + i] || out[kWave + i] != sum;
    }
    report("registers indexed through M0 (v_movrels, v_movreld)", bad, kWave);
  }
  {
    int in[kWave], out[4 * kWave];
    for (int i = 0; i < kWave; ++i) in[i] = static_cast<int>(next() % 100000);
    int* din = upload(in, kWave);
    int* dout = upload(out, 4 * kWave);
    lanes<<<1, kWave>>>(din, dout);
    CHECK(hipDeviceSynchronize());
    download(out, dout, 4 * kWave);
    int bad[4] = {};
    for (int i = 0; i < kWave; ++i) {
      const int row = i & ~15, sel = (i % 16 * 5 + 3) % 16;
      bad[0] += out[i] != in[row + sel];
      bad[1] += out[kWave + i] != in[(row ^ 16) + sel];
      bad[2] += out[2 * kWave + i] != in[row + 5];
      bad[3] += out[3 * kWave + i] != in[row + ((i % 16) ^ 3)];
    }
    report("v_permlane16_b32", bad[0], kWave);
    int* dsum = upload(out, kWave);
    stack_by_pointer<<<1, kWave>>>(din, dsum);
    CHECK(hipDeviceSynchronize());
    int sums[kWave];
    download(sums, dsum, kWave);
    int bad_stack = 0;
    for (int l = 0; l < kWave; ++l) {
      int want = 0;
      for (int i = 0; i < 24 - l % 8; ++i) want += (in[(l + i) % kWave] + i) * (i + 1);
      bad_stack += sums[l] != want;
    }
    report("the stack through a flat pointer (FLAT_SCRATCH set first)", bad_stack, kWave);
    report("v_permlanex16_b32", bad[1], kWave);
    report("DPP row_share", bad[2], kWave);
    report("DPP row_xmask", bad[3], kWave);
  }
  std::printf("%s\n", wrong ? "FAIL" : "PASS");
  return wrong ? 1 : 0;
}
