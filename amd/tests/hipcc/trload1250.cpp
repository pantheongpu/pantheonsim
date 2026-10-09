// gfx1250 (CDNA 5, MI455X) transposing loads, each against the host (trload1250.gfx1250, wave32). The oracle is AMD's
// CDNA5 ISA (section 10.9, 11.2.4): a matrix held in memory with its K index running slowest (A stored column by
// column, B row by row) is loaded by global_load_tr* or ds_load_tr* straight into the registers a WMMA reads, so the
// products come out right only if the loads put every element in the lane and register the WMMA expects. The answers
// are exact (small multiples of 1/8, small integers), so any difference is a wrong lane, register or element.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
// The builtins take clang's own half vector, which is not _Float16's.
typedef __fp16 v8g __attribute__((vector_size(16)));

#define GP(T, p) ((__attribute__((address_space(1))) T*)(p))
#define LP(T, p) ((__attribute__((address_space(3))) T*)(p))

// A is 16x32 and stored [k][m]; B is 32x16 and stored [k][n]. Each of the two 16-wide halves of K is one load.
__global__ void k_tr16(const _Float16* a, const _Float16* b, const float* c, float* d, int via_lds) {
  __shared__ _Float16 sa[32 * 16], sb[32 * 16];
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  if (via_lds) {
    for (int i = l; i < 32 * 16; i += 32) sa[i] = a[i], sb[i] = b[i];
    __syncthreads();
  }
  v16h va, vb;
  for (int ch = 0; ch < 2; ++ch) {
    const int k = ch * 16 + (l % 8) + 8 * (l / 16), half = (l % 16) / 8;
    const int off = k * 16 + 8 * half;
    v8g x, y;
    if (via_lds) {
      x = __builtin_amdgcn_ds_load_tr16_b128_v8f16(LP(v8g, sa + off));
      y = __builtin_amdgcn_ds_load_tr16_b128_v8f16(LP(v8g, sb + off));
    } else {
      x = __builtin_amdgcn_global_load_tr16_b128_v8f16(GP(v8g, a + off));
      y = __builtin_amdgcn_global_load_tr16_b128_v8f16(GP(v8g, b + off));
    }
    for (int j = 0; j < 8; ++j) va[ch * 8 + j] = static_cast<_Float16>(x[j]), vb[ch * 8 + j] = static_cast<_Float16>(y[j]);
  }
  v8f vc;
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8f vd = __builtin_amdgcn_wmma_f32_16x16x32_f16(false, va, false, vb, (short)0, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}

// The same for 8-bit elements: A is 16x64 stored [k][m], B 64x16 stored [k][n], and four loads cover K.
__global__ void k_tr8(const int8_t* a, const int8_t* b, const int* c, int* d, int via_lds) {
  __shared__ int8_t sa[64 * 16], sb[64 * 16];
  const int l = threadIdx.x, r = l % 16, h = l / 16;
  if (via_lds) {
    for (int i = l; i < 64 * 16; i += 32) sa[i] = a[i], sb[i] = b[i];
    __syncthreads();
  }
  v8i va, vb;
  for (int ch = 0; ch < 4; ++ch) {
    const int k = ch * 16 + 4 * (l / 8) + (l % 4), half = (l % 8) / 4;
    const int off = k * 16 + 8 * half;
    v2i x, y;
    if (via_lds) {
      x = __builtin_amdgcn_ds_load_tr8_b64_v2i32(LP(v2i, sa + off));
      y = __builtin_amdgcn_ds_load_tr8_b64_v2i32(LP(v2i, sb + off));
    } else {
      x = __builtin_amdgcn_global_load_tr8_b64_v2i32(GP(v2i, a + off));
      y = __builtin_amdgcn_global_load_tr8_b64_v2i32(GP(v2i, b + off));
    }
    va[2 * ch] = x[0], va[2 * ch + 1] = x[1];
    vb[2 * ch] = y[0], vb[2 * ch + 1] = y[1];
  }
  v8i vc;
  for (int e = 0; e < 8; ++e) vc[e] = c[(e + 8 * h) * 16 + r];
  const v8i vd = __builtin_amdgcn_wmma_i32_16x16x64_iu8(true, va, true, vb, vc, false, false);
  for (int e = 0; e < 8; ++e) d[(e + 8 * h) * 16 + r] = vd[e];
}

static uint32_t seed = 12345;
static int rnd(int n) {
  seed = seed * 1664525u + 1013904223u;
  return static_cast<int>((seed >> 8) % static_cast<uint32_t>(n));
}
template <class T>
static T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  if (hipMalloc(&p, v.size() * sizeof(T)) != hipSuccess) return nullptr;
  if (hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) return nullptr;
  return p;
}
static void report(const char* what, int wrong, int of) { std::printf("%s: %d of %d wrong\n", what, wrong, of); }

int main() {
  int failed = 0;
  for (int via_lds = 0; via_lds < 2; ++via_lds) {
    {
      std::vector<float> fa(16 * 32), fb(32 * 16), fc(256);
      std::vector<_Float16> ha(16 * 32), hb(32 * 16);
      for (int i = 0; i < 16 * 32; ++i) {
        fa[i] = (rnd(33) - 16) / 8.0f, fb[i] = (rnd(33) - 16) / 8.0f;
        ha[i] = static_cast<_Float16>(fa[i]), hb[i] = static_cast<_Float16>(fb[i]);   // [k][m] and [k][n]
      }
      for (auto& x : fc) x = (rnd(129) - 64) / 8.0f;
      float* dd = nullptr;
      CHECK(hipMalloc(&dd, 256 * sizeof(float)));
      k_tr16<<<1, 32>>>(upload(ha), upload(hb), upload(fc), dd, via_lds);
      CHECK(hipDeviceSynchronize());
      std::vector<float> out(256);
      CHECK(hipMemcpy(out.data(), dd, 256 * sizeof(float), hipMemcpyDeviceToHost));
      int wrong = 0;
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          double s = fc[m * 16 + n];
          for (int k = 0; k < 32; ++k) s += double{fa[k * 16 + m]} * fb[k * 16 + n];
          wrong += out[m * 16 + n] != static_cast<float>(s);
        }
      report(via_lds ? "ds_load_tr16_b128 into a 16x16x32 f16 WMMA" : "global_load_tr16_b128 into a 16x16x32 f16 WMMA", wrong, 256);
      failed += wrong != 0;
    }
    {
      std::vector<int8_t> a8(64 * 16), b8(64 * 16);
      std::vector<int> c32(256);
      for (auto& x : a8) x = static_cast<int8_t>(rnd(256) - 128);
      for (auto& x : b8) x = static_cast<int8_t>(rnd(256) - 128);
      for (auto& x : c32) x = rnd(2001) - 1000;
      int* dd = nullptr;
      CHECK(hipMalloc(&dd, 256 * sizeof(int)));
      k_tr8<<<1, 32>>>(upload(a8), upload(b8), upload(c32), dd, via_lds);
      CHECK(hipDeviceSynchronize());
      std::vector<int> out(256);
      CHECK(hipMemcpy(out.data(), dd, 256 * sizeof(int), hipMemcpyDeviceToHost));
      int wrong = 0;
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          int64_t s = c32[m * 16 + n];
          for (int k = 0; k < 64; ++k) s += int64_t{a8[k * 16 + m]} * b8[k * 16 + n];
          wrong += out[m * 16 + n] != static_cast<int>(s);
        }
      report(via_lds ? "ds_load_tr8_b64 into a 16x16x64 iu8 WMMA" : "global_load_tr8_b64 into a 16x16x64 iu8 WMMA", wrong, 256);
      failed += wrong != 0;
    }
  }
  std::printf("trload1250: 4 checks, %d failed\n", failed);
  return failed != 0;
}
