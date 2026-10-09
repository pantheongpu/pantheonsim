// Helpers the narrow-precision probes share (nvidia/tests/e2e/lowprec_*.cu): exact
// test values in every type, the block-scale tile layout, an FNV-1a hash and the
// optional dump of every case's data (PROBE_DUMP=<dir>).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace lp {
constexpr int T_E4M3 = 28, T_E5M2 = 29, T_FP4 = 33, T_BF16 = CUDA_R_16BF, T_F16 = CUDA_R_16F, T_F32 = CUDA_R_32F;

int bits_of(int t) {
  switch (t) {
    case T_FP4: return 4;
    case T_E4M3: case T_E5M2: case CUDA_R_8I: return 8;
    case T_F16: case T_BF16: return 16;
    case T_F32: case CUDA_R_32I: return 32;
    default: return 32;
  }
}
size_t bytes_of(int t, size_t n) { return (n * bits_of(t) + 7) / 8; }

// The small set of values every type holds exactly (E2M1 holds {0,.5,1,1.5,2,3,4,6}).
const float kVals[12] = {1, -1, 0.5f, -0.5f, 2, -2, 1.5f, -1.5f, 0, 3, -3, 4};

uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

uint8_t e4m3_of(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3); }
uint8_t e5m2_of(float f) { return (uint8_t)__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E5M2); }
uint8_t e2m1_of(float f) {
  static const float t[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int i = 0; i < 8; ++i)
    if (t[i] == std::fabs(f)) return (uint8_t)(i | (f < 0 ? 8 : 0));
  return 0;
}


// Element i of a packed array of type t, as a double.
inline double dec_elem(int t, const uint8_t* p, size_t i) {
  switch (t) {
    case T_E4M3: { __half_raw h = __nv_cvt_fp8_to_halfraw(p[i], __NV_E4M3); return (double)__half2float(__half(h)); }
    case T_E5M2: { __half_raw h = __nv_cvt_fp8_to_halfraw(p[i], __NV_E5M2); return (double)__half2float(__half(h)); }
    case T_F16: { __half h; std::memcpy(&h, p + 2 * i, 2); return (double)__half2float(h); }
    case T_BF16: { __nv_bfloat16 h; std::memcpy(&h, p + 2 * i, 2); return (double)__bfloat162float(h); }
    case T_FP4: { static const double v[8] = {0, .5, 1, 1.5, 2, 3, 4, 6}; const int c = (p[i / 2] >> (4 * (i & 1))) & 15; return (c & 8) ? -v[c & 7] : v[c & 7]; }
    case CUDA_R_8I: return (int8_t)p[i];
    default: { float f; std::memcpy(&f, p + 4 * i, 4); return f; }
  }
}

// n elements of type t from the value table, salted.
std::vector<uint8_t> make_matrix(int t, size_t n, uint32_t salt, bool big) {
  std::vector<uint8_t> out(bytes_of(t, n) + 64, 0);
  for (size_t i = 0; i < n; ++i) {
    float v = kVals[mix((uint32_t)i * 2654435761u + salt) % 12];
    if (big) v *= (t == T_E5M2 ? 4096.f : 128.f);
    if (t == T_E4M3 && big && std::fabs(v) > 448.f) v = v < 0 ? -448.f : 448.f;
    if (t == T_FP4 && big) v = v < 0 ? -6.f : 6.f;
    switch (t) {
      case T_E4M3: out[i] = e4m3_of(v); break;
      case T_E5M2: out[i] = e5m2_of(v); break;
      case T_FP4: out[i / 2] |= (uint8_t)(e2m1_of(v) << (4 * (i & 1))); break;
      case CUDA_R_8I: out[i] = (uint8_t)(int8_t)(v * 2); break;
      case CUDA_R_32I: { int32_t x = (int32_t)(v * 2); std::memcpy(&out[i * 4], &x, 4); break; }
      case T_F16: { __half h = __float2half_rn(v); std::memcpy(&out[i * 2], &h, 2); break; }
      case T_BF16: { __nv_bfloat16 h = __float2bfloat16_rn(v); std::memcpy(&out[i * 2], &h, 2); break; }
      default: std::memcpy(&out[i * 4], &v, 4); break;
    }
  }
  return out;
}

// ---- scale tensors ----
size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }
size_t tiled_bytes(size_t outer, size_t blocks) { return cdiv(outer, 128) * 128 * cdiv(blocks, 4) * 4; }
uint64_t fnv(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ULL;
  return h;
}

const char* dump_dir() { return std::getenv("PROBE_DUMP"); }
void dump(const std::string& name, const char* what, const void* p, size_t n) {
  const char* d = dump_dir();
  if (!d) return;
  std::string base = name;
  for (auto& c : base) if (c == ' ' || c == '/') c = '_';
  const std::string path = std::string(d) + "/" + base + "." + what;
  if (FILE* f = std::fopen(path.c_str(), "wb")) { std::fwrite(p, 1, n, f); std::fclose(f); }
}

struct Dev {
  void* p = nullptr;
  size_t n = 0;
  Dev() = default;
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  ~Dev() { if (p) cudaFree(p); }
  void* put(const std::vector<uint8_t>& h) {
    n = h.size();
    cudaMalloc(&p, n + 4096);
    cudaMemset(p, 0, n + 4096);
    cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice);
    return p;
  }
  void* fill(size_t bytes, int byte) {
    n = bytes;
    cudaMalloc(&p, n + 4096);
    cudaMemset(p, byte, n + 4096);
    return p;
  }
  std::vector<uint8_t> get() const { std::vector<uint8_t> h(n + 4096); cudaMemcpy(h.data(), p, n + 4096, cudaMemcpyDeviceToHost); return h; }
};

}  // namespace lp
