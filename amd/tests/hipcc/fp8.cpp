// gfx942's 8-bit floats, converted on the device by the instructions HIP's
// header uses there, and on the host by the same header's software
// conversion, which is how AMD says they come out: every float in a sweep
// narrowed to fp8 and bf8 (rounded to nearest, saturating or not, in pairs,
// and stochastically), and every one of the 256 bytes widened back.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr __hip_fp8_interpretation_t kTypes[2] = {__HIP_E4M3_FNUZ, __HIP_E5M2_FNUZ};

// Per float and type: to nearest, saturating, as the low and the high of a
// pair, then stochastically, with and without saturating.
constexpr int kWays = 6;

// The device's conversions exist only where the device code is compiled.
__global__ void narrow(const float* x, const unsigned* random, int n, unsigned char* out) {
#if __HIP_DEVICE_COMPILE__
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float y = x[(i + 1) % n];
  for (int t = 0; t < 2; ++t) {
    unsigned char* o = out + (size_t(i) * 2 + t) * kWays;
    o[0] = __hip_cvt_float_to_fp8(x[i], __HIP_NOSAT, kTypes[t]);
    o[1] = __hip_cvt_float_to_fp8(x[i], __HIP_SATFINITE, kTypes[t]);
    const __hip_fp8x2_storage_t two = __hip_cvt_float2_to_fp8x2(float2{x[i], y}, __HIP_NOSAT, kTypes[t]);
    o[2] = two & 0xFF;
    o[3] = two >> 8;
    o[4] = internal::cast_to_f8_from_f32<true>(x[i], false, kTypes[t], random[i]);
    o[5] = internal::cast_to_f8_from_f32<true>(x[i], true, kTypes[t], random[i]);
  }
#endif
}

__global__ void widen(float* out) {
#if __HIP_DEVICE_COMPILE__
  const int b = threadIdx.x;
  for (int t = 0; t < 2; ++t) out[t * 256 + b] = internal::cast_to_f32_from_f8(b, kTypes[t]);
  // And two at once, from the low half of a register.
  if (b % 2 == 0)
    for (int t = 0; t < 2; ++t) {
      const float2 two = internal::cast_to_f32x2_from_f8x2(b | (b + 1) << 8, kTypes[t]);
      out[512 + t * 256 + b] = two.x;
      out[512 + t * 256 + b + 1] = two.y;
    }
#endif
}

static unsigned host_narrow(float x, int t, bool saturate, bool stochastic, unsigned random) {
  const int we = t ? 5 : 4, wm = t ? 2 : 3;
  return internal::cast_to_f8<float, true>(x, wm, we, saturate, stochastic, random);
}

static bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, 4) == 0; }

int main() {
  // Zeroes, infinities and a NaN; each type's largest value and just past it
  // (240 and 57344); the least normals and subnormals and the halfway points
  // between them; and floats of every size in between.
  std::vector<float> x = {0.0f,     -0.0f,     INFINITY,   -INFINITY, NAN,       240.0f,   241.0f,   247.9f,
                          248.0f,   -250.0f,   57344.0f,   57345.0f,  61439.0f,  61440.0f, 1e9f,     -1e-30f,
                          0x1p-7f,  0x1p-10f,  0x1.8p-11f, 0x1p-11f,  0x1p-12f,  0x1p-15f, 0x1p-17f, 0x1.8p-18f,
                          0x1p-18f, 0x1p-149f, 1.0625f,    1.1875f,   -1.0625f,  3.0f,     -7.5f,    0.3f};
  unsigned seed = 12345;
  const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
  for (int i = 0; i < 4064; ++i) {
    const int e = int(next() % 50) - 25;
    x.push_back(std::ldexp(float(next() % 100000) / 100000.0f + 0.5f, e) * (next() & 1 ? -1.0f : 1.0f));
  }
  const int n = int(x.size());
  std::vector<unsigned> random(n);
  for (auto& r : random) r = next();
  float* dx;
  unsigned* dr;
  unsigned char* dout;
  float* dwide;
  if (hipMalloc(&dx, n * 4) || hipMalloc(&dr, n * 4) || hipMalloc(&dout, size_t(n) * 2 * kWays) ||
      hipMalloc(&dwide, 1024 * 4))
    return 1;
  if (hipMemcpy(dx, x.data(), n * 4, hipMemcpyHostToDevice) || hipMemcpy(dr, random.data(), n * 4, hipMemcpyHostToDevice))
    return 1;
  narrow<<<(n + 255) / 256, 256>>>(dx, dr, n, dout);
  widen<<<1, 256>>>(dwide);
  std::vector<unsigned char> got(size_t(n) * 2 * kWays);
  std::vector<float> wide(1024);
  if (hipMemcpy(got.data(), dout, got.size(), hipMemcpyDeviceToHost) ||
      hipMemcpy(wide.data(), dwide, 1024 * 4, hipMemcpyDeviceToHost))
    return 1;

  static const char* kType[2] = {"fp8", "bf8"};
  static const char* kWay[kWays] = {"to nearest", "saturating", "the low of a pair", "the high of a pair",
                                    "stochastically", "stochastically, saturating"};
  for (int t = 0; t < 2; ++t)
    for (int way = 0; way < kWays; ++way) {
      int wrong = 0;
      for (int i = 0; i < n; ++i) {
        const float in = way == 3 ? x[(i + 1) % n] : x[i];
        const unsigned want = host_narrow(in, t, way == 1 || way == 5, way >= 4, random[i]);
        const unsigned have = got[(size_t(i) * 2 + t) * kWays + way];
        if (have != want && wrong++ < 3)
          std::printf("  %s %s: %a gives 0x%02x, not 0x%02x\n", kType[t], kWay[way], in, have, want);
      }
      std::printf("floats to %s, %s: %d of %d wrong\n", kType[t], kWay[way], wrong, n);
    }
  for (int t = 0; t < 2; ++t)
    for (int two = 0; two < 2; ++two) {
      int wrong = 0;
      for (int b = 0; b < 256; ++b) {
        const float want = internal::cast_from_f8<float, true>(b, t ? 2 : 3, t ? 5 : 4);
        const float have = wide[two * 512 + t * 256 + b];
        if (!same(have, want) && wrong++ < 3) std::printf("  %s 0x%02x widens to %a, not %a\n", kType[t], b, have, want);
      }
      std::printf("%s to floats%s: %d of 256 wrong\n", kType[t], two ? ", two at once" : "", wrong);
    }
  return 0;
}
