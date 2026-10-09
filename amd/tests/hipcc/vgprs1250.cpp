// gfx1250 (CDNA 5, MI455X) vector registers past v255 (vgprs1250.gfx1250, wave32). A wave may be given up to 1024 of them,
// and the compiler reaches those above v255 by s_set_vgpr_msb, which sets the two high bits of the numbers the instructions
// after it name. The kernel keeps hundreds of values live at once (each held in a register by an empty asm), then uses them
// all; the answers are small integers, so any value read from or written to the wrong register shows.
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

constexpr int K = 600;

__global__ __attribute__((amdgpu_waves_per_eu(1, 1))) __launch_bounds__(32) void k_many(const uint32_t* in, uint32_t* out) {
  const int l = threadIdx.x;
  uint32_t v[K];
  #pragma unroll
  for (int i = 0; i < K; ++i) {
    v[i] = in[i * 32 + l] + static_cast<uint32_t>(i);
    asm volatile("" : "+v"(v[i]));
  }
  // Every value is used after all of them were made, in an order that pairs far-apart registers.
  uint32_t a = 0, b = 0;
  #pragma unroll
  for (int i = 0; i < K; ++i) {
    a += v[i] * static_cast<uint32_t>(i + 1);
    b ^= v[K - 1 - i] + a;
  }
  out[l] = a;
  out[32 + l] = b;
}

int main() {
  std::vector<uint32_t> in(K * 32);
  for (int i = 0; i < K; ++i)
    for (int l = 0; l < 32; ++l) in[i * 32 + l] = static_cast<uint32_t>((i * 37 + l * 11) % 101);
  uint32_t *d_in = nullptr, *d_out = nullptr;
  CHECK(hipMalloc(&d_in, in.size() * 4));
  CHECK(hipMalloc(&d_out, 64 * 4));
  CHECK(hipMemcpy(d_in, in.data(), in.size() * 4, hipMemcpyHostToDevice));
  k_many<<<1, 32>>>(d_in, d_out);
  CHECK(hipDeviceSynchronize());
  std::vector<uint32_t> out(64);
  CHECK(hipMemcpy(out.data(), d_out, 64 * 4, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int l = 0; l < 32; ++l) {
    std::vector<uint32_t> v(K);
    for (int i = 0; i < K; ++i) v[i] = in[i * 32 + l] + static_cast<uint32_t>(i);
    uint32_t a = 0, b = 0;
    for (int i = 0; i < K; ++i) {
      a += v[i] * static_cast<uint32_t>(i + 1);
      b ^= v[K - 1 - i] + a;
    }
    wrong += out[l] != a || out[32 + l] != b;
  }
  std::printf("%d live values, past v255: %d of 32 lanes wrong\n", K, wrong);
  std::printf("vgprs1250: %d failed\n", wrong != 0);
  return wrong != 0;
}
