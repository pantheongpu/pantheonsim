// v_sin_f32 and v_cos_f32 of the multiples of 1/64 turn from -4 to 4, and of the floats next to the multiples from 1/64 to 1, on one AMD card.
// Built:  hipcc -O2 --offload-arch=<gfx> amd_trig_probe.cpp -o amd_trig_probe ; ROCR_VISIBLE_DEVICES=<n> ./amd_trig_probe
// Output: one line a result, 'input sin cos' in hex; tools/numerics/amd/gfx1030.trig64ths.txt is the RX 6800's (2026-11-23, ROCm 7.2.0).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
__global__ void k(const uint32_t* in, uint32_t* s, uint32_t* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = __uint_as_float(in[i]), rs, rc;
  asm volatile("v_sin_f32 %0, %1" : "=v"(rs) : "v"(x));
  asm volatile("v_cos_f32 %0, %1" : "=v"(rc) : "v"(x));
  s[i] = __float_as_uint(rs);
  c[i] = __float_as_uint(rc);
}
int main() {
  std::vector<uint32_t> in;
  for (int j = -256; j <= 256; ++j) { float f = j / 64.0f; uint32_t b; memcpy(&b, &f, 4); in.push_back(b); }
  for (int j = 1; j <= 64; ++j) { float f = j / 64.0f; uint32_t b; memcpy(&b, &f, 4); in.push_back(b + 1); in.push_back(b - 1); }
  int n = in.size();
  uint32_t *d_in, *d_s, *d_c;
  hipMalloc(&d_in, n * 4); hipMalloc(&d_s, n * 4); hipMalloc(&d_c, n * 4);
  hipMemcpy(d_in, in.data(), n * 4, hipMemcpyHostToDevice);
  k<<<(n + 63) / 64, 64>>>(d_in, d_s, d_c, n);
  std::vector<uint32_t> s(n), c(n);
  hipMemcpy(s.data(), d_s, n * 4, hipMemcpyDeviceToHost);
  hipMemcpy(c.data(), d_c, n * 4, hipMemcpyDeviceToHost);
  char name[256]; hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  printf("# %s %s\n", p.name, p.gcnArchName);
  for (int i = 0; i < n; ++i) printf("%08x %08x %08x\n", in[i], s[i], c[i]);
}
