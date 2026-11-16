// A kernel that reaches the same stack slot two ways, as unoptimized code does: it stores a local through the private
// segment buffer (buffer_store_dword off, s[0:3], s33) and hands its address to a function that is not inlined, which
// loads it through a flat pointer (src_private_base + (s33 >> 5) + offset in a wave32 kernel, >> 6 in a wave64 one).
// Both must name the same byte of the work-item's private memory (amd/tests/unit/test_hipcc_scratch_flat.cpp).
//
// Built -O0 as device code only, for gfx1030 (wave32), without the device libraries (the kernel calls none, and the object
// stays small: no HIP header, the lane's number from the compiler's builtin):
//   hipcc -O0 -nogpulib -std=c++17 --offload-arch=gfx1030 --offload-device-only --no-gpu-bundle-output -c scratch_flat.cpp -o scratch_flat.O0.gfx1030.o
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

__device__ __attribute__((noinline)) float weigh(const float* a, const float* b, const float* c, const float* table) {
  return *a + *b * 2.0f + *c * 4.0f + table[3] * 8.0f;
}

// The locals whose addresses are taken live in a function that is called, not in the kernel: the kernel's own frame starts at
// stack pointer 0, where the scaled pointer and the flat pointer agree whatever the stride is. The function calls twice: the
// addresses it computed on entry are spilled across the first call (through the buffer) and reloaded for the second, after
// flat stores have gone all over the frame.
__device__ __attribute__((noinline)) float frame(float v) {
  float x = v;
  float y = v + 1.0f;
  float z = v + 2.0f;
  float table[4] = {x, y, z, x + 3.0f};
  const float first = weigh(&x, &y, &z, table);
  float more[16];
  for (int k = 0; k < 16; ++k) more[k] = first + static_cast<float>(k);
  const float second = weigh(&y, &z, &x, more);
  return first + second;
}

__global__ void scratch_flat(const float* in, float* out) {
  const int i = __builtin_amdgcn_workitem_id_x();
  float pad[6] = {in[i], 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};   // (the kernel's frame is not empty, so the callee's does not start at 0)
  out[i] = frame(pad[0]) + (pad[1] - 1.0f);
}
