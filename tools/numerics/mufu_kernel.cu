// The two kernels of tools/numerics/mufu_probe.cu: one approximate-math instruction (MUFU) applied to consecutive 32-bit
// patterns. The instruction is written as the PTX form `rcp.approx.ftz.f32`, which ptxas makes a lone `MUFU.RCP` of (no
// instruction before or after it touches the value: the probe checks the disassembly); the probe then rewrites the
// sub-operation field of that instruction in the compiled code to the one asked for (COS, SIN, EX2, LG2, RCP, RSQ, RCP64H,
// RSQ64H, SQRT, TANH): PTX has no form that yields every one of them alone (`sin.approx` is a multiply by 1/2pi and
// MUFU.SIN; the 64H forms are half of a double), and one kernel for all ten keeps the code around the instruction the same.
//
//   nvcc -cubin -arch=sm_<card> mufu_kernel.cu -o mufu_kernel.cubin
#include <cstdint>

__device__ __forceinline__ uint32_t mufu(uint32_t bits) {
  const float x = __uint_as_float(bits);
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return __float_as_uint(r);
}

// out[i] = MUFU(base + i), i < n.
extern "C" __global__ void mufu_dump(uint32_t* out, uint32_t base, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = mufu(base + i);
}

// out[t] = a hash of MUFU over the 2^log2_block inputs from base + t * 2^log2_block, in order: FNV-1a over the 32-bit results
// taken as one unit each (h = (h ^ result) * prime).
extern "C" __global__ void mufu_hash(uint64_t* out, uint32_t base, uint32_t log2_block, uint32_t n) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  const uint32_t start = base + (t << log2_block);
  uint64_t h = 0xcbf29ce484222325ull;
  for (uint32_t j = 0; j < (1u << log2_block); ++j) h = (h ^ mufu(start + j)) * 0x100000001b3ull;
  out[t] = h;
}
