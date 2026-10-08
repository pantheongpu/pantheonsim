// ldmatrix in every form sm_100a and sm_120a have, as their SASS LDSM runs
// it: the 16-bit 8x8 (.M88, .MT88), the 8-bit 16x16 transposed (.8.MT1616),
// and the packed fp4 and fp6 rows unpacked to a byte a value
// (.U4x16P64TO8, .U6x16P32TO8, as .M816 and .MT1616) that CUTLASS's
// f8f6f4 GEMMs feed their mma. Shared memory holds a byte pattern; each
// form's registers are summed per lane and printed, so the SASS run and the
// PTX run must print the same.
#include <cstdio>
#include <cuda_runtime.h>

__device__ unsigned mix(unsigned h, unsigned v) { return (h ^ v) * 16777619u; }

// The 8-bit and packed forms are the family-specific targets' (sm_100a,
// sm_120a and their kin); nvcc also builds the plain target, which has none.
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define HAVE_FORMS 1
#endif

__global__ void forms(unsigned* out) {
#ifdef HAVE_FORMS
  __shared__ __align__(16) unsigned char s[32 * 16 * 2];
  for (unsigned i = threadIdx.x; i < sizeof s; i += 32) s[i] = static_cast<unsigned char>(i * 37 + 11);
  __syncwarp();
  const unsigned lane = threadIdx.x;
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(s)) + lane * 16;   // lane's row
  unsigned r[4], h;
  unsigned* o = out + lane * 12;

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  h = 2166136261u; for (unsigned x : r) h = mix(h, x); o[0] = h;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  h = 2166136261u; for (unsigned x : r) h = mix(h, x); o[1] = h;

  asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  h = 2166136261u; for (unsigned x : r) h = mix(h, x); o[2] = h;
  asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  h = 2166136261u; for (unsigned x : r) h = mix(h, x); o[3] = h;
  asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
  o[4] = mix(mix(2166136261u, r[0]), r[1]);
  asm volatile("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {%0}, [%1];" : "=r"(r[0]) : "r"(a));
  o[5] = r[0];

  asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
  o[6] = mix(mix(2166136261u, r[0]), r[1]);
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  h = 2166136261u; for (unsigned x : r) h = mix(h, x); o[7] = h;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
  h = 2166136261u; for (unsigned x : r) h = mix(h, x); o[8] = h;
  asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
  o[9] = mix(mix(2166136261u, r[0]), r[1]);
  // Two raw words, so a difference shows what moved and not only that it did.
  asm volatile("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {%0}, [%1];" : "=r"(r[0]) : "r"(a));
  o[10] = r[0];
  asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
  o[11] = r[1];
#else
  (void)out;
#endif
}

int main() {
  unsigned* out = nullptr;
  cudaMalloc(&out, 32 * 12 * sizeof(unsigned));
  forms<<<1, 32>>>(out);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  unsigned host[32 * 12];
  cudaMemcpy(host, out, sizeof host, cudaMemcpyDeviceToHost);
  static const char* const names[] = {"m8n8.x4.b16", "m8n8.x4.trans.b16", "m8n16.x4.b4x16_p64", "m8n16.x4.b6x16_p32",
                                      "m8n16.x2.b4x16_p64", "m8n16.x1.b6x16_p32", "m16n16.x1.trans.b8",
                                      "m16n16.x2.trans.b8", "m16n16.x2.trans.b4x16_p64", "m16n16.x1.trans.b6x16_p32",
                                      "m8n16.x1.b4x16_p64 (lane 5's word)", "m16n16.x1.trans.b6x16_p32 (lane 5's 2nd)"};
  for (int f = 0; f < 12; ++f) {
    unsigned h = 2166136261u;
    for (int l = 0; l < 32; ++l) h = (h ^ host[l * 12 + f]) * 16777619u;
    if (f < 10) std::printf("%-28s %08x\n", names[f], h);
    else std::printf("%-28s %08x\n", names[f], host[5 * 12 + f]);
  }
  cudaFree(out);
  return 0;
}
