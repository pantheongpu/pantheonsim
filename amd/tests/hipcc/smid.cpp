// Which compute unit a work-group runs on, as HIP's __smid() says, on every
// simulated AMD GPU. It reads the hardware registers that hold where a wave
// runs: HW_ID on CDNA, with XCC_ID's compute die on gfx942 and gfx950, and
// HW_ID1's workgroup processor on RDNA2 and RDNA3. On gfx12 HIP's header
// still reads HW_ID's fields from register 4, which gfx12 made STATE_PRIV, so
// there this reads HW_ID1 itself, as __smid does on RDNA3.
//
// Four work-groups to a multiprocessor, of 256 work-items: every one should
// find a multiprocessor of its own among them all, and every wave of a group
// the same one. Built for each target into one program (build.sh).
//
// On gfx1030 and gfx1031 the waves of a group are spread over the SIMDs of one
// workgroup processor, and __smid's lowest bit is the SIMD number's (HIP
// reads it as a CU in a WGP), so a real RX 6800 or RX 6700 XT gives two
// values per group; the workgroup processor is what stays the same, so that
// is what is compared there.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <set>
#include <vector>

__device__ unsigned where() {
#if defined(__gfx1200__) || defined(__gfx1201__)
  return __builtin_amdgcn_s_getreg((10 << 11) | (10 << 6) | 23);   // hwreg(HW_REG_HW_ID1, 10, 11): 20:10
#elif defined(__gfx1030__) || defined(__gfx1031__)
  return __smid() >> 1;
#else
  return __smid();
#endif
}

__global__ void smid(unsigned* out, unsigned* disagree) {
  __shared__ unsigned first;
  const unsigned id = where();
  if (threadIdx.x == 0) first = id;
  __syncthreads();
  if (id != first) atomicAdd(disagree, 1u);
  if (threadIdx.x == 0) out[blockIdx.x] = id;
}

int main() {
  hipDeviceProp_t p;
  if (hipGetDeviceProperties(&p, 0) != hipSuccess) return 1;
  const int groups = 4 * p.multiProcessorCount;
  unsigned *out, *disagree;
  if (hipMalloc(&out, groups * sizeof(unsigned)) != hipSuccess || hipMalloc(&disagree, sizeof(unsigned)) != hipSuccess ||
      hipMemset(disagree, 0, sizeof(unsigned)) != hipSuccess)
    return 1;
  smid<<<groups, 256>>>(out, disagree);
  std::vector<unsigned> ids(groups);
  unsigned wrong = 0;
  if (hipMemcpy(ids.data(), out, groups * sizeof(unsigned), hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(&wrong, disagree, sizeof(unsigned), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  const std::set<unsigned> distinct(ids.begin(), ids.end());
  std::printf("%s: %zu of %d multiprocessors, %u work-items disagree\n", p.gcnArchName, distinct.size(),
              p.multiProcessorCount, wrong);
  return 0;
}
