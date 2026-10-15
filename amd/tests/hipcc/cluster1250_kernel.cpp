// The kernel of cluster1250.cpp: built for a cluster of 4 x 2 work-groups (build.sh adds the "amdgpu-cluster-dims" function attribute to the
// kernel's IR, as ROCm 7.2's clang has no source attribute for it), and compiled to a code object the host program embeds.
#include <hip/hip_runtime.h>

// Per work-group, eight words of o: where the work-group is by the compiler's own account (blockIdx, from TTMP6/7/9 and IB_STS2), by the
// cluster builtins, and the sum of every flag of its cluster's work-groups read after the cluster barrier (each wrote its own before).
extern "C" __global__ void k_cluster(unsigned* o, unsigned* flags) {
  const unsigned group = blockIdx.y * gridDim.x + blockIdx.x;
  if (threadIdx.x == 0) flags[group] = group + 1;
  __syncthreads();
  __builtin_amdgcn_s_cluster_barrier();
  if (threadIdx.x == 0) {
    const unsigned cx = __builtin_amdgcn_cluster_id_x(), cy = __builtin_amdgcn_cluster_id_y();
    const unsigned wx = __builtin_amdgcn_cluster_workgroup_id_x(), wy = __builtin_amdgcn_cluster_workgroup_id_y();
    const unsigned flat = __builtin_amdgcn_cluster_workgroup_flat_id();
    unsigned sum = 0;
    for (unsigned y = 0; y < 2; ++y)
      for (unsigned x = 0; x < 4; ++x) sum += flags[(cy * 2 + y) * gridDim.x + cx * 4 + x];
    unsigned* mine = o + group * 8;
    mine[0] = blockIdx.x, mine[1] = blockIdx.y, mine[2] = cx, mine[3] = cy, mine[4] = wx, mine[5] = wy, mine[6] = flat, mine[7] = sum;
  }
}
