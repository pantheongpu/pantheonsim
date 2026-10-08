// Work-group shapes: every work-item of a block with more than one dimension
// is reached, with its own ids, and the waves of one work-group see each
// other's LDS writes after __syncthreads. A block of 64x2 on a wave-64 GPU is
// two waves, the second of which is y = 1; if the simulator hands a wave the
// wrong ids that wave runs as y = 0 and the rows overwrite each other. Written
// for this repository, in HIP. It knows nothing of VirtualGPU.
//
//   workgroup          prints one line per block shape and exits 1 if any is wrong
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

namespace {

// Each work-item stores its own three ids at its linear index, then the
// waves swap halves of the block through LDS: item i reads what item
// (i + n/2) % n wrote, from another wave when the block has several.
__global__ void ids_and_lds(unsigned* ids, unsigned* swapped) {
  __shared__ unsigned lds[1024];
  const unsigned n = blockDim.x * blockDim.y * blockDim.z;
  const unsigned lin = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  ids[lin] = (threadIdx.z << 20) | (threadIdx.y << 10) | threadIdx.x;
  lds[lin] = 1000 + lin;
  __syncthreads();
  swapped[lin] = lds[(lin + n / 2) % n];
}

bool run(dim3 block) {
  const unsigned n = block.x * block.y * block.z;
  unsigned *ids = nullptr, *swapped = nullptr;
  if (hipMalloc(&ids, n * 4) != hipSuccess || hipMalloc(&swapped, n * 4) != hipSuccess) return false;
  hipMemset(ids, 0xff, n * 4);
  hipMemset(swapped, 0xff, n * 4);
  ids_and_lds<<<1, block>>>(ids, swapped);
  std::vector<unsigned> h_ids(n), h_swapped(n);
  const bool ok = hipMemcpy(h_ids.data(), ids, n * 4, hipMemcpyDeviceToHost) == hipSuccess &&
                  hipMemcpy(h_swapped.data(), swapped, n * 4, hipMemcpyDeviceToHost) == hipSuccess;
  unsigned bad_ids = 0, bad_lds = 0;
  for (unsigned z = 0, lin = 0; z < block.z; ++z)
    for (unsigned y = 0; y < block.y; ++y)
      for (unsigned x = 0; x < block.x; ++x, ++lin) {
        if (!ok || h_ids[lin] != ((z << 20) | (y << 10) | x)) ++bad_ids;
        if (!ok || h_swapped[lin] != 1000 + (lin + n / 2) % n) ++bad_lds;
      }
  std::printf("block %ux%ux%u: work-items with the wrong ids %u of %u, LDS exchange wrong %u\n", block.x, block.y,
              block.z, bad_ids, n, bad_lds);
  hipFree(ids);
  hipFree(swapped);
  return bad_ids == 0 && bad_lds == 0;
}

}  // namespace

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 2;
  std::printf("device %s warp %d\n", prop.name, prop.warpSize);
  bool all = true;
  for (dim3 block : {dim3(128, 1, 1), dim3(64, 2, 1), dim3(32, 4, 1), dim3(16, 8, 1), dim3(64, 4, 1), dim3(32, 8, 1),
                     dim3(8, 8, 2), dim3(4, 4, 16), dim3(40, 5, 1)})
    all &= run(block);
  std::printf(all ? "PASS\n" : "FAIL\n");
  return all ? 0 : 1;
}
