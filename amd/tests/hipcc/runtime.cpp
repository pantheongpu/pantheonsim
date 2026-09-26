// What a HIP program asks its runtime about a device and a kernel, built by
// hipcc: how many work-groups of a kernel a compute unit holds (which the
// kernel's registers and LDS decide), the block size that fills the device,
// each device attribute against the property it names, and a kernel on one
// device reading and writing another's memory once peer access is enabled.
#include <hip/hip_runtime.h>

#include <cstdio>

#define CHECK(x)                                                                    \
  do {                                                                              \
    hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                         \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                      \
      return 1;                                                                     \
    }                                                                               \
  } while (0)

__global__ void plain(int* out) { out[blockIdx.x * blockDim.x + threadIdx.x] = threadIdx.x; }

// 24 KB of LDS a work-group: two fit in a compute unit's 64 KB.
__global__ void tiled(float* out) {
  __shared__ float tile[6144];
  tile[threadIdx.x] = threadIdx.x;
  __syncthreads();
  out[blockIdx.x * blockDim.x + threadIdx.x] = tile[(threadIdx.x * 7) % blockDim.x];
}

// Told that it overwrites v0 to v127, the compiler gives it 128 vector
// registers, so a SIMD holds four of its waves rather than eight.
__global__ void __launch_bounds__(256) wide(int* out) {
  asm volatile("" ::: "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
               "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
               "v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44",
               "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53", "v54", "v55", "v56", "v57", "v58", "v59",
               "v60", "v61", "v62", "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73", "v74",
               "v75", "v76", "v77", "v78", "v79", "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89",
               "v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100", "v101", "v102", "v103",
               "v104", "v105", "v106", "v107", "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116",
               "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125", "v126", "v127");
  out[blockIdx.x * blockDim.x + threadIdx.x] = 1;
}

// Runs on device 0 with two of its pointers into device 1.
__global__ void from_peer(const int* peer_in, int* peer_out, int* local_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  local_out[i] = peer_in[i] + 1;
  peer_out[i] = peer_in[i] * 2;
}

int main() {
  // Work-groups a compute unit holds.
  int plain256 = 0, plain65 = 0, plain_lds = 0, tiled256 = 0, wide256 = 0;
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&plain256, plain, 256, 0));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&plain65, plain, 65, 0));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&plain_lds, plain, 256, 20000));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&tiled256, tiled, 256, 0));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&wide256, wide, 256, 0));
  std::printf("work-groups a compute unit holds: plain %d, of 65 threads %d, with 20000 bytes of LDS %d, "
              "tiled %d, wide %d\n", plain256, plain65, plain_lds, tiled256, wide256);
  // The block size, and the grid of it, that fills the device.
  int grid = 0, block = 0, wide_grid = 0, wide_block = 0;
  CHECK(hipOccupancyMaxPotentialBlockSize(&grid, &block, plain, 0, 0));
  CHECK(hipOccupancyMaxPotentialBlockSize(&wide_grid, &wide_block, wide, 0, 256));
  std::printf("filling the device: plain %d of %d, wide %d of %d\n", grid, block, wide_grid, wide_block);

  // Every attribute against the property it names.
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  const struct {
    hipDeviceAttribute_t attribute;
    long long property;
  } pairs[] = {
      {hipDeviceAttributeMaxThreadsPerBlock, p.maxThreadsPerBlock},
      {hipDeviceAttributeMaxBlockDimX, p.maxThreadsDim[0]},
      {hipDeviceAttributeMaxBlockDimZ, p.maxThreadsDim[2]},
      {hipDeviceAttributeMaxGridDimX, p.maxGridSize[0]},
      {hipDeviceAttributeMaxSharedMemoryPerBlock, static_cast<long long>(p.sharedMemPerBlock)},
      {hipDeviceAttributeTotalConstantMemory, static_cast<long long>(p.totalConstMem)},
      {hipDeviceAttributeWarpSize, p.warpSize},
      {hipDeviceAttributeMaxRegistersPerBlock, p.regsPerBlock},
      {hipDeviceAttributeClockRate, p.clockRate},
      {hipDeviceAttributeMemoryClockRate, p.memoryClockRate},
      {hipDeviceAttributeMemoryBusWidth, p.memoryBusWidth},
      {hipDeviceAttributeMultiprocessorCount, p.multiProcessorCount},
      {hipDeviceAttributeComputeMode, p.computeMode},
      {hipDeviceAttributeL2CacheSize, p.l2CacheSize},
      {hipDeviceAttributeMaxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor},
      {hipDeviceAttributeComputeCapabilityMajor, p.major},
      {hipDeviceAttributeComputeCapabilityMinor, p.minor},
      {hipDeviceAttributeConcurrentKernels, p.concurrentKernels},
      {hipDeviceAttributePciBusId, p.pciBusID},
      {hipDeviceAttributePciDeviceId, p.pciDeviceID},
      {hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, static_cast<long long>(p.maxSharedMemoryPerMultiProcessor)},
      {hipDeviceAttributeIsMultiGpuBoard, p.isMultiGpuBoard},
      {hipDeviceAttributeIntegrated, p.integrated},
      {hipDeviceAttributeCooperativeLaunch, p.cooperativeLaunch},
      {hipDeviceAttributeCooperativeMultiDeviceLaunch, p.cooperativeMultiDeviceLaunch},
      {hipDeviceAttributeEccEnabled, p.ECCEnabled},
      {hipDeviceAttributeManagedMemory, p.managedMemory},
      {hipDeviceAttributeIsLargeBar, p.isLargeBar},
      {hipDeviceAttributeAsicRevision, p.asicRevision},
      {hipDeviceAttributeUnifiedAddressing, p.unifiedAddressing},
  };
  int agree = 0;
  for (const auto& a : pairs) {
    int v = -1;
    if (hipDeviceGetAttribute(&v, a.attribute, 0) == hipSuccess && v == a.property) ++agree;
    else std::printf("attribute %d says %d, and the property %lld\n", static_cast<int>(a.attribute), v, a.property);
  }
  std::printf("%d of %zu attributes agree with the properties\n", agree, sizeof pairs / sizeof pairs[0]);
  int v = 0;
  std::printf("a device that is not there: %s\n", hipGetErrorName(hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, 99)));
  // HIP keeps the last call that failed until hipGetLastError reads it: a
  // call that succeeds in between does not clear it, and reading it does.
  int count = 0;
  CHECK(hipGetDeviceCount(&count));
  const hipError_t kept = hipPeekAtLastError(), read = hipGetLastError(), after = hipGetLastError();
  std::printf("the last error outlives a call that succeeds: %s, %s, then %s\n", hipGetErrorName(kept),
              hipGetErrorName(read), hipGetErrorName(after));

  // A kernel on device 0 reaching into device 1.
  if (count < 2) {
    std::printf("one device: no peer to reach\n");
    return 0;
  }
  const int n = 256;
  int host[n];
  for (int i = 0; i < n; ++i) host[i] = 3 * i - 100;
  int *peer_in = nullptr, *peer_out = nullptr, *local_out = nullptr;
  CHECK(hipSetDevice(1));
  CHECK(hipMalloc(&peer_in, sizeof host));
  CHECK(hipMalloc(&peer_out, sizeof host));
  CHECK(hipMemcpy(peer_in, host, sizeof host, hipMemcpyHostToDevice));
  CHECK(hipSetDevice(0));
  CHECK(hipMalloc(&local_out, sizeof host));
  CHECK(hipDeviceEnablePeerAccess(1, 0));
  from_peer<<<n / 64, 64>>>(peer_in, peer_out, local_out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  int local[n], remote[n];
  CHECK(hipMemcpy(local, local_out, sizeof local, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(remote, peer_out, sizeof remote, hipMemcpyDeviceToHost));
  int right = 0;
  for (int i = 0; i < n; ++i) right += local[i] == host[i] + 1 && remote[i] == host[i] * 2;
  std::printf("a kernel on device 0 read and wrote device 1's memory right for %d of %d elements\n", right, n);
  CHECK(hipDeviceDisablePeerAccess(1));
  std::printf("disabling it again: %s\n", hipGetErrorName(hipDeviceDisablePeerAccess(1)));
  return 0;
}
