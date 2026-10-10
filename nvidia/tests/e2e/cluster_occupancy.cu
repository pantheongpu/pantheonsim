// cudaOccupancyMaxActiveClusters, cudaOccupancyMaxPotentialClusterSize and the driver's
// cuOccupancyMaxActiveClusters / cuOccupancyMaxPotentialClusterSize (nvidia/docs/clusters.md).
//
// Prints what each answers for a plain kernel, cluster sizes from 1 to 17, with and without
// cudaFuncAttributeNonPortableClusterSizeAllowed. On an RTX 3060 (no clusters) the output is
// the card's own, recorded in cluster_occupancy.rtx3060.expected (`run_cluster_occupancy.sh
// --card --update` writes it on a machine with one). On a part with clusters the output is
// checked by the run script against the layout NVIDIA publishes for the part (derived: the
// SMs of a part with TPCs disabled are spread over its GPCs evenly), not against a card.
//
// Not called with a null configuration or an unregistered function pointer: the card's
// runtime crashes on both.
#include <cstdio>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void plain(float* p) { p[threadIdx.x + blockIdx.x * blockDim.x] += 1.0f; }

static const char* err(cudaError_t e) { return cudaGetErrorName(e); }
static const char* err(CUresult r) {
  const char* s = "?";
  cuGetErrorName(r, &s);
  return s;
}

int main() {
  cudaFree(nullptr);
  int clusters = 0;
  cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, 0);
  std::printf("cluster launch %d\n", clusters);
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, reinterpret_cast<const void*>(plain), 128, 0);
  std::printf("blocks per sm %d\n", per_sm);

  cudaLaunchConfig_t cfg;
  std::memset(&cfg, 0, sizeof cfg);
  cfg.gridDim = dim3(2048);
  cfg.blockDim = dim3(128);
  cudaLaunchAttribute attr;
  std::memset(&attr, 0, sizeof attr);
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;

  CUlaunchConfig dcfg;
  std::memset(&dcfg, 0, sizeof dcfg);
  dcfg.gridDimX = 2048;
  dcfg.gridDimY = dcfg.gridDimZ = 1;
  dcfg.blockDimX = 128;
  dcfg.blockDimY = dcfg.blockDimZ = 1;
  CUlaunchAttribute dattr;
  std::memset(&dattr, 0, sizeof dattr);
  dattr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  dattr.value.clusterDim.y = 1;
  dattr.value.clusterDim.z = 1;
  // The driver's twins take a CUfunction: the same kernel as PTX in a module.
  static const char kPtx[] = R"PTX(
.version 7.0
.target sm_75
.address_size 64
.visible .entry plain(.param .u64 p)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<5>;
  .reg .f32 %f<3>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd4, %rd2, %rd3;
  ld.global.f32 %f1, [%rd4];
  add.f32 %f2, %f1, 0f3F800000;
  st.global.f32 [%rd4], %f2;
  ret;
}
)PTX";
  CUmodule module = nullptr;
  CUfunction dfn = nullptr;
  if (cuModuleLoadData(&module, kPtx) != CUDA_SUCCESS || cuModuleGetFunction(&dfn, module, "plain") != CUDA_SUCCESS) {
    std::printf("the driver module did not load\n");
    return 1;
  }

  for (int nonportable = 0; nonportable < 2; ++nonportable) {
    cudaFuncSetAttribute(reinterpret_cast<const void*>(plain), cudaFuncAttributeNonPortableClusterSizeAllowed, nonportable);
    cuFuncSetAttribute(dfn, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, nonportable);
    std::printf("non-portable sizes %s\n", nonportable ? "allowed" : "not allowed");
    // No cluster dimension in the configuration.
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    dcfg.attrs = nullptr;
    dcfg.numAttrs = 0;
    int n = -7, m = -7;
    cudaError_t e = cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(plain), &cfg);
    CUresult r = cuOccupancyMaxActiveClusters(&m, dfn, &dcfg);
    std::printf("active, no dimension: %s %d | driver %s %d\n", err(e), n, err(r), m);
    n = m = -7;
    e = cudaOccupancyMaxPotentialClusterSize(&n, reinterpret_cast<const void*>(plain), &cfg);
    r = cuOccupancyMaxPotentialClusterSize(&m, dfn, &dcfg);
    std::printf("potential, no dimension: %s %d | driver %s %d\n", err(e), n, err(r), m);
    // Each size in x.
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    dcfg.attrs = &dattr;
    dcfg.numAttrs = 1;
    for (int size : {1, 2, 3, 4, 5, 8, 9, 12, 16, 17}) {
      attr.val.clusterDim.x = static_cast<unsigned>(size);
      dattr.value.clusterDim.x = static_cast<unsigned>(size);
      n = m = -7;
      e = cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(plain), &cfg);
      r = cuOccupancyMaxActiveClusters(&m, dfn, &dcfg);
      std::printf("active, cluster %2d: %s %d | driver %s %d\n", size, err(e), n, err(r), m);
    }
    // The dimensions multiply.
    attr.val.clusterDim.x = 2;
    attr.val.clusterDim.y = 4;
    dattr.value.clusterDim.x = 2;
    dattr.value.clusterDim.y = 4;
    n = m = -7;
    e = cudaOccupancyMaxActiveClusters(&n, reinterpret_cast<const void*>(plain), &cfg);
    r = cuOccupancyMaxActiveClusters(&m, dfn, &dcfg);
    std::printf("active, cluster 2x4: %s %d | driver %s %d\n", err(e), n, err(r), m);
    attr.val.clusterDim.y = 1;
    dattr.value.clusterDim.y = 1;
    // The dimension of the configuration is ignored by the potential size.
    attr.val.clusterDim.x = 2;
    dattr.value.clusterDim.x = 2;
    n = m = -7;
    e = cudaOccupancyMaxPotentialClusterSize(&n, reinterpret_cast<const void*>(plain), &cfg);
    r = cuOccupancyMaxPotentialClusterSize(&m, dfn, &dcfg);
    std::printf("potential, cluster 2: %s %d | driver %s %d\n", err(e), n, err(r), m);
    // A block that does not fit: no clusters, and no size.
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
    cfg.dynamicSmemBytes = 1u << 20;
    n = -7;
    e = cudaOccupancyMaxPotentialClusterSize(&n, reinterpret_cast<const void*>(plain), &cfg);
    std::printf("potential, 1 MiB of shared memory: %s %d\n", err(e), n);
    cfg.dynamicSmemBytes = 0;
  }
  // Null pointers the card answers.
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;
  cudaError_t e = cudaOccupancyMaxActiveClusters(nullptr, reinterpret_cast<const void*>(plain), &cfg);
  std::printf("null count: %s\n", err(e));
  e = cudaOccupancyMaxPotentialClusterSize(nullptr, reinterpret_cast<const void*>(plain), &cfg);
  std::printf("null size: %s\n", err(e));
  int n = -7;
  e = cudaOccupancyMaxActiveClusters(&n, nullptr, &cfg);
  std::printf("null function: %s %d\n", err(e), n);
  cuModuleUnload(module);
  return 0;
}
