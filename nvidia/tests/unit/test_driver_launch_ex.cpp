// cuLaunchKernelEx and its attribute list. Triton launches a clustered kernel
// (num_ctas > 1) through this entry point with CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
// and CUTLASS's cluster launcher does the same: an entry point that dropped
// the list would run every block as a cluster of one -- a launch that succeeds
// and computes the wrong thing -- and one that refused it (as this one did)
// turned the program away.
//   - the cluster shape reaches %cluster_nctarank, %cluster_ctarank and %clusterid;
//   - a cooperative launch is what cuLaunchCooperativeKernel does;
//   - attributes that change nothing a kernel computes are accepted;
//   - a malformed list is refused.
#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "vtest.hpp"

namespace {

const char* kPtx = R"(
.version 8.0
.target sm_90
.address_size 64
.visible .entry where(.param .u64 out)
{
  .reg .b32 %r<8>;
  .reg .b64 %rd<6>;
  ld.param.u64 %rd1, [out];
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %cluster_ctarank;
  mov.u32 %r3, %cluster_nctarank;
  mov.u32 %r4, %clusterid.x;
  mul.wide.u32 %rd2, %r1, 12;
  add.u64 %rd3, %rd1, %rd2;
  st.global.u32 [%rd3], %r2;
  st.global.u32 [%rd3+4], %r3;
  st.global.u32 [%rd3+8], %r4;
  ret;
}
.visible .entry fill(.param .u64 out)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [out];
  mov.u32 %r1, %ctaid.x;
  mul.wide.u32 %rd2, %r1, 4;
  add.u64 %rd3, %rd1, %rd2;
  add.u32 %r2, %r1, 1;
  st.global.u32 [%rd3], %r2;
  ret;
}
)";

struct Env {
  CUcontext ctx = nullptr;
  CUmodule mod = nullptr;
  CUfunction where = nullptr, fill = nullptr;
  CUdeviceptr out = 0;
  Env() {
    cuInit(0);
    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);
    cuModuleLoadData(&mod, kPtx);
    cuModuleGetFunction(&where, mod, "where");
    cuModuleGetFunction(&fill, mod, "fill");
    cuMemAlloc(&out, 4096);
  }
  ~Env() {
    cuMemFree(out);
    cuModuleUnload(mod);
    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRelease(dev);
  }
  std::vector<uint32_t> read(size_t words) {
    std::vector<uint32_t> v(words);
    cuMemcpyDtoH(v.data(), out, words * 4);
    return v;
  }
};

CUlaunchConfig config(unsigned grid, unsigned block, CUlaunchAttribute* attrs, unsigned n) {
  CUlaunchConfig c{};
  c.gridDimX = grid;
  c.gridDimY = c.gridDimZ = 1;
  c.blockDimX = block;
  c.blockDimY = c.blockDimZ = 1;
  c.attrs = attrs;
  c.numAttrs = n;
  return c;
}

CUlaunchAttribute cluster_attr(unsigned x, unsigned y, unsigned z) {
  CUlaunchAttribute a;
  std::memset(&a, 0, sizeof a);
  a.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  a.value.clusterDim.x = x;
  a.value.clusterDim.y = y;
  a.value.clusterDim.z = z;
  return a;
}

}  // namespace

VTEST(the_attribute_entries_are_72_bytes_as_the_driver_abi_has_them) {
  // The stride cuLaunchKernelEx walks the list by. A toolkit that moved it
  // would be reported here, against the header this test is built with.
  VCHECK_EQ(sizeof(CUlaunchAttribute), size_t{72});
}

VTEST(a_cluster_dimension_attribute_shapes_the_clusters) {
  Env e;
  void* args[] = {&e.out};
  CUlaunchAttribute attr = cluster_attr(4, 1, 1);
  CUlaunchConfig cfg = config(8, 32, &attr, 1);
  VCHECK_EQ(cuLaunchKernelEx(&cfg, e.where, args, nullptr), CUDA_SUCCESS);
  const auto v = e.read(8 * 3);
  for (uint32_t b = 0; b < 8; ++b) {
    VCHECK_EQ(v[3 * b + 0], b % 4);    // %cluster_ctarank
    VCHECK_EQ(v[3 * b + 1], 4u);       // %cluster_nctarank
    VCHECK_EQ(v[3 * b + 2], b / 4);    // %clusterid.x
  }
}

VTEST(the_cluster_attribute_is_found_wherever_it_stands_in_the_list) {
  Env e;
  void* args[] = {&e.out};
  CUlaunchAttribute attrs[3];
  std::memset(attrs, 0, sizeof attrs);
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_PRIORITY;
  attrs[0].value.priority = 0;
  attrs[1].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attrs[1].value.programmaticStreamSerializationAllowed = 1;
  attrs[2] = cluster_attr(2, 1, 1);
  CUlaunchConfig cfg = config(4, 32, attrs, 3);
  VCHECK_EQ(cuLaunchKernelEx(&cfg, e.where, args, nullptr), CUDA_SUCCESS);
  const auto v = e.read(4 * 3);
  for (uint32_t b = 0; b < 4; ++b) {
    VCHECK_EQ(v[3 * b + 1], 2u);
    VCHECK_EQ(v[3 * b + 2], b / 2);
  }
}

VTEST(without_attributes_every_block_is_a_cluster_of_one) {
  Env e;
  void* args[] = {&e.out};
  CUlaunchConfig cfg = config(3, 32, nullptr, 0);
  VCHECK_EQ(cuLaunchKernelEx(&cfg, e.where, args, nullptr), CUDA_SUCCESS);
  const auto v = e.read(3 * 3);
  for (uint32_t b = 0; b < 3; ++b) {
    VCHECK_EQ(v[3 * b + 0], 0u);
    VCHECK_EQ(v[3 * b + 1], 1u);
    VCHECK_EQ(v[3 * b + 2], b);
  }
}

VTEST(attributes_that_change_no_result_are_accepted) {
  Env e;
  void* args[] = {&e.out};
  CUlaunchAttribute attrs[2];
  std::memset(attrs, 0, sizeof attrs);
  attrs[0].id = CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY;
  attrs[0].value.syncPolicy = CU_SYNC_POLICY_SPIN;
  attrs[1].id = CU_LAUNCH_ATTRIBUTE_PRIORITY;
  attrs[1].value.priority = 0;
  CUlaunchConfig cfg = config(5, 32, attrs, 2);
  VCHECK_EQ(cuLaunchKernelEx(&cfg, e.fill, args, nullptr), CUDA_SUCCESS);
  const auto v = e.read(5);
  for (uint32_t b = 0; b < 5; ++b) VCHECK_EQ(v[b], b + 1);
}

VTEST(a_cooperative_attribute_holds_a_grid_to_what_can_be_resident) {
  Env e;
  void* args[] = {&e.out};
  CUlaunchAttribute attr;
  std::memset(&attr, 0, sizeof attr);
  attr.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  attr.value.cooperative = 1;
  // A grid no device can keep resident: a cooperative launch of it would hang
  // on the grid barrier, so it is refused, as cuLaunchCooperativeKernel does.
  CUlaunchConfig big = config(1u << 24, 32, &attr, 1);
  VCHECK(cuLaunchKernelEx(&big, e.fill, args, nullptr) != CUDA_SUCCESS);
  // The same attribute on a grid that fits runs.
  CUlaunchConfig small = config(4, 32, &attr, 1);
  VCHECK_EQ(cuLaunchKernelEx(&small, e.fill, args, nullptr), CUDA_SUCCESS);
  // And with the attribute off, the large grid is an ordinary launch.
  attr.value.cooperative = 0;
  CUlaunchConfig plain = config(64, 32, &attr, 1);
  VCHECK_EQ(cuLaunchKernelEx(&plain, e.fill, args, nullptr), CUDA_SUCCESS);
}

VTEST(a_malformed_launch_configuration_is_refused) {
  Env e;
  void* args[] = {&e.out};
  VCHECK_EQ(cuLaunchKernelEx(nullptr, e.fill, args, nullptr), CUDA_ERROR_INVALID_VALUE);
  CUlaunchConfig cfg = config(1, 32, nullptr, 2);   // two attributes promised, no list
  VCHECK_EQ(cuLaunchKernelEx(&cfg, e.fill, args, nullptr), CUDA_ERROR_INVALID_VALUE);
}

VTEST(a_cluster_the_grid_does_not_divide_into_is_refused) {
  Env e;
  void* args[] = {&e.out};
  CUlaunchAttribute attr = cluster_attr(4, 1, 1);
  CUlaunchConfig cfg = config(6, 32, &attr, 1);   // 6 blocks do not form clusters of 4
  VCHECK(cuLaunchKernelEx(&cfg, e.where, args, nullptr) != CUDA_SUCCESS);
}

VTEST_MAIN
