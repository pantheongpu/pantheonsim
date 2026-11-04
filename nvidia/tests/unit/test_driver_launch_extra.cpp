// The `extra` form of the driver's launch calls (pantheonsim #471): kernelParams NULL and
//   extra = {CU_LAUNCH_PARAM_BUFFER_POINTER, &buffer, CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END}
// where the buffer holds the kernel's parameters laid out as its parameter block is (each at the next offset its
// alignment allows). cudarc and Dr.Jit launch this way. The cases:
//   - the arguments reach the kernel, through cuLaunchKernel, cuLaunchKernelEx and a graph kernel node;
//   - a parameter that needs padding sits at its aligned offset, not packed against the one before;
//   - a malformed list (a size short of the parameters, kernelParams as well as extra, no buffer, no size, a key
//     given twice, an unknown key, no END) is CUDA_ERROR_INVALID_VALUE and runs nothing.
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vtest.hpp"

namespace {

// Parameters: u8 at 0, u64 at 8, u32 at 16, f64 at 24; 32 bytes in all.
const char* kPtx = R"(
.version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u8 a, .param .u64 out, .param .u32 n, .param .f64 d)
{
  .reg .b16 %h<2>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<2>;
  .reg .f64 %fd<2>;
  ld.param.u8 %r1, [a];
  ld.param.u64 %rd1, [out];
  ld.param.u32 %r2, [n];
  ld.param.f64 %fd1, [d];
  st.global.u32 [%rd1], %r1;
  st.global.u32 [%rd1+4], %r2;
  st.global.f64 [%rd1+8], %fd1;
  ret;
}
.visible .entry nop()
{
  ret;
}
)";

struct Args {   // the layout the kernel's parameter block has
  uint8_t a;
  uint8_t pad0[7];
  CUdeviceptr out;
  uint32_t n;
  uint8_t pad1[4];
  double d;
};
static_assert(sizeof(Args) == 32 && offsetof(Args, out) == 8 && offsetof(Args, n) == 16 && offsetof(Args, d) == 24,
              "the layout under test");

struct Env {
  CUcontext ctx = nullptr;
  CUmodule mod = nullptr;
  CUfunction k = nullptr, nop = nullptr;
  CUdeviceptr out = 0;
  Env() {
    cuInit(0);
    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);
    cuModuleLoadData(&mod, kPtx);
    cuModuleGetFunction(&k, mod, "k");
    cuModuleGetFunction(&nop, mod, "nop");
    cuMemAlloc(&out, 64);
    clear();
  }
  ~Env() {
    cuMemFree(out);
    cuModuleUnload(mod);
    CUdevice dev = 0;
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRelease(dev);
  }
  void clear() { cuMemsetD8(out, 0, 64); }
  struct Out {
    uint32_t a, n;
    double d;
  };
  Out read() {
    cuCtxSynchronize();
    Out o{};
    cuMemcpyDtoH(&o, out, sizeof o);
    return o;
  }
  Args args(uint8_t a, uint32_t n, double d) const {
    Args v;
    std::memset(&v, 0, sizeof v);
    v.a = a;
    v.out = out;
    v.n = n;
    v.d = d;
    return v;
  }
};

CUresult launch_extra(CUfunction f, void** extra, void** kernelParams = nullptr) {
  return cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernelParams, extra);
}

}  // namespace

VTEST(the_extra_buffer_reaches_the_kernel_at_each_parameters_aligned_offset) {
  Env e;
  Args a = e.args(7, 42, 2.5);
  size_t sz = sizeof a;
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, extra), CUDA_SUCCESS);
  const Env::Out o = e.read();
  VCHECK_EQ(o.a, 7u);
  VCHECK_EQ(o.n, 42u);
  VCHECK_EQ(o.d, 2.5);
}

VTEST(the_size_may_come_first_and_a_larger_buffer_is_fine) {
  Env e;
  Args a = e.args(3, 9, -1.0);
  size_t sz = 4096;
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, extra), CUDA_SUCCESS);
  const Env::Out o = e.read();
  VCHECK_EQ(o.a, 3u);
  VCHECK_EQ(o.n, 9u);
  VCHECK_EQ(o.d, -1.0);
}

VTEST(extra_agrees_with_kernelParams) {
  Env e;
  uint8_t a8 = 5;
  uint32_t n = 77;
  double d = 8.25;
  void* params[] = {&a8, &e.out, &n, &d};
  VCHECK_EQ(cuLaunchKernel(e.k, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr), CUDA_SUCCESS);
  const Env::Out via_params = e.read();
  e.clear();
  Args a = e.args(5, 77, 8.25);
  size_t sz = sizeof a;
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, extra), CUDA_SUCCESS);
  const Env::Out via_extra = e.read();
  VCHECK(std::memcmp(&via_params, &via_extra, sizeof via_params) == 0);
  VCHECK_EQ(via_extra.n, 77u);
}

VTEST(a_kernel_with_no_parameters_takes_a_list_that_is_only_END_or_a_buffer) {
  Env e;
  void* extra[] = {CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.nop, extra), CUDA_SUCCESS);
  Args a = e.args(0, 0, 0);
  size_t sz = 0;
  void* with_buffer[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.nop, with_buffer), CUDA_SUCCESS);
}

VTEST(cuLaunchKernelEx_takes_extra_too) {
  Env e;
  Args a = e.args(11, 12, 13.5);
  size_t sz = sizeof a;
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  CUlaunchConfig cfg{};
  cfg.gridDimX = cfg.gridDimY = cfg.gridDimZ = 1;
  cfg.blockDimX = cfg.blockDimY = cfg.blockDimZ = 1;
  VCHECK_EQ(cuLaunchKernelEx(&cfg, e.k, nullptr, extra), CUDA_SUCCESS);
  const Env::Out o = e.read();
  VCHECK_EQ(o.a, 11u);
  VCHECK_EQ(o.n, 12u);
  VCHECK_EQ(o.d, 13.5);
}

VTEST(a_graph_kernel_node_takes_extra_too) {
  Env e;
  CUgraph probe = nullptr;
  if (cuGraphCreate(&probe, 0) != CUDA_SUCCESS) {
    std::printf("SKIP: no runtime beside the driver to build graphs with\n");
    return;
  }
  cuGraphDestroy(probe);
  Args a = e.args(21, 22, 23.5);
  size_t sz = sizeof a;
  void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  CUgraph g = nullptr;
  VCHECK_EQ(cuGraphCreate(&g, 0), CUDA_SUCCESS);
  CUDA_KERNEL_NODE_PARAMS np{};
  np.func = e.k;
  np.gridDimX = np.gridDimY = np.gridDimZ = 1;
  np.blockDimX = np.blockDimY = np.blockDimZ = 1;
  np.extra = extra;
  CUgraphNode node = nullptr;
  VCHECK_EQ(cuGraphAddKernelNode(&node, g, nullptr, 0, &np), CUDA_SUCCESS);
  CUgraphExec exec = nullptr;
  VCHECK_EQ(cuGraphInstantiate(&exec, g, 0), CUDA_SUCCESS);
  VCHECK_EQ(cuGraphLaunch(exec, nullptr), CUDA_SUCCESS);
  const Env::Out o = e.read();
  VCHECK_EQ(o.a, 21u);
  VCHECK_EQ(o.n, 22u);
  VCHECK_EQ(o.d, 23.5);
  cuGraphExecDestroy(exec);
  cuGraphDestroy(g);
}

VTEST(a_malformed_extra_list_is_refused_and_runs_nothing) {
  Env e;
  Args a = e.args(9, 9, 9.0);
  size_t sz = sizeof a, short_sz = sizeof a - 1, zero = 0;
  void* params_unused[] = {nullptr};
  // A size short of the parameters' end (the last byte of the f64 is missing).
  void* too_short[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &short_sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, too_short), CUDA_ERROR_INVALID_VALUE);
  void* zero_size[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &zero, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, zero_size), CUDA_ERROR_INVALID_VALUE);
  // kernelParams and extra together.
  void* good[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, good, params_unused), CUDA_ERROR_INVALID_VALUE);
  // No size, no buffer.
  void* no_size[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, no_size), CUDA_ERROR_INVALID_VALUE);
  void* no_buffer[] = {CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, no_buffer), CUDA_ERROR_INVALID_VALUE);
  void* nothing[] = {CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, nothing), CUDA_ERROR_INVALID_VALUE);
  // A key given twice, and an unknown key (no END to reach in either).
  void* twice[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz};
  VCHECK_EQ(launch_extra(e.k, twice), CUDA_ERROR_INVALID_VALUE);
  void* unknown[] = {reinterpret_cast<void*>(0x7), &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, CU_LAUNCH_PARAM_END};
  VCHECK_EQ(launch_extra(e.k, unknown), CUDA_ERROR_INVALID_VALUE);
  // No END: the pointer and size and then a third key that is not END.
  void* no_end[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &a, CU_LAUNCH_PARAM_BUFFER_SIZE, &sz, reinterpret_cast<void*>(0x9), nullptr};
  VCHECK_EQ(launch_extra(e.k, no_end), CUDA_ERROR_INVALID_VALUE);
  // None of them ran a kernel: the output is still zeroes.
  const Env::Out o = e.read();
  VCHECK_EQ(o.a, 0u);
  VCHECK_EQ(o.n, 0u);
}

VTEST_MAIN
