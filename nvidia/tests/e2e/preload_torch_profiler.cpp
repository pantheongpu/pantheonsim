// PyTorch's profiler under `vgpu run --preload`, with no PyTorch: the simulator's profiling engine (what CUPTI's activity records
// are made from) switched on, and a cuBLAS call. cuBLAS is a host routine here, and the copies it makes to reach its operands are
// inside a Silence: they are not the program's calls, and a profiler must not list them. Each shim carries its own copy of the
// engine, and the process has to use one. It did only while the unversioned libcuda.so.1 was in the global scope: with the driver
// loaded privately (#597) cuBLAS's Silence was its own while the runtime recorded into the runtime's, so PyTorch's profiler listed
// cudaMemcpy. Run by run_preload_torch_like.sh; names no CUDA library, so it loads the runtime and cuBLAS with dlopen (argv: their
// sonames), and reaches the engine as CUPTI does, through the first definition in the global scope.
#include <dlfcn.h>
#include <cstdio>
#include <string>
#include <vector>

#include "vgpu/profiling.hpp"

namespace {
void* sym(void* h, const char* name) {
  void* p = dlsym(h, name);
  if (!p) std::printf("missing %s\n", name);
  return p;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  void* rt = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* blas = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (!rt || !blas) {
    std::printf("cannot load the runtime or cuBLAS: %s\n", dlerror());
    return 2;
  }
  using SetEnabled = void (*)(bool);
  using Drain = std::vector<vgpu::profiling::Event> (*)();
  using Malloc = int (*)(void**, size_t);
  using Free = int (*)(void*);
  using Sync = int (*)();
  using Create = int (*)(void**);
  using Destroy = int (*)(void*);
  using Sgemm = int (*)(void*, int, int, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
  const auto set_enabled = reinterpret_cast<SetEnabled>(sym(RTLD_DEFAULT, "_ZN4vgpu9profiling11set_enabledEb"));
  const auto drain = reinterpret_cast<Drain>(sym(RTLD_DEFAULT, "_ZN4vgpu9profiling5drainEv"));
  const auto cuda_malloc = reinterpret_cast<Malloc>(sym(rt, "cudaMalloc"));
  const auto cuda_free = reinterpret_cast<Free>(sym(rt, "cudaFree"));
  const auto sync = reinterpret_cast<Sync>(sym(rt, "cudaDeviceSynchronize"));
  const auto create = reinterpret_cast<Create>(sym(blas, "cublasCreate_v2"));
  const auto destroy = reinterpret_cast<Destroy>(sym(blas, "cublasDestroy_v2"));
  const auto sgemm = reinterpret_cast<Sgemm>(sym(blas, "cublasSgemm_v2"));
  if (!set_enabled || !drain || !cuda_malloc || !cuda_free || !sync || !create || !destroy || !sgemm) return 2;

  void* handle = nullptr;
  float *a = nullptr, *b = nullptr, *c = nullptr;
  const int n = 16;
  if (create(&handle) != 0 || cuda_malloc(reinterpret_cast<void**>(&a), n * n * sizeof(float)) != 0 ||
      cuda_malloc(reinterpret_cast<void**>(&b), n * n * sizeof(float)) != 0 ||
      cuda_malloc(reinterpret_cast<void**>(&c), n * n * sizeof(float)) != 0) {
    std::printf("cuBLAS or the runtime does not start\n");
    return 1;
  }
  drain();
  set_enabled(true);
  const float one = 1.0f, zero = 0.0f;
  const int gemm = sgemm(handle, 0, 0, n, n, n, &one, a, n, b, n, &zero, c, n);
  void* probe = nullptr;
  cuda_malloc(&probe, 256);   // a call of the program's own: recorded
  sync();
  set_enabled(false);
  const std::vector<vgpu::profiling::Event> events = drain();
  cuda_free(probe);
  cuda_free(a);
  cuda_free(b);
  cuda_free(c);
  destroy(handle);

  int mallocs = 0, memcpys = 0;
  for (const auto& e : events) {
    if (e.kind != vgpu::profiling::EventKind::Api || e.domain_driver) continue;
    if (e.name == "cudaMalloc") ++mallocs;
    if (e.name == "cudaMemcpy") ++memcpys;
  }
  int fails = 0;
  const auto check = [&](bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
  };
  check(gemm == 0, "cublasSgemm");
  check(mallocs == 1, "the program's own cudaMalloc is recorded");
  check(memcpys == 0, "no cudaMemcpy is recorded: cuBLAS's copies of its operands are not the program's calls");
  if (memcpys) std::printf("     (%d cudaMemcpy recorded)\n", memcpys);
  std::printf("%s\n", fails == 0 ? "ok" : "failed");
  return fails == 0 ? 0 : 1;
}
