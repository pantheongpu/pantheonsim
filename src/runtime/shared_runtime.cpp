// One simulated machine per process, whichever CUDA library asks first.
//
// libcudart and libcuda each carry a copy of this core, and each used to make
// its own Runtime -- its own devices, its own device memory, the same
// addresses. A program that mixes the two APIs, as frameworks do (CUTLASS's
// cuMemsetD32Async on a cudaMalloc'd workspace, PyTorch's driver-API kernel
// loads on caching-allocator memory), then handed one library a pointer only
// the other knew. Now both ask for the machine through
// vgpu_shared_runtime_v1, looked up in the process's global scope: the loader
// gives both libraries the same definition -- the first one loaded -- and so
// the same Runtime. A library loaded privately (RTLD_LOCAL) finds only its
// own, as before.
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu/telemetry.hpp"

extern "C" __attribute__((visibility("default"))) vgpu::runtime::Runtime* vgpu_shared_runtime_v1() {
  static std::mutex mu;
  // Owned here, in the library whose definition the loader chose, and
  // destroyed with it at exit -- which is when its telemetry segment goes.
  static std::unique_ptr<vgpu::runtime::Runtime> rt;
  std::lock_guard<std::mutex> lock(mu);
  if (!rt) {
    const char* gpu = std::getenv("VGPU_GPU");
    const std::string id = gpu && gpu[0] ? gpu : "nvidia/h100";
    int count = 1;
    if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && c[0]) count = std::atoi(c);
    vgpu::DeviceProfile profile = vgpu::load_gpu(id);
    // Optional: shrink advertised VRAM so VRAM-proportional stress tests run
    // at laptop scale (vgpu/registry.hpp).
    vgpu::apply_vram_override(profile);
    rt = std::make_unique<vgpu::runtime::Runtime>(profile, count);
    const char* q = std::getenv("VGPU_QUIET");
    if (!(q && q[0] == '1'))
      std::fprintf(stderr, "[vgpu] virtual GPU platform initialized: %d x %s (%s)\n", count, profile.id.c_str(),
                   profile.model.c_str());
  }
  return rt.get();
}

// And one lock for the two libraries' API calls. Each used to lock only its
// own calls, which was enough while each had its own machine; sharing one, a
// program calling the runtime on one thread and the driver on another could
// load modules into the same device, or run kernels whose atomics each
// library's interpreter serialized with a table of its own, at the same time.
// Never destroyed: a library's exit handlers may still take it.
extern "C" __attribute__((visibility("default"))) std::recursive_mutex* vgpu_shared_api_mutex_v1() {
  static auto* mu = new std::recursive_mutex();
  return mu;
}

namespace vgpu::runtime {

std::recursive_mutex& shared_api_mutex() {
  using Fn = std::recursive_mutex* (*)();
  static const Fn fn = [] {
    const auto found = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "vgpu_shared_api_mutex_v1"));
    return found ? found : &vgpu_shared_api_mutex_v1;
  }();
  return *fn();
}

Runtime* shared_runtime() {
  using Fn = Runtime* (*)();
  static const Fn fn = [] {
    const auto found = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "vgpu_shared_runtime_v1"));
    return found ? found : &vgpu_shared_runtime_v1;
  }();
  return fn();
}

}  // namespace vgpu::runtime
