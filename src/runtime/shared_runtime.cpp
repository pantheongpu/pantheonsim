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
// the same Runtime. A library loaded privately (RTLD_LOCAL) is found by name
// instead (owner(), below).
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu/runtime/visible_devices.hpp"
#include "vgpu/telemetry.hpp"

namespace {
std::mutex g_rt_mu;
// Owned here, in the library whose definition the loader chose, and
// destroyed with it at exit -- which is when its telemetry segment goes.
std::unique_ptr<vgpu::runtime::Runtime> g_rt;
}  // namespace

// This copy's machine, if it has made one, without making one.
extern "C" __attribute__((visibility("default"))) vgpu::runtime::Runtime* vgpu_shared_runtime_peek_v1() {
  std::lock_guard<std::mutex> lock(g_rt_mu);
  return g_rt.get();
}

extern "C" __attribute__((visibility("default"))) vgpu::runtime::Runtime* vgpu_shared_runtime_v1() {
  auto& rt = g_rt;
  std::lock_guard<std::mutex> lock(g_rt_mu);
  if (!rt) {
    const char* gpu = std::getenv("VGPU_GPU");
    const std::string id = gpu && gpu[0] ? gpu : "nvidia/h100";
    int count = 1;
    if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && c[0]) count = std::atoi(c);
    vgpu::DeviceProfile profile = vgpu::load_gpu(id);
    // Optional: shrink advertised VRAM so VRAM-proportional stress tests run
    // at laptop scale (vgpu/registry.hpp).
    vgpu::apply_vram_override(profile);
    // The devices the program is shown: CUDA_VISIBLE_DEVICES picks among the
    // machine's, in the order it lists them. NVML, and the machine's
    // telemetry, still list them all, as on a real host.
    const vgpu::runtime::VisibleDevices shown = vgpu::runtime::cuda_visible_devices(
        profile, count, std::getenv("CUDA_VISIBLE_DEVICES"), std::getenv("CUDA_DEVICE_ORDER"));
    if (shown.error == 0) {
      rt = std::make_unique<vgpu::runtime::Runtime>(profile, static_cast<int>(shown.physical.size()),
                                                    shown.physical, count);
    } else {
      rt = std::make_unique<vgpu::runtime::Runtime>(profile, 1, std::vector<int>{0}, count);
      rt->set_visibility_error(shown.error);
    }
    const char* q = std::getenv("VGPU_QUIET");
    if (!(q && q[0] == '1')) {
      if (shown.error == 0)
        std::fprintf(stderr, "[vgpu] virtual GPU platform initialized: %d x %s (%s)\n",
                     static_cast<int>(shown.physical.size()), profile.id.c_str(), profile.model.c_str());
      else
        std::fprintf(stderr, "[vgpu] virtual GPU platform initialized: no device is visible (CUDA_VISIBLE_DEVICES) -> %d\n",
                     shown.error);
    }
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

namespace {
// The copy of the core whose machine and lock this one uses, chosen once.
//
// First, a copy that has already made the machine, among the two libraries
// that carry the core, found by name if loaded. That is for a library loaded
// privately (RTLD_LOCAL), which the global lookup cannot see: dlsym's default
// search from inside it finds its own definition first. Rust's libloading
// loads that way, and cudarc loads libcuda first and libcudart (under
// libcurand) later, so each made a machine of its own and cuRAND wrote to an
// address only libcuda had allocated. A loaded NVIDIA libcuda has none of
// these symbols and is passed over.
//
// Otherwise the global definition -- the first library loaded, for libraries
// loaded globally -- or this copy's own.
struct Owner {
  Runtime* (*rt)();
  std::recursive_mutex* (*mu)();
};
const Owner& owner() {
  static const Owner o = [] {
    using Peek = Runtime* (*)();
    for (const char* lib : {"libcuda.so.1", "libcudart.so.12", "libcudart.so.13"}) {
      void* h = dlopen(lib, RTLD_LAZY | RTLD_NOLOAD);
      if (!h) continue;
      const auto peek = reinterpret_cast<Peek>(dlsym(h, "vgpu_shared_runtime_peek_v1"));
      const auto rt = reinterpret_cast<Runtime* (*)()>(dlsym(h, "vgpu_shared_runtime_v1"));
      const auto mu = reinterpret_cast<std::recursive_mutex* (*)()>(dlsym(h, "vgpu_shared_api_mutex_v1"));
      dlclose(h);   // NOLOAD took a reference; the library stays loaded
      if (peek && rt && mu && peek()) return Owner{rt, mu};
    }
    const auto rt = reinterpret_cast<Runtime* (*)()>(dlsym(RTLD_DEFAULT, "vgpu_shared_runtime_v1"));
    const auto mu = reinterpret_cast<std::recursive_mutex* (*)()>(dlsym(RTLD_DEFAULT, "vgpu_shared_api_mutex_v1"));
    if (rt && mu) return Owner{rt, mu};
    return Owner{&vgpu_shared_runtime_v1, &vgpu_shared_api_mutex_v1};
  }();
  return o;
}
}  // namespace

std::recursive_mutex& shared_api_mutex() { return *owner().mu(); }

Runtime* shared_runtime() { return owner().rt(); }

}  // namespace vgpu::runtime
