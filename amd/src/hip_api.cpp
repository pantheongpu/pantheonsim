// libamdhip64: the HIP runtime API, over VirtualGPU's devices.
//
// A HIP program asks for devices and memory, loads a code object, and
// launches the kernels in it. That is what this implements: the devices come
// from a profile (VGPU_GPU, an AMD one), the memory from the device's manager,
// and a launch decodes and runs the kernel's own CDNA instructions
// (vgpu/amd_exec.hpp).
//
// The interface is the documented HIP one (amd/include/vgpu_hip.h, written
// from AMD's public documentation); nothing here is AMD's code. What is not
// implemented is refused by name rather than ignored, because a HIP program
// that believes a launch happened will compare wrong answers.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu_hip.h"

namespace {

using vgpu::amd::CodeObject;
using vgpu::amd::Kernel;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}

// Says what went wrong and why, once, where a program would otherwise only
// see an error code.
hipError_t fail(hipError_t code, const std::string& what) {
  if (!quiet()) std::fprintf(stderr, "VirtualGPU HIP: %s\n", what.c_str());
  return code;
}

struct Module {
  CodeObject object;
  uint64_t globals = 0;   // where the module's own variables were placed
};

// A kernel a program has looked up: the module it came from, and which kernel.
struct Function {
  Module* module = nullptr;
  const Kernel* kernel = nullptr;
};

struct State {
  std::mutex mutex;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  std::vector<std::unique_ptr<Module>> modules;
  std::vector<std::unique_ptr<Function>> functions;
  int current = 0;
  hipError_t last = hipSuccess;
  std::string profile_id;
};

State& state() {
  static State s;
  return s;
}

// The rack the environment describes. An NVIDIA profile is refused here: a
// HIP program on an NVIDIA GPU is a mistake worth naming, not a silent
// translation.
hipError_t ensure_runtime(State& s) {
  if (s.rt) return hipSuccess;
  const char* gpu = std::getenv("VGPU_GPU");
  const std::string id = gpu && *gpu ? gpu : "amd/mi300x";
  int count = 1;
  if (const char* c = std::getenv("VGPU_DEVICE_COUNT"); c && *c) count = std::atoi(c);
  if (count < 1) count = 1;
  try {
    vgpu::DeviceProfile p = vgpu::load_gpu(id);
    if (p.vendor != "amd")
      return fail(hipErrorInvalidDevice, "VGPU_GPU=" + id +
                                             " is not an AMD GPU, and HIP runs on AMD GPUs. Set VGPU_GPU to an "
                                             "amd/ profile (for example VGPU_GPU=amd/mi300x).");
    vgpu::apply_vram_override(p);
    s.rt = std::make_unique<vgpu::runtime::Runtime>(p, count);
    s.profile_id = p.id;
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidDevice, std::string("no usable GPU profile: ") + e.what());
  }
  return hipSuccess;
}

vgpu::runtime::Device* device(State& s) {
  if (ensure_runtime(s) != hipSuccess) return nullptr;
  if (s.current < 0 || s.current >= s.rt->device_count()) return nullptr;
  return &s.rt->device(s.current);
}

hipError_t record(State& s, hipError_t e) {
  s.last = e;
  return e;
}

// The kernel's arguments, as HIP passes them: either one packed buffer
// (`extra`), or one pointer per argument (`kernelParams`), which are packed
// here at the offsets the code object's metadata gives.
hipError_t build_kernargs(const Kernel& k, void** params, void** extra, std::vector<uint8_t>* out) {
  out->assign(k.kernarg_size, 0);
  if (extra) {
    const void* buffer = nullptr;
    size_t size = 0;
    for (size_t i = 0; extra[i] != HIP_LAUNCH_PARAM_END; ++i) {
      if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_POINTER) buffer = extra[++i];
      else if (extra[i] == HIP_LAUNCH_PARAM_BUFFER_SIZE) size = *static_cast<size_t*>(extra[++i]);
      else return fail(hipErrorInvalidValue, "extra[] holds something that is not a launch parameter");
    }
    if (!buffer) return fail(hipErrorInvalidValue, "extra[] has no argument buffer");
    if (size > out->size()) out->resize(size);
    std::memcpy(out->data(), buffer, size ? size : out->size());
    return hipSuccess;
  }
  if (!params) {
    // A kernel that takes nothing needs neither.
    return k.kernarg_size && !k.args.empty()
               ? fail(hipErrorInvalidValue, "the kernel takes arguments, and the launch passed none")
               : hipSuccess;
  }
  for (size_t i = 0; i < k.args.size(); ++i) {
    const vgpu::amd::KernelArg& a = k.args[i];
    if (a.hidden()) continue;   // what the compiler adds is not the program's to pass
    if (!params[i]) return fail(hipErrorInvalidValue, "argument " + std::to_string(i) + " is a null pointer");
    if (a.offset + a.size > out->size())
      return fail(hipErrorInvalidValue, "argument " + std::to_string(i) + " is past the kernarg segment");
    std::memcpy(out->data() + a.offset, params[i], a.size);
  }
  return hipSuccess;
}

}  // namespace

extern "C" {

hipError_t hipInit(unsigned int) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, ensure_runtime(s));
}

hipError_t hipGetDeviceCount(int* count) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!count) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  *count = s.rt->device_count();
  return record(s, hipSuccess);
}

hipError_t hipSetDevice(int d) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (d < 0 || d >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  s.current = d;
  return record(s, hipSuccess);
}

hipError_t hipGetDevice(int* d) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!d) return record(s, hipErrorInvalidValue);
  *d = s.current;
  return record(s, hipSuccess);
}

hipError_t hipGetDeviceProperties(hipDeviceProp_t* props, int ordinal) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  const vgpu::DeviceProfile& p = s.rt->device(ordinal).profile();
  *props = {};
  std::snprintf(props->name, sizeof props->name, "%s", p.model.c_str());
  std::snprintf(props->gcnArchName, sizeof props->gcnArchName, "%s", p.gcn_arch.c_str());
  props->totalGlobalMem = static_cast<size_t>(p.vram_bytes);
  props->sharedMemPerBlock = static_cast<size_t>(p.limits.shared_mem_per_block);
  props->warpSize = static_cast<int>(p.warp_size);
  props->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
  for (int i = 0; i < 3; ++i) {
    props->maxThreadsDim[i] = static_cast<int>(p.limits.max_block_dim[i]);
    props->maxGridSize[i] = static_cast<int>(p.limits.max_grid_dim[i]);
  }
  props->clockRate = static_cast<int>(p.telemetry.sm_clock_max_mhz) * 1000;
  props->multiProcessorCount = static_cast<int>(p.limits.multiprocessors);
  props->major = p.cc_major;
  props->minor = p.cc_minor;
  return record(s, hipSuccess);
}

hipError_t hipDeviceSynchronize(void) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return record(s, ensure_runtime(s));   // every launch here has already finished
}

hipError_t hipDeviceReset(void) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  s.functions.clear();
  s.modules.clear();
  for (int i = 0; i < s.rt->device_count(); ++i) s.rt->device(i).reset();
  return record(s, hipSuccess);
}

hipError_t hipMalloc(void** ptr, size_t size) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  if (!size) {
    *ptr = nullptr;
    return record(s, hipSuccess);
  }
  try {
    *ptr = reinterpret_cast<void*>(d->memory().alloc(size));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorOutOfMemory, e.what()));
  }
  return record(s, hipSuccess);
}

hipError_t hipFree(void* ptr) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!ptr) return record(s, hipSuccess);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  try {
    d->memory().free(reinterpret_cast<uint64_t>(ptr));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidDevicePointer, e.what()));
  }
  return record(s, hipSuccess);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t bytes, hipMemcpyKind kind) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  if (!bytes) return record(s, hipSuccess);
  if (!dst || !src) return record(s, hipErrorInvalidValue);
  vgpu::MemoryManager& mem = d->memory();
  const uint64_t dst_va = reinterpret_cast<uint64_t>(dst), src_va = reinterpret_cast<uint64_t>(src);
  // hipMemcpyDefault asks the runtime to tell device memory from host memory,
  // which it does by which addresses the device owns.
  bool to_device = kind == hipMemcpyHostToDevice, from_device = kind == hipMemcpyDeviceToHost;
  if (kind == hipMemcpyDeviceToDevice) to_device = from_device = true;
  if (kind == hipMemcpyDefault) {
    to_device = mem.owns(dst_va);
    from_device = mem.owns(src_va);
  } else if (kind != hipMemcpyHostToHost && kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost &&
             kind != hipMemcpyDeviceToDevice) {
    return record(s, hipErrorInvalidMemcpyDirection);
  }
  try {
    if (to_device && from_device) {
      std::vector<uint8_t> buf(bytes);
      mem.read(src_va, buf.data(), bytes);
      mem.write(dst_va, buf.data(), bytes);
    } else if (to_device) {
      mem.write(dst_va, src, bytes);
    } else if (from_device) {
      mem.read(src_va, dst, bytes);
    } else {
      std::memcpy(dst, src, bytes);
    }
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  d->note_transfer(bytes, 0.0);
  return record(s, hipSuccess);
}

hipError_t hipMemset(void* dst, int value, size_t bytes) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  if (!bytes) return record(s, hipSuccess);
  if (!dst) return record(s, hipErrorInvalidValue);
  const uint8_t byte = static_cast<uint8_t>(value);
  try {
    d->memory().fill(reinterpret_cast<uint64_t>(dst), &byte, 1, bytes);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}

// Every launch and copy here finishes before it returns, so the asynchronous
// forms are the synchronous ones: a stream is a handle, and there is nothing
// for it to be waiting on.
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t) {
  return hipMemcpy(dst, src, bytes, kind);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t bytes, hipStream_t) {
  return hipMemset(dst, value, bytes);
}

hipError_t hipMemGetInfo(size_t* free, size_t* total) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  const uint64_t capacity = d->memory().capacity(), used = d->memory().used();
  if (total) *total = static_cast<size_t>(capacity);
  if (free) *free = static_cast<size_t>(capacity > used ? capacity - used : 0);
  return record(s, hipSuccess);
}

hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t ordinal) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!bytes) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *bytes = static_cast<size_t>(s.rt->device(ordinal).profile().vram_bytes);
  return record(s, hipSuccess);
}

hipError_t hipDeviceGetName(char* name, int len, hipDevice_t ordinal) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!name || len <= 0) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  std::snprintf(name, static_cast<size_t>(len), "%s", s.rt->device(ordinal).profile().model.c_str());
  return record(s, hipSuccess);
}

hipError_t hipDeviceGet(hipDevice_t* device_out, int ordinal) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!device_out) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  *device_out = ordinal;   // a device is its ordinal here
  return record(s, hipSuccess);
}

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!module || !image) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  // A code object's length is in its own header; the ELF says where its
  // sections end, and the last of them is where the image stops.
  const uint8_t* bytes = static_cast<const uint8_t*>(image);
  if (std::memcmp(bytes, "\x7F" "ELF", 4) != 0)
    return record(s, fail(hipErrorInvalidImage, "the image is not an ELF code object"));
  uint64_t shoff = 0;
  std::memcpy(&shoff, bytes + 0x28, 8);
  uint16_t shentsize = 0, shnum = 0;
  std::memcpy(&shentsize, bytes + 0x3A, 2);
  std::memcpy(&shnum, bytes + 0x3C, 2);
  const uint64_t size = shoff + uint64_t{shentsize} * shnum;
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  try {
    auto m = std::make_unique<Module>();
    m->object = vgpu::amd::load_code_object(std::string(reinterpret_cast<const char*>(bytes), size), "the image");
    // The module's own variables go on the device, and the code is told where
    // they are: until that is done, a kernel reaching one reads nothing.
    if (!m->object.data.empty()) {
      m->globals = d->memory().alloc(m->object.data.size());
      d->memory().write(m->globals, m->object.data.data(), m->object.data.size());
    }
    vgpu::amd::place_globals(m->object, m->globals);
    *module = reinterpret_cast<hipModule_t>(m.get());
    s.modules.push_back(std::move(m));
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidImage, e.what()));
  }
  return record(s, hipSuccess);
}

hipError_t hipModuleLoad(hipModule_t* module, const char* path) {
  if (!path) return hipErrorInvalidValue;
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail(hipErrorFileNotFound, std::string("no code object at ") + path);
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  if (bytes.empty()) return fail(hipErrorInvalidImage, std::string(path) + " is empty");
  return hipModuleLoadData(module, bytes.data());
}

hipError_t hipModuleUnload(hipModule_t module) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < s.modules.size(); ++i)
    if (reinterpret_cast<hipModule_t>(s.modules[i].get()) == module) {
      for (size_t f = s.functions.size(); f-- > 0;)
        if (s.functions[f]->module == s.modules[i].get()) s.functions.erase(s.functions.begin() + f);
      if (s.modules[i]->globals) {
        if (vgpu::runtime::Device* d = device(s)) {
          try {
            d->memory().free(s.modules[i]->globals);
          } catch (const std::exception&) {
          }
        }
      }
      s.modules.erase(s.modules.begin() + i);
      return record(s, hipSuccess);
    }
  return record(s, hipErrorNotFound);
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* name) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!function || !module || !name) return record(s, hipErrorInvalidValue);
  Module* m = reinterpret_cast<Module*>(module);
  const Kernel* k = vgpu::amd::find_kernel(m->object, name);
  if (!k) return record(s, fail(hipErrorNotFound, std::string("the code object has no kernel named ") + name));
  auto f = std::make_unique<Function>();
  f->module = m;
  f->kernel = k;
  *function = reinterpret_cast<hipFunction_t>(f.get());
  s.functions.push_back(std::move(f));
  return record(s, hipSuccess);
}

hipError_t hipModuleGetGlobal(void** dptr, size_t* bytes, hipModule_t module, const char* name) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!module || !name) return record(s, hipErrorInvalidValue);
  Module* m = reinterpret_cast<Module*>(module);
  const vgpu::amd::GlobalVar* g = vgpu::amd::find_global(m->object, name);
  if (!g) return record(s, fail(hipErrorNotFound, std::string("the module has no variable named ") + name));
  if (dptr) *dptr = reinterpret_cast<void*>(m->globals + g->offset);
  if (bytes) *bytes = static_cast<size_t>(g->size);
  return record(s, hipSuccess);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                                 hipStream_t, void** params, void** extra) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!f) return record(s, hipErrorInvalidValue);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  Function* fn = reinterpret_cast<Function*>(f);
  if (!bx || !by || !bz || !gx || !gy || !gz) return record(s, hipErrorInvalidConfiguration);
  std::vector<uint8_t> args;
  if (const hipError_t e = build_kernargs(*fn->kernel, params, extra, &args); e != hipSuccess)
    return record(s, e);

  vgpu::MemoryManager& mem = d->memory();
  uint64_t kernarg = 0;
  try {
    kernarg = mem.alloc(args.empty() ? 1 : args.size());
    if (!args.empty()) mem.write(kernarg, args.data(), args.size());
    vgpu::amd::Dispatch dispatch;
    dispatch.object = &fn->module->object;
    dispatch.kernel = fn->kernel;
    dispatch.kernarg = kernarg;
    dispatch.groups[0] = gx;
    dispatch.groups[1] = gy;
    dispatch.groups[2] = gz;
    dispatch.group_size[0] = bx;
    dispatch.group_size[1] = by;
    dispatch.group_size[2] = bz;
    dispatch.wave_size = static_cast<uint32_t>(d->profile().warp_size);
    dispatch.dynamic_lds = shared;   // what the launch adds to the kernel's own LDS
    const vgpu::amd::DispatchStats stats = vgpu::amd::execute(dispatch, mem);
    mem.free(kernarg);
    // What the device spent, as telemetry reports a kernel: the instructions
    // a wave retires, at the profile's clock.
    const uint32_t mhz = d->profile().telemetry.sm_clock_max_mhz;
    const double clock = mhz ? mhz * 1e6 : 1e9;
    d->note_busy(static_cast<double>(stats.instructions) / clock);
  } catch (const std::exception& e) {
    if (kernarg) {
      try {
        mem.free(kernarg);
      } catch (const std::exception&) {
      }
    }
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  return record(s, hipSuccess);
}

const char* hipGetErrorName(hipError_t e) {
  switch (e) {
    case hipSuccess: return "hipSuccess";
    case hipErrorInvalidValue: return "hipErrorInvalidValue";
    case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
    case hipErrorNotInitialized: return "hipErrorNotInitialized";
    case hipErrorDeinitialized: return "hipErrorDeinitialized";
    case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";
    case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
    case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
    case hipErrorInvalidImage: return "hipErrorInvalidImage";
    case hipErrorInvalidContext: return "hipErrorInvalidContext";
    case hipErrorFileNotFound: return "hipErrorFileNotFound";
    case hipErrorNotFound: return "hipErrorNotFound";
    case hipErrorNotSupported: return "hipErrorNotSupported";
    case hipErrorUnknown: break;
  }
  return "hipErrorUnknown";
}

const char* hipGetErrorString(hipError_t e) {
  switch (e) {
    case hipSuccess: return "no error";
    case hipErrorInvalidValue: return "invalid argument";
    case hipErrorOutOfMemory: return "out of memory";
    case hipErrorNotInitialized: return "invalid device ordinal";
    case hipErrorDeinitialized: return "driver shutting down";
    case hipErrorInvalidConfiguration: return "invalid configuration argument";
    case hipErrorInvalidSymbol: return "invalid device symbol";
    case hipErrorInvalidDevicePointer: return "invalid device pointer";
    case hipErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
    case hipErrorInvalidDevice: return "invalid device ordinal";
    case hipErrorInvalidImage: return "invalid device function image";
    case hipErrorInvalidContext: return "invalid device context";
    case hipErrorFileNotFound: return "file not found";
    case hipErrorNotFound: return "named symbol not found";
    case hipErrorNotSupported: return "operation not supported";
    case hipErrorUnknown: break;
  }
  return "unknown error";
}

hipError_t hipGetLastError(void) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const hipError_t e = s.last;
  s.last = hipSuccess;
  return e;
}

hipError_t hipPeekAtLastError(void) { return state().last; }

int hipGetStreamDeviceId(hipStream_t) { return state().current; }

// Streams: every launch here finishes before it returns, so a stream is a
// handle and nothing more. The null stream is what a program gets by default.
hipError_t hipStreamCreate(hipStream_t* stream) {
  if (!stream) return hipErrorInvalidValue;
  static int next = 1;
  *stream = reinterpret_cast<hipStream_t>(static_cast<intptr_t>(next++));
  return hipSuccess;
}
hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  // Neither flag changes anything here: there is no other work for a stream
  // to be blocking on, and nothing to synchronise against.
  if (flags & ~static_cast<unsigned>(hipStreamNonBlocking)) return hipErrorInvalidValue;
  return hipStreamCreate(stream);
}
hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }

// Events: a program records one before its work and one after, and asks how
// long there was between them. Every launch here has finished by the time it
// returns, so an event is recorded the moment the call is made and the time
// between two of them is the time the simulator took -- not what a card would
// have taken, which this does not claim to know.
namespace {

struct Event {
  bool timing = true;
  bool recorded = false;
  std::chrono::steady_clock::time_point when{};
};

std::mutex g_event_mutex;
std::map<hipEvent_t, std::unique_ptr<Event>> g_events;

Event* find_event(hipEvent_t e) {   // the caller holds g_event_mutex
  const auto it = g_events.find(e);
  return it == g_events.end() ? nullptr : it->second.get();
}

}  // namespace

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags) {
  if (!event) return hipErrorInvalidValue;
  if (flags & ~static_cast<unsigned>(hipEventBlockingSync | hipEventDisableTiming)) return hipErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_event_mutex);
  static intptr_t next = 1;
  const hipEvent_t handle = reinterpret_cast<hipEvent_t>(next++);
  auto e = std::make_unique<Event>();
  e->timing = (flags & hipEventDisableTiming) == 0;
  g_events.emplace(handle, std::move(e));
  *event = handle;
  return hipSuccess;
}

hipError_t hipEventCreate(hipEvent_t* event) { return hipEventCreateWithFlags(event, hipEventDefault); }

hipError_t hipEventDestroy(hipEvent_t event) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  return g_events.erase(event) ? hipSuccess : hipErrorInvalidHandle;
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  Event* e = find_event(event);
  if (!e) return hipErrorInvalidHandle;
  e->recorded = true;
  e->when = std::chrono::steady_clock::now();
  return hipSuccess;
}

// There is never work left behind an event, so both of these answer at once.
hipError_t hipEventSynchronize(hipEvent_t event) {
  std::lock_guard<std::mutex> lock(g_event_mutex);
  return find_event(event) ? hipSuccess : hipErrorInvalidHandle;
}

hipError_t hipEventQuery(hipEvent_t event) { return hipEventSynchronize(event); }

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end) {
  if (!ms) return hipErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_event_mutex);
  const Event* a = find_event(start);
  const Event* b = find_event(end);
  // An event that was never recorded, or one created without timing, has no
  // time to give.
  if (!a || !b || !a->recorded || !b->recorded || !a->timing || !b->timing) return hipErrorInvalidHandle;
  *ms = std::chrono::duration<float, std::milli>(b->when - a->when).count();
  return hipSuccess;
}

// The versions a program checks before it trusts a feature. HIP reports
// these as major * 10000000 + minor * 100000 + patch.
hipError_t hipRuntimeGetVersion(int* version) {
  if (!version) return hipErrorInvalidValue;
  *version = 60443483;   // 6.4.43483, a ROCm 6.4 runtime
  return hipSuccess;
}
hipError_t hipDriverGetVersion(int* version) { return hipRuntimeGetVersion(version); }

}  // extern "C"
