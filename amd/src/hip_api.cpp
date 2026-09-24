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
#include <set>
#include <mutex>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vgpu/hip_abi.hpp"
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

// ---- Programs built by hipcc ---------------------------------------------
//
// A program built by hipcc never loads a module. Its device code is inside the
// executable, and before main it hands that to the runtime and says which of
// its host-side functions stands for which kernel; a chevron launch then hands
// the runtime the host-side function. The device code is a clang offload
// bundle -- one entry per target -- and a device takes the entry for its own
// gfx target, loaded and placed the first time a kernel runs on it, since each
// device has memory of its own for the module's variables.
struct FatBinary {
  std::map<std::string, std::string> targets;   // "gfx942" -> its code object
  std::map<int, std::unique_ptr<Module>> on_device;
};

struct HostFunction {
  FatBinary* binary = nullptr;
  std::string kernel;
};

// A kernel launch a graph holds: what to run and with what, copied when it was
// captured, since a graph replays what the stream was asked to do then.
struct Node {
  int device = 0;
  const void* host_function = nullptr;
  vgpu::amd::abi::Dim3 grid{1, 1, 1}, block{1, 1, 1};
  uint32_t shared = 0;
  std::vector<uint8_t> args;
};
struct Graph {
  std::vector<Node> nodes;
};

struct State {
  std::mutex mutex;
  std::unique_ptr<vgpu::runtime::Runtime> rt;
  std::vector<std::unique_ptr<Module>> modules;
  std::vector<std::unique_ptr<Function>> functions;
  std::vector<std::unique_ptr<FatBinary>> fat_binaries;
  std::map<const void*, HostFunction> host_functions;
  std::set<std::pair<int, int>> peers;           // (device, peer) pairs with access enabled
  std::map<hipStream_t, Graph> capturing;        // streams recording rather than running
  std::vector<std::unique_ptr<Graph>> graphs, graph_execs;
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

// Runs one kernel on one device: the arguments go into device memory as its
// kernarg segment, and the dispatch runs to completion before this returns.
hipError_t dispatch_kernel(State& s, vgpu::runtime::Device& d, const CodeObject& object, const Kernel& kernel,
                           vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, uint32_t shared,
                           const std::vector<uint8_t>& args) {
  if (!block.x || !block.y || !block.z || !grid.x || !grid.y || !grid.z) return hipErrorInvalidConfiguration;
  vgpu::MemoryManager& mem = d.memory();
  uint64_t kernarg = 0;
  try {
    kernarg = mem.alloc(args.empty() ? 1 : args.size());
    if (!args.empty()) mem.write(kernarg, args.data(), args.size());
    vgpu::amd::Dispatch dispatch;
    dispatch.object = &object;
    dispatch.kernel = &kernel;
    dispatch.kernarg = kernarg;
    dispatch.groups[0] = grid.x;
    dispatch.groups[1] = grid.y;
    dispatch.groups[2] = grid.z;
    dispatch.group_size[0] = block.x;
    dispatch.group_size[1] = block.y;
    dispatch.group_size[2] = block.z;
    dispatch.wave_size = static_cast<uint32_t>(d.profile().warp_size);
    dispatch.dynamic_lds = shared;   // what the launch adds to the kernel's own LDS
    const vgpu::amd::DispatchStats stats = vgpu::amd::execute(dispatch, mem);
    mem.free(kernarg);
    // What the device spent, as telemetry reports a kernel: the instructions
    // a wave retires, at the profile's clock.
    const uint32_t mhz = d.profile().telemetry.sm_clock_max_mhz;
    const double clock = mhz ? mhz * 1e6 : 1e9;
    d.note_busy(static_cast<double>(stats.instructions) / clock);
  } catch (const std::exception& e) {
    if (kernarg) {
      try {
        mem.free(kernarg);
      } catch (const std::exception&) {
      }
    }
    return fail(hipErrorLaunchFailure, s.profile_id + ": " + e.what());
  }
  return hipSuccess;
}

// The module a registered binary becomes on one device: the code object for
// that device's gfx target, loaded, with its variables placed in the device's
// own memory. A program built for other targets only is told so by name.
hipError_t module_on(State& s, FatBinary& fb, int ordinal, Module** out) {
  if (auto it = fb.on_device.find(ordinal); it != fb.on_device.end()) {
    *out = it->second.get();
    return hipSuccess;
  }
  vgpu::runtime::Device& d = s.rt->device(ordinal);
  const std::string& gfx = d.profile().gcn_arch;
  const auto target = fb.targets.find(gfx);
  if (target == fb.targets.end()) {
    std::string built;
    for (const auto& t : fb.targets) built += (built.empty() ? "" : ", ") + t.first;
    return fail(hipErrorNoBinaryForGpu, "the program carries device code for " + (built.empty() ? "no GPU" : built) +
                                            ", and this device is " + gfx + ". Build it with --offload-arch=" + gfx +
                                            ".");
  }
  try {
    auto m = std::make_unique<Module>();
    m->object = vgpu::amd::load_code_object(target->second, "the program's " + gfx + " code");
    if (!m->object.data.empty()) {
      m->globals = d.memory().alloc(m->object.data.size());
      d.memory().write(m->globals, m->object.data.data(), m->object.data.size());
    }
    vgpu::amd::place_globals(m->object, m->globals);
    *out = m.get();
    fb.on_device.emplace(ordinal, std::move(m));
  } catch (const std::exception& e) {
    return fail(hipErrorInvalidImage, e.what());
  }
  return hipSuccess;
}

// Reads the targets out of a clang offload bundle: a magic string, a count,
// and for each entry where its bytes are and the target it is for.
bool read_bundle(const uint8_t* b, std::map<std::string, std::string>* out) {
  static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  if (std::memcmp(b, kMagic, 24) != 0) return false;
  uint64_t count = 0;
  std::memcpy(&count, b + 24, 8);
  uint64_t at = 32;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t offset = 0, size = 0, triple_len = 0;
    std::memcpy(&offset, b + at, 8);
    std::memcpy(&size, b + at + 8, 8);
    std::memcpy(&triple_len, b + at + 16, 8);
    const std::string triple(reinterpret_cast<const char*>(b + at + 24), triple_len);
    at += 24 + triple_len;
    // "hipv4-amdgcn-amd-amdhsa--gfx942", perhaps with ":sramecc+:xnack-" after
    // it: the processor is what follows the last "--", up to any feature.
    const size_t dashes = triple.rfind("--");
    if (triple.rfind("hip", 0) != 0 || dashes == std::string::npos || !size) continue;
    std::string gfx = triple.substr(dashes + 2);
    if (const size_t colon = gfx.find(':'); colon != std::string::npos) gfx.resize(colon);
    (*out)[gfx] = std::string(reinterpret_cast<const char*>(b + offset), size);
  }
  return true;
}

// Each thread's pending chevron launch: hipcc pushes the configuration, then
// calls the kernel's host-side function, which pops it and launches.
struct CallConfiguration {
  vgpu::amd::abi::Dim3 grid, block;
  size_t shared;
  hipStream_t stream;
};
thread_local std::vector<CallConfiguration> g_call_configurations;

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

  return record(s, dispatch_kernel(s, *d, fn->module->object, *fn->kernel, {gx, gy, gz}, {bx, by, bz}, shared, args));
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
    case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction";
    case hipErrorNoBinaryForGpu: return "hipErrorNoBinaryForGpu";
    case hipErrorInvalidHandle: return "hipErrorInvalidHandle";
    case hipErrorIllegalState: return "hipErrorIllegalState";
    case hipErrorNotReady: return "hipErrorNotReady";
    case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";
    case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
    case hipErrorStreamCaptureUnsupported: return "hipErrorStreamCaptureUnsupported";
    case hipErrorStreamCaptureUnmatched: return "hipErrorStreamCaptureUnmatched";
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
    case hipErrorInvalidDeviceFunction: return "invalid device function";
    case hipErrorNoBinaryForGpu: return "no kernel image is available for execution on the device";
    case hipErrorInvalidHandle: return "invalid resource handle";
    case hipErrorIllegalState: return "the operation cannot be performed in the present state";
    case hipErrorNotReady: return "device not ready";
    case hipErrorPeerAccessAlreadyEnabled: return "peer access is already enabled";
    case hipErrorPeerAccessNotEnabled: return "peer access has not been enabled";
    case hipErrorLaunchFailure: return "unspecified launch failure";
    case hipErrorStreamCaptureUnsupported: return "operation not permitted when stream is capturing";
    case hipErrorStreamCaptureUnmatched: return "the capture was not initiated in this stream";
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
// ---- The launch ABI hipcc compiles a program against ------------------------

// Before main: the program's device code. The handle it is given back is what
// it names the binary by when it registers functions and when it exits.
void** __hipRegisterFatBinary(const void* data) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto fb = std::make_unique<FatBinary>();
  const auto* wrapper = static_cast<const vgpu::amd::abi::FatbinWrapper*>(data);
  if (!wrapper || wrapper->magic != vgpu::amd::abi::kFatbinMagic || !wrapper->binary ||
      !read_bundle(static_cast<const uint8_t*>(wrapper->binary), &fb->targets))
    fail(hipErrorInvalidImage, "the program's device code is not a clang offload bundle this can read");
  void** handle = reinterpret_cast<void**>(fb.get());
  s.fat_binaries.push_back(std::move(fb));
  return handle;
}

// Before main, once per kernel: which host-side function stands for which
// kernel in which binary.
void __hipRegisterFunction(void** modules, const void* host_function, char*, const char* device_name, unsigned int,
                           void*, void*, void*, void*, int*) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!modules || !host_function || !device_name) return;
  s.host_functions[host_function] = HostFunction{reinterpret_cast<FatBinary*>(modules), device_name};
}

// At exit: the binary's modules go, and their variables with them.
void __hipUnregisterFatBinary(void** modules) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  FatBinary* fb = reinterpret_cast<FatBinary*>(modules);
  for (auto it = s.host_functions.begin(); it != s.host_functions.end();)
    it = it->second.binary == fb ? s.host_functions.erase(it) : std::next(it);
  for (size_t i = 0; i < s.fat_binaries.size(); ++i)
    if (s.fat_binaries[i].get() == fb) {
      for (auto& [ordinal, m] : fb->on_device)
        if (m->globals && s.rt && ordinal < s.rt->device_count()) {
          try {
            s.rt->device(ordinal).memory().free(m->globals);
          } catch (const std::exception&) {
          }
        }
      s.fat_binaries.erase(s.fat_binaries.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
}

hipError_t __hipPushCallConfiguration(vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block, size_t shared,
                                      hipStream_t stream) {
  g_call_configurations.push_back({grid, block, shared, stream});
  return hipSuccess;
}

hipError_t __hipPopCallConfiguration(vgpu::amd::abi::Dim3* grid, vgpu::amd::abi::Dim3* block, size_t* shared,
                                     hipStream_t* stream) {
  if (g_call_configurations.empty()) return hipErrorInvalidConfiguration;
  const CallConfiguration c = g_call_configurations.back();
  g_call_configurations.pop_back();
  if (grid) *grid = c.grid;
  if (block) *block = c.block;
  if (shared) *shared = c.shared;
  if (stream) *stream = c.stream;
  return hipSuccess;
}

// A chevron launch, and hipLaunchKernel called by hand: the kernel is named by
// its host-side function, and each argument by a pointer to its value. On a
// stream that is capturing, the launch is recorded rather than run.
hipError_t hipLaunchKernel(const void* host_function, vgpu::amd::abi::Dim3 grid, vgpu::amd::abi::Dim3 block,
                           void** args, size_t shared, hipStream_t stream) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  vgpu::runtime::Device* d = device(s);
  if (!d) return record(s, hipErrorInvalidDevice);
  const auto hf = s.host_functions.find(host_function);
  if (hf == s.host_functions.end())
    return record(s, fail(hipErrorInvalidDeviceFunction, "no kernel was registered for that function"));
  Module* m = nullptr;
  if (const hipError_t e = module_on(s, *hf->second.binary, s.current, &m); e != hipSuccess) return record(s, e);
  const Kernel* k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
  if (!k)
    return record(s, fail(hipErrorInvalidDeviceFunction, "the program's device code has no kernel named " +
                                                             hf->second.kernel));
  std::vector<uint8_t> packed;
  if (const hipError_t e = build_kernargs(*k, args, nullptr, &packed); e != hipSuccess) return record(s, e);
  if (auto cap = s.capturing.find(stream); stream && cap != s.capturing.end()) {
    cap->second.nodes.push_back(Node{s.current, host_function, grid, block, static_cast<uint32_t>(shared), packed});
    return record(s, hipSuccess);
  }
  return record(s, dispatch_kernel(s, *d, m->object, *k, grid, block, static_cast<uint32_t>(shared), packed));
}

// ---- What the device is, in the layout the HIP headers give it ---------------

hipError_t hipGetDevicePropertiesR0600(vgpu::amd::abi::DevicePropR0600* props, int ordinal) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!props) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (ordinal < 0 || ordinal >= s.rt->device_count()) return record(s, hipErrorInvalidDevice);
  const vgpu::DeviceProfile& p = s.rt->device(ordinal).profile();
  std::memset(props, 0, sizeof *props);
  std::snprintf(props->name, sizeof props->name, "%s", p.model.c_str());
  std::snprintf(props->gcnArchName, sizeof props->gcnArchName, "%s", p.gcn_arch.c_str());
  props->totalGlobalMem = static_cast<size_t>(p.vram_bytes);
  props->sharedMemPerBlock = static_cast<size_t>(p.limits.shared_mem_per_block);
  props->sharedMemPerBlockOptin = static_cast<size_t>(p.limits.shared_mem_per_block_optin);
  props->sharedMemPerMultiprocessor = static_cast<size_t>(p.limits.shared_mem_per_sm);
  props->maxSharedMemoryPerMultiProcessor = static_cast<size_t>(p.limits.shared_mem_per_sm);
  props->regsPerBlock = static_cast<int>(p.limits.registers_per_block);
  props->regsPerMultiprocessor = static_cast<int>(p.limits.registers_per_sm);
  props->warpSize = static_cast<int>(p.warp_size);
  props->maxThreadsPerBlock = static_cast<int>(p.limits.max_threads_per_block);
  for (int i = 0; i < 3; ++i) {
    props->maxThreadsDim[i] = static_cast<int>(p.limits.max_block_dim[i]);
    props->maxGridSize[i] = static_cast<int>(p.limits.max_grid_dim[i]);
  }
  props->clockRate = static_cast<int>(p.telemetry.sm_clock_max_mhz) * 1000;
  props->memoryClockRate = static_cast<int>(p.telemetry.mem_clock_max_mhz) * 1000;
  props->multiProcessorCount = static_cast<int>(p.limits.multiprocessors);
  props->maxThreadsPerMultiProcessor = static_cast<int>(p.limits.max_threads_per_sm);
  props->maxBlocksPerMultiProcessor = static_cast<int>(p.limits.max_blocks_per_sm);
  props->l2CacheSize = static_cast<int>(p.limits.l2_cache_bytes);
  props->major = p.cc_major;
  props->minor = p.cc_minor;
  props->pciDeviceID = ordinal;
  props->concurrentKernels = 1;
  props->unifiedAddressing = 1;
  props->ECCEnabled = p.telemetry.ecc ? 1 : 0;
  return record(s, hipSuccess);
}

// Nothing here waits on anything, so how a program would like to wait changes
// nothing; the flags it may ask for are accepted and any other is refused.
hipError_t hipSetDeviceFlags(unsigned int flags) {
  const unsigned int known = 0x7 /* schedule */ | 0x8 /* map host */ | 0x10 /* lmem resize */;
  return record(state(), (flags & ~known) ? hipErrorInvalidValue : hipSuccess);
}

// ---- Host memory -------------------------------------------------------------
//
// On a card this is pinned host memory the GPU can also reach. Here a copy
// reaches it like any host memory, and a kernel cannot: a kernel's addresses
// are the device's own. The workloads use it to stage copies, which is what
// this supports.
hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int) {
  if (!ptr) return hipErrorInvalidValue;
  *ptr = size ? std::aligned_alloc(4096, (size + 4095) / 4096 * 4096) : nullptr;
  return (size && !*ptr) ? hipErrorOutOfMemory : hipSuccess;
}

hipError_t hipHostFree(void* ptr) {
  std::free(ptr);
  return hipSuccess;
}

// ---- One device reaching another --------------------------------------------

hipError_t hipDeviceCanAccessPeer(int* can, int ordinal, int peer) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!can) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int n = s.rt->device_count();
  if (ordinal < 0 || ordinal >= n || peer < 0 || peer >= n) return record(s, hipErrorInvalidDevice);
  *can = ordinal != peer;   // every device here can reach every other; none reaches itself as a peer
  return record(s, hipSuccess);
}

hipError_t hipDeviceEnablePeerAccess(int peer, unsigned int flags) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (flags) return record(s, hipErrorInvalidValue);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  if (peer < 0 || peer >= s.rt->device_count() || peer == s.current) return record(s, hipErrorInvalidDevice);
  if (!s.peers.insert({s.current, peer}).second) return record(s, hipErrorPeerAccessAlreadyEnabled);
  return record(s, hipSuccess);
}

// A copy from one device's memory to another's. The address says which device
// owns it, and the device numbers the program gave have to agree.
hipError_t hipMemcpyPeerAsync(void* dst, int dst_device, const void* src, int src_device, size_t bytes,
                              hipStream_t) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (const hipError_t e = ensure_runtime(s); e != hipSuccess) return record(s, e);
  const int n = s.rt->device_count();
  if (dst_device < 0 || dst_device >= n || src_device < 0 || src_device >= n) return record(s, hipErrorInvalidDevice);
  if (!bytes) return record(s, hipSuccess);
  if (!dst || !src) return record(s, hipErrorInvalidValue);
  vgpu::MemoryManager& to = s.rt->device(dst_device).memory();
  vgpu::MemoryManager& from = s.rt->device(src_device).memory();
  const uint64_t dst_va = reinterpret_cast<uint64_t>(dst), src_va = reinterpret_cast<uint64_t>(src);
  if (!to.owns(dst_va) || !from.owns(src_va))
    return record(s, fail(hipErrorInvalidValue, "a peer copy's addresses are not on the devices it names"));
  try {
    std::vector<uint8_t> buf(bytes);
    from.read(src_va, buf.data(), bytes);
    to.write(dst_va, buf.data(), bytes);
  } catch (const std::exception& e) {
    return record(s, fail(hipErrorInvalidValue, e.what()));
  }
  s.rt->device(src_device).note_transfer(bytes, 0.0);
  return record(s, hipSuccess);
}

// ---- Graphs a stream is recorded into ----------------------------------------
//
// Between beginning and ending a capture, kernels launched on the stream are
// recorded rather than run. The graph that comes out can be instantiated and
// launched, which runs what was recorded, in order, with the arguments it was
// recorded with.
hipError_t hipStreamBeginCapture(hipStream_t stream, int) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!stream) return record(s, fail(hipErrorStreamCaptureUnsupported, "the null stream cannot be captured"));
  if (!s.capturing.emplace(stream, Graph{}).second) return record(s, hipErrorIllegalState);
  return record(s, hipSuccess);
}

hipError_t hipStreamEndCapture(hipStream_t stream, void** graph) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  const auto it = s.capturing.find(stream);
  if (it == s.capturing.end()) return record(s, hipErrorStreamCaptureUnmatched);
  if (!graph) return record(s, hipErrorInvalidValue);
  s.graphs.push_back(std::make_unique<Graph>(std::move(it->second)));
  s.capturing.erase(it);
  *graph = s.graphs.back().get();
  return record(s, hipSuccess);
}

hipError_t hipGraphInstantiate(void** exec, void* graph, void*, char*, size_t) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!exec || !graph) return record(s, hipErrorInvalidValue);
  s.graph_execs.push_back(std::make_unique<Graph>(*static_cast<Graph*>(graph)));
  *exec = s.graph_execs.back().get();
  return record(s, hipSuccess);
}

hipError_t hipGraphLaunch(void* exec, hipStream_t) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (!exec) return record(s, hipErrorInvalidValue);
  for (const Node& node : static_cast<Graph*>(exec)->nodes) {
    const auto hf = s.host_functions.find(node.host_function);
    if (hf == s.host_functions.end()) return record(s, hipErrorInvalidDeviceFunction);
    Module* m = nullptr;
    if (const hipError_t e = module_on(s, *hf->second.binary, node.device, &m); e != hipSuccess) return record(s, e);
    const Kernel* k = vgpu::amd::find_kernel(m->object, hf->second.kernel);
    if (!k) return record(s, hipErrorInvalidDeviceFunction);
    if (const hipError_t e = dispatch_kernel(s, s.rt->device(node.device), m->object, *k, node.grid, node.block,
                                             node.shared, node.args);
        e != hipSuccess)
      return record(s, e);
  }
  return record(s, hipSuccess);
}

hipError_t hipGraphDestroy(void* graph) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < s.graphs.size(); ++i)
    if (s.graphs[i].get() == graph) {
      s.graphs.erase(s.graphs.begin() + static_cast<std::ptrdiff_t>(i));
      return record(s, hipSuccess);
    }
  return record(s, hipErrorInvalidValue);
}

hipError_t hipGraphExecDestroy(void* exec) {
  State& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (size_t i = 0; i < s.graph_execs.size(); ++i)
    if (s.graph_execs[i].get() == exec) {
      s.graph_execs.erase(s.graph_execs.begin() + static_cast<std::ptrdiff_t>(i));
      return record(s, hipSuccess);
    }
  return record(s, hipErrorInvalidValue);
}

hipError_t hipRuntimeGetVersion(int* version) {
  if (!version) return hipErrorInvalidValue;
  *version = 60443483;   // 6.4.43483, a ROCm 6.4 runtime
  return hipSuccess;
}
hipError_t hipDriverGetVersion(int* version) { return hipRuntimeGetVersion(version); }

}  // extern "C"
