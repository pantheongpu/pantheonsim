#include "vgpu/runtime/runtime.hpp"

#include <chrono>
#include <cstdio>

#include "vgpu/error.hpp"
#include "vgpu/ptx/parser.hpp"

namespace vgpu::runtime {

namespace {

// ".target sm_86" -> 86. Returns 0 for anything that is not a plain sm_NN,
// which is treated as "no opinion" rather than as an error.
int target_arch(const std::string& target) {
  const size_t at = target.find("sm_");
  if (at == std::string::npos) return 0;
  int v = 0;
  for (size_t i = at + 3; i < target.size() && target[i] >= '0' && target[i] <= '9'; ++i)
    v = v * 10 + (target[i] - '0');
  return v;
}

}  // namespace

uint64_t Device::load_module(const std::string& ptx_src) {
  try {
    auto mod = std::make_shared<ptx::Module>(ptx::parse(ptx_src));
    // PTX is forward compatible but not backward: a module built for a newer
    // architecture than the device is rejected by the real driver with
    // CUDA_ERROR_INVALID_PTX, and a simulator that loaded it anyway would let
    // a program pass here and fail on the hardware it is standing in for.
    const int want = target_arch(mod->target);
    const int have = profile_.cc_major * 10 + profile_.cc_minor;
    if (want && have && want > have)
      throw Error::make(Err::PtxParse, "module targets ", mod->target,
                        " but this device is compute capability ", profile_.cc_major, ".",
                        profile_.cc_minor,
                        "; PTX runs on newer architectures, not older ones");
    LoadedModule lm;
    lm.id = next_module_id_++;
    // Materialize module .global variables into device memory.
    for (const auto& g : mod->globals) {
      uint64_t va = mem_.alloc(g.size);
      if (!g.init.empty()) mem_.write(va, g.init.data(), g.init.size());
      lm.symbols[g.name] = va;
      lm.global_vas.push_back(va);
    }
    // Second pass: a global initialised with another symbol's address can only
    // be filled in once every global has one. A symbol that names a kernel
    // rather than a variable has no address in this model and stays zero --
    // taking a kernel's address is a host-side operation, and PTX that only
    // uses it to carry a mangled name (which is what NVRTC's name expressions
    // compile to) never dereferences it.
    for (const auto& g : mod->globals) {
      if (g.init_symbol.empty()) continue;
      auto it = lm.symbols.find(g.init_symbol);
      const uint64_t target = it == lm.symbols.end() ? 0 : it->second;
      const uint64_t slot = lm.symbols[g.name];
      const uint64_t bytes = g.size < sizeof(uint64_t) ? g.size : sizeof(uint64_t);
      mem_.write(slot, &target, bytes);
    }
    lm.mod = std::move(mod);
    uint64_t id = lm.id;
    modules_.push_back(std::move(lm));
    return id;
  } catch (const Error& e) {
    if (e.code() == Err::UnsupportedPtx)
      throw Error::make(e.code(), e.message(), "\n  GPU profile: ", profile_.id);
    throw;
  }
}

void Device::unload_module(uint64_t module_id) {
  for (auto it = modules_.begin(); it != modules_.end(); ++it) {
    if (it->id == module_id) {
      for (uint64_t va : it->global_vas) mem_.free(va);
      modules_.erase(it);
      return;
    }
  }
  throw Error::make(Err::NotFound, "module handle ", module_id, " is not loaded on device ", ordinal_);
}

const ptx::EntryFn* Device::get_function(uint64_t module_id, const std::string& name) const {
  for (const auto& lm : modules_) {
    if (lm.id != module_id) continue;
    const ptx::EntryFn* fn = lm.mod->find_entry(name);
    if (!fn) {
      std::string names;
      for (const auto& e : lm.mod->entries) names += "\n  " + e.name;
      throw Error::make(Err::NotFound, "no kernel named '", name, "' in module. Kernels present:", names);
    }
    return fn;
  }
  throw Error::make(Err::NotFound, "module handle ", module_id, " is not loaded on device ", ordinal_);
}

const exec::SymbolTable* Device::symbols(uint64_t module_id) const {
  for (const auto& lm : modules_)
    if (lm.id == module_id) return &lm.symbols;
  throw Error::make(Err::NotFound, "module handle ", module_id, " is not loaded on device ", ordinal_);
}

void Device::launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg,
                    const std::vector<std::vector<uint8_t>>& args, const exec::SymbolTable* syms) {
  if (!telemetry_) {
    exec::launch(fn, cfg, args, mem_, profile_, syms);
    return;
  }
  // Utilization is the real fraction of wall time spent executing kernels. The
  // progress hook publishes it *during* the launch so a long kernel still shows
  // live telemetry rather than freezing until it returns.
  uint32_t ord = static_cast<uint32_t>(ordinal_);
  telemetry::Publisher* pub = telemetry_;
  auto start = std::chrono::steady_clock::now();
  exec::launch(fn, cfg, args, mem_, profile_, syms,
               [pub, ord](double dt) { pub->note_kernel(ord, dt); });
  double busy = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  pub->note_kernel(ord, 0.0);
  (void)busy;
}

void Runtime::publish_identity(const DeviceProfile& p, int ordinal) {
  telemetry::DeviceSample* d = telemetry_.device(static_cast<uint32_t>(ordinal));
  if (!d) return;
  telemetry::describe_device(p, ordinal, d);
  telemetry_.refresh(static_cast<uint32_t>(ordinal));
}


Runtime::Runtime(const DeviceProfile& profile, int device_count) {
  if (device_count < 1) throw Error::make(Err::InvalidValue, "device_count must be >= 1");
  if (device_count > telemetry::kMaxDevices)
    throw Error::make(Err::InvalidValue, "device_count must be <= ", telemetry::kMaxDevices);
  telemetry_.set_device_count(static_cast<uint32_t>(device_count));
  for (int i = 0; i < device_count; ++i) {
    devices_.push_back(std::make_unique<Device>(profile, i, telemetry_.active() ? &telemetry_ : nullptr));
    publish_identity(profile, i);
  }
}

Device& Runtime::device(int ordinal) {
  if (ordinal < 0 || ordinal >= device_count())
    throw Error::make(Err::InvalidValue, "invalid device ordinal ", ordinal, " (have ", device_count(),
                      " virtual device", device_count() == 1 ? "" : "s", ")");
  return *devices_[ordinal];
}

}  // namespace vgpu::runtime
