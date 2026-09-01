#include "vgpu/runtime/runtime.hpp"

#include "vgpu/error.hpp"
#include "vgpu/ptx/parser.hpp"

namespace vgpu::runtime {

uint64_t Device::load_module(const std::string& ptx_src) {
  try {
    auto mod = std::make_shared<ptx::Module>(ptx::parse(ptx_src));
    LoadedModule lm;
    lm.id = next_module_id_++;
    // Materialize module .global variables into device memory.
    for (const auto& g : mod->globals) {
      uint64_t va = mem_.alloc(g.size);
      if (!g.init.empty()) mem_.write(va, g.init.data(), g.init.size());
      lm.symbols[g.name] = va;
      lm.global_vas.push_back(va);
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
  exec::launch(fn, cfg, args, mem_, profile_, syms);
}

Runtime::Runtime(const DeviceProfile& profile, int device_count) {
  if (device_count < 1) throw Error::make(Err::InvalidValue, "device_count must be >= 1");
  for (int i = 0; i < device_count; ++i) devices_.push_back(std::make_unique<Device>(profile, i));
}

Device& Runtime::device(int ordinal) {
  if (ordinal < 0 || ordinal >= device_count())
    throw Error::make(Err::InvalidValue, "invalid device ordinal ", ordinal, " (have ", device_count(),
                      " virtual device", device_count() == 1 ? "" : "s", ")");
  return *devices_[ordinal];
}

}  // namespace vgpu::runtime
