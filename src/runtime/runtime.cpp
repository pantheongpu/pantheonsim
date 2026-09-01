#include "vgpu/runtime/runtime.hpp"

#include <chrono>
#include <cstdio>

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
  std::snprintf(d->name, sizeof d->name, "%s", p.model.c_str());
  std::snprintf(d->architecture, sizeof d->architecture, "%s", p.architecture.c_str());
  std::snprintf(d->vendor, sizeof d->vendor, "%s", p.vendor.c_str());
  // Deterministic synthetic UUID/bus id, stable for a given profile+ordinal.
  uint32_t h = 2166136261u;
  for (char c : p.id) h = (h ^ static_cast<unsigned char>(c)) * 16777619u;
  std::snprintf(d->uuid, sizeof d->uuid, "GPU-%08x-%04x-%04x-%04x-%08x%04x", h, (h >> 16) & 0xFFFF,
                0x4000 | (h & 0x0FFF), 0x8000 | ((h >> 4) & 0x3FFF), h * 2654435761u,
                static_cast<unsigned>(ordinal));
  // Each virtual device gets its own PCI slot on a synthetic bus.
  std::snprintf(d->bus_id, sizeof d->bus_id, "00000000:%02X:00.0", ordinal + 1);
  d->pci_device_id = (p.telemetry.pci_device_id << 16) | p.telemetry.pci_vendor_id;
  d->pci_subsystem_id = d->pci_device_id;
  d->cc_major = p.cc_major;
  d->cc_minor = p.cc_minor;
  d->multiprocessors = p.limits.multiprocessors;
  d->vram_total_bytes = p.vram_bytes;
  d->vram_used_bytes = 0;
  d->power_limit_mw = p.telemetry.power_limit_w * 1000;
  d->temperature_max_c = p.telemetry.temperature_max_c;
  d->sm_clock_max_mhz = p.telemetry.sm_clock_max_mhz;
  d->mem_clock_max_mhz = p.telemetry.mem_clock_max_mhz;
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
