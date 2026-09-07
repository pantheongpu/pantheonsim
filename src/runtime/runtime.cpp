#include "vgpu/runtime/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

int Device::module_arch(uint64_t module_id) const {
  for (const auto& lm : modules_) {
    if (lm.id != module_id) continue;
    // ".target sm_86" -- and it may carry more, as in ".target sm_86, debug".
    const std::string& t = lm.mod->target;
    const size_t at = t.find("sm_");
    if (at == std::string::npos) return 0;
    int v = 0;
    for (size_t i = at + 3; i < t.size() && t[i] >= '0' && t[i] <= '9'; ++i) v = v * 10 + (t[i] - '0');
    return v;
  }
  return 0;
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

namespace {

// VGPU_COUNTERS=1 prints the per-launch counters. These are exact counts of
// what executed, not samples: a simulator can report every instruction and
// every memory access, which is the one thing hardware counters cannot do.
// Nothing here is timing-derived, because there is no timing model to derive
// it from.
bool counters_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("VGPU_COUNTERS");
    return e && e[0] && e[0] != '0';
  }();
  return on;
}

void report_counters(const std::string& kernel, const exec::LaunchConfig& cfg,
                     const exec::LaunchStats& st) {
  if (!counters_enabled()) return;
  const double lanes = st.instructions ? static_cast<double>(st.thread_instructions) /
                                             static_cast<double>(st.instructions)
                                       : 0.0;
  // Sectors per request against the fewest a request of this shape could have
  // needed. 100% means every byte fetched was asked for; 25% means three
  // quarters of the traffic was the memory system rounding up to 32 bytes.
  const double sectors_per_request =
      st.global_requests ? static_cast<double>(st.global_sectors) /
                               static_cast<double>(st.global_requests)
                         : 0.0;
  const double ideal_sectors =
      st.global_requests
          ? static_cast<double>(st.global_bytes_read + st.global_bytes_written) / 32.0
          : 0.0;
  const auto cls = [&st](exec::InstClass c) { return st.inst_by_class[static_cast<size_t>(c)]; };
  const double coalescing_pct =
      st.global_sectors ? 100.0 * ideal_sectors / static_cast<double>(st.global_sectors) : 0.0;
  std::fprintf(stderr,
               "[vgpu][counters] %s  grid=%ux%ux%u block=%ux%ux%u\n"
               "    blocks=%llu warps=%llu\n"
               "    inst_executed=%llu  thread_inst_executed=%llu  lanes_active_avg=%.2f/32\n"
               "    divergent_branches=%llu  barriers=%llu  atomics=%llu (%llu B)\n"
               "    global  ld=%llu st=%llu  read=%llu B write=%llu B\n"
               "    shared  ld=%llu st=%llu  read=%llu B write=%llu B\n"
               "    local   ld=%llu st=%llu  read=%llu B write=%llu B\n"
               "    sectors global=%llu over %llu requests (%.2f per request, %.0f%% of ideal)\n"
               "    shared  bank_conflicts=%llu over %llu requests\n"
               "    mix     fp16=%llu fp32=%llu fp64=%llu int=%llu cvt=%llu\n"
               "            ctrl=%llu mem=%llu tensor=%llu misc=%llu  (tensor issues=%llu)\n",
               kernel.c_str(), cfg.grid[0], cfg.grid[1], cfg.grid[2], cfg.block[0], cfg.block[1],
               cfg.block[2], (unsigned long long)st.blocks, (unsigned long long)st.warps,
               (unsigned long long)st.instructions, (unsigned long long)st.thread_instructions,
               lanes, (unsigned long long)st.divergent_branches, (unsigned long long)st.barriers,
               (unsigned long long)st.atomics, (unsigned long long)st.atomic_bytes,
               (unsigned long long)st.global_loads,
               (unsigned long long)st.global_stores, (unsigned long long)st.global_bytes_read,
               (unsigned long long)st.global_bytes_written, (unsigned long long)st.shared_loads,
               (unsigned long long)st.shared_stores, (unsigned long long)st.shared_bytes_read,
               (unsigned long long)st.shared_bytes_written, (unsigned long long)st.local_loads,
               (unsigned long long)st.local_stores,
               (unsigned long long)st.local_bytes_read,
               (unsigned long long)st.local_bytes_written,
               (unsigned long long)st.global_sectors, (unsigned long long)st.global_requests,
               sectors_per_request, coalescing_pct,
               (unsigned long long)st.shared_bank_conflicts,
               (unsigned long long)st.shared_requests,
               (unsigned long long)cls(exec::InstClass::Fp16),
               (unsigned long long)cls(exec::InstClass::Fp32),
               (unsigned long long)cls(exec::InstClass::Fp64),
               (unsigned long long)cls(exec::InstClass::Integer),
               (unsigned long long)cls(exec::InstClass::BitConvert),
               (unsigned long long)cls(exec::InstClass::Control),
               (unsigned long long)cls(exec::InstClass::Memory),
               (unsigned long long)cls(exec::InstClass::Tensor),
               (unsigned long long)cls(exec::InstClass::Misc),
               (unsigned long long)st.tensor_instructions);
}

}  // namespace

void Device::launch(const ptx::EntryFn& fn, const exec::LaunchConfig& in_cfg,
                    const std::vector<std::vector<uint8_t>>& args, const exec::SymbolTable* syms) {
  // Texture objects belong to the device, so the launch does not have to be
  // told about them by every caller. A caller that set them explicitly keeps
  // its own table.
  exec::LaunchConfig cfg = in_cfg;
  if (!cfg.textures && !textures_.empty()) cfg.textures = &textures_;
  if (!telemetry_) {
    report_counters(fn.name, cfg, exec::launch(fn, cfg, args, mem_, profile_, syms));
    return;
  }
  // Utilization is the real fraction of wall time spent executing kernels. The
  // progress hook publishes it *during* the launch so a long kernel still shows
  // live telemetry rather than freezing until it returns.
  uint32_t ord = static_cast<uint32_t>(ordinal_);
  telemetry::Publisher* pub = telemetry_;
  auto start = std::chrono::steady_clock::now();
  const exec::LaunchStats st = exec::launch(fn, cfg, args, mem_, profile_, syms,
               [pub, ord](double dt) { pub->note_kernel(ord, dt); });
  report_counters(fn.name, cfg, st);
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
