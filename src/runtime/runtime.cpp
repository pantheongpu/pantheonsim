#include "vgpu/ras.hpp"
#include <unistd.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <algorithm>
#include <utility>
#include "vgpu/runtime/runtime.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdio>

#include "vgpu/error.hpp"
#include <cctype>

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
    // The suffixed targets narrow that. sm_90a code uses features only
    // compute capability 9.0 has (wgmma is gone on Blackwell), and sm_100f
    // code runs within its family, 10.x, and no further.
    const size_t sm = mod->target.find("sm_");
    const char suffix = [&]() -> char {
      if (sm == std::string::npos) return 0;
      size_t i = sm + 3;
      while (i < mod->target.size() && std::isdigit(static_cast<unsigned char>(mod->target[i]))) ++i;
      return i < mod->target.size() ? mod->target[i] : 0;
    }();
    if (want && have && ((suffix == 'a' && want != have) || (suffix == 'f' && want / 10 != have / 10)))
      throw Error::make(Err::PtxParse, "module targets ", mod->target,
                        ", which is specific to compute capability ",
                        suffix == 'a' ? std::to_string(want / 10) + "." + std::to_string(want % 10)
                                      : std::to_string(want / 10) + ".x",
                        ", and this device is ", profile_.cc_major, ".", profile_.cc_minor);
    LoadedModule lm;
    lm.id = next_module_id_++;
    // Materialize module .global variables into device memory.
    for (const auto& g : mod->globals) {
      uint64_t va = mem_.alloc(g.size);
      if (!g.init.empty()) mem_.write(va, g.init.data(), g.init.size());
      lm.symbols[g.name] = va;
      lm.global_vas.push_back(va);
    }
    // Device functions get an address so a function pointer can be stored in a
    // global, loaded and called. The address encodes the index, which is what
    // an indirect call decodes to find the function again.
    for (size_t i = 0; i < mod->funcs.size(); ++i)
      lm.symbols[mod->funcs[i]->name] = kFuncVaBase + i * kFuncVaStride;
    // Second pass: a global initialised with another symbol's address can only
    // be filled in once every global has one. A symbol that names a kernel
    // rather than a variable has no address in this model and stays zero --
    // taking a kernel's address is a host-side operation, and PTX that only
    // uses it to carry a mangled name (which is what NVRTC's name expressions
    // compile to) never dereferences it.
    for (const auto& g : mod->globals) {
      if (g.init_symbols.empty()) continue;
      const uint64_t slot = lm.symbols[g.name];
      for (const auto& si : g.init_symbols) {
        auto it = lm.symbols.find(si.name);
        const uint64_t target = it == lm.symbols.end() ? 0 : it->second;
        const uint64_t room = g.size > si.offset ? g.size - si.offset : 0;
        const uint64_t bytes = room < sizeof(uint64_t) ? room : sizeof(uint64_t);
        if (bytes) mem_.write(slot + si.offset, &target, bytes);
      }
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

void Device::reset() {
  // Modules first: their globals are allocations, and unloading frees them by
  // handle. Whatever is left afterwards -- cudaMalloc, arrays, pitched
  // buffers -- goes in one sweep.
  while (!modules_.empty()) unload_module(modules_.back().id);
  textures_.clear();
  mem_.free_all();
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

// One record per launch, as a line of JSON, for a tool that reads counters
// rather than a person: `vgpu ncu` sets VGPU_COUNTERS_FILE, runs the program,
// and reports from the file. Every counter is written under the name it has in
// LaunchStats, so the reader decides what each means rather than this file.
void record_counters(int device, const std::string& kernel, const exec::LaunchConfig& cfg,
                     const exec::LaunchStats& st) {
  static const char* path = std::getenv("VGPU_COUNTERS_FILE");
  if (!path || !*path) return;
  static std::mutex mu;
  static std::atomic<uint64_t> seq{0};
  char buf[4096];
  const auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
  const int n = std::snprintf(
      buf, sizeof buf,
      "{\"launch\":%llu,\"pid\":%d,\"device\":%d,\"kernel\":\"%s\","
      "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
      "\"blocks\":%llu,\"warps\":%llu,\"instructions\":%llu,\"thread_instructions\":%llu,"
      "\"global_loads\":%llu,\"global_stores\":%llu,"
      "\"global_bytes_read\":%llu,\"global_bytes_written\":%llu,"
      "\"shared_loads\":%llu,\"shared_stores\":%llu,"
      "\"shared_bytes_read\":%llu,\"shared_bytes_written\":%llu,"
      "\"global_sectors_ld\":%llu,\"global_sectors_st\":%llu,"
      "\"global_requests_ld\":%llu,\"global_requests_st\":%llu,"
      "\"shared_requests_ld\":%llu,\"shared_requests_st\":%llu,"
      "\"shared_bank_conflicts_ld\":%llu,\"shared_bank_conflicts_st\":%llu,"
      "\"global_bytes_ld\":%llu,\"global_bytes_st\":%llu}\n",
      u(seq.fetch_add(1)), static_cast<int>(getpid()), device, kernel.c_str(), cfg.grid[0],
      cfg.grid[1], cfg.grid[2], cfg.block[0], cfg.block[1], cfg.block[2], u(st.blocks),
      u(st.warps), u(st.instructions), u(st.thread_instructions), u(st.global_loads),
      u(st.global_stores), u(st.global_bytes_read), u(st.global_bytes_written),
      u(st.shared_loads), u(st.shared_stores), u(st.shared_bytes_read),
      u(st.shared_bytes_written), u(st.global_sectors_ld), u(st.global_sectors_st),
      u(st.global_requests_ld), u(st.global_requests_st), u(st.shared_requests_ld),
      u(st.shared_requests_st), u(st.shared_bank_conflicts_ld), u(st.shared_bank_conflicts_st),
      u(st.global_bytes_ld), u(st.global_bytes_st));
  if (n <= 0 || static_cast<size_t>(n) >= sizeof buf) return;   // a name too long to record
  std::lock_guard<std::mutex> lock(mu);
  if (std::FILE* f = std::fopen(path, "a")) {
    std::fwrite(buf, 1, static_cast<size_t>(n), f);
    std::fclose(f);
  }
}

void report_counters(int device, const std::string& kernel, const exec::LaunchConfig& cfg,
                     const exec::LaunchStats& st) {
  record_counters(device, kernel, cfg, st);
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

  // Per-opcode issues, most-used first. The class histogram says a kernel is
  // memory-heavy; this says it is memory-heavy because of ld.global.nc, which
  // is the difference between a number and a lead. Truncated because a long
  // tail of ones is noise, and the total says what was left out.
  const auto& names = ptx::opcode_names();
  std::vector<std::pair<uint64_t, std::string>> ops;
  for (size_t i = 1; i < st.inst_by_opcode.size() && i < names.size(); ++i)
    if (st.inst_by_opcode[i]) ops.emplace_back(st.inst_by_opcode[i], names[i]);
  if (!ops.empty()) {
    std::sort(ops.begin(), ops.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string line = "    by opcode";
    const size_t show = std::min<size_t>(ops.size(), 10);
    for (size_t i = 0; i < show; ++i)
      line += " " + ops[i].second + "=" + std::to_string(ops[i].first);
    if (ops.size() > show) line += "  (+" + std::to_string(ops.size() - show) + " more)";
    std::fprintf(stderr, "%s\n", line.c_str());
  }
}

}  // namespace

// Takes faults armed with `vgpu fault arm` on this device's kernel accesses. A
// corrected error is counted and the access goes on; an uncorrectable one is
// counted, logged as the driver logs it (Xid 48, then 63 for the page or row
// it takes out of service when it is in device memory), and ends the kernel;
// a bit flip corrupts the value silently -- on a store, the value written, so
// every later read finds it. Accesses on several host threads may race to take
// the last armed fault; ArmedFaults::take lets exactly one have it.
class FaultHook final : public MemoryManager::AccessFault {
 public:
  FaultHook(const DeviceProfile& p, int ordinal) : faults_(identify(p, ordinal)) {
    load_pending = faults_.pending(ras::Target::Load);
    store_pending = faults_.pending(ras::Target::Store);
    shared_pending = faults_.pending(ras::Target::Shared);
    stuck_pending = faults_.stuck_pending();
    alu_pending = faults_.pending(ras::Target::Alu);
    copy_pending = faults_.pending(ras::Target::Copy);
  }

  // A GPU that has fallen off the bus runs nothing.
  void check_lost(const std::string& kernel) {
    if (faults_.lost())
      throw Error::make(Err::DeviceLost, "kernel '", kernel,
                        "' was not run: the GPU has fallen off the bus (`vgpu fault lose`)");
  }

  // A hang armed for this launch: it stalls the launch, with the device shown
  // fully busy as a hung card is, for `seconds` -- or, with 0, until the
  // process is stopped -- and then fails it the way a timed-out launch fails.
  void maybe_hang(const std::string& kernel, telemetry::Publisher* pub, uint32_t ord) {
    uint64_t seconds = 0;
    if (!faults_.take_hang(&seconds)) return;
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      if (pub) pub->note_kernel(ord, 0.25);
      const double waited =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      if (seconds && waited >= static_cast<double>(seconds)) break;
    }
    if (pub) pub->note_kernel(ord, 0.0);
    ras::report_xid(uuid_, bus_id_, 8,
                    "pid=" + std::to_string(::getpid()) + ", name=" + process_name(),
                    "GPU stopped processing");
    throw Error::make(Err::ExecLimit, "kernel '", kernel, "' hung for ", seconds,
                      " s and was stopped (armed with `vgpu fault arm --hang`)");
  }

  // A kernel that faulted on its own -- a bad address, a misaligned access --
  // is logged the way the driver logs it: Xid 31 for an address the MMU has no
  // mapping for, Xid 13 for an exception the SM raises itself. Faults this
  // simulator finds that hardware would not (a data race, an uninitialized
  // register) and ones logged where they happen (ECC, hangs) are not logged.
  void report_kernel_fault(const Error& e) {
    const std::string& m = e.message();
    const std::string who = "pid=" + std::to_string(::getpid()) + ", name=" + process_name();
    const bool on_chip = m.rfind("shared memory", 0) == 0 || m.rfind("local memory", 0) == 0;
    switch (e.code()) {
      case Err::InvalidPointer:
      case Err::UseAfterFree:
      case Err::OutOfBounds: {
        if (on_chip) {
          ras::report_xid(uuid_, bus_id_, 13, who,
                          "Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): Out Of Range Address");
          return;
        }
        // The page the access fell in, printed as the driver prints it.
        uint64_t addr = 0;
        if (const size_t at = m.find(" at 0x"); at != std::string::npos)
          addr = std::strtoull(m.c_str() + at + 4, nullptr, 16);
        char where[32];
        std::snprintf(where, sizeof where, "0x%llx_%08llx",
                      static_cast<unsigned long long>(addr >> 32),
                      static_cast<unsigned long long>(addr & 0xFFFFF000u));
        // Past the end of an allocation, inside its last page, is a page with
        // no valid entry; anywhere else no page table covers the address.
        const char* type = e.code() == Err::OutOfBounds ? "FAULT_PTE" : "FAULT_PDE";
        const char* access = m.find("write") != std::string::npos ? "ACCESS_TYPE_VIRT_WRITE"
                                                                   : "ACCESS_TYPE_VIRT_READ";
        ras::report_xid(uuid_, bus_id_, 31, who,
                        std::string("Ch 00000008, intr 00000000. MMU Fault: ENGINE GRAPHICS GPC0 "
                                    "GPCCLIENT_T1_0 faulted @ ") +
                            where + ". Fault is of type " + type + " " + access);
        return;
      }
      case Err::MisalignedAccess:
        ras::report_xid(uuid_, bus_id_, 13, who,
                        "Graphics SM Warp Exception on (GPC 0, TPC 0, SM 0): Misaligned Address");
        return;
      default:
        return;
    }
  }

  uint64_t on_load(uint64_t addr, uint32_t size, uint64_t value) override {
    return take(ras::Target::Load, addr, size, value);
  }
  uint64_t on_store(uint64_t addr, uint32_t size, uint64_t value) override {
    return take(ras::Target::Store, addr, size, value);
  }
  uint64_t on_shared_load(uint64_t offset, uint32_t size, uint64_t value) override {
    return take(ras::Target::Shared, offset, size, value);
  }
  void on_read(uint64_t offset, uint8_t* bytes, uint64_t len) override {
    faults_.apply_stuck(offset, bytes, len);
  }
  void on_copy(uint64_t addr, uint8_t* bytes, uint64_t len) override {
    switch (faults_.take(ras::Target::Copy)) {
      case ras::Armed::Corrected:
        ras::inject_ecc(uuid_, ras::Severity::Corrected, ras::Location::DeviceMemory, 1, scheme_);
        return;
      case ras::Armed::Bitflip: {
        // One bit of one byte of what this copy delivers; memory keeps its value.
        const uint64_t n = flips_.fetch_add(1);
        bytes[(n * 2654435761u) % len] ^= static_cast<uint8_t>(1u << (n % 8));
        faults_.note_bitflip();
        return;
      }
      case ras::Armed::Uncorrected:
        uncorrectable_in_memory(addr, " during a copy");
      default:
        return;
    }
  }

  // A flip in the upper half of the result's bits -- an integer's high bits, a
  // float's exponent or leading mantissa -- so the error is one a result check
  // can see rather than a last-place difference inside its tolerance.
  uint64_t on_alu(uint64_t value, uint32_t bits) override {
    if (faults_.take(ras::Target::Alu) != ras::Armed::Bitflip) return value;
    const uint32_t half = std::max(bits / 2, 1u);
    const uint32_t bit = bits - 2 - static_cast<uint32_t>(flips_.fetch_add(1) % (half - 1 ? half - 1 : 1));
    faults_.note_bitflip();
    return value ^ (uint64_t{1} << bit);
  }

 private:
  uint64_t take(ras::Target at, uint64_t addr, uint32_t size, uint64_t value) {
    // Shared memory and L1 are one SRAM in the SM: its errors are the L1
    // cache's, and no page or row of device memory is taken out of service.
    const bool shared = at == ras::Target::Shared;
    const ras::Location where = shared ? ras::Location::L1Cache : ras::Location::DeviceMemory;
    switch (faults_.take(at)) {
      case ras::Armed::None:
      case ras::Armed::Hang:
        return value;
      case ras::Armed::Corrected:
        ras::inject_ecc(uuid_, ras::Severity::Corrected, where, 1, scheme_);
        return value;
      case ras::Armed::Bitflip: {
        const uint32_t bit = static_cast<uint32_t>((addr * 8 + flips_.fetch_add(1)) % (size * 8u));
        faults_.note_bitflip();
        return value ^ (uint64_t{1} << bit);
      }
      case ras::Armed::Uncorrected:
        break;
    }
    if (!shared) uncorrectable_in_memory(addr, "");
    ras::inject_ecc(uuid_, ras::Severity::Uncorrected, where, 1, scheme_);
    char at_text[32];
    std::snprintf(at_text, sizeof at_text, "0x%llx", static_cast<unsigned long long>(addr));
    ras::report_xid(uuid_, bus_id_, 48, "pid=" + std::to_string(::getpid()) + ", name=" + process_name(),
                    "An uncorrectable double bit error (DBE) has been detected on GPU in the SM "
                    "L1 cache.");
    throw Error::make(Err::EccUncorrectable, "uncorrectable ECC error in shared memory at offset ",
                      at_text, " (armed with `vgpu fault arm --on shared`)");
  }

  // An uncorrectable error in device memory, taken by a kernel's load or by a
  // copy: counted, logged as Xid 48 and then Xid 63 for the page or row it
  // takes out of service, and the operation fails.
  [[noreturn]] void uncorrectable_in_memory(uint64_t addr, const char* during) {
    ras::inject_ecc(uuid_, ras::Severity::Uncorrected, ras::Location::DeviceMemory, 1, scheme_);
    ras::report_xid(uuid_, bus_id_, 48, "pid=" + std::to_string(::getpid()) + ", name=" + process_name(),
                    "An uncorrectable double bit error (DBE) has been detected on GPU "
                    "in the framebuffer at partition 0, subpartition 0.");
    if (scheme_ == ras::Retirement::Pages)
      ras::report_xid(uuid_, bus_id_, 63, "",
                      "ECC page retirement recording event: a page is pending "
                      "retirement, reboot to activate.");
    else if (scheme_ == ras::Retirement::Rows)
      ras::report_xid(uuid_, bus_id_, 63, "",
                      "Row Remapper: New row marked for remapping, reset gpu to activate.");
    char at_text[32];
    std::snprintf(at_text, sizeof at_text, "0x%llx", static_cast<unsigned long long>(addr));
    throw Error::make(Err::EccUncorrectable, "uncorrectable ECC error in device memory at ", at_text,
                      during, " (armed with `vgpu fault arm`)");
  }

  // The device's UUID and bus id, the ones nvidia-smi reports, and the way it
  // takes failing memory out of service.
  std::string identify(const DeviceProfile& p, int ordinal) {
    telemetry::DeviceSample d{};
    telemetry::describe_device(p, ordinal, &d);
    uuid_ = d.uuid;
    bus_id_ = d.bus_id;
    scheme_ = static_cast<ras::Retirement>(d.memory_retirement);
    return uuid_;
  }
  static std::string process_name() {
    std::string name = "<unknown>";
    if (std::FILE* f = std::fopen("/proc/self/comm", "r")) {
      char buf[64] = {0};
      if (std::fgets(buf, sizeof buf, f)) {
        name = buf;
        while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) name.pop_back();
      }
      std::fclose(f);
    }
    return name;
  }

  std::string uuid_, bus_id_;
  ras::Retirement scheme_ = ras::Retirement::None;
  ras::ArmedFaults faults_;
  std::atomic<uint64_t> flips_{0};
};

Device::Device(DeviceProfile profile, int ordinal, telemetry::Publisher* telemetry)
    : profile_(std::move(profile)), ordinal_(ordinal),
      mem_(profile_.vram_bytes, static_cast<uint32_t>(ordinal)), telemetry_(telemetry) {
  if (telemetry_) {
    int ord = ordinal_;
    telemetry::Publisher* pub = telemetry_;
    mem_.set_usage_observer(
        [pub, ord](uint64_t used) { pub->note_memory(static_cast<uint32_t>(ord), used); });
  }
  install_fault_hook();
}

Device::~Device() = default;

void Device::install_fault_hook() {
  // Without a writable runtime directory no fault can be armed for this device;
  // everything else works exactly as before.
  try {
    fault_ = std::make_unique<FaultHook>(profile_, ordinal_);
    mem_.set_access_fault(fault_.get());
  } catch (const std::exception&) {
    fault_.reset();
  }
}

void Device::launch(const ptx::EntryFn& fn, const exec::LaunchConfig& in_cfg,
                    const std::vector<std::vector<uint8_t>>& args, const exec::SymbolTable* syms) {
  // Texture objects belong to the device, so the launch does not have to be
  // told about them by every caller. A caller that set them explicitly keeps
  // its own table.
  exec::LaunchConfig cfg = in_cfg;
  if (!cfg.textures && !textures_.empty()) cfg.textures = &textures_;
  if (fault_) {
    fault_->check_lost(fn.name);
    fault_->maybe_hang(fn.name, telemetry_, static_cast<uint32_t>(ordinal_));
  }
  try {
    run_kernel(fn, cfg, args, syms);
  } catch (const Error& e) {
    if (fault_) {
      try {
        fault_->report_kernel_fault(e);
      } catch (const std::exception&) {
        // No runtime directory to log to: the kernel's own error still stands.
      }
    }
    throw;
  }
}

void Device::run_kernel(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg,
                        const std::vector<std::vector<uint8_t>>& args, const exec::SymbolTable* syms) {
  if (!telemetry_) {
    report_counters(ordinal_, fn.name, cfg, exec::launch(fn, cfg, args, mem_, profile_, syms));
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
  report_counters(ordinal_, fn.name, cfg, st);
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
  // Every device needs an address window of its own (memory.hpp).
  static_assert(telemetry::kMaxDevices <= vgpu::kDeviceVaWindows);
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
