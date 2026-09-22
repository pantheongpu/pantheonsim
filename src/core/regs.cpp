#include "vgpu/regs.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "vgpu/embedded_registers.hpp"
#include "vgpu/error.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/shared_state.hpp"
#include "vgpu/yamlish.hpp"

namespace vgpu::regs {
namespace {

// ---- The database ---------------------------------------------------------------

uint32_t parse_hex(const std::string& s, const std::string& where) {
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s.c_str(), &end, 16);
  if (s.size() < 3 || s.compare(0, 2, "0x") != 0 || !end || *end || v > 0xFFFFFFFFull)
    throw Error::make(Err::ProfileParse, where, ": expected a hex value like \"0x0010\", got '", s, "'");
  return static_cast<uint32_t>(v);
}

std::vector<Register> load(const char* yaml, const std::string& origin, uint32_t size) {
  const yamlish::Value doc = yamlish::parse(yaml, origin);
  std::vector<Register> out;
  for (const auto& [name, v] : doc.map) {
    const std::string where = origin + ": " + name;
    if (v.kind != yamlish::Value::Kind::Map) throw Error::make(Err::ProfileParse, where, ": expected a map");
    auto str = [&](const char* key, bool required) -> std::string {
      const auto it = v.map.find(key);
      if (it == v.map.end()) {
        if (required) throw Error::make(Err::ProfileParse, where, ": missing ", key);
        return "";
      }
      return it->second.kind == yamlish::Value::Kind::Int ? std::to_string(it->second.i) : it->second.str;
    };
    auto list = [&](const char* key) {
      std::vector<std::string> l;
      const auto it = v.map.find(key);
      if (it != v.map.end())
        for (const auto& e : it->second.list) l.push_back(e.str);
      return l;
    };
    Register r;
    r.name = name;
    r.offset = parse_hex(str("offset", true), where);
    r.width = static_cast<uint32_t>(std::atoi(str("width", true).c_str()));
    if (r.width != 8 && r.width != 16 && r.width != 24 && r.width != 32)
      throw Error::make(Err::ProfileParse, where, ": width is 8, 16, 24 or 32");
    const std::string a = str("access", true);
    r.access = a == "ro" ? Access::Ro : a == "rw" ? Access::Rw : a == "rw1c" ? Access::Rw1c
             : a == "bar" ? Access::Bar : throw Error::make(Err::ProfileParse, where, ": unknown access '", a, "'");
    if (const std::string reset = str("reset", false); !reset.empty()) r.reset = parse_hex(reset, where);
    if (const std::string mask = str("write_mask", false); !mask.empty()) r.write_mask = parse_hex(mask, where);
    r.backing = str("backing", false);
    r.status = str("status", true);
    r.source = str("source", false);
    r.measured = str("measured", false);
    r.fields = list("fields");
    r.surfaces = list("surfaces");
    if (r.access == Access::Rw && !r.write_mask)
      throw Error::make(Err::ProfileParse, where, ": a rw register needs a write_mask");
    if ((r.access == Access::Rw1c || r.access == Access::Bar) && r.backing.empty())
      throw Error::make(Err::ProfileParse, where, ": a ", a, " register needs a backing");
    if (r.offset + r.width / 8 > size) throw Error::make(Err::ProfileParse, where, ": past the end of the space");
    out.push_back(std::move(r));
  }
  std::sort(out.begin(), out.end(), [](const Register& x, const Register& y) { return x.offset < y.offset; });
  for (size_t i = 1; i < out.size(); ++i)
    if (out[i].offset < out[i - 1].offset + out[i - 1].width / 8)
      throw Error::make(Err::ProfileParse, origin, ": ", out[i].name, " overlaps ", out[i - 1].name);
  return out;
}

// The database file each space is declared in.
const char* database_name(Space s) { return s == Space::Config ? "pci-config" : "amd-mmio"; }

const char* space_yaml(Space s) {
  for (const auto& e : embedded::kRegisterSpaces)
    if (std::strcmp(e.space, database_name(s)) == 0) return e.yaml;
  throw Error::make(Err::Internal, "no ", database_name(s), " register database was embedded");
}

// Changes when the database does, so a state file written against another
// layout is started again rather than misread.
uint32_t database_version(Space s) {
  uint32_t h = 2166136261u;
  for (const char* p = space_yaml(s); *p; ++p) h = (h ^ static_cast<unsigned char>(*p)) * 16777619u;
  return h | 1u;
}

// ---- Base address registers -----------------------------------------------------

struct BarLayout {
  uint64_t size = 0;   // 0: not implemented
  bool io = false, is64 = false, pref = false;
  bool high = false;   // the upper half of the 64-bit BAR before it
  uint64_t base = 0;   // where firmware put it
};

uint64_t pow2_at_least(uint64_t v) {
  uint64_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

bool is_amd(const telemetry::DeviceSample& d) { return std::strcmp(d.vendor, "amd") == 0; }
bool is_geforce(const telemetry::DeviceSample& d) {
  return std::strstr(d.name, "GeForce") || std::strstr(d.name, "RTX 30");
}

unsigned bus_number(const telemetry::DeviceSample& d) {
  // "00000000:01:00.0"
  const char* colon = std::strchr(d.bus_id, ':');
  return colon ? static_cast<unsigned>(std::strtoul(colon + 1, nullptr, 16)) : 1;
}

// NVIDIA: BAR0 16 MiB of registers; BAR1 the framebuffer aperture, 64-bit and
// prefetchable -- 256 MiB on a GeForce, the framebuffer rounded up to a power
// of two on data-center cards; BAR3 32 MiB, 64-bit; an I/O BAR on a GeForce.
// AMD: BAR0 the framebuffer aperture, BAR2 the doorbells, BAR5 the registers.
// Sizes are a model, not measured per card.
std::array<BarLayout, 6> bar_layout(const telemetry::DeviceSample& d) {
  std::array<BarLayout, 6> b{};
  const uint64_t slot = bus_number(d) ? bus_number(d) - 1 : 0;
  const uint64_t mmio32 = 0xE0000000ull + slot * 0x02000000ull;
  const uint64_t mmio64 = 0x200000000000ull + slot * 0x10000000000ull;
  const uint64_t fb = pow2_at_least(d.vram_total_bytes ? d.vram_total_bytes : (1ull << 28));
  if (is_amd(d)) {
    b[0] = {fb, false, true, true, false, mmio64};
    b[1].high = true;
    b[2] = {2ull << 20, false, true, true, false, mmio64 + 0x8000000000ull};
    b[3].high = true;
    b[5] = {512ull << 10, false, false, false, false, mmio32 + 0x01000000ull};
  } else {
    b[0] = {16ull << 20, false, false, false, false, mmio32};
    b[1] = {is_geforce(d) ? (256ull << 20) : fb, false, true, true, false, mmio64};
    b[2].high = true;
    b[3] = {32ull << 20, false, true, true, false, mmio64 + 0x8000000000ull};
    b[4].high = true;
    if (is_geforce(d)) b[5] = {128, true, false, false, false, 0x3000ull + slot * 0x1000ull};
  }
  return b;
}

// ---- Shared state ------------------------------------------------------------------

constexpr uint32_t kMagic = 0x56524547;  // "VREG"
constexpr uint32_t kMaxRegisters = 128;
constexpr uint32_t kLogWords = 7;        // seq, time, pid, access, value, process (16 bytes)

struct State {
  uint64_t rw_set[kMaxRegisters];        // by register index: written since reset
  uint64_t rw_value[kMaxRegisters];
  uint64_t bar_set[6];
  uint64_t bar_value[6];
  uint64_t base_correctable[32];         // write-1-to-clear: the count when each bit was cleared
  uint64_t base_uncorrectable[32];
  uint64_t base_devsta[4];
  uint64_t mailbox[4];                   // the SMU's: message, argument, response, response set
  uint64_t log_seq;
  uint64_t log[kLogEntries][kLogWords];
};

std::string state_path(const std::string& uuid, Space s) {
  return telemetry::default_path() + "/regs-" + uuid + (s == Space::Config ? "" : ".mmio");
}

// The name the process was started under: argv[0]'s last component, which is
// the tool's own name even where it is a script that execs vgpu under it
// (nvidia-smi, rocm-smi, amd-smi). The kernel's short name when that fails.
std::string process_name() {
  char buf[256] = {0};
  if (std::FILE* f = std::fopen("/proc/self/cmdline", "r")) {
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    buf[n] = 0;
    std::fclose(f);
  }
  std::string name = buf;   // up to the first NUL: argv[0]
  if (const size_t slash = name.rfind('/'); slash != std::string::npos) name = name.substr(slash + 1);
  if (name.empty()) {
    if (std::FILE* f = std::fopen("/proc/self/comm", "r")) {
      if (std::fgets(buf, sizeof buf, f)) name = buf;
      std::fclose(f);
    }
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) name.pop_back();
  }
  return name;
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

// ---- Error status, from the reliability counts --------------------------------------

// The AER correctable-status bit each injected PCIe counter sets (see
// ras::publish_session for the same mapping in the kernel's AER stats).
uint64_t correctable_count(const ras::Counters& c, uint32_t bit) {
  const auto n = [&](ras::Pcie p) { return c.pcie[static_cast<uint32_t>(p)]; };
  switch (bit) {
    case 0: return n(ras::Pcie::Correctable);
    case 6: return n(ras::Pcie::BadTlp) + n(ras::Pcie::Lcrc);
    case 7: return n(ras::Pcie::BadDllp);
    case 8: return n(ras::Pcie::ReplayRollover);
    case 12: return n(ras::Pcie::Replay);
    default: return 0;
  }
}
uint64_t uncorrectable_count(const ras::Counters& c, uint32_t bit) {
  const auto n = [&](ras::Pcie p) { return c.pcie[static_cast<uint32_t>(p)]; };
  switch (bit) {
    case 4: return n(ras::Pcie::Fatal);        // data link protocol error
    case 14: return n(ras::Pcie::NonFatal);    // completion timeout
    default: return 0;
  }
}
uint64_t devsta_count(const ras::Counters& c, uint32_t bit) {
  if (bit == 0) {
    uint64_t sum = 0;
    for (uint32_t b : {0u, 6u, 7u, 8u, 12u}) sum += correctable_count(c, b);
    return sum;
  }
  if (bit == 1) return uncorrectable_count(c, 14);
  if (bit == 2) return uncorrectable_count(c, 4);
  return 0;
}

// A status bit is set while its count has grown since it was last cleared. A
// count below the mark means the counts started again (a driver reload).
bool raised(uint64_t count, uint64_t cleared_at) {
  return count > (cleared_at <= count ? cleared_at : 0);
}

// The SMU mailbox, as the driver drives it: clear the response, write the
// argument, write the message; the SMU answers in the response register, and
// in the argument register when the message returns a value.
void smu_write(State* s, const std::string& backing, uint32_t v) {
  if (backing == "smu.argument") {
    __atomic_store_n(&s->mailbox[1], uint64_t{v}, __ATOMIC_RELAXED);
  } else if (backing == "smu.response") {
    __atomic_store_n(&s->mailbox[2], uint64_t{v}, __ATOMIC_RELAXED);
    __atomic_store_n(&s->mailbox[3], uint64_t{1}, __ATOMIC_RELEASE);
  } else if (backing == "smu.message") {
    __atomic_store_n(&s->mailbox[0], uint64_t{v}, __ATOMIC_RELAXED);
    const uint32_t arg = static_cast<uint32_t>(__atomic_load_n(&s->mailbox[1], __ATOMIC_RELAXED));
    uint32_t reply = arg, result = kSmuResultOk;
    switch (v) {
      case kSmuTestMessage: reply = arg + 1; break;
      // A firmware version in the form the driver prints as 85.111.0; a model.
      case kSmuGetSmuVersion: reply = 0x00556F00u; break;
      // The interface version the driver expects (SMU13_0_6_DRIVER_IF_VERSION).
      case kSmuGetDriverIfVersion: reply = 0x08042024u; break;
      case kSmuGetMetricsVersion: reply = 0x11u; break;   // SMU_METRICS_TABLE_VERSION
      default: result = kSmuResultUnknownCmd; break;
    }
    __atomic_store_n(&s->mailbox[1], uint64_t{reply}, __ATOMIC_RELAXED);
    __atomic_store_n(&s->mailbox[2], uint64_t{result}, __ATOMIC_RELAXED);
    __atomic_store_n(&s->mailbox[3], uint64_t{1}, __ATOMIC_RELEASE);
  }
}

}  // namespace

const char* access_name(Access a) {
  switch (a) {
    case Access::Ro: return "ro";
    case Access::Rw: return "rw";
    case Access::Rw1c: return "rw1c";
    case Access::Bar: return "bar";
  }
  return "?";
}

const char* space_name(Space s) { return s == Space::Config ? "config" : "mmio"; }

bool parse_space(const std::string& name, Space* out) {
  if (name == "config") *out = Space::Config;
  else if (name == "mmio") *out = Space::AmdMmio;
  else return false;
  return true;
}

uint32_t space_size(Space s) { return s == Space::Config ? kConfigSize : 512u << 10; }

bool has_space(const telemetry::DeviceSample& d, Space s) { return s == Space::Config || is_amd(d); }

const std::vector<Register>& registers(Space space) {
  // Each loaded on first use, so one space's database cannot stop another's.
  const auto loaded = [](Space s) {
    auto r = load(space_yaml(s), std::string("registers/") + database_name(s) + ".yaml", space_size(s));
    if (r.size() > kMaxRegisters) throw Error::make(Err::Internal, "more than ", kMaxRegisters, " registers");
    return r;
  };
  if (space == Space::Config) {
    static const std::vector<Register> config = loaded(Space::Config);
    return config;
  }
  static const std::vector<Register> mmio = loaded(Space::AmdMmio);
  return mmio;
}

const Register* find(Space space, const std::string& key) {
  const auto& regs = registers(space);
  if (key.rfind("0x", 0) == 0) {
    char* end = nullptr;
    const unsigned long off = std::strtoul(key.c_str(), &end, 16);
    for (const auto& r : regs)
      if (end && !*end && r.offset == off) return &r;
    return nullptr;
  }
  for (const auto& r : regs)
    if (r.name == key) return &r;
  return nullptr;
}

struct RegisterSpace::Impl {
  Impl(Space sp, const telemetry::DeviceSample& dev)
      : space(sp),
        d(dev),
        file(state_path(dev.uuid, sp), true, kMagic, database_version(sp), sizeof(State), "register state"),
        bars(bar_layout(dev)) {}
  Space space;
  telemetry::DeviceSample d;
  SharedState file;
  std::array<BarLayout, 6> bars;

  State* state() { return static_cast<State*>(file.payload()); }

  ras::Counters counts() {
    try {
      return ras::read(d.uuid).since_load;
    } catch (const std::exception&) {
      return {};
    }
  }

  uint32_t bar_value(uint32_t n) {
    const BarLayout& b = bars[n];
    State* s = state();
    const bool set = __atomic_load_n(&s->bar_set[n], __ATOMIC_ACQUIRE);
    const uint64_t programmed = __atomic_load_n(&s->bar_value[n], __ATOMIC_RELAXED);
    if (b.high) {
      const BarLayout& low = bars[n - 1];
      if (!low.size || !low.is64) return 0;
      const uint32_t v = set ? static_cast<uint32_t>(programmed) : static_cast<uint32_t>(low.base >> 32);
      return v & static_cast<uint32_t>(~((low.size - 1) >> 32));
    }
    if (!b.size) return 0;
    const uint32_t flags = b.io ? 0x1u : (b.is64 ? 0x4u : 0u) | (b.pref ? 0x8u : 0u);
    const uint32_t addr = set ? static_cast<uint32_t>(programmed) : static_cast<uint32_t>(b.base);
    const uint32_t keep = static_cast<uint32_t>(~(b.size - 1)) & (b.io ? ~0x3u : ~0xFu);
    return (addr & keep) | flags;
  }

  uint32_t backed(const Register& r, const ras::Counters& c) {
    const std::string& k = r.backing;
    const uint32_t vendor = d.pci_device_id & 0xFFFF, device = d.pci_device_id >> 16;
    if (k == "profile.vendor_id" || k == "profile.subsystem_vendor_id") return vendor;
    if (k == "profile.device_id" || k == "profile.subsystem_id") return device;
    if (k == "profile.revision") return is_amd(d) ? 0x00 : 0xa1;
    // As a bound driver leaves it: memory and bus mastering on, INTx off for
    // MSI, and I/O on where there is an I/O BAR (measured: RTX 3080 Ti, 0x0407).
    if (k == "profile.command") return is_geforce(d) ? 0x0407 : 0x0406;
    // A GeForce is function 0 of two, the second its HDMI audio controller
    // (measured: RTX 3080 Ti, 0x80).
    if (k == "profile.header_type") return is_geforce(d) ? 0x80 : 0x00;
    // AMD's Instinct cards are processing accelerators; NVIDIA's data-center
    // cards 3D controllers; a GeForce a VGA controller.
    if (k == "profile.class") return is_amd(d) ? 0x120000 : is_geforce(d) ? 0x030000 : 0x030200;
    if (k.rfind("bar.", 0) == 0) return bar_value(static_cast<uint32_t>(k[4] - '0'));
    if (k == "link.capabilities")
      // Max speed and width, ASPM L1 supported, L1 exit latency under 32 us.
      return (d.pcie_gen_max & 0xF) | ((d.pcie_width_max & 0x3F) << 4) | (2u << 10) | (6u << 15);
    if (k == "link.status") {
      // Current speed and width, and the slot's reference clock. A GeForce
      // drops its link to Gen1 while idle and retrains when work arrives
      // (measured: RTX 3080 Ti, 2.5 GT/s idle of 16 GT/s).
      const uint32_t gen = is_geforce(d) && d.utilization_gpu == 0 && d.pcie_gen ? 1 : d.pcie_gen;
      return (gen & 0xF) | ((d.pcie_width & 0x3F) << 4) | (1u << 12);
    }
    if (k == "link.capabilities2")
      return d.pcie_gen_max ? (((1u << d.pcie_gen_max) - 1) << 1) : 0;
    if (k == "link.control2") return d.pcie_gen_max & 0xF;
    // Engine status: the units a busy GPU has working, and the idle state's
    // command FIFOs available and DB and CB clean. Values are a model.
    const bool busy = d.utilization_gpu > 0;
    if (k == "engine.grbm_status")
      return busy ? 0xE7D84008u   // GUI active, CP, CB, DB, PA, SC, BCI, SPI, SX, IA, TA busy
                  : 0x00003028u;  // FIFOs available, DB and CB clean
    if (k == "engine.grbm_status2") return busy ? 0x73018008u : 0x00000008u;
    if (k == "engine.cp_stat") return busy ? 0x80000000u : 0u;
    if (k == "engine.rlc_stat") return busy ? 0x00000005u : 0u;
    State* s = state();
    if (k == "smu.message") return static_cast<uint32_t>(__atomic_load_n(&s->mailbox[0], __ATOMIC_RELAXED));
    if (k == "smu.argument") return static_cast<uint32_t>(__atomic_load_n(&s->mailbox[1], __ATOMIC_RELAXED));
    if (k == "smu.response")
      // Ready after reset, as the SMU is once the driver has loaded.
      return __atomic_load_n(&s->mailbox[3], __ATOMIC_ACQUIRE)
                 ? static_cast<uint32_t>(__atomic_load_n(&s->mailbox[2], __ATOMIC_RELAXED))
                 : kSmuResultOk;
    if (k == "aer.correctable" || k == "aer.uncorrectable" || k == "devsta") {
      uint32_t v = 0;
      for (uint32_t bit = 0; bit < (k == "devsta" ? 3u : 32u); ++bit) {
        uint64_t n, at;
        if (k == "aer.correctable") {
          n = correctable_count(c, bit);
          at = __atomic_load_n(&s->base_correctable[bit], __ATOMIC_RELAXED);
        } else if (k == "aer.uncorrectable") {
          n = uncorrectable_count(c, bit);
          at = __atomic_load_n(&s->base_uncorrectable[bit], __ATOMIC_RELAXED);
        } else {
          n = devsta_count(c, bit);
          at = __atomic_load_n(&s->base_devsta[bit], __ATOMIC_RELAXED);
        }
        if (raised(n, at)) v |= 1u << bit;
      }
      return v;
    }
    throw Error::make(Err::Internal, "register ", r.name, ": unknown backing '", k, "'");
  }

  const std::vector<Register>& regs() const { return registers(space); }

  uint32_t evaluate(const Register& r, const ras::Counters& c) {
    const size_t i = static_cast<size_t>(&r - regs().data());
    State* s = state();
    switch (r.access) {
      case Access::Rw:
        // The SMU mailbox is live; any other backing is the reset value, which
        // follows the device until something writes the register.
        if (r.backing.rfind("smu.", 0) == 0) return backed(r, c);
        return __atomic_load_n(&s->rw_set[i], __ATOMIC_ACQUIRE)
                   ? static_cast<uint32_t>(__atomic_load_n(&s->rw_value[i], __ATOMIC_RELAXED))
                   : r.backing.empty() ? r.reset : backed(r, c);
      case Access::Ro:
        return r.backing.empty() ? r.reset : backed(r, c);
      case Access::Rw1c:
      case Access::Bar:
        return backed(r, c);
    }
    return 0;
  }

  void fill(uint8_t* out, uint32_t len) {
    std::memset(out, 0, len);
    const ras::Counters c = counts();
    for (const Register& r : regs()) {
      if (r.offset >= len) break;
      const uint32_t v = evaluate(r, c);
      for (uint32_t b = 0; b < r.width / 8 && r.offset + b < len; ++b)
        out[r.offset + b] = static_cast<uint8_t>(v >> (8 * b));
    }
  }

  void log(bool write, uint32_t offset, uint32_t size, uint32_t value) {
    State* s = state();
    const uint64_t seq = __atomic_add_fetch(&s->log_seq, 1, __ATOMIC_ACQ_REL);
    uint64_t* e = s->log[(seq - 1) % kLogEntries];
    uint64_t name[2] = {0, 0};
    const std::string p = process_name();
    std::memcpy(name, p.data(), std::min<size_t>(p.size(), sizeof name - 1));
    __atomic_store_n(&e[1], now_ns(), __ATOMIC_RELAXED);
    __atomic_store_n(&e[2], static_cast<uint64_t>(::getpid()), __ATOMIC_RELAXED);
    __atomic_store_n(&e[3], (uint64_t{write} << 63) | (uint64_t{size} << 32) | offset, __ATOMIC_RELAXED);
    __atomic_store_n(&e[4], value, __ATOMIC_RELAXED);
    __atomic_store_n(&e[5], name[0], __ATOMIC_RELAXED);
    __atomic_store_n(&e[6], name[1], __ATOMIC_RELAXED);
    __atomic_store_n(&e[0], seq, __ATOMIC_RELEASE);   // published last
  }
};

namespace {
void check_access(Space space, uint32_t offset, uint32_t size) {
  if (space == Space::AmdMmio) {
    if (size != 4 || offset % 4 || offset + size > space_size(space))
      throw std::invalid_argument("an MMIO access is 4 bytes, aligned, inside the 512 KiB BAR");
    return;
  }
  if ((size != 1 && size != 2 && size != 4) || offset % size || offset + size > kConfigSize)
    throw std::invalid_argument("a configuration access is 1, 2 or 4 bytes, naturally aligned, inside "
                                "the 4096-byte space");
}
}  // namespace

RegisterSpace::RegisterSpace(Space s, const telemetry::DeviceSample& d) {
  if (!has_space(d, s))
    throw std::invalid_argument(std::string(d.name) + " has no " + space_name(s) + " registers modelled");
  impl_ = std::make_unique<Impl>(s, d);
}
RegisterSpace::~RegisterSpace() = default;
Space RegisterSpace::space() const { return impl_->space; }

uint32_t RegisterSpace::value(const Register& r) { return impl_->evaluate(r, impl_->counts()); }

uint32_t RegisterSpace::read(uint32_t offset, uint32_t size) {
  check_access(impl_->space, offset, size);
  // The registers the access covers; bytes no register declares read as 0.
  const ras::Counters c = impl_->counts();
  uint32_t v = 0;
  for (const Register& r : impl_->regs()) {
    const uint32_t rbytes = r.width / 8;
    if (r.offset + rbytes <= offset || r.offset >= offset + size) continue;
    const uint32_t rv = impl_->evaluate(r, c);
    for (uint32_t b = 0; b < rbytes; ++b) {
      const uint32_t at = r.offset + b;
      if (at >= offset && at < offset + size) v |= ((rv >> (8 * b)) & 0xFFu) << (8 * (at - offset));
    }
  }
  impl_->log(false, offset, size, v);
  return v;
}

std::vector<uint8_t> RegisterSpace::image_unlogged(uint32_t len) {
  len = std::min(len, space_size(impl_->space));
  std::vector<uint8_t> out(len);
  impl_->fill(out.data(), len);
  return out;
}

std::vector<uint8_t> RegisterSpace::image(uint32_t len) {
  len = std::min(len, space_size(impl_->space));
  std::vector<uint8_t> out(len);
  impl_->fill(out.data(), len);
  impl_->log(false, 0, len, 0);
  return out;
}

void RegisterSpace::write(uint32_t offset, uint32_t size, uint32_t value) {
  check_access(impl_->space, offset, size);
  impl_->log(true, offset, size, value);
  State* s = impl_->state();
  const ras::Counters c = impl_->counts();
  const auto& regs = impl_->regs();
  for (size_t i = 0; i < regs.size(); ++i) {
    const Register& r = regs[i];
    const uint32_t rbytes = r.width / 8;
    if (r.offset + rbytes <= offset || r.offset >= offset + size) continue;
    // The bytes of this register the write covers, and what it writes there.
    uint32_t covered = 0, bits = 0;
    for (uint32_t b = 0; b < rbytes; ++b) {
      const uint32_t at = r.offset + b;
      if (at < offset || at >= offset + size) continue;
      covered |= 0xFFu << (8 * b);
      bits |= ((value >> (8 * (at - offset))) & 0xFFu) << (8 * b);
    }
    switch (r.access) {
      case Access::Ro:
        break;
      case Access::Rw: {
        if (r.backing.rfind("smu.", 0) == 0) {
          smu_write(s, r.backing, (impl_->evaluate(r, c) & ~covered) | (bits & covered));
          break;
        }
        const uint32_t old = impl_->evaluate(r, c);
        const uint32_t mask = r.write_mask & covered;
        __atomic_store_n(&s->rw_value[i], uint64_t{(old & ~mask) | (bits & mask)}, __ATOMIC_RELAXED);
        __atomic_store_n(&s->rw_set[i], uint64_t{1}, __ATOMIC_RELEASE);
        break;
      }
      case Access::Rw1c:
        // Each bit written as 1 is cleared: its count as of now is the mark.
        for (uint32_t bit = 0; bit < 32; ++bit) {
          if (!(bits & (1u << bit))) continue;
          if (r.backing == "aer.correctable")
            __atomic_store_n(&s->base_correctable[bit], correctable_count(c, bit), __ATOMIC_RELAXED);
          else if (r.backing == "aer.uncorrectable")
            __atomic_store_n(&s->base_uncorrectable[bit], uncorrectable_count(c, bit), __ATOMIC_RELAXED);
          else if (r.backing == "devsta" && bit < 4)
            __atomic_store_n(&s->base_devsta[bit], devsta_count(c, bit), __ATOMIC_RELAXED);
        }
        break;
      case Access::Bar: {
        // A BAR keeps whatever was written; reading it back masks off the bits
        // below its size, which is how software sizes it (write all ones, read).
        const uint32_t n = static_cast<uint32_t>(r.backing[4] - '0');
        const uint32_t old = impl_->bar_value(n);
        __atomic_store_n(&s->bar_value[n], uint64_t{(old & ~covered) | (bits & covered)}, __ATOMIC_RELAXED);
        __atomic_store_n(&s->bar_set[n], uint64_t{1}, __ATOMIC_RELEASE);
        break;
      }
    }
  }
  ras::publish_session(impl_->d.uuid);   // the session's config file, rewritten
}

std::vector<LogEntry> access_log(const std::string& uuid, Space space) {
  std::vector<LogEntry> out;
  SharedState file(state_path(uuid, space), false, kMagic, database_version(space), sizeof(State),
                   "register state");
  const State* s = static_cast<const State*>(file.payload());
  if (!s) return out;
  const uint64_t head = __atomic_load_n(&s->log_seq, __ATOMIC_ACQUIRE);
  const uint64_t first = head > kLogEntries ? head - kLogEntries + 1 : 1;
  for (uint64_t seq = first; seq <= head; ++seq) {
    const uint64_t* e = s->log[(seq - 1) % kLogEntries];
    if (__atomic_load_n(&e[0], __ATOMIC_ACQUIRE) != seq) continue;   // overwritten or unfinished
    LogEntry l;
    l.seq = seq;
    l.time_ns = e[1];
    l.pid = static_cast<uint32_t>(e[2]);
    l.write = e[3] >> 63;
    l.size = static_cast<uint32_t>((e[3] >> 32) & 0x7FFFFFFF);
    l.offset = static_cast<uint32_t>(e[3]);
    l.value = static_cast<uint32_t>(e[4]);
    char name[17] = {0};
    std::memcpy(name, &e[5], 8);
    std::memcpy(name + 8, &e[6], 8);
    l.process = name;
    out.push_back(l);
  }
  return out;
}

Link link(RegisterSpace& cs) {
  const uint32_t cap = cs.read(0x84, 4);
  const uint32_t sta = cs.read(0x8a, 2);
  Link l;
  l.max_gen = cap & 0xF;
  l.max_width = (cap >> 4) & 0x3F;
  l.gen = sta & 0xF;
  l.width = (sta >> 4) & 0x3F;
  return l;
}

const char* const kSysfsFiles[12] = {"config", "resource", "vendor", "device", "class",
                                     "subsystem_vendor", "subsystem_device", "revision",
                                     "current_link_speed", "current_link_width", "max_link_speed",
                                     "max_link_width"};

namespace {
// Replaces a file whole, so a reader never sees half of one.
void replace_file(const std::string& path, const std::string& bytes) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return;
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
}

// The kernel's names for the link speeds, by generation.
const char* speed_name(uint32_t gen) {
  static const char* const kNames[] = {"Unknown", "2.5 GT/s PCIe", "5.0 GT/s PCIe", "8.0 GT/s PCIe",
                                       "16.0 GT/s PCIe", "32.0 GT/s PCIe", "64.0 GT/s PCIe"};
  return gen < 7 ? kNames[gen] : kNames[0];
}

std::string hex(uint32_t v, int digits) {
  char b[16];
  std::snprintf(b, sizeof b, "0x%0*x\n", digits, v);
  return b;
}
}  // namespace

void write_sysfs_files(const telemetry::DeviceSample& d, const std::string& dir) {
  ConfigSpace cs(d);
  const std::vector<uint8_t> cfg = cs.image_unlogged(kConfigSize);
  const auto word = [&](uint32_t off) { return static_cast<uint32_t>(cfg[off] | cfg[off + 1] << 8); };
  const auto dword = [&](uint32_t off) { return word(off) | word(off + 2) << 16; };
  replace_file(dir + "/config", std::string(cfg.begin(), cfg.end()));
  replace_file(dir + "/vendor", hex(word(0x00), 4));
  replace_file(dir + "/device", hex(word(0x02), 4));
  replace_file(dir + "/class", hex(dword(0x08) >> 8, 6));
  replace_file(dir + "/revision", hex(cfg[0x08], 2));
  replace_file(dir + "/subsystem_vendor", hex(word(0x2c), 4));
  replace_file(dir + "/subsystem_device", hex(word(0x2e), 4));
  const uint32_t cap = dword(0x84), sta = word(0x8a);
  replace_file(dir + "/current_link_speed", std::string(speed_name(sta & 0xF)) + "\n");
  replace_file(dir + "/current_link_width", std::to_string((sta >> 4) & 0x3F) + "\n");
  replace_file(dir + "/max_link_speed", std::string(speed_name(cap & 0xF)) + "\n");
  replace_file(dir + "/max_link_width", std::to_string((cap >> 4) & 0x3F) + "\n");
  // resource: start, end and flags of each BAR, then the expansion ROM and the
  // SR-IOV BARs, as the kernel prints them; an unimplemented one is all zeros.
  std::string res;
  for (int n = 0; n < 13; ++n) {
    uint64_t start = 0, end = 0, flags = 0;
    if (n < 6) {
      const Bar b = bar(cs, d, n);
      if (b.size) {
        start = b.base;
        end = b.base + b.size - 1;
        // IORESOURCE_IO or _MEM, _PREFETCH, _MEM_64 and _SIZEALIGN, with the
        // BAR's own low bits, as the kernel keeps them.
        flags = b.io ? 0x40101 : 0x40200 | (b.prefetchable ? 0x2000 : 0) | (b.is64 ? 0x100000 : 0) |
                                     (b.prefetchable ? 0x8 : 0) | (b.is64 ? 0x4 : 0);
      }
    }
    char line[80];
    std::snprintf(line, sizeof line, "0x%016llx 0x%016llx 0x%016llx\n", static_cast<unsigned long long>(start),
                  static_cast<unsigned long long>(end), static_cast<unsigned long long>(flags));
    res += line;
  }
  replace_file(dir + "/resource", res);
}

Bar bar(RegisterSpace& cs, const telemetry::DeviceSample& d, int n) {
  const auto layout = bar_layout(d);
  Bar b;
  if (n < 0 || n > 5) return b;
  const BarLayout& l = layout[static_cast<size_t>(n)];
  if (l.high || !l.size) return b;
  const Register* reg = find_config("bar" + std::to_string(n));
  const uint32_t lo = cs.value(*reg);
  uint64_t base = lo & (l.io ? ~0x3u : ~0xFu);
  if (l.is64) base |= uint64_t{cs.value(*find_config("bar" + std::to_string(n + 1)))} << 32;
  b.base = base;
  b.size = l.size;
  b.io = l.io;
  b.is64 = l.is64;
  b.prefetchable = l.pref;
  return b;
}

}  // namespace vgpu::regs
