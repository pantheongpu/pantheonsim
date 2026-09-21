#include "vgpu/ras.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include "vgpu/error.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::ras {
namespace {

constexpr uint32_t kMagic = 0x56524153;  // "VRAS"
constexpr uint32_t kVersion = 5;  // 2: armed faults; 3: hangs and clock-event reasons; 4: events;
                                  // 5: faults armed on stores and shared memory

struct File {
  uint32_t magic;
  uint32_t version;
  uint64_t reserved;
  Counters counters;
};
static_assert(std::is_standard_layout_v<Counters> && sizeof(Counters) % sizeof(uint64_t) == 0,
              "Counters is read and written a word at a time");
constexpr size_t kWords = sizeof(Counters) / sizeof(uint64_t);

std::string volatile_path(const std::string& uuid) {
  return telemetry::default_path() + "/ras-" + uuid + ".volatile";
}
std::string aggregate_path(const std::string& uuid) {
  return state_dir() + "/ras-" + uuid + ".aggregate";
}

// A state file mapped shared, so every process that touches the machine sees
// one set of counts; unmapped when it goes out of scope. Opened without
// `create`, a file that does not exist maps to nothing and reads as zero.
class Mapped {
 public:
  Mapped(const std::string& path, bool create) {
    if (create) {
      std::error_code ec;
      std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    }
    fd_ = ::open(path.c_str(), create ? (O_RDWR | O_CREAT) : O_RDWR, 0600);
    if (fd_ < 0) {
      if (!create && errno == ENOENT) return;
      throw Error::make(Err::Internal, "cannot open reliability state ", path, ": ",
                        std::strerror(errno));
    }
    struct stat st {};
    if (::fstat(fd_, &st) == 0 && st.st_size < static_cast<off_t>(sizeof(File)) &&
        ::ftruncate(fd_, sizeof(File)) != 0)
      throw Error::make(Err::Internal, "cannot size reliability state ", path, ": ",
                        std::strerror(errno));
    void* p = ::mmap(nullptr, sizeof(File), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED)
      throw Error::make(Err::Internal, "cannot map reliability state ", path, ": ",
                        std::strerror(errno));
    file_ = static_cast<File*>(p);
    // A new file is all zeros, and the first process to see it claims it. Two
    // processes creating it at once must not both clear it -- the second would
    // erase the first one's increment -- hence the compare-and-swap. A file
    // from an incompatible build is started again rather than misread.
    uint32_t expected = 0;
    if (__atomic_compare_exchange_n(&file_->magic, &expected, kMagic, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE)) {
      __atomic_store_n(&file_->version, kVersion, __ATOMIC_RELEASE);
    } else if (expected != kMagic || __atomic_load_n(&file_->version, __ATOMIC_ACQUIRE) != kVersion) {
      uint64_t* w = words();
      for (size_t i = 0; i < kWords; ++i) __atomic_store_n(&w[i], 0, __ATOMIC_RELAXED);
      __atomic_store_n(&file_->version, kVersion, __ATOMIC_RELEASE);
      __atomic_store_n(&file_->magic, kMagic, __ATOMIC_RELEASE);
    }
  }
  ~Mapped() {
    if (file_) ::munmap(file_, sizeof(File));
    if (fd_ >= 0) ::close(fd_);
  }
  Mapped(const Mapped&) = delete;
  Mapped& operator=(const Mapped&) = delete;

  Counters* counters() { return file_ ? &file_->counters : nullptr; }
  uint64_t* words() { return file_ ? reinterpret_cast<uint64_t*>(&file_->counters) : nullptr; }

 private:
  int fd_ = -1;
  File* file_ = nullptr;
};

Counters load(const std::string& path) {
  Counters out{};
  Mapped m(path, false);
  if (uint64_t* w = m.words()) {
    uint64_t* dst = reinterpret_cast<uint64_t*>(&out);
    for (size_t i = 0; i < kWords; ++i) dst[i] = __atomic_load_n(&w[i], __ATOMIC_RELAXED);
  }
  return out;
}

void add(uint64_t* field, uint64_t n) { __atomic_fetch_add(field, n, __ATOMIC_RELAXED); }
void set(uint64_t* field, uint64_t v) { __atomic_store_n(field, v, __ATOMIC_RELAXED); }

constexpr const char* kLocationNames[kLocations] = {
    "device_memory", "register_file", "l1_cache", "l2_cache", "texture_memory", "cbu", "sram"};
constexpr const char* kPcieNames[kPcieCounters] = {
    "replay", "replay_rollover", "l0_to_recovery", "correctable", "naks_received", "bad_tlp",
    "naks_sent", "bad_dllp", "non_fatal", "fatal", "lcrc", "lane"};

}  // namespace

uint64_t Counters::ecc_total(Severity s) const {
  uint64_t sum = 0;
  for (uint32_t l = 0; l < kLocations; ++l) sum += ecc[static_cast<uint32_t>(s)][l];
  return sum;
}

State read(const std::string& uuid) {
  State s;
  s.since_load = load(volatile_path(uuid));
  s.lifetime = load(aggregate_path(uuid));
  return s;
}

void inject_ecc(const std::string& uuid, Severity s, Location l, uint64_t n, Retirement scheme) {
  if (n == 0) return;
  const auto si = static_cast<uint32_t>(s), li = static_cast<uint32_t>(l);
  {
    Mapped v(volatile_path(uuid), true);
    add(&v.counters()->ecc[si][li], n);
  }
  Mapped a(aggregate_path(uuid), true);
  Counters* c = a.counters();
  add(&c->ecc[si][li], n);
  record_event(uuid, s == Severity::Corrected ? kEventSingleBitEcc : kEventDoubleBitEcc, 0);
  // An uncorrectable error in device memory takes that memory out of service.
  // Corrected errors do not here: a real card retires a page only after
  // several at the same address, and injection does not model addresses yet.
  if (s == Severity::Uncorrected && l == Location::DeviceMemory) {
    if (scheme == Retirement::Pages) {
      add(&c->retired_dbe, n);
      set(&c->retired_pending, 1);
    } else if (scheme == Retirement::Rows) {
      add(&c->rows_uncorrectable, n);
      set(&c->rows_pending, 1);
    }
  }
}

void inject_pcie(const std::string& uuid, Pcie c, uint64_t n) {
  if (n == 0) return;
  Mapped v(volatile_path(uuid), true);
  add(&v.counters()->pcie[static_cast<uint32_t>(c)], n);
}

void reset_volatile(const std::string& uuid, bool driver_reload) {
  {
    Mapped v(volatile_path(uuid), false);
    if (uint64_t* w = v.words())
      for (size_t i = 0; i < kWords; ++i) set(&w[i], 0);
  }
  if (!driver_reload) return;
  // Pending retirements and remaps take effect at a driver load.
  Mapped a(aggregate_path(uuid), false);
  if (Counters* c = a.counters()) {
    set(&c->retired_pending, 0);
    set(&c->rows_pending, 0);
  }
}

void reset_aggregate(const std::string& uuid) {
  Mapped a(aggregate_path(uuid), false);
  if (Counters* c = a.counters())
    for (uint32_t s = 0; s < kSeverities; ++s)
      for (uint32_t l = 0; l < kLocations; ++l) set(&c->ecc[s][l], 0);
}

const char* location_name(Location l) { return kLocationNames[static_cast<uint32_t>(l)]; }

bool parse_location(const std::string& s, Location* out) {
  if (s == "dram") {
    *out = Location::DeviceMemory;
    return true;
  }
  for (uint32_t i = 0; i < kLocations; ++i)
    if (s == kLocationNames[i]) {
      *out = static_cast<Location>(i);
      return true;
    }
  return false;
}

const char* pcie_name(Pcie c) { return kPcieNames[static_cast<uint32_t>(c)]; }

bool parse_pcie(const std::string& s, Pcie* out) {
  for (uint32_t i = 0; i < kPcieCounters; ++i)
    if (s == kPcieNames[i]) {
      *out = static_cast<Pcie>(i);
      return true;
    }
  return false;
}

std::string state_dir() {
  if (const char* d = std::getenv("VGPU_STATE_DIR"); d && *d) return d;
  if (const char* x = std::getenv("XDG_STATE_HOME"); x && *x) return std::string(x) + "/vgpu";
  if (const char* h = std::getenv("HOME"); h && *h) return std::string(h) + "/.local/state/vgpu";
  return "/tmp/vgpu-state-" + std::to_string(::getuid());
}

// ---- Faults delivered to a running kernel -----------------------------------

namespace {
// The slot of a fault kind in Counters::armed.
uint32_t armed_slot(Armed kind) {
  return kind == Armed::Corrected ? 0 : kind == Armed::Uncorrected ? 1 : 2;
}
constexpr const char* kTargetNames[kTargets] = {"load", "store", "shared"};
}  // namespace

const char* target_name(Target t) { return kTargetNames[static_cast<uint32_t>(t)]; }

bool parse_target(const std::string& s, Target* out) {
  for (uint32_t i = 0; i < kTargets; ++i)
    if (s == kTargetNames[i]) {
      *out = static_cast<Target>(i);
      return true;
    }
  return false;
}

uint64_t Counters::armed_total() const {
  uint64_t n = 0;
  for (uint64_t p : armed_pending) n += p;
  return n;
}

void arm(const std::string& uuid, Armed kind, uint64_t n, Target at) {
  if (n == 0 || kind == Armed::None || kind == Armed::Hang) return;
  if (at == Target::Store && kind != Armed::Bitflip)
    throw std::invalid_argument("an ECC error is found when memory is read, not written");
  Mapped v(volatile_path(uuid), true);
  Counters* c = v.counters();
  const uint32_t t = static_cast<uint32_t>(at);
  add(&c->armed[t][armed_slot(kind)], n);
  add(&c->armed_pending[t], n);
}

struct ArmedFaults::Impl {
  explicit Impl(const std::string& uuid) : file(volatile_path(uuid), true) {}
  Mapped file;
};

ArmedFaults::ArmedFaults(const std::string& uuid) : impl_(std::make_unique<Impl>(uuid)) {}
ArmedFaults::~ArmedFaults() = default;

const uint64_t* ArmedFaults::pending(Target at) const {
  return &impl_->file.counters()->armed_pending[static_cast<uint32_t>(at)];
}

Armed ArmedFaults::take(Target at) {
  Counters* c = impl_->file.counters();
  uint64_t* armed = c->armed[static_cast<uint32_t>(at)];
  // Decrements a counter only if it is above zero, so two threads taking the
  // last one cannot both have it.
  auto take_one = [](uint64_t* word) {
    uint64_t have = __atomic_load_n(word, __ATOMIC_RELAXED);
    while (have > 0)
      if (__atomic_compare_exchange_n(word, &have, have - 1, false, __ATOMIC_ACQ_REL,
                                      __ATOMIC_RELAXED))
        return true;
    return false;
  };
  Armed got = Armed::None;
  if (take_one(&armed[armed_slot(Armed::Uncorrected)])) got = Armed::Uncorrected;
  else if (take_one(&armed[armed_slot(Armed::Bitflip)])) got = Armed::Bitflip;
  else if (take_one(&armed[armed_slot(Armed::Corrected)])) got = Armed::Corrected;
  if (got != Armed::None) take_one(&c->armed_pending[static_cast<uint32_t>(at)]);
  return got;
}

void ArmedFaults::note_bitflip() { add(&impl_->file.counters()->bitflips_delivered, 1); }

// ---- The kernel log ---------------------------------------------------------

void log_kernel(const std::string& message) {
  const char* session = std::getenv("VGPU_SESSION");
  if (!session || !*session) return;
  const std::string path = std::string(session) + "/dmesg.log";
  // After the last stamp in the file, by as long as it has been since the file
  // was last written: the boot lines are stamped seconds after "boot", and an
  // event must not appear before them.
  double last = 0.0;
  if (std::ifstream in{path, std::ios::ate}) {
    const std::streamoff size = in.tellg();
    in.seekg(size > 4096 ? size - 4096 : 0);
    std::string line;
    while (std::getline(in, line)) {
      double t = 0;
      if (std::sscanf(line.c_str(), "[%lf]", &t) == 1) last = t;
    }
  }
  double since = 0.0;
  std::error_code ec;
  const auto mtime = std::filesystem::last_write_time(path, ec);
  if (!ec)
    since = std::chrono::duration<double>(std::filesystem::file_time_type::clock::now() - mtime).count();
  char line[1024];
  std::snprintf(line, sizeof line, "[%12.6f] %s\n", last + (since > 1e-6 ? since : 1e-6), message.c_str());
  if (std::FILE* f = std::fopen(path.c_str(), "a")) {
    std::fputs(line, f);
    std::fclose(f);
  }
}

namespace {
// "00000000:01:00.0" -> "0000:01:00.0": sysfs and the kernel log use a
// four-digit domain where nvidia-smi uses eight.
std::string kernel_bdf(const std::string& bus_id) {
  unsigned domain = 0, bus = 0, dev = 0, fn = 0;
  std::sscanf(bus_id.c_str(), "%x:%x:%x.%x", &domain, &bus, &dev, &fn);
  char out[32];
  std::snprintf(out, sizeof out, "%04x:%02x:%02x.%x", domain, bus, dev, fn);
  return out;
}
}  // namespace

std::string xid_line(const std::string& bus_id, int xid, const std::string& process,
                     const std::string& detail) {
  // The driver names the device by domain, bus and slot, without the function.
  const std::string bdf = kernel_bdf(bus_id);
  char out[1024];
  std::snprintf(out, sizeof out, "NVRM: Xid (PCI:%s): %d, %s, %s", bdf.substr(0, bdf.rfind('.')).c_str(),
                xid, process.empty() ? "pid='<unknown>', name=<unknown>" : process.c_str(),
                detail.c_str());
  return out;
}

std::string aer_line(const std::string& bus_id, Pcie c) {
  const char* kind = nullptr;
  switch (c) {
    case Pcie::Correctable: case Pcie::Replay: case Pcie::ReplayRollover:
    case Pcie::BadTlp: case Pcie::BadDllp: case Pcie::Lcrc:
      kind = "Corrected error received";
      break;
    case Pcie::NonFatal: kind = "Uncorrected (Non-Fatal) error received"; break;
    case Pcie::Fatal: kind = "Uncorrected (Fatal) error received"; break;
    default: return "";
  }
  return std::string("pcieport 0000:00:01.0: AER: ") + kind + ": " + kernel_bdf(bus_id);
}

// ---- Hangs --------------------------------------------------------------------

void arm_hang(const std::string& uuid, uint64_t seconds) {
  Mapped v(volatile_path(uuid), true);
  Counters* c = v.counters();
  set(&c->hang_seconds, seconds);
  add(&c->armed_hang, 1);
}

bool ArmedFaults::take_hang(uint64_t* seconds) {
  Counters* c = impl_->file.counters();
  uint64_t have = __atomic_load_n(&c->armed_hang, __ATOMIC_RELAXED);
  while (have > 0)
    if (__atomic_compare_exchange_n(&c->armed_hang, &have, have - 1, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_RELAXED)) {
      *seconds = __atomic_load_n(&c->hang_seconds, __ATOMIC_RELAXED);
      return true;
    }
  return false;
}

// ---- Clock-event reasons ------------------------------------------------------

namespace {

// Where each reason's time accumulates, by bit position.
int reason_slot(uint64_t bit) {
  for (int i = 0; i < 8; ++i)
    if (bit == (uint64_t{1} << i)) return i;
  return -1;
}

uint64_t now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}

// The end of the current window, as of `now`: its expiry if that has passed.
uint64_t window_end(const Counters& c, uint64_t now) {
  return c.throttle_until_ns && c.throttle_until_ns < now ? c.throttle_until_ns : now;
}

// Moves the current window's time into the counters and ends it.
void close_window(Counters* c) {
  const uint64_t mask = __atomic_load_n(&c->throttle_mask, __ATOMIC_RELAXED);
  if (mask) {
    const uint64_t end = window_end(*c, now_ns());
    const uint64_t since = __atomic_load_n(&c->throttle_since_ns, __ATOMIC_RELAXED);
    const uint64_t us = end > since ? (end - since) / 1000 : 0;
    for (int i = 0; i < 8; ++i)
      if (mask & (uint64_t{1} << i)) add(&c->throttle_us[i], us);
  }
  set(&c->throttle_mask, 0);
  set(&c->throttle_since_ns, 0);
  set(&c->throttle_until_ns, 0);
}

}  // namespace

uint64_t reason_bit(const std::string& name) {
  if (name == "sw_power_cap") return kSwPowerCap;
  if (name == "hw_slowdown") return kHwSlowdown;
  if (name == "sw_thermal_slowdown") return kSwThermalSlowdown;
  if (name == "hw_thermal_slowdown") return kHwThermalSlowdown;
  if (name == "hw_power_brake_slowdown") return kHwPowerBrakeSlowdown;
  return 0;
}

void throttle(const std::string& uuid, uint64_t reasons, uint64_t seconds) {
  Mapped v(volatile_path(uuid), true);
  Counters* c = v.counters();
  close_window(c);
  const uint64_t now = now_ns();
  set(&c->throttle_since_ns, now);
  set(&c->throttle_until_ns, seconds ? now + seconds * 1000000000ull : 0);
  set(&c->throttle_mask, reasons);
}

void clear_throttle(const std::string& uuid) {
  Mapped v(volatile_path(uuid), false);
  if (Counters* c = v.counters()) close_window(c);
}

uint64_t apply_throttle(telemetry::DeviceSample& d) {
  const Counters c = read(d.uuid).since_load;
  const uint64_t now = now_ns();
  const bool live = c.throttle_mask && (!c.throttle_until_ns || now < c.throttle_until_ns);
  const uint64_t active = live ? c.throttle_mask : 0;
  d.clock_event_reasons = active;
  if (!active) return 0;
  if (active & (kSwThermalSlowdown | kHwThermalSlowdown)) {
    const uint32_t slowdown = d.temperature_max_c ? d.temperature_max_c : 85;
    if (d.temperature_c < slowdown) d.temperature_c = slowdown;
  }
  if ((active & kSwPowerCap) && d.power_limit_mw) d.power_mw = d.power_limit_mw;
  double factor = 1.0;
  if (active & (kHwSlowdown | kHwThermalSlowdown | kHwPowerBrakeSlowdown)) factor = 0.5;
  else if (active & (kSwThermalSlowdown | kSwPowerCap)) factor = 0.75;
  const auto cap = static_cast<uint32_t>(d.sm_clock_max_mhz * factor);
  if (d.sm_clock_mhz > cap) d.sm_clock_mhz = cap;
  return active;
}

uint64_t throttle_time_us(const std::string& uuid, uint64_t reason) {
  const int slot = reason_slot(reason);
  if (slot < 0) return 0;
  const Counters c = read(uuid).since_load;
  uint64_t us = c.throttle_us[slot];
  if (c.throttle_mask & reason) {
    const uint64_t end = window_end(c, now_ns());
    if (end > c.throttle_since_ns) us += (end - c.throttle_since_ns) / 1000;
  }
  return us;
}

// ---- NVML events --------------------------------------------------------------

void record_event(const std::string& uuid, uint64_t type, uint64_t data) {
  Mapped v(volatile_path(uuid), true);
  Counters* c = v.counters();
  const uint64_t seq = __atomic_add_fetch(&c->event_seq, 1, __ATOMIC_ACQ_REL);
  uint64_t* e = c->events[(seq - 1) % kEvents];
  set(&e[1], type);
  set(&e[2], data);
  set(&e[3], now_ns());
  // Published last, so a reader that sees this sequence sees the event too.
  __atomic_store_n(&e[0], seq, __ATOMIC_RELEASE);
}

uint64_t event_head(const std::string& uuid) {
  Mapped v(volatile_path(uuid), false);
  const Counters* c = v.counters();
  return c ? __atomic_load_n(&c->event_seq, __ATOMIC_ACQUIRE) : 0;
}

bool next_event(const std::string& uuid, uint64_t mask, uint64_t* after, Event* out) {
  Mapped v(volatile_path(uuid), false);
  Counters* c = v.counters();
  if (!c) return false;
  const uint64_t head = __atomic_load_n(&c->event_seq, __ATOMIC_ACQUIRE);
  if (head < *after) *after = 0;                                    // the ring was reset
  if (head > kEvents && *after < head - kEvents) *after = head - kEvents;  // overwritten
  while (*after < head) {
    const uint64_t want = *after + 1;
    uint64_t* e = c->events[(want - 1) % kEvents];
    const uint64_t seq = __atomic_load_n(&e[0], __ATOMIC_ACQUIRE);
    if (seq < want) return false;   // claimed but not yet published: next time
    *after = want;
    if (seq > want) continue;       // overwritten while we looked
    const Event ev{seq, __atomic_load_n(&e[1], __ATOMIC_RELAXED),
                   __atomic_load_n(&e[2], __ATOMIC_RELAXED), __atomic_load_n(&e[3], __ATOMIC_RELAXED)};
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (__atomic_load_n(&e[0], __ATOMIC_RELAXED) != want) continue;   // overwritten as we read it
    if (ev.type & mask) {
      *out = ev;
      return true;
    }
  }
  return false;
}

void report_xid(const std::string& uuid, const std::string& bus_id, int xid,
                const std::string& process, const std::string& detail) {
  log_kernel(xid_line(bus_id, xid, process, detail));
  record_event(uuid, kEventXid, static_cast<uint64_t>(xid));
}

}  // namespace vgpu::ras
