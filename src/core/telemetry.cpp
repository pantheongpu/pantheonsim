#include "vgpu/telemetry.hpp"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vgpu::telemetry {
namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool disabled() {
  const char* t = std::getenv("VGPU_TELEMETRY");
  return t && t[0] == '0';
}

constexpr double kWindowSeconds = 0.5;  // telemetry integration window

}  // namespace

std::string default_path() {
  if (const char* p = std::getenv("VGPU_TELEMETRY_PATH"); p && p[0]) return p;
  if (const char* r = std::getenv("XDG_RUNTIME_DIR"); r && r[0])
    return std::string(r) + "/vgpu-telemetry.d";
  return "/tmp/vgpu-telemetry-" + std::to_string(getuid()) + ".d";
}

Publisher::Publisher() {
  if (disabled()) return;
  size_ = sizeof(Shared);
  std::string dir = default_path();
  ::mkdir(dir.c_str(), 0700);
  // One segment per publisher instance -- keyed by pid *and* a process-local
  // sequence, because a single process can hold more than one Runtime (the
  // driver-API and runtime-API shims each create their own).
  static std::atomic<unsigned> seq{0};
  std::string path = dir + "/pub-" + std::to_string(getpid()) + "-" +
                     std::to_string(seq.fetch_add(1));
  path_ = path;
  fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (fd_ < 0) return;
  if (::ftruncate(fd_, static_cast<off_t>(size_)) != 0) {
    ::close(fd_);
    fd_ = -1;
    return;
  }
  void* p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    return;
  }
  shared_ = static_cast<Shared*>(p);
  std::memset(shared_, 0, size_);
  shared_->magic = kMagic;
  shared_->version = kVersion;
  shared_->writer_pid = static_cast<uint32_t>(getpid());
  shared_->update_ns = now_ns();
  // The real nvidia-smi refuses to run against an NVML whose version it does
  // not recognise, so both strings are overridable to match a given install.
  const char* drv = std::getenv("VGPU_DRIVER_VERSION");
  const char* cuda = std::getenv("VGPU_CUDA_VERSION");
  std::snprintf(shared_->driver_version, sizeof shared_->driver_version, "%s",
                drv && drv[0] ? drv : "580.00.00");
  std::snprintf(shared_->cuda_version, sizeof shared_->cuda_version, "%s",
                cuda && cuda[0] ? cuda : "13.0");
}

Publisher::~Publisher() {
  if (shared_) {
    shared_->device_count = 0;
    ::munmap(shared_, size_);
  }
  if (fd_ >= 0) ::close(fd_);
  // Remove this process's segment so it cannot linger as phantom hardware.
  if (!path_.empty()) ::unlink(path_.c_str());
}

void Publisher::begin_update() {
  if (shared_) __atomic_add_fetch(&shared_->update_seq, 1, __ATOMIC_ACQ_REL);
}
void Publisher::end_update() {
  if (shared_) {
    shared_->update_ns = now_ns();
    __atomic_add_fetch(&shared_->update_seq, 1, __ATOMIC_ACQ_REL);
  }
}

void Publisher::set_device_count(uint32_t n) {
  if (!shared_) return;
  begin_update();
  shared_->device_count = std::min<uint32_t>(n, kMaxDevices);
  end_update();
}

DeviceSample* Publisher::device(uint32_t ordinal) {
  if (!shared_ || ordinal >= kMaxDevices) return nullptr;
  return &shared_->devices[ordinal];
}

void Publisher::set_driver_version(const std::string& driver, const std::string& cuda) {
  if (!shared_) return;
  std::snprintf(shared_->driver_version, sizeof shared_->driver_version, "%s", driver.c_str());
  std::snprintf(shared_->cuda_version, sizeof shared_->cuda_version, "%s", cuda.c_str());
}

void Publisher::note_memory(uint32_t ordinal, uint64_t used_bytes) {
  DeviceSample* d = device(ordinal);
  if (!d) return;
  begin_update();
  d->vram_used_bytes = used_bytes;
  end_update();
  refresh(ordinal);
}

void Publisher::note_kernel(uint32_t ordinal, double seconds_busy) {
  DeviceSample* d = device(ordinal);
  if (!d) return;
  Accum& a = accum_[ordinal];
  a.busy_seconds += seconds_busy;
  begin_update();
  ++d->kernels_launched;
  end_update();
  refresh(ordinal);
}

void Publisher::note_transfer(uint32_t ordinal, uint64_t bytes, double seconds_busy) {
  DeviceSample* d = device(ordinal);
  if (!d) return;
  Accum& a = accum_[ordinal];
  a.busy_seconds += seconds_busy;
  a.mem_busy_seconds += seconds_busy;
  begin_update();
  d->bytes_moved += bytes;
  end_update();
  refresh(ordinal);
}

void Publisher::refresh(uint32_t ordinal) {
  DeviceSample* d = device(ordinal);
  if (!d) return;
  Accum& a = accum_[ordinal];
  int64_t t = now_ns();
  if (!a.primed) {
    a.window_start_ns = t;
    a.power_w = d->power_limit_mw / 1000.0 * 0.08;  // idle draw
    a.temp_c = 32.0;                                // idle/ambient
    a.primed = true;
    // Publish idle readings straight away: a device that has done no work
    // should look idle, not powered off.
    d->power_mw = static_cast<uint32_t>(a.power_w * 1000.0);
    d->temperature_c = static_cast<uint32_t>(a.temp_c);
    d->sm_clock_mhz = d->sm_clock_max_mhz / 6;
    d->mem_clock_mhz = d->mem_clock_max_mhz / 4;
    d->voltage_mv = 700;
    d->fan_percent = 0;
    d->perf_state = 8;
  }
  double elapsed = static_cast<double>(t - a.window_start_ns) / 1e9;
  if (elapsed < kWindowSeconds) return;  // integrate over the window
  // How many windows this update covers. Applying the lag once per elapsed
  // window keeps the curves correct when updates arrive irregularly (a single
  // long-running kernel, say) instead of freezing the readings.
  double windows = std::max(1.0, elapsed / kWindowSeconds);

  // REAL: fraction of wall time the device spent executing.
  double util = elapsed > 0 ? std::clamp(a.busy_seconds / elapsed, 0.0, 1.0) : 0.0;
  double mem_util = elapsed > 0 ? std::clamp(a.mem_busy_seconds / elapsed, 0.0, 1.0) : 0.0;

  // SYNTHETIC: a first-order model so the derived readings behave like a real
  // card's (rise under load, lag behind it, decay when idle). Not a prediction
  // of any physical device.
  double idle_w = d->power_limit_mw / 1000.0 * 0.08;
  double target_w = idle_w + (d->power_limit_mw / 1000.0 - idle_w) * util;
  a.power_w += (target_w - a.power_w) * (1.0 - std::pow(1.0 - 0.45, windows));  // fast electrical

  // A device that reports no thermal threshold still needs a plausible synthetic
  // curve; 85 C is the common slowdown point and the whole reading is labelled
  // synthetic anyway. Without this the curve ran below room temperature.
  const double t_max = d->temperature_max_c ? static_cast<double>(d->temperature_max_c) : 85.0;
  double target_c = 32.0 + (t_max - 42.0) * util;
  a.temp_c += (target_c - a.temp_c) * (1.0 - std::pow(1.0 - 0.08, windows));  // slow thermal mass

  d->utilization_gpu = static_cast<uint32_t>(std::lround(util * 100.0));
  d->utilization_mem = static_cast<uint32_t>(std::lround(mem_util * 100.0));
  d->power_mw = static_cast<uint32_t>(std::lround(a.power_w * 1000.0));
  d->temperature_c = static_cast<uint32_t>(std::lround(a.temp_c));
  // Clocks and voltage step with load, mimicking a boost curve.
  d->sm_clock_mhz = static_cast<uint32_t>(
      std::lround(d->sm_clock_max_mhz * (util > 0.02 ? (0.55 + 0.45 * util) : 0.15)));
  d->mem_clock_mhz = util > 0.02 ? d->mem_clock_max_mhz : d->mem_clock_max_mhz / 4;
  d->voltage_mv = static_cast<uint32_t>(std::lround(700.0 + 350.0 * util));
  d->fan_percent = static_cast<uint32_t>(
      std::lround(std::clamp((a.temp_c - 30.0) / 50.0, 0.0, 1.0) * 100.0));
  // Performance state: P0 busy .. P8 idle.
  d->perf_state = util > 0.5 ? 0 : util > 0.15 ? 2 : util > 0.02 ? 5 : 8;

  begin_update();
  end_update();
  a.window_start_ns = t;
  a.busy_seconds = 0;
  a.mem_busy_seconds = 0;
}

namespace {

// Reads one publisher's segment. Seqlock-style: retry while a writer is
// mid-update, so a snapshot is never half-applied.
bool read_one(const std::string& path, Shared* out) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  struct stat st {};
  if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < sizeof(Shared)) {
    ::close(fd);
    return false;
  }
  void* p = ::mmap(nullptr, sizeof(Shared), PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) return false;
  const Shared* s = static_cast<const Shared*>(p);
  for (int attempt = 0; attempt < 64; ++attempt) {
    uint64_t before = __atomic_load_n(&s->update_seq, __ATOMIC_ACQUIRE);
    if (before & 1) continue;
    std::memcpy(out, s, sizeof(Shared));
    uint64_t after = __atomic_load_n(&s->update_seq, __ATOMIC_ACQUIRE);
    if (before == after) break;
  }
  ::munmap(p, sizeof(Shared));
  if (out->magic != kMagic || out->version != kVersion) return false;
  // A publisher that died leaves its file behind; clean it up rather than
  // reporting phantom hardware.
  if (out->writer_pid != 0 && ::kill(static_cast<pid_t>(out->writer_pid), 0) != 0 &&
      errno == ESRCH) {
    ::unlink(path.c_str());
    return false;
  }
  return out->device_count > 0;
}

}  // namespace

void describe_device(const DeviceProfile& p, int ordinal, DeviceSample* d) {
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
}

Shared idle_snapshot(const DeviceProfile& p, int device_count) {
  Shared s{};
  s.magic = kMagic;
  s.version = kVersion;
  s.device_count = static_cast<uint32_t>(
      device_count < 1 ? 1 : (device_count > kMaxDevices ? kMaxDevices : device_count));
  for (uint32_t i = 0; i < s.device_count; ++i) {
    describe_device(p, static_cast<int>(i), &s.devices[i]);
    DeviceSample& d = s.devices[i];
    // An idle device: cool, near its floor clock, drawing its idle power.
    d.utilization_gpu = 0;
    d.utilization_mem = 0;
    d.temperature_c = 32;
    d.power_mw = d.power_limit_mw / 12;
    d.voltage_mv = 700;
    d.sm_clock_mhz = d.sm_clock_max_mhz / 5;
    d.mem_clock_mhz = d.mem_clock_max_mhz / 5;
    d.fan_percent = 0;
    d.perf_state = 8;   // P8 is the idle state a real device reports
    d.proc_count = 0;
  }
  return s;
}

bool read_snapshot(Shared* out, const std::string& dir) {
  if (!out) return false;
  DIR* d = ::opendir(dir.c_str());
  if (!d) return false;
  std::vector<Shared> pubs;
  while (struct dirent* e = ::readdir(d)) {
    if (std::strncmp(e->d_name, "pub-", 4) != 0) continue;
    Shared s{};
    if (read_one(dir + "/" + e->d_name, &s)) pubs.push_back(s);
  }
  ::closedir(d);
  if (pubs.empty()) return false;

  // Several processes share one virtual machine, exactly as they share a
  // physical GPU. Identity comes from the publisher holding the most devices
  // (the session, when there is one); memory and counters add up; each
  // contributing process shows up in the per-device process list.
  size_t base = 0;
  for (size_t i = 1; i < pubs.size(); ++i)
    if (pubs[i].device_count > pubs[base].device_count) base = i;
  *out = pubs[base];

  for (uint32_t dev = 0; dev < out->device_count; ++dev) {
    DeviceSample& agg = out->devices[dev];
    agg.vram_used_bytes = 0;
    agg.kernels_launched = 0;
    agg.bytes_moved = 0;
    agg.proc_count = 0;
    uint32_t util = 0, mem_util = 0;
    for (const Shared& p : pubs) {
      if (dev >= p.device_count) continue;
      const DeviceSample& s = p.devices[dev];
      agg.vram_used_bytes += s.vram_used_bytes;
      agg.kernels_launched += s.kernels_launched;
      agg.bytes_moved += s.bytes_moved;
      util = std::max(util, s.utilization_gpu);
      mem_util = std::max(mem_util, s.utilization_mem);
      // Report the busiest publisher's derived readings for this device.
      if (s.utilization_gpu >= util) {
        agg.temperature_c = s.temperature_c;
        agg.power_mw = s.power_mw;
        agg.sm_clock_mhz = s.sm_clock_mhz;
        agg.mem_clock_mhz = s.mem_clock_mhz;
        agg.voltage_mv = s.voltage_mv;
        agg.fan_percent = s.fan_percent;
        agg.perf_state = s.perf_state;
      }
      if (s.vram_used_bytes > 0 && agg.proc_count < kMaxProcs) {
        ProcSample& ps = agg.procs[agg.proc_count++];
        ps.pid = p.writer_pid;
        ps.used_bytes = s.vram_used_bytes;
        std::snprintf(ps.name, sizeof ps.name, "pid %u", p.writer_pid);
      }
    }
    agg.utilization_gpu = util;
    agg.utilization_mem = mem_util;
    if (agg.vram_used_bytes > agg.vram_total_bytes) agg.vram_used_bytes = agg.vram_total_bytes;
  }
  return true;
}

}  // namespace vgpu::telemetry
