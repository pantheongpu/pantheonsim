// Shared machinery of the NVML entry points that live in the nvml_*.inc files
// (included by nvml_api.cpp, in its translation unit, after the helpers it uses).
//
//  * Header tiers. CI builds this against four nvml.h versions, and the later
//    ones declare functions and types the earlier ones lack. Each tier macro is
//    defined when the header it names is in use, tested through a macro that
//    header introduced (the struct-version and size macros are NVML's own
//    feature tests). A function first declared in CUDA 12.8 sits inside
//    #if VGPU_NVML_12_8; an entry point from CUDA 12.0 needs no guard.
//  * The device idioms: NVML_DEV validates the handle the way every other entry
//    point does, NVML_ARG a result pointer.
//  * Setters. The real driver needs root. Here a setter takes effect when the
//    process is root (or VGPU_NVML_ROOT=1 says it is, which is how tests and CI,
//    which are not root, exercise them) and is NVML_ERROR_NO_PERMISSION
//    otherwise -- see setters_allowed().
//  * The settings overlay. What a setter changes lives in a small file per
//    device, keyed by UUID, so a later nvidia-smi or NVML process sees it and
//    the getters read it back. Two lifetimes, like the reliability state
//    (vgpu/ras.hpp): settings the real driver keeps across a driver reload
//    (the pending ECC mode, MIG mode, the compute mode in the inforom) live in
//    the state directory; the rest (power limit, clock locks, persistence mode,
//    accounting) die with the session's runtime directory, as a driver reload
//    would reset them.

#if defined(NVML_CC_GPU_ATTESTATION_REPORT_SIZE)
#define VGPU_NVML_12_4 1
#endif
#if defined(NVML_DEVICE_ARCH_BLACKWELL)
#define VGPU_NVML_12_8 1
#endif
#if defined(NVML_DEVICE_UUID_ASCII_LEN)
#define VGPU_NVML_12_9 1
#endif
#if defined(NVML_DEVICE_HOSTNAME_BUFFER_SIZE)
#define VGPU_NVML_13_0 1
#endif
#if defined(NVML_INIT_FLAG_FORCE_INIT)
#define VGPU_NVML_13_2 1
#endif

#include <fcntl.h>
#include <sched.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "enum_value.hpp"   // an enum a caller may have filled with a value it does not declare

// Validates a device handle and refreshes the snapshot; leaves `d` pointing at
// the device's sample. The lock is held for the rest of the function.
#define NVML_DEV(device)                \
  REQUIRE_INIT();                       \
  refresh();                            \
  const auto* d = sample(device);       \
  if (!d) return bad(device)
#define NVML_ARG(p) \
  if (!(p)) return NVML_ERROR_INVALID_ARGUMENT
// A versioned structure whose version word is not the one this library speaks.
#define NVML_VERSION(p, expected) \
  if ((p)->version != (expected)) return NVML_ERROR_ARGUMENT_VERSION_MISMATCH

// An entry point whose answer is NOT_SUPPORTED on every device we simulate, for
// a reason the comment on it gives. The handle is still validated first, which
// is what the real library does: a bad handle is INVALID_ARGUMENT, not
// "not supported".
#define VGPU_NVML_NS_DEV(name, device, params)       \
  VGPU_EXPORT nvmlReturn_t name params {             \
    NVML_DEV(device);                                \
    return NVML_ERROR_NOT_SUPPORTED;                 \
  }
// The same without a device (system-level calls).
#define VGPU_NVML_NS(name, params)                   \
  VGPU_EXPORT nvmlReturn_t name params {             \
    REQUIRE_INIT();                                  \
    return NVML_ERROR_NOT_SUPPORTED;                 \
  }

namespace {

// ---- profile facts ----------------------------------------------------------

// The profile a device was made from, found by its model name: the telemetry
// sample carries the name, and every built-in profile's is unique. Loaded once.
const vgpu::DeviceProfile* profile_of(const vgpu::telemetry::DeviceSample& d) {
  static std::mutex mu;
  static std::map<std::string, vgpu::DeviceProfile> cache;
  std::lock_guard<std::mutex> lock(mu);
  if (cache.empty()) {
    for (const std::string& id : vgpu::available_gpus()) {
      try {
        vgpu::DeviceProfile p = vgpu::load_gpu(id);
        if (p.vendor == "nvidia") cache.emplace(p.model, std::move(p));
      } catch (const std::exception&) {
      }
    }
  }
  const auto it = cache.find(d.name);
  return it == cache.end() ? nullptr : &it->second;
}
// The nvml: facts of a device's profile, or an empty record.
const vgpu::NvmlClass& nvml_facts(const vgpu::telemetry::DeviceSample& d) {
  static const vgpu::NvmlClass none{};
  const vgpu::DeviceProfile* p = profile_of(d);
  return p ? p->nvml : none;
}

// ---- setters ------------------------------------------------------------------

// Whether this process may change a device's settings. The driver demands root
// for every setter; a simulated machine is a user's, and the processes that
// manage it are rarely root, so VGPU_NVML_ROOT overrides in both directions:
// "1" lets a non-root process (a test, CI) change settings, "0" refuses even
// root. Unset, it is the process's own effective uid.
bool setters_allowed() {
  if (const char* v = std::getenv("VGPU_NVML_ROOT"); v && v[0]) return v[0] != '0';
  return ::geteuid() == 0;
}
#define NVML_SETTER_PERMISSION() \
  if (!setters_allowed()) return NVML_ERROR_NO_PERMISSION

// ---- the settings overlay -------------------------------------------------------

enum class Life { Volatile, Persistent };

std::string overlay_path(const std::string& uuid, Life life) {
  return (life == Life::Persistent ? vgpu::ras::state_dir() : vgpu::telemetry::default_path()) +
         "/nvml-" + uuid + ".conf";
}

using Settings = std::map<std::string, std::string>;

Settings overlay_read(const std::string& uuid, Life life) {
  Settings out;
  std::ifstream in(overlay_path(uuid, life));
  for (std::string line; std::getline(in, line);) {
    const size_t eq = line.find('=');
    if (eq != std::string::npos) out[line.substr(0, eq)] = line.substr(eq + 1);
  }
  return out;
}

// Rewrites the settings file under an exclusive lock, so two setters in two
// processes do not lose each other's change. An empty value removes the key.
void overlay_update(const std::string& uuid, Life life,
                    const std::vector<std::pair<std::string, std::string>>& changes) {
  const std::string path = overlay_path(uuid, life);
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
  if (fd < 0) return;
  ::flock(fd, LOCK_EX);
  Settings s = overlay_read(uuid, life);
  for (const auto& [k, v] : changes) {
    if (v.empty()) s.erase(k);
    else s[k] = v;
  }
  std::string text;
  for (const auto& [k, v] : s) text += k + "=" + v + "\n";
  if (::ftruncate(fd, 0) == 0) {
    size_t done = 0;
    while (done < text.size()) {
      const ssize_t n = ::pwrite(fd, text.data() + done, text.size() - done, static_cast<off_t>(done));
      if (n <= 0) break;
      done += static_cast<size_t>(n);
    }
  }
  ::flock(fd, LOCK_UN);
  ::close(fd);
}
void overlay_set(const std::string& uuid, Life life, const std::string& key, const std::string& value) {
  overlay_update(uuid, life, {{key, value}});
}

// A setting's value, or `dflt` when no setter has touched it.
std::string setting(const std::string& uuid, Life life, const char* key, const char* dflt) {
  const Settings s = overlay_read(uuid, life);
  const auto it = s.find(key);
  return it == s.end() ? dflt : it->second;
}
long long setting_int(const std::string& uuid, Life life, const char* key, long long dflt) {
  const Settings s = overlay_read(uuid, life);
  const auto it = s.find(key);
  if (it == s.end() || it->second.empty()) return dflt;
  char* end = nullptr;
  const long long v = std::strtoll(it->second.c_str(), &end, 10);
  return end && *end == '\0' ? v : dflt;
}

}  // namespace
