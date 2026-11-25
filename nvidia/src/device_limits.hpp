// The defaults and ranges of the device limits that follow the device (cudaDeviceGetLimit / cuCtxGetLimit and their setters),
// shared by the runtime and the driver API. Evidence: an A10G (sm_86, 80 SMs, 6 MiB L2) and a Tesla T4 (sm_75, 40 SMs, 4 MiB L2), both
// Linux driver 595.91, CUDA 13.5 runtime, 2026-11-23 (nvidia/tests/data/card/xarch_limits.*.txt). Everything else is inferred from them.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vgpu/profile.hpp"

namespace vgpu::nv_limits {

// A limit nobody has set: the answer is the device's default, which depends on the profile.
constexpr size_t kUnset = ~size_t{0};

// cudaLimitPrintfFifoSize before anyone sets it. NVIDIA documents 1 MiB; the cards say otherwise: the T4 gives 1310720 (40 SMs x 1024
// threads x 32 bytes) and the A10G 3932160 (80 SMs x 1536 threads x 32 bytes). A buffer of 32 bytes for every thread the device can hold.
inline size_t default_printf_fifo(const DeviceProfile& p) {
  const uint64_t v = uint64_t{p.limits.multiprocessors} * p.limits.max_threads_per_sm * 32;
  return v ? static_cast<size_t>(v) : size_t{1} << 20;
}

// The persisting-L2 limit counts in sixteenths of the L2: the A10G rounds a request up to a multiple of 393216 (6 MiB / 16), and every
// profile whose persisting maximum was read from a card (RTX 3060, A10G, A100, H100, L4, L40S, RTX 4090 ...) is a whole number of them.
inline uint64_t persisting_unit(const DeviceProfile& p) { return p.limits.l2_cache_bytes / 16; }

// Whether the device has persisting-L2 limit and an access policy window at all. The T4 (compute capability 7.5, persisting maximum 0)
// answers cudaErrorUnsupportedLimit to both the get and the set, cudaSuccess to cudaCtxResetPersistingL2Cache, and cudaErrorInvalidValue
// to a stream's accessPolicyWindow of any size.
inline bool persisting_supported(const DeviceProfile& p) { return p.cuda.persisting_l2_bytes > 0; }

// The A10G's default is three sixteenths of its L2 (1179648), kept below the maximum; the other devices are inferred from it.
inline size_t default_persisting(const DeviceProfile& p) {
  const uint64_t unit = persisting_unit(p), most = p.cuda.persisting_l2_bytes;
  if (!persisting_supported(p)) return 0;
  const uint64_t v = unit ? 3 * unit : most;
  return static_cast<size_t>(v < most ? v : most);
}

enum class Set { Ok, InvalidValue };

// A request: above the maximum is cudaErrorInvalidValue and changes nothing; otherwise it is rounded up to a whole unit (0 stays 0).
inline Set set_persisting(const DeviceProfile& p, size_t value, size_t* out) {
  const uint64_t most = p.cuda.persisting_l2_bytes, unit = persisting_unit(p);
  if (value > most) return Set::InvalidValue;
  *out = unit ? static_cast<size_t>((value + unit - 1) / unit * unit) : value;
  if (*out > most) *out = static_cast<size_t>(most);
  return Set::Ok;
}

// accessPolicyMaxWindowSize: 0 before compute capability 8.0 (no access policy window at all), 134213632 (128 MiB less 4 KiB) on the A10G
// (measured) and the other compute-capability-8 and 9 parts, 128 MiB on compute capability 12 (from the RTX PRO 6000 card file).
inline uint64_t max_access_window(const DeviceProfile& p) {
  return p.cc_major < 8 ? 0 : p.cc_major >= 12 ? uint64_t{134217728} : uint64_t{134213632};
}

// A stream's accessPolicyWindow.num_bytes may not exceed it (the A10G: one byte over is cudaErrorInvalidValue; the T4: any window is).
inline bool window_fits(uint64_t num_bytes, uint64_t max_window) { return num_bytes <= max_window; }

}  // namespace vgpu::nv_limits
