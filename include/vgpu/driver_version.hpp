// The CUDA version the simulated driver reports, in CUDA's own encoding
// (major * 1000 + minor * 10, so 12.4 is 12040).
//
// One source, because on real hardware there is one: cuDriverGetVersion,
// cudaDriverGetVersion and nvidia-smi's "CUDA Version" all come from the same
// installed driver and cannot disagree. Here they used to come from three
// places -- a constant 13000 in the driver shim, whatever toolkit the runtime
// shim happened to be built with, and the session's --cuda flag -- so one
// session reported CUDA 13.0, 12.6 and 12.4 depending on how it was asked, and
// --cuda changed only the one a person reads, not the ones programs check.
//
// The session's declaration wins: `vgpu shell --cuda X` exports
// VGPU_CUDA_VERSION, and that is the machine the person asked for. Outside a
// shell, or given something that is not a version, the driver reports the
// default below.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace vgpu {

// A driver must be at least as new as the runtime a program links, or the
// program refuses to start ("driver version is insufficient for CUDA runtime
// version"). The runtime shim reports the CUDART_VERSION of the toolkit it was
// built with, which CMake passes down as VGPU_TOOLKIT_CUDART_VERSION, so the
// default driver is never older than that. 13.0 stays the floor for builds
// without a toolkit, where there is no runtime shim to be older than.
#if defined(VGPU_TOOLKIT_CUDART_VERSION)
inline constexpr int kDefaultDriverVersion =
    VGPU_TOOLKIT_CUDART_VERSION > 13000 ? VGPU_TOOLKIT_CUDART_VERSION : 13000;
#else
inline constexpr int kDefaultDriverVersion = 13000;  // CUDA 13.0
#endif

// The driver release printed beside it when nothing declares one. A 580-series
// driver is the first to carry CUDA 13.0.
inline constexpr const char* kDefaultDriverRelease = "580.00.00";

// "12.4" for 12040: the form nvidia-smi prints.
inline std::string cuda_version_string(int v) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "%d.%d", v / 1000, (v % 1000) / 10);
  return buf;
}

inline int driver_version() {
  const char* v = std::getenv("VGPU_CUDA_VERSION");
  if (!v || !*v) return kDefaultDriverVersion;
  char* end = nullptr;
  const long major = std::strtol(v, &end, 10);
  if (end == v || *end != '.') return kDefaultDriverVersion;
  const char* m = end + 1;
  const long minor = std::strtol(m, &end, 10);
  if (end == m || *end != '\0') return kDefaultDriverVersion;
  // CUDA has shipped majors 1 through 13; allow headroom, refuse nonsense.
  if (major < 1 || major > 99 || minor < 0 || minor > 99) return kDefaultDriverVersion;
  return static_cast<int>(major * 1000 + minor * 10);
}

// The same version as nvidia-smi prints it, "12.4". Derived from the number,
// never copied from the environment, so that a value the driver rejects is
// not shown as if it had been accepted.
inline std::string driver_version_string() { return cuda_version_string(driver_version()); }

}  // namespace vgpu
