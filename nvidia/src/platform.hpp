// Which kind of machine the simulated driver is answering as.
//
// VGPU_PLATFORM=wsl (the default) or linux. The expected files of the CUPTI cases were printed by an
// RTX 3060 under WSL2, and a native Linux driver answers a few things differently (measured on an
// RTX 4090 and an RTX 5090 with CUDA 13.0, on Vast.ai; nvidia/tests/data/aws/README.md):
//   - sharing handles that were not opened: the import calls fail with cudaErrorUnknown (999), where
//     WSL says not supported (801);
//   - cudaDeviceFlushGPUDirectRDMAWrites succeeds, where WSL says not supported;
//   - the driver makes seven streams of its own in a context, where WSL makes eight.
// Nothing else is switched: the device profile, the attributes and everything else a program can read
// keep their one set of answers.
#pragma once

#include <cstdlib>
#include <cstring>

namespace vgpu::platform {

inline bool native_linux() {
  static const bool v = [] {
    const char* e = std::getenv("VGPU_PLATFORM");
    return e != nullptr && std::strcmp(e, "linux") == 0;
  }();
  return v;
}

// The streams the driver makes for itself in each context.
inline int internal_streams() { return native_linux() ? 7 : 8; }

}  // namespace vgpu::platform
