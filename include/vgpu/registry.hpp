// Registry of built-in device profiles (embedded at build time from profiles/).
#pragma once

#include <string>
#include <vector>

#include "vgpu/profile.hpp"

namespace vgpu {

// All built-in GPU ids, e.g. {"nvidia/a10", ..., "nvidia/b200"}.
std::vector<std::string> available_gpus();

// The ids load_gpu() also resolves but available_gpus() does not list: parts the engine runs and the register
// database, the monitors and the CLI have no data for (Intel's, today).
std::vector<std::string> available_engine_gpus();

// Loads a built-in profile. Ids are case-insensitive. Throws Err::UnknownGpu
// with the list of available ids on failure.
DeviceProfile load_gpu(const std::string& id);

}  // namespace vgpu
