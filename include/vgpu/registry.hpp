// Registry of built-in device profiles (embedded at build time from profiles/).
#pragma once

#include <string>
#include <vector>

#include "vgpu/profile.hpp"

namespace vgpu {

// All built-in GPU ids, e.g. {"nvidia/a10", ..., "nvidia/b200"}.
std::vector<std::string> available_gpus();

// Loads a built-in profile. Ids are case-insensitive. Throws Err::UnknownGpu
// with the list of available ids on failure.
DeviceProfile load_gpu(const std::string& id);

}  // namespace vgpu
