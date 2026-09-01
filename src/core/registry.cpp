#include "vgpu/registry.hpp"

#include <algorithm>
#include <cctype>

#include "vgpu/embedded_profiles.hpp"
#include "vgpu/error.hpp"

namespace vgpu {
namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

}  // namespace

std::vector<std::string> available_gpus() {
  std::vector<std::string> out;
  for (const auto& p : embedded::kProfiles) out.emplace_back(p.id);
  return out;
}

DeviceProfile load_gpu(const std::string& id) {
  std::string want = lower(id);
  for (const auto& p : embedded::kProfiles) {
    if (want == p.id) {
      DeviceProfile prof = DeviceProfile::from_yaml(p.yaml, std::string("profiles/") + p.id + ".yaml");
      if (lower(prof.id) != want)
        throw Error::make(Err::Internal, "profile file for ", p.id, " declares mismatched id ", prof.id);
      return prof;
    }
  }
  std::string avail;
  for (const auto& p : embedded::kProfiles) avail += std::string("\n  ") + p.id;
  throw Error::make(Err::UnknownGpu, "no device profile named '", id, "'. Available GPUs:", avail);
}

}  // namespace vgpu
