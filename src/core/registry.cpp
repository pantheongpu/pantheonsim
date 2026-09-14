#include "vgpu/registry.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "vgpu/embedded_profiles.hpp"
#include "vgpu/error.hpp"

namespace vgpu {
namespace {
// Where a built-in profile came from, for its error messages: nvidia/h100 is
// nvidia/profiles/h100.yaml.
std::string profile_file(const std::string& id) {
  const auto slash = id.find('/');
  if (slash == std::string::npos) return "profiles/" + id + ".yaml";
  return id.substr(0, slash) + "/profiles" + id.substr(slash) + ".yaml";
}
}  // namespace

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
      DeviceProfile prof = DeviceProfile::from_yaml(p.yaml, profile_file(p.id));
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
