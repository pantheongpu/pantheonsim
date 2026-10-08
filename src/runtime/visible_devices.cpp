#include "vgpu/runtime/visible_devices.hpp"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#include "vgpu/telemetry.hpp"

namespace vgpu::runtime {

namespace {
constexpr int kNoDevice = 100;       // cudaErrorNoDevice
constexpr int kInvalidDevice = 101;  // cudaErrorInvalidDevice

std::string trimmed(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}
std::string lowered(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
}  // namespace

VisibleDevices cuda_visible_devices(const DeviceProfile& profile, int machine, const char* visible, const char* order) {
  VisibleDevices out;
  // A machine of AMD's GPUs has no CUDA device: the CUDA libraries are here
  // because the simulator's directory holds every vendor's, but on the machine
  // it models there is no NVIDIA driver, and a program asking CUDA for devices
  // (offload-arch, a framework probing both vendors) is told there are none.
  if (profile.vendor != "nvidia") {
    out.error = kNoDevice;
    return out;
  }
  const auto all = [&] {
    for (int i = 0; i < machine; ++i) out.physical.push_back(i);
  };
  if (order && *order && std::strcmp(order, "FASTEST_FIRST") != 0 && std::strcmp(order, "PCI_BUS_ID") != 0) {
    out.error = kInvalidDevice;
    return out;
  }
  if (!visible) {
    all();
    return out;
  }
  // The UUIDs the devices have, as NVML prints them.
  std::vector<std::string> uuids;
  for (int i = 0; i < machine; ++i) {
    telemetry::DeviceSample d{};
    telemetry::describe_device(profile, i, &d);
    uuids.push_back(lowered(d.uuid));
  }
  // A list names its devices all by index or all by UUID: the first element
  // says which, and one of the other kind ends the list. A device named twice
  // is an error.
  enum Kind { kNone, kIndex, kUuid };
  Kind list_kind = kNone;
  std::vector<bool> named(static_cast<size_t>(machine), false);
  std::string list = visible;
  size_t at = 0;
  for (bool more = true; more;) {
    const size_t comma = list.find(',', at);
    more = comma != std::string::npos;
    const std::string element = trimmed(list.substr(at, more ? comma - at : std::string::npos));
    at = more ? comma + 1 : list.size();
    int device = -1;
    Kind kind = kIndex;
    if (lowered(element).rfind("gpu-", 0) == 0) {
      kind = kUuid;
      const std::string want = lowered(element);
      for (int i = 0; i < machine; ++i) {
        if (uuids[static_cast<size_t>(i)].rfind(want, 0) != 0) continue;
        device = device < 0 ? i : -2;   // -2: more than one device has it
      }
    } else if (!element.empty() && std::isdigit(static_cast<unsigned char>(element[0]))) {
      errno = 0;
      const long v = std::strtol(element.c_str(), nullptr, 10);
      if (errno == 0 && v < machine) device = static_cast<int>(v);
    }
    if (device < 0) break;   // names no device: the list ends here
    if (list_kind == kNone) list_kind = kind;
    else if (kind != list_kind) break;
    if (named[static_cast<size_t>(device)]) {
      out.error = kInvalidDevice;
      out.physical.clear();
      return out;
    }
    named[static_cast<size_t>(device)] = true;
    out.physical.push_back(device);
  }
  if (out.physical.empty()) out.error = kNoDevice;
  return out;
}

}  // namespace vgpu::runtime
