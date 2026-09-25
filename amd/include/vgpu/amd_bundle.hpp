// A clang offload bundle: the device code a HIP program or library carries,
// one code object per target it was built for. hipcc writes one into every
// executable; ROCm's libraries write theirs compressed.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace vgpu::amd {

struct Bundle {
  // The bundle's bytes, where it came compressed: the code objects below are
  // views into them. Otherwise they are views into the bytes it was read from,
  // which have to outlive it.
  std::string inflated;
  std::map<std::string, std::string_view> targets;   // "gfx942:xnack-" -> its code object
};

// Whether b starts a bundle, compressed or not.
bool is_bundle(const uint8_t* b);

// Reads one. A compressed bundle is inflated with the system's zstd or zlib,
// whichever it was compressed with; where that library is not installed, or
// the bundle is not one of these, this throws.
std::unique_ptr<Bundle> read_bundle(const uint8_t* b);

// Which of a bundle's code objects a device runs, as HIP chooses: one built
// for the device's target, with every feature it names set the way the device
// runs ("gfx942:xnack-" on a device that is "gfx942:sramecc+:xnack-"; one that
// names none runs either way). Of those, the one that names the most. Null
// where none will do.
const std::string_view* code_for(const Bundle& b, const std::string& device);

// The targets a bundle carries, for saying so: "gfx90a, gfx942:xnack-".
std::string target_list(const Bundle& b);

}  // namespace vgpu::amd
