// Writing a resolved address into the instruction a code relocation names (shared by the module loader, src/sass/module.cpp, and the
// host-GPU mode's card image, vgpu/host_gpu_image.hpp).
#pragma once

#include <cstdint>
#include <cstring>

namespace vgpu::sass {

// The code relocations the loader resolves: the low (56, 62) or high (57, 63) half of an address in a MOV's 32-bit immediate, a
// CALL.ABS.NOINC's target, 49 bits at 32 (58), or from sm_90 in 4-byte units at 16-23 and 34-80 (75).
inline bool code_reloc_type(uint32_t type) { return type == 56 || type == 57 || type == 58 || type == 62 || type == 63 || type == 75; }

// Writes `value` into the 16-byte instruction at `at` as relocation `type` places it (one of code_reloc_type's).
inline void patch_code_word(uint8_t* at, uint32_t type, uint64_t value) {
  uint64_t hi, lo;
  std::memcpy(&lo, at, 8);
  std::memcpy(&hi, at + 8, 8);
  if (type == 58) {          // bits 32-80
    lo = (lo & 0xffffffffull) | (value << 32);
    hi = (hi & ~uint64_t{0x1ffff}) | ((value >> 32) & 0x1ffff);
  } else if (type == 75) {   // words: the low eight bits at 16-23, the rest at 34-80
    const uint64_t words = value >> 2;
    lo = (lo & ~(uint64_t{0xff} << 16) & 0x3ffffffffull) | ((words & 0xff) << 16) | ((words >> 8) << 34);
    hi = (hi & ~uint64_t{0x1ffff}) | ((words >> 8 >> 30) & 0x1ffff);
  } else {                   // bits 32-63: the low half (56, 62) or the high one (57, 63)
    const uint64_t half = type == 56 || type == 62 ? (value & 0xffffffffull) : (value >> 32);
    lo = (lo & 0xffffffffull) | (half << 32);
  }
  std::memcpy(at, &lo, 8);
  std::memcpy(at + 8, &hi, 8);
}

}  // namespace vgpu::sass
