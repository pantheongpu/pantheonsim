// NVIDIA fatbin container parsing (clean-room).
//
// Layout derived empirically from binaries produced by the locally installed
// CUDA toolkit plus the public fatbinary_section.h wrapper struct. A fatbin is
// a sequence of containers:
//   container: { u32 magic=0xBA55ED50; u16 version; u16 header_size; u64 size; }
//   entries:   { u16 kind (1=PTX, 2=ELF/SASS); u16 version; u32 header_size;
//                u64 padded_payload_size; u32 payload_size; u32 unknown;
//                u16 minor; u16 major; u32 arch; ...; u64 flags; ... }
// flags bit 0x2000 = LZ4-compressed payload, bit 0x8000 = zstd-compressed.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::cuda {

struct FatbinPtx {
  uint32_t arch = 0;  // e.g. 86 for compute_86
  std::string text;   // decompressed PTX source
};

// Extracts every PTX image from a fatbin blob. `data` may point at the
// __fatBinC_Wrapper_t (magic 0x466243B1) or directly at a container.
// Throws vgpu::Error with a precise reason on malformed/unsupported input.
std::vector<FatbinPtx> extract_ptx(const void* data);

}  // namespace vgpu::cuda
