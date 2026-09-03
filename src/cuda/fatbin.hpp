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
#include <cstddef>
#include <vector>

namespace vgpu::cuda {

struct FatbinPtx {
  uint32_t arch = 0;  // e.g. 86 for compute_86
  std::string text;   // decompressed PTX source
};

// Largest fatbin we will walk. The loader APIs hand us a bare pointer with no
// length, so a corrupt or truncated image cannot be bounds-checked against the
// real allocation; instead every offset must be internally consistent and stay
// under this cap. 512 MiB is far above any real fatbin.
inline constexpr uint64_t kMaxFatbinBytes = 512ull * 1024 * 1024;

// Extracts every PTX image from a fatbin blob. `data` may point at the
// __fatBinC_Wrapper_t (magic 0x466243B1) or directly at a container.
//
// Robustness matters here: this parses a binary structure whose offsets and
// sizes come from the file, and a malformed one must fail cleanly rather than
// read out of bounds or spin. Throws vgpu::Error with a precise reason on
// malformed or unsupported input.
// `bytes` is the real size of the buffer at `data`. Pass it whenever it is
// known: it is the only thing that can catch a truncated image, because every
// other bound in the format is a number read out of the image itself. A
// container whose declared size is larger than the buffer is then rejected
// instead of walked off the end.
std::vector<FatbinPtx> extract_ptx(const void* data, size_t bytes);

// Unbounded form, for the CUDA entry points that have no length to give:
// __cudaRegisterFatBinary and cuModuleLoadFatBinary both take a bare pointer.
// Offsets are still checked for internal consistency and against
// kMaxFatbinBytes, but a truncated image cannot be detected here -- prefer the
// two-argument form anywhere a size exists.
std::vector<FatbinPtx> extract_ptx(const void* data);

}  // namespace vgpu::cuda
