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
#include <utility>
#include <vector>

namespace vgpu::cuda {

struct FatbinPtx {
  uint32_t arch = 0;  // e.g. 86 for compute_86
  std::string text;   // decompressed PTX source
};

// Entry kinds, as nvcc and NVIDIA's libnvfatbin write them (measured on CUDA
// 13.0's output): PTX, an ELF cubin, NVVM bitcode for link-time optimisation,
// and the relocatable PTX nvFatbinAddReloc copies out of a host object -- which
// the RTX 3060's driver JITs exactly as it does kind 1, so it is PTX here too.
inline constexpr uint16_t kFatbinPtx = 1;
inline constexpr uint16_t kFatbinElf = 2;
inline constexpr uint16_t kFatbinLtoIr = 8;
inline constexpr uint16_t kFatbinRelocPtx = 64;
inline bool is_ptx_kind(uint16_t kind) { return kind == kFatbinPtx || kind == kFatbinRelocPtx; }

// One entry of a fatbin, decompressed.
struct FatbinImage {
  uint16_t kind = 0;
  uint32_t arch = 0;              // the XX of sm_XX / compute_XX
  uint16_t major = 0, minor = 0;  // PTX ISA version for PTX; ELF ABI for a cubin
  std::string name;               // the entry's identifier, where it has one
  std::string data;               // the payload (PTX text keeps any trailing NULs)
  std::string options;            // compile options recorded with a PTX entry
  bool stored = false;            // a non-PTX payload that would not decompress, as stored
  bool is_ptx() const { return is_ptx_kind(kind); }
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

// The fatbin's cubins (ELF images of SASS), by the same walk: `arch` is the
// SASS architecture (86 for sm_86), `text` the ELF bytes.
std::vector<FatbinPtx> extract_elf(const void* data, size_t bytes);
std::vector<FatbinPtx> extract_elf(const void* data);

// The cubin a device of compute capability `cc` runs from a fatbin, as the
// driver picks it: SASS built for exactly this architecture, or else the
// newest the device can run (the same major, an older minor). Only linked
// cubins (ET_EXEC): relocatable ones (-rdc) need a device link first, and
// the PTX path links those. Empty when there is none, when VGPU_SASS=0
// asks for PTX only, or when the cubin holds an instruction the SASS
// executor does not run yet and the fatbin has PTX (VGPU_SASS=1: the cubin
// regardless). In fatbin_sass.cpp, apart from the rest: it needs the SASS
// executor, which nvJitLink and nvFatbin, built from this file, do not carry.
std::string pick_cubin(const void* fatbin, uint32_t cc);

// Every entry of one fatbin container, not only the PTX: what nvJitLink and
// nvFatbin need to tell a SASS-only or LTO-IR-only input from one they can
// use. `consumed`, when given, receives the container's size in bytes, so a
// caller can walk the several containers a host object's section holds back
// to back. `kinds` -- kind values or'd together, which are powers of two --
// selects entries; the rest are skipped without being decompressed, so a SASS
// image the PTX path never reads cannot fail it.
std::vector<FatbinImage> extract_images(const void* data, size_t bytes, size_t* consumed = nullptr,
                                        uint16_t kinds = 0xffff);

// A fatbin container holding `images`, uncompressed, laid out as NVIDIA's
// libnvfatbin lays out an uncompressed one (-compress=false): the 16-byte
// container header, then per entry a 64-byte header, the identifier, for PTX
// an options record, and the payload padded to 8 bytes. extract_images reads
// it back, and so does NVIDIA's driver.
std::string write_fatbin(const std::vector<FatbinImage>& images);

// The sections of a 64-bit ELF file, as name and contents (empty for a
// section that occupies no file bytes); empty when the buffer is not ELF.
// Throws vgpu::Error on a section table or section that does not fit.
std::vector<std::pair<std::string, std::string>> elf_sections(const void* data, size_t bytes);

// The fatbin containers a host object carries: the contents of its
// .nv_fatbin section (whole-program device code) and its __nv_relfatbin
// section (relocatable device code, -rdc / -dc), each container separately --
// only the second with `relocatable_only`. Empty for an object with no device
// code.
std::vector<std::string> host_object_fatbins(const void* data, size_t bytes,
                                             bool relocatable_only = false);

// The members of an ar archive (a static library), as name and bytes.
// Throws vgpu::Error on a malformed archive.
std::vector<std::pair<std::string, std::string>> archive_members(const void* data, size_t bytes);

// What a buffer holds, judged by its first bytes the way nvJitLink's
// NVJITLINK_INPUT_ANY judges it.
enum class BlobKind { Unknown, Ptx, Fatbin, Cubin, HostObject, Archive, LtoIr };
BlobKind classify_blob(const void* data, size_t bytes);

// The architecture a cubin was built for (86 for sm_86), read from its ELF
// header; 0 when the buffer is not a CUDA ELF image.
uint32_t cubin_arch(const void* data, size_t bytes);

// Whether a cubin is linked (ELF type ET_EXEC: what nvcc -cubin writes) rather
// than relocatable (ET_REL: -rdc / -dc, which needs a device link).
bool cubin_linked(const void* data, size_t bytes);

// Which of a fatbin's PTX images the driver would JIT for a device of compute
// capability `cc` (e.g. 90), as an index into `ptxs`; ptxs.size() if empty.
//
// The newest image the device can run wins. What "can run" means depends on
// the target suffix, which only the PTX's own .target line carries -- the
// entry header reports sm_90 and sm_90a alike as 90:
//   sm_XY   runs on XY and everything newer;
//   sm_XYf  runs within the family: the same major, minor XY or newer;
//   sm_XYa  runs on XY only.
// Between images of the same XY the more specific target wins, since it is
// the one a build aimed at this exact device (-arch=sm_90a embeds both an
// sm_90 and an sm_90a image, and only the second has wgmma in it). When no
// image qualifies, the newest of all, which the load then refuses with the
// reason.
size_t pick_ptx(const std::vector<FatbinPtx>& ptxs, uint32_t cc);

}  // namespace vgpu::cuda
