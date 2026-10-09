// libvgpunvfatbin -- VirtualGPU's nvFatbin, presented as libnvfatbin.so.13
// (.so.12 under a CUDA 12 toolkit that has one).
//
// nvFatbin builds the container nvcc's -fatbin builds -- PTX, cubins and
// LTO-IR for several architectures in one image -- at run time, for a program
// to load with cuModuleLoadData or hand to nvJitLink. Nothing in it needs a
// GPU, so this one is complete in the way that matters: what it writes is a
// fatbin as NVIDIA's lays one out with -compress=false (fatbin.cpp's
// write_fatbin, checked against libnvfatbin 13.0's output), which VirtualGPU's
// loaders read back and NVIDIA's driver loads too.
//
// What it writes follows libnvfatbin 13.0 as measured: the entry flag word
// (the options' bitness, host, CUDA/OpenCL and debug bits, and an arch
// suffix's a/f bit), PTX stored without its comments, LTO-IR entries
// carrying the bitcode's own version, a relocatable entry copied from its
// object as it was stored there -- and compression: by default PTX, LTO-IR and
// the relocatable PTX are compressed with zstd (libzstd is opened when
// needed; without it an entry is stored uncompressed), cubins only with
// -compress-all, -compress-mode=size or -g, and -compress-mode=speed uses LZ4
// instead; -compress=false and -compress-mode=none switch it off. The
// compressed bytes are a valid zstd frame or LZ4 block, not NVIDIA's own
// (another zstd, another level), so sizes differ a little.
//
// Two differences, each written down where it happens:
//   - A "cubin" that is PTX text -- what VirtualGPU's NVRTC (with the PTX
//     engine) and nvJitLink hand out as one, since PTX is what that engine
//     runs -- goes in as the PTX it is, so that a program which packages its
//     JIT output still gets a fatbin it can load.
//   - nvFatbinAddIndex refuses every input with NVFATBIN_ERROR_INVALID_INDEX:
//     it takes an index file, a format no NVIDIA tool documents or writes
//     ("no method of creating an index file is available", says nvFatbin.h),
//     and NVIDIA's own library answers that for each input measured -- text,
//     cubins, fatbins, zeros.
// LTO-IR itself is packaged faithfully, since packaging needs no compiler;
// nothing on VirtualGPU can run it, and nvJitLink hands it to the toolkit's.
//
// Result codes follow NVIDIA's library as measured on CUDA 13.0's, including
// the ones that surprise: an architecture is the bare number ("86", not
// "sm_86"), PTX whose .target differs from it is accepted, and a host object
// with no device code adds nothing without failing.
#include "../include/vgpu_nvfatbin.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "fatbin.hpp"
#include "vgpu/error.hpp"

// CUDA_VERSION of the toolkit this library stands in for (CMake passes it).
#ifndef VGPU_NVFATBIN_VERSION
#define VGPU_NVFATBIN_VERSION 13000
#endif

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

struct Fatbin {
  // From the options: the entry flag word, and how entries are compressed.
  uint64_t flags = 0x11;
  bool compress = true, compress_all = false, debug = false, speed = false, size = false, none = false;
  std::vector<vgpu::cuda::FatbinImage> entries;
  std::set<std::pair<std::string, uint32_t>> reloc_ids;   // (identifier, arch) of relocatable PTX
  std::string image;   // the container, built on demand
  bool dirty = true;
};

std::mutex g_mu;
std::set<const void*> g_live;

Fatbin* get(nvFatbinHandle h) {
  std::lock_guard<std::mutex> l(g_mu);
  return g_live.count(h) ? reinterpret_cast<Fatbin*>(h) : nullptr;
}

const std::string& image_of(Fatbin& F) {
  if (F.dirty) {
    F.image = vgpu::cuda::write_fatbin(F.entries);
    F.dirty = false;
  }
  return F.image;
}

// "86", "90a", "100f": the XX of sm_XX, which is what nvFatbin takes, read as C's strtol
// reads a number (measured: "", "+80", " 80" and "99999" are taken; "sm_80", "80 ", "-5",
// "0x50" and "80af" are NVFATBIN_ERROR_INVALID_ARCH) with an optional a or f after it, which
// goes into the entry's flags as 0x100000 and 0x200000.
bool parse_arch(const char* arch, uint32_t* number, uint64_t* flag_bits) {
  char* end = nullptr;
  const long v = std::strtol(arch, &end, 10);
  if (v < 0) return false;
  *flag_bits = 0;
  if (*end == 'a' || *end == 'f') {
    *flag_bits = *end == 'a' ? 0x100000 : 0x200000;
    ++end;
  }
  if (*end) return false;
  *number = static_cast<uint32_t>(v);
  return true;
}

// Compresses an entry's payload per the handle's options: the kind decides whether it is (PTX, LTO-IR
// by default, a cubin only with -compress-all, -compress-mode=size or -g). Leaves the entry alone when
// it should not be, or cannot be (no libzstd).
void finish_entry(const Fatbin& F, vgpu::cuda::FatbinImage& im) {
  im.flags = F.flags;
  if (!F.compress || F.none) return;
  const bool wanted = im.kind == vgpu::cuda::kFatbinElf ? (F.compress_all || F.size || F.debug) : true;
  if (!wanted) return;
  std::string payload = im.data;
  if (im.is_ptx() && (payload.empty() || payload.back() != '\0')) payload.push_back('\0');
  const std::string packed = F.speed ? vgpu::cuda::compress_lz4(payload) : vgpu::cuda::compress_zstd(payload, F.size);
  if (packed.empty()) return;
  im.flags |= F.speed ? vgpu::cuda::kFatbinLz4 : vgpu::cuda::kFatbinZstd;
  if (im.kind == vgpu::cuda::kFatbinLtoIr) im.flags |= vgpu::cuda::kFatbinLtoFlag;
  im.uncompressed_size = payload.size();
  im.data = packed;
}

// The ISA version a PTX module declares, as nvcc records it in the entry.
bool ptx_version(const std::string& text, uint16_t* major, uint16_t* minor) {
  const size_t at = text.find(".version");
  if (at == std::string::npos) return false;
  unsigned ma = 0, mi = 0;
  if (std::sscanf(text.c_str() + at, ".version %u.%u", &ma, &mi) < 1) return false;
  *major = static_cast<uint16_t>(ma);
  *minor = static_cast<uint16_t>(mi);
  return true;
}

nvFatbinResult add_ptx_entry(Fatbin& F, std::string text, uint32_t arch, uint64_t arch_flag, const char* identifier,
                             const char* options) {
  while (!text.empty() && text.back() == '\0') text.pop_back();
  vgpu::cuda::FatbinImage im;
  if (!ptx_version(text, &im.major, &im.minor)) return NVFATBIN_ERROR_MISSING_PTX_VERSION;
  im.kind = vgpu::cuda::kFatbinPtx;
  im.arch = arch;
  im.name = identifier ? identifier : "";
  im.options = options ? options : "";
  im.data = vgpu::cuda::strip_ptx_comments(text);   // stored without its comments, as NVIDIA's does
  finish_entry(F, im);
  im.flags |= arch_flag;
  F.entries.push_back(std::move(im));
  F.dirty = true;
  return NVFATBIN_SUCCESS;
}

}  // namespace

// NVIDIA's strings, as 13.0 returns them (NULL for success, "unknown error" past the last).
VGPU_EXPORT const char* nvFatbinGetErrorString(nvFatbinResult result) {
  switch (static_cast<int>(result)) {
    case NVFATBIN_SUCCESS: return nullptr;
    case NVFATBIN_ERROR_INTERNAL: return "internal error";
    case NVFATBIN_ERROR_ELF_ARCH_MISMATCH: return "fatbinary elf mismatch: elf arch does not match user-specified arch";
    case NVFATBIN_ERROR_ELF_SIZE_MISMATCH: return "fatbinary elf mismatch: elf size doesn't match user-specified size";
    case NVFATBIN_ERROR_MISSING_PTX_VERSION: return "could not find ptx version";
    case NVFATBIN_ERROR_NULL_POINTER: return "input contained null pointer";
    case NVFATBIN_ERROR_COMPRESSION_FAILED: return "compression failed";
    case NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED: return "compressed size is bigger than max size of compressed size field";
    case NVFATBIN_ERROR_UNRECOGNIZED_OPTION: return "unknown option";
    case NVFATBIN_ERROR_INVALID_ARCH: return "invalid architecture";
    case NVFATBIN_ERROR_INVALID_NVVM: return "invalid NVVM input";
    case NVFATBIN_ERROR_EMPTY_INPUT: return "empty input";
    case NVFATBIN_ERROR_MISSING_PTX_ARCH: return "missing ptx architecture";
    case NVFATBIN_ERROR_PTX_ARCH_MISMATCH: return "ptx architecture incompatible with specified architecture";
    case NVFATBIN_ERROR_MISSING_FATBIN: return "host object doesn't contain a fatbin";
    case NVFATBIN_ERROR_INVALID_INDEX: return "invalid index input";
    case NVFATBIN_ERROR_IDENTIFIER_REUSE:
      return "each relocatable entry must have a unique identifier name per architecture";
    case NVFATBIN_ERROR_INTERNAL_PTX_OPTION: return "used ptx options include internal options";
  }
  return "unknown error";
}

VGPU_EXPORT nvFatbinResult nvFatbinCreate(nvFatbinHandle* handle_indirect, const char** options,
                                          size_t optionsCount) {
  if (!handle_indirect || (optionsCount && !options)) return NVFATBIN_ERROR_NULL_POINTER;
  auto* F = new Fatbin;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.insert(F);
  }
  *handle_indirect = reinterpret_cast<nvFatbinHandle>(F);
  bool bits32 = false, bits64 = false, cuda = false, opencl = false;
  for (size_t i = 0; i < optionsCount; ++i) {
    if (!options[i]) return NVFATBIN_ERROR_NULL_POINTER;
    const std::string o = options[i];
    // Measured on libnvfatbin 13.0: -32 and -64, and -cuda and -opencl, exclude each other; -compress takes
    // true or false, -compress-mode one of five words, -host linux, windows or mac, all lower case.
    if (o == "-32") {
      bits32 = true;
    } else if (o == "-64") {
      bits64 = true;
    } else if (o == "-c") {
    } else if (o == "-cuda") {
      cuda = true;
    } else if (o == "-opencl") {
      opencl = true;
    } else if (o == "-g") {
      F->debug = true;
    } else if (o == "-compress=true") {
      F->compress = true;
    } else if (o == "-compress=false") {
      F->compress = false;
    } else if (o == "-compress-all") {
      F->compress_all = true;
    } else if (o.rfind("-compress-mode=", 0) == 0) {
      const std::string m = o.substr(15);
      F->none = m == "none";
      F->speed = m == "speed";
      F->size = m == "size";
      if (m != "none" && m != "speed" && m != "size" && m != "default" && m != "balance")
        return NVFATBIN_ERROR_UNRECOGNIZED_OPTION;
    } else if (o == "-host=linux") {
      F->flags = (F->flags & ~uint64_t{0x70}) | 0x10;
    } else if (o == "-host=windows") {
      F->flags = (F->flags & ~uint64_t{0x70}) | 0x40;
    } else if (o == "-host=mac") {
      F->flags = (F->flags & ~uint64_t{0x70}) | 0x20;
    } else {
      return NVFATBIN_ERROR_UNRECOGNIZED_OPTION;
    }
  }
  if ((bits32 && bits64) || (cuda && opencl)) return NVFATBIN_ERROR_UNRECOGNIZED_OPTION;
  if (bits32) F->flags &= ~uint64_t{1};
  if (F->debug) F->flags |= 0x2;
  if (cuda) F->flags |= 0x4;
  if (opencl) F->flags |= 0x8;
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinDestroy(nvFatbinHandle* handle_indirect) {
  if (!handle_indirect || !*handle_indirect) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(*handle_indirect);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(F);
  }
  delete F;
  *handle_indirect = nullptr;
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char* code, size_t size,
                                          const char* arch, const char* identifier,
                                          const char* optionsCmdLine) {
  if (!handle || !code || !arch) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  uint32_t a = 0;
  uint64_t arch_flag = 0;
  if (!parse_arch(arch, &a, &arch_flag)) return NVFATBIN_ERROR_INVALID_ARCH;
  return add_ptx_entry(*F, std::string(code, size), a, arch_flag, identifier, optionsCmdLine);
}

VGPU_EXPORT nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size,
                                            const char* arch, const char* identifier) {
  if (!handle || !code || !arch) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  uint32_t a = 0;
  uint64_t arch_flag = 0;
  if (!parse_arch(arch, &a, &arch_flag)) return NVFATBIN_ERROR_INVALID_ARCH;
  using vgpu::cuda::BlobKind;
  const BlobKind kind = vgpu::cuda::classify_blob(code, size);
  // VirtualGPU's cubins are PTX (see the top of this file).
  if (kind == BlobKind::Ptx)
    return add_ptx_entry(*F, std::string(static_cast<const char*>(code), size), a, arch_flag, identifier,
                         nullptr);
  // Anything that is not a CUDA ELF image NVIDIA's reports as a size mismatch.
  if (kind != BlobKind::Cubin) return NVFATBIN_ERROR_ELF_SIZE_MISMATCH;
  if (vgpu::cuda::cubin_arch(code, size) != a) return NVFATBIN_ERROR_ELF_ARCH_MISMATCH;
  vgpu::cuda::FatbinImage im;
  im.kind = vgpu::cuda::kFatbinElf;
  im.arch = a;
  im.major = 1;   // as nvcc and libnvfatbin record a cubin: 1, and the ELF ABI version
  im.minor = static_cast<const uint8_t*>(code)[8];
  im.name = identifier ? identifier : "";
  im.data.assign(static_cast<const char*>(code), size);
  finish_entry(*F, im);
  im.flags |= arch_flag;
  F->entries.push_back(std::move(im));
  F->dirty = true;
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinAddLTOIR(nvFatbinHandle handle, const void* code, size_t size,
                                            const char* arch, const char* identifier,
                                            const char* optionsCmdLine) {
  if (!handle || !code || !arch) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  uint32_t a = 0;
  uint64_t arch_flag = 0;
  if (!parse_arch(arch, &a, &arch_flag)) return NVFATBIN_ERROR_INVALID_ARCH;
  // Bytes that are not LTO-IR: NVIDIA's answers NVFATBIN_ERROR_INTERNAL.
  if (vgpu::cuda::classify_blob(code, size) != vgpu::cuda::BlobKind::LtoIr)
    return NVFATBIN_ERROR_INTERNAL;
  vgpu::cuda::FatbinImage im;
  im.kind = vgpu::cuda::kFatbinLtoIr;
  im.arch = a;
  // The entry's version is the bitcode's own: bytes 4 and 5 of NVVM's header (1.65 under CUDA 13.0).
  if (size >= 6) {
    im.major = static_cast<const uint8_t*>(code)[4];
    im.minor = static_cast<const uint8_t*>(code)[5];
  }
  im.name = identifier ? identifier : "";
  im.options = optionsCmdLine ? optionsCmdLine : "";
  if (size >= 0x2c) im.uncompressed_head.assign(static_cast<const char*>(code) + 0x28, 4);
  im.data.assign(static_cast<const char*>(code), size);
  finish_entry(*F, im);
  im.flags |= arch_flag;
  F->entries.push_back(std::move(im));
  F->dirty = true;
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinAddIndex(nvFatbinHandle handle, const void* code, size_t size,
                                            const char*) {
  if (!handle || !code) return NVFATBIN_ERROR_NULL_POINTER;
  if (!get(handle)) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  // (NVIDIA's own library answers this for text, cubins, fatbins and zeros alike: no tool writes an
  // index file that could be offered it.)
  return NVFATBIN_ERROR_INVALID_INDEX;
}

// The relocatable PTX of a host object built with -dc / -rdc: each PTX image
// of its __nv_relfatbin section, under the object's module identifier (its
// __nv_module_id section), as NVIDIA's does. An identifier may be used once
// per architecture.
VGPU_EXPORT nvFatbinResult nvFatbinAddReloc(nvFatbinHandle handle, const void* code, size_t size) {
  if (!handle || !code) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  if (vgpu::cuda::classify_blob(code, size) != vgpu::cuda::BlobKind::HostObject)
    return NVFATBIN_ERROR_MISSING_FATBIN;
  std::string id;
  std::vector<vgpu::cuda::FatbinImage> found;
  try {
    for (const auto& [name, contents] : vgpu::cuda::elf_sections(code, size))
      if (name == "__nv_module_id") id.assign(contents.c_str());
    for (const std::string& f : vgpu::cuda::host_object_fatbins(code, size, /*relocatable_only=*/true))
      for (auto& im : vgpu::cuda::extract_images(f.data(), f.size()))
        if (im.is_ptx()) found.push_back(std::move(im));
  } catch (const vgpu::Error& e) {
    std::fprintf(stderr, "[vgpu] nvFatbinAddReloc: %s\n", e.message().c_str());
    return NVFATBIN_ERROR_INTERNAL;
  }
  for (const auto& im : found)
    if (F->reloc_ids.count({id, im.arch})) return NVFATBIN_ERROR_IDENTIFIER_REUSE;
  for (auto& im : found) {
    F->reloc_ids.insert({id, im.arch});
    im.kind = vgpu::cuda::kFatbinRelocPtx;
    im.name = id;
    // Copied on as the object stored it, compressed or not, flags and all (measured: even under
    // -compress=false the entry stays as nvcc compressed it).
    im.data = std::move(im.stored_payload);
    im.flags = im.stored_flags;
    im.uncompressed_size = im.stored_uncompressed;
    F->entries.push_back(std::move(im));
    F->dirty = true;
  }
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t* size) {
  if (!handle || !size) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  *size = image_of(*F).size();
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void* buffer) {
  if (!handle || !buffer) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  const std::string& image = image_of(*F);
  std::memcpy(buffer, image.data(), image.size());
  return NVFATBIN_SUCCESS;
}

// For tests: whether entries get zstd-compressed (libzstd could be opened) -- NVIDIA's always can.
extern "C" __attribute__((visibility("default"))) int vgpu_nvfatbin_compresses_with_zstd(void) {
  return !vgpu::cuda::compress_zstd(std::string(64, 'z'), false).empty();
}

VGPU_EXPORT nvFatbinResult nvFatbinVersion(unsigned int* major, unsigned int* minor) {
  if (!major || !minor) return NVFATBIN_ERROR_NULL_POINTER;
  *major = VGPU_NVFATBIN_VERSION / 1000;
  *minor = VGPU_NVFATBIN_VERSION % 1000 / 10;
  return NVFATBIN_SUCCESS;
}
