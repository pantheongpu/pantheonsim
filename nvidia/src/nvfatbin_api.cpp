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
// Three differences, each written down where it happens:
//   - Nothing is compressed. -compress and its relatives are accepted; the
//     container is simply larger than NVIDIA's.
//   - A "cubin" that is PTX text -- what VirtualGPU's NVRTC and nvJitLink
//     hand out as one, since PTX is what the simulator runs -- goes in as the
//     PTX it is, so that a program which packages its JIT output still gets
//     a fatbin it can load.
//   - nvFatbinAddIndex is refused: an index names LTO-IR libraries, which only
//     NVIDIA's compiler can use.
// LTO-IR itself is packaged faithfully, since packaging needs no compiler;
// nothing on VirtualGPU can run it, and nvJitLink says so when it meets it.
//
// Result codes follow NVIDIA's library as measured on CUDA 13.0's, including
// the ones that surprise: an architecture is the bare number ("86", not
// "sm_86"), PTX whose .target differs from it is accepted, and a host object
// with no device code adds nothing without failing.
#include "../include/vgpu_nvfatbin.h"

#include <cctype>
#include <cstdio>
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

// "86", "90a", "100f": the XX of sm_XX, which is what nvFatbin takes. NVIDIA's
// refuses "sm_86" with NVFATBIN_ERROR_INVALID_ARCH, and so does this.
bool parse_arch(const char* arch, uint32_t* number) {
  const size_t n = std::strlen(arch);
  size_t i = 0;
  uint32_t v = 0;
  while (i < n && std::isdigit(static_cast<unsigned char>(arch[i])))
    v = v * 10 + static_cast<uint32_t>(arch[i++] - '0');
  if (i == 0 || i + (i < n && (arch[i] == 'a' || arch[i] == 'f')) != n) return false;
  *number = v;
  return true;
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

nvFatbinResult add_ptx_entry(Fatbin& F, std::string text, uint32_t arch, const char* identifier,
                             const char* options) {
  while (!text.empty() && text.back() == '\0') text.pop_back();
  vgpu::cuda::FatbinImage im;
  if (!ptx_version(text, &im.major, &im.minor)) return NVFATBIN_ERROR_MISSING_PTX_VERSION;
  im.kind = vgpu::cuda::kFatbinPtx;
  im.arch = arch;
  im.name = identifier ? identifier : "";
  im.options = options ? options : "";
  im.data = std::move(text);
  F.entries.push_back(std::move(im));
  F.dirty = true;
  return NVFATBIN_SUCCESS;
}

}  // namespace

VGPU_EXPORT const char* nvFatbinGetErrorString(nvFatbinResult result) {
  switch (result) {
    case NVFATBIN_SUCCESS: return "no error";
    case NVFATBIN_ERROR_INTERNAL: return "internal error";
    case NVFATBIN_ERROR_ELF_ARCH_MISMATCH: return "the cubin's architecture is not the one given";
    case NVFATBIN_ERROR_ELF_SIZE_MISMATCH: return "the cubin is not an ELF image of the size given";
    case NVFATBIN_ERROR_MISSING_PTX_VERSION: return "the PTX has no .version directive";
    case NVFATBIN_ERROR_NULL_POINTER: return "a required pointer is NULL";
    case NVFATBIN_ERROR_COMPRESSION_FAILED: return "compression failed";
    case NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED: return "the compressed size does not fit its field";
    case NVFATBIN_ERROR_UNRECOGNIZED_OPTION: return "unrecognized option";
    case NVFATBIN_ERROR_INVALID_ARCH: return "invalid architecture (give the number, e.g. \"86\")";
    case NVFATBIN_ERROR_INVALID_NVVM: return "the input is not NVVM LTO-IR";
    case NVFATBIN_ERROR_EMPTY_INPUT: return "empty input";
    case NVFATBIN_ERROR_MISSING_PTX_ARCH: return "the PTX has no architecture";
    case NVFATBIN_ERROR_PTX_ARCH_MISMATCH: return "the PTX's architecture is not the one given";
    case NVFATBIN_ERROR_MISSING_FATBIN: return "the host object contains no fatbin";
    case NVFATBIN_ERROR_INVALID_INDEX: return "invalid index input";
    case NVFATBIN_ERROR_IDENTIFIER_REUSE:
      return "a relocatable entry's identifier is already used for this architecture";
    case NVFATBIN_ERROR_INTERNAL_PTX_OPTION: return "the PTX options include internal options";
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
  for (size_t i = 0; i < optionsCount; ++i) {
    if (!options[i]) return NVFATBIN_ERROR_NULL_POINTER;
    const std::string o = options[i];
    // Word size, host and language, debug info, and compression: none of
    // them changes what a simulated GPU loads, and nothing is compressed.
    if (o == "-32" || o == "-64" || o == "-c" || o == "-g" || o == "-cuda" || o == "-opencl" ||
        o == "-compress-all" || o.rfind("-compress=", 0) == 0 || o.rfind("-compress-mode=", 0) == 0 ||
        o.rfind("-host=", 0) == 0)
      continue;
    return NVFATBIN_ERROR_UNRECOGNIZED_OPTION;
  }
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
  if (!parse_arch(arch, &a)) return NVFATBIN_ERROR_INVALID_ARCH;
  return add_ptx_entry(*F, std::string(code, size), a, identifier, optionsCmdLine);
}

VGPU_EXPORT nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void* code, size_t size,
                                            const char* arch, const char* identifier) {
  if (!handle || !code || !arch) return NVFATBIN_ERROR_NULL_POINTER;
  Fatbin* F = get(handle);
  if (!F) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  uint32_t a = 0;
  if (!parse_arch(arch, &a)) return NVFATBIN_ERROR_INVALID_ARCH;
  using vgpu::cuda::BlobKind;
  const BlobKind kind = vgpu::cuda::classify_blob(code, size);
  // VirtualGPU's cubins are PTX (see the top of this file).
  if (kind == BlobKind::Ptx)
    return add_ptx_entry(*F, std::string(static_cast<const char*>(code), size), a, identifier,
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
  if (!parse_arch(arch, &a)) return NVFATBIN_ERROR_INVALID_ARCH;
  // Bytes that are not LTO-IR: NVIDIA's answers NVFATBIN_ERROR_INTERNAL.
  if (vgpu::cuda::classify_blob(code, size) != vgpu::cuda::BlobKind::LtoIr)
    return NVFATBIN_ERROR_INTERNAL;
  vgpu::cuda::FatbinImage im;
  im.kind = vgpu::cuda::kFatbinLtoIr;
  im.arch = a;
  im.name = identifier ? identifier : "";
  im.options = optionsCmdLine ? optionsCmdLine : "";
  im.data.assign(static_cast<const char*>(code), size);
  F->entries.push_back(std::move(im));
  F->dirty = true;
  return NVFATBIN_SUCCESS;
}

VGPU_EXPORT nvFatbinResult nvFatbinAddIndex(nvFatbinHandle handle, const void* code, size_t size,
                                            const char*) {
  if (!handle || !code) return NVFATBIN_ERROR_NULL_POINTER;
  if (!get(handle)) return NVFATBIN_ERROR_INTERNAL;
  if (size == 0) return NVFATBIN_ERROR_EMPTY_INPUT;
  std::fprintf(stderr,
               "[vgpu] nvFatbinAddIndex: an index names LTO-IR libraries, which only NVIDIA's "
               "compiler can use; VirtualGPU runs PTX, so the index is not added\n");
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
    while (!im.data.empty() && im.data.back() == '\0') im.data.pop_back();
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

VGPU_EXPORT nvFatbinResult nvFatbinVersion(unsigned int* major, unsigned int* minor) {
  if (!major || !minor) return NVFATBIN_ERROR_NULL_POINTER;
  *major = VGPU_NVFATBIN_VERSION / 1000;
  *minor = VGPU_NVFATBIN_VERSION % 1000 / 10;
  return NVFATBIN_SUCCESS;
}
