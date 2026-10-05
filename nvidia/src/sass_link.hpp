// The device linker for machine code: relocatable cubins (nvcc -rdc / -dc,
// SASS) linked into one cubin the SASS loader runs, as nvJitLink and nvlink
// link them. See sass_link.cpp for what is merged, how, and where each rule
// was measured.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu::cuda {

struct SassLinkInput {
  std::string label;            // the name errors use (a file name, or the caller's)
  std::vector<uint8_t> cubin;   // a relocatable cubin (ELF type ET_REL)
};

struct SassLinkResult {
  bool ok = false;
  std::vector<uint8_t> cubin;   // the linked cubin (ET_EXEC)
  std::string errors;           // nvJitLink's error-log lines, one per problem
};

// Links `inputs` for SASS architecture `arch` (86 for sm_86). Fails -- with
// the reason in `errors` -- on an undefined reference, a relocation or section
// it does not know, and anything malformed; a second definition of a symbol is
// reported in `errors` and dropped, and the link succeeds, as NVIDIA's does.
SassLinkResult link_sass(const std::vector<SassLinkInput>& inputs, uint32_t arch);

// True for a relocatable cubin (ELF type ET_REL), the kind link_sass takes.
bool cubin_relocatable(const void* data, size_t size);

}  // namespace vgpu::cuda
