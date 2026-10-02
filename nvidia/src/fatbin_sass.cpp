// pick_cubin: the cubin the driver runs from a fatbin. Apart from fatbin.cpp
// because it asks the SASS executor what it can run, and nvJitLink and
// nvFatbin, which are built from fatbin.cpp, carry no simulator.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fatbin.hpp"
#include "vgpu/error.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::cuda {

std::string pick_cubin(const void* fatbin, uint32_t cc) {
  if (const char* v = std::getenv("VGPU_SASS"); v && v[0] == '0') return {};
  std::vector<FatbinPtx> elfs;
  try {
    elfs = extract_elf(fatbin);
  } catch (const Error&) {
    return {};
  }
  const FatbinPtx* best = nullptr;
  for (const FatbinPtx& e : elfs) {
    if (e.text.size() < 64 || e.text[0] != 0x7f) continue;
    uint16_t type;
    std::memcpy(&type, e.text.data() + 16, 2);
    if (type != 2 /* ET_EXEC */) continue;
    if (e.arch / 10 != cc / 10 || e.arch > cc) continue;
    if (!best || e.arch > best->arch) best = &e;
  }
  if (!best) return {};
  // SASS the executor cannot run all of yet gives way to the fatbin's PTX,
  // when it has some; VGPU_SASS=1 insists on the SASS (to find what is
  // missing), and VGPU_SASS_LOG=1 says when a fatbin falls back.
  const char* force = std::getenv("VGPU_SASS");
  if (!(force && force[0] == '1')) {
    std::string why;
    try {
      why = vgpu::sass::unsupported(reinterpret_cast<const uint8_t*>(best->text.data()), best->text.size());
    } catch (const Error& e) {
      why = e.message();
    }
    if (!why.empty()) {
      bool has_ptx = false;
      try {
        has_ptx = !extract_ptx(fatbin).empty();
      } catch (const Error&) {
      }
      if (has_ptx) {
        if (const char* log = std::getenv("VGPU_SASS_LOG"); log && log[0] == '1')
          std::fprintf(stderr, "[vgpu] running PTX instead of SASS: %s\n", why.c_str());
        return {};
      }
    }
  }
  if (const char* log = std::getenv("VGPU_SASS_LOG"); log && log[0] == '1')
    std::fprintf(stderr, "[vgpu] running SASS: an sm_%u cubin\n", best->arch);
  return best->text;
}

}  // namespace vgpu::cuda
