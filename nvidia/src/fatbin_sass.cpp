// pick_cubin: the cubin the driver runs from a fatbin. Apart from fatbin.cpp
// because it asks the SASS executor what it can run, and nvJitLink and
// nvFatbin, which are built from fatbin.cpp, carry no simulator.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fatbin.hpp"
#include "vgpu/error.hpp"
#include "vgpu/sass/cubin.hpp"
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
  // The image the driver runs: the newest the device can run (the same major
  // and a minor no newer than the device's), where an sm_XYa image runs on XY
  // alone -- so on a cc 10.3 or 10.7 profile an sm_100a cubin is no candidate,
  // and the fatbin's PTX (when it has some the device can use) is what runs.
  // Of two images for one architecture the arch-specific one wins, as for PTX
  // (pick_ptx).
  const FatbinPtx* best = nullptr;
  bool best_specific = false;
  for (const FatbinPtx& e : elfs) {
    if (e.text.size() < 64 || e.text[0] != 0x7f) continue;
    uint16_t type;
    std::memcpy(&type, e.text.data() + 16, 2);
    if (type != 2 /* ET_EXEC */) continue;
    const bool specific =
        vgpu::sass::cubin_arch_specific(reinterpret_cast<const uint8_t*>(e.text.data()), e.text.size());
    if (!vgpu::sass::runs_on(static_cast<int>(e.arch), specific, static_cast<int>(cc))) continue;
    if (!best || e.arch > best->arch || (e.arch == best->arch && specific && !best_specific)) {
      best = &e;
      best_specific = specific;
    }
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
