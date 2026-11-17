// pick_cubin: the cubin the driver runs from a fatbin. Apart from fatbin.cpp
// because it asks the SASS executor what it can run, and nvJitLink and
// nvFatbin, which are built from fatbin.cpp, carry no simulator.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fatbin.hpp"
#include "vgpu/error.hpp"
#include "vgpu/host_gpu_rt.hpp"
#include "vgpu/sass/cubin.hpp"
#include "vgpu/sass/exec.hpp"

namespace vgpu::cuda {

std::string pick_cubin(const void* fatbin, uint32_t cc) {
  if (const char* v = std::getenv("VGPU_SASS"); v && v[0] == '0') return {};
  std::vector<FatbinPtx> elfs;
  try {
    elfs = extract_elf_for(fatbin, cc);
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
      // PTX this device can JIT, that is: a fatbin whose only PTX is for a newer architecture (an sm_80 build
      // that also embeds compute_90) has nothing to fall back on here, and the SASS runs.
      bool has_ptx = false;
      try {
        for (const FatbinPtx& p : extract_ptx(fatbin)) has_ptx = has_ptx || ptx_runs_on(p, cc);
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

// ---- host-GPU turbo mode ----

namespace {
bool turbo_selected() { return vgpu::hostgpu::turbo_selected(); }

void add_cubin(vgpu::hostgpu::CardSource& src, const char* data, size_t n, uint32_t arch_hint) {
  if (n < 64 || static_cast<unsigned char>(data[0]) != 0x7f) return;
  uint16_t type;
  std::memcpy(&type, data + 16, 2);
  if (type != 2 /* ET_EXEC: a linked image; the driver cannot load a relocatable one */) return;
  vgpu::hostgpu::CardCandidate c;
  c.kind = vgpu::hostgpu::CardCandidate::Cubin;
  const uint32_t arch = arch_hint ? arch_hint : cubin_arch(data, n);
  if (!arch) return;
  c.arch = static_cast<int>(arch);
  c.suffix = vgpu::sass::cubin_arch_specific(reinterpret_cast<const uint8_t*>(data), n) ? 'a' : 0;
  c.bytes.assign(data, n);
  src.candidates.push_back(std::move(c));
}

void add_ptx(vgpu::hostgpu::CardSource& src, const std::string& text, uint32_t header_arch) {
  vgpu::hostgpu::CardCandidate c;
  c.kind = vgpu::hostgpu::CardCandidate::Ptx;
  int arch = 0;
  char suffix = 0;
  if (!vgpu::hostgpu::ptx_target_of(text, &arch, &suffix)) arch = static_cast<int>(header_arch);
  if (!arch) return;
  c.arch = arch;
  c.suffix = suffix;
  c.bytes = text;
  src.candidates.push_back(std::move(c));
}

std::shared_ptr<const vgpu::hostgpu::CardSource> finish(std::shared_ptr<vgpu::hostgpu::CardSource> s) {
  if (s->candidates.empty()) return nullptr;
  return s;
}
}  // namespace

std::shared_ptr<const vgpu::hostgpu::CardSource> card_source_from_fatbin(const void* fatbin) {
  if (!turbo_selected() || !fatbin) return nullptr;
  auto s = std::make_shared<vgpu::hostgpu::CardSource>();
  try {
    for (const FatbinPtx& e : extract_elf(fatbin)) add_cubin(*s, e.text.data(), e.text.size(), e.arch);
  } catch (const Error&) {
  }
  try {
    for (const FatbinPtx& p : extract_ptx(fatbin)) add_ptx(*s, p.text, p.arch);
  } catch (const Error&) {
  }
  return finish(std::move(s));
}

std::shared_ptr<const vgpu::hostgpu::CardSource> card_source_from_cubin(const void* cubin, size_t bytes) {
  if (!turbo_selected() || !cubin) return nullptr;
  auto s = std::make_shared<vgpu::hostgpu::CardSource>();
  add_cubin(*s, static_cast<const char*>(cubin), bytes, 0);
  return finish(std::move(s));
}

std::shared_ptr<const vgpu::hostgpu::CardSource> card_source_from_ptx(const std::string& ptx) {
  if (!turbo_selected() || ptx.empty()) return nullptr;
  auto s = std::make_shared<vgpu::hostgpu::CardSource>();
  add_ptx(*s, ptx, 0);
  return finish(std::move(s));
}

}  // namespace vgpu::cuda
