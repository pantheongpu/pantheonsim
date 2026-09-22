// The register C API (vgpu_regs.h) over the register engine (vgpu/regs.hpp).
#include "vgpu_regs.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "vgpu/machine.hpp"
#include "vgpu/regs.hpp"

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

struct vgpu_regs {
  int gpu;
  vgpu::regs::Space space;
};

namespace {

thread_local std::string g_error;

int fail(int code, std::string message) {
  g_error = std::move(message);
  return code;
}

// Runs an access against the device as it reads now, so a register that
// follows the device -- engine status, the link, error status -- is current.
template <class F>
int with_space(vgpu_regs* h, F&& f) {
  if (!h) return fail(VGPU_REGS_EINVAL, "no register handle");
  try {
    auto snap = std::make_unique<vgpu::telemetry::Shared>();
    if (!vgpu::read_machine(snap.get())) return fail(VGPU_REGS_ENODEV, "no machine to describe");
    if (h->gpu < 0 || static_cast<uint32_t>(h->gpu) >= snap->device_count)
      return fail(VGPU_REGS_ENODEV, "no GPU " + std::to_string(h->gpu));
    vgpu::regs::RegisterSpace rs(h->space, snap->devices[h->gpu]);
    f(rs);
  } catch (const std::invalid_argument& e) {
    return fail(VGPU_REGS_EINVAL, e.what());
  } catch (const std::exception& e) {
    return fail(VGPU_REGS_ERROR, e.what());
  }
  g_error.clear();
  return VGPU_REGS_OK;
}

}  // namespace

VGPU_EXPORT int vgpu_regs_open(int gpu, const char* space, vgpu_regs** out) {
  if (!out) return fail(VGPU_REGS_EINVAL, "no place to put the handle");
  *out = nullptr;
  vgpu::regs::Space s{};
  if (!space || !vgpu::regs::parse_space(space, &s))
    return fail(VGPU_REGS_EINVAL, std::string("unknown register space '") + (space ? space : "") + "'");
  try {
    auto snap = std::make_unique<vgpu::telemetry::Shared>();
    if (!vgpu::read_machine(snap.get())) return fail(VGPU_REGS_ENODEV, "no machine to describe");
    if (gpu < 0 || static_cast<uint32_t>(gpu) >= snap->device_count)
      return fail(VGPU_REGS_ENODEV, "no GPU " + std::to_string(gpu));
    if (!vgpu::regs::has_space(snap->devices[gpu], s))
      return fail(VGPU_REGS_ENOTSUP, std::string(snap->devices[gpu].name) + " has no " + space +
                                         " registers modelled");
  } catch (const std::exception& e) {
    return fail(VGPU_REGS_ERROR, e.what());
  }
  *out = new vgpu_regs{gpu, s};
  g_error.clear();
  return VGPU_REGS_OK;
}

VGPU_EXPORT void vgpu_regs_close(vgpu_regs* regs) { delete regs; }

VGPU_EXPORT int vgpu_regs_read(vgpu_regs* regs, uint32_t offset, uint32_t size, uint32_t* value) {
  if (!value) return fail(VGPU_REGS_EINVAL, "no place to put the value");
  return with_space(regs, [&](vgpu::regs::RegisterSpace& rs) { *value = rs.read(offset, size); });
}

VGPU_EXPORT int vgpu_regs_write(vgpu_regs* regs, uint32_t offset, uint32_t size, uint32_t value) {
  return with_space(regs, [&](vgpu::regs::RegisterSpace& rs) { rs.write(offset, size, value); });
}

VGPU_EXPORT int vgpu_regs_find(const char* space, const char* name, uint32_t* offset, uint32_t* width_bits) {
  vgpu::regs::Space s{};
  if (!space || !vgpu::regs::parse_space(space, &s))
    return fail(VGPU_REGS_EINVAL, std::string("unknown register space '") + (space ? space : "") + "'");
  if (!name) return fail(VGPU_REGS_EINVAL, "no register name");
  try {
    const vgpu::regs::Register* r = vgpu::regs::find(s, name);
    if (!r) return fail(VGPU_REGS_ENOENT, std::string("no register '") + name + "' in " + space);
    if (offset) *offset = r->offset;
    if (width_bits) *width_bits = r->width;
  } catch (const std::exception& e) {
    return fail(VGPU_REGS_ERROR, e.what());
  }
  g_error.clear();
  return VGPU_REGS_OK;
}

VGPU_EXPORT int vgpu_regs_locate(vgpu_regs* regs, const char* name, uint32_t* offset, uint32_t* width_bits) {
  if (!name) return fail(VGPU_REGS_EINVAL, "no register name");
  const vgpu::regs::Register* r = nullptr;
  uint32_t at = vgpu::regs::RegisterSpace::kAbsent;
  const int rc = with_space(regs, [&](vgpu::regs::RegisterSpace& rs) {
    if ((r = vgpu::regs::find(rs.space(), name))) at = rs.offset_of(*r);
  });
  if (rc != VGPU_REGS_OK) return rc;
  if (!r) return fail(VGPU_REGS_ENOENT, std::string("no register '") + name + "'");
  if (at == vgpu::regs::RegisterSpace::kAbsent)
    return fail(VGPU_REGS_ENOENT, std::string("this GPU has no ") + r->capability + " capability");
  if (offset) *offset = at;
  if (width_bits) *width_bits = r->width;
  return VGPU_REGS_OK;
}

VGPU_EXPORT const char* vgpu_regs_last_error(void) { return g_error.c_str(); }
