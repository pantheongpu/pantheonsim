// The RDNA (gfx11) decoder behind gcn::decode and gcn::to_text, which hand
// RDNA targets to it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/amd_gcn.hpp"

namespace vgpu::amd::gcn::rdna {

Inst decode(const std::vector<uint8_t>& code, uint64_t at, uint64_t pc, Target target);
std::string to_text(const Inst& i);

}  // namespace vgpu::amd::gcn::rdna
