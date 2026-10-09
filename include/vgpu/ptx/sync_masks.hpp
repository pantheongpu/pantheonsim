// Member masks the code generator can see through: a *.sync instruction whose
// mask register is set once, from a constant, is compiled as if the constant
// were written in it (see src/ptx/sync_masks.cpp for why that matters).
#pragma once

#include <cstddef>

#include "vgpu/ptx/ast.hpp"

namespace vgpu::ptx {

// Replaces such mask registers with the constant; returns how many.
size_t fold_constant_sync_masks(EntryFn& fn);

}  // namespace vgpu::ptx
