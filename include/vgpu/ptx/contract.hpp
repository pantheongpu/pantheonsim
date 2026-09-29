// Multiply-add contraction as the code generator performs it: a mul and the
// add or sub that consumes its product, neither with a rounding modifier,
// become one fma (see src/ptx/contract.cpp for the rules and how they were
// measured).
#pragma once

#include <cstddef>

#include "vgpu/ptx/ast.hpp"

namespace vgpu::ptx {

// Rewrites the function's qualifying adds and subs into fmas; returns how
// many. The multiplies stay, their products now unread.
size_t contract_mul_add(EntryFn& fn);

}  // namespace vgpu::ptx
