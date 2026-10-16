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

// Records, for each f64 add, sub, mul, min, max and fma, which of its two commutative operands ptxas puts first, which decides which of two
// NaN operands an RTX 3060 returns: it orders them by the definition order of their registers, not by the text (probed with PTX
// loaded through the driver: the operand whose register was defined later wins, whatever its name or its position in the instruction;
// a register redefined counts from its latest definition before the instruction; a constant counts as latest). Sets
// OpFloatBin::nan_a_first and OpFma::nan_swap_ab. Run after contract_mul_add; independent of VGPU_PTX_CONTRACT.
void mark_nan_order(EntryFn& fn);

}  // namespace vgpu::ptx
