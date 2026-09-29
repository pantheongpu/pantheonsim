// Number formats both engines convert the same way: defined with the PTX
// interpreter's arithmetic (src/exec/interpreter.cpp) and used by the SASS
// executor too, so a value means the same thing to either.
#pragma once

#include <cstdint>

namespace vgpu::exec {

// FP8: e4m3 (no infinity, max 448) or e5m2 (IEEE-shaped, max 57344).
double fp8_value(uint32_t byte, bool e5m2);
// Round to nearest even; .satfinite clamps to the largest finite value
// instead of giving infinity or NaN.
uint32_t fp8_bits(double v, bool e5m2, bool satfinite);

// Where stored element k of chunk `chunk` in row `row` of a structured-sparse
// mma A operand goes, as a column of the K-wide row, by the metadata the 32
// lanes hold and the selector (mma.sp's rule, as an RTX 3060 places it).
// `bits` is the element width.
uint32_t mma_sparse_column(uint32_t bits, uint32_t K, uint32_t sel, const uint32_t meta[32], uint32_t row,
                           uint32_t chunk, uint32_t k);

}  // namespace vgpu::exec
