// PTX parser entry point.
#pragma once

#include <string>

#include "vgpu/ptx/ast.hpp"

namespace vgpu::ptx {

// Parses PTX source into a Module. Throws Err::PtxParse for malformed input
// and Err::UnsupportedPtx for well-formed PTX outside the implemented subset.
Module parse(const std::string& src);

}  // namespace vgpu::ptx
