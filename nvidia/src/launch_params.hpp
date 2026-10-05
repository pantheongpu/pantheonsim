// The layout of a kernel's parameter block: each parameter at the next offset
// its alignment allows, as the driver packs it for cuLaunchKernel's `extra`
// (CU_LAUNCH_PARAM_BUFFER_POINTER) and cuParamSet*, and as cuFuncGetParamInfo
// reports it. A struct passed by value is a .b8 array whose alignment the PTX
// gives; a scalar aligns to its size.
#ifndef VGPU_LAUNCH_PARAMS_HPP_
#define VGPU_LAUNCH_PARAMS_HPP_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vgpu/ptx/ast.hpp"

namespace vgpu_launch {

// Where each parameter of `fn` starts in a packed block, and the block's size
// when every parameter is present.
inline std::vector<size_t> param_offsets(const vgpu::ptx::EntryFn& fn, size_t* total = nullptr) {
  std::vector<size_t> at;
  size_t end = 0;
  for (const auto& p : fn.params) {
    const size_t align = p.align ? p.align : std::max<uint32_t>(1, p.ty.bytes());
    const size_t o = (end + align - 1) / align * align;
    at.push_back(o);
    end = o + p.size;
  }
  if (total) *total = end;
  return at;
}

// Splits a packed block into one buffer per parameter. Bytes past the end of
// the block read as zero.
inline std::vector<std::vector<uint8_t>> unpack(const vgpu::ptx::EntryFn& fn, const void* block, size_t size) {
  const std::vector<size_t> at = param_offsets(fn);
  std::vector<std::vector<uint8_t>> out(fn.params.size());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i].assign(fn.params[i].size, 0);
    if (block && at[i] < size)
      std::memcpy(out[i].data(), static_cast<const uint8_t*>(block) + at[i],
                  std::min<size_t>(fn.params[i].size, size - at[i]));
  }
  return out;
}

// cuLaunchKernel's `extra`: CU_LAUNCH_PARAM_BUFFER_POINTER (1) then the block,
// CU_LAUNCH_PARAM_BUFFER_SIZE (2) then a pointer to its size, ending with
// CU_LAUNCH_PARAM_END (0). Returns false for anything else.
inline bool read_extra(void** extra, const void** block, size_t* size) {
  *block = nullptr;
  *size = 0;
  bool have_block = false;
  for (size_t i = 0; extra[i] != nullptr || i == 0; i += 2) {
    const auto tag = reinterpret_cast<uintptr_t>(extra[i]);
    if (tag == 0) break;
    if (tag == 1) {
      *block = extra[i + 1];
      have_block = true;
    } else if (tag == 2) {
      if (!extra[i + 1]) return false;
      *size = *static_cast<const size_t*>(extra[i + 1]);
    } else {
      return false;
    }
    if (i > 64) return false;
  }
  return have_block;
}

}  // namespace vgpu_launch

#endif  // VGPU_LAUNCH_PARAMS_HPP_
