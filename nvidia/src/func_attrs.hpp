// What a kernel's attributes are and what changes them: cudaFuncGetAttributes / cudaFuncSetAttribute in the
// runtime shim, cuFuncGetAttribute / cuFuncSetAttribute / cuKernelGetAttribute / cuKernelSetAttribute in the
// driver shim. One set of rules for both, measured on an RTX 3060 (sm_86, driver 596.36, CUDA 13.0 runtime;
// nvidia/tests/data/exports_sweep_runtime.rtx3060.expected and nvidia/tests/e2e/func_attributes.cu):
//
//  * The dynamic shared memory a launch may ask for (cudaFuncAttributeMaxDynamicSharedMemorySize,
//    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) is, until the program sets it, the default per-block limit
//    (48 KiB) less the kernel's static shared memory: 49152 for none, 48896 for 256 bytes, 0 for a kernel with
//    the whole 48 KiB static, 9152 for 40000. It is not the opt-in limit of the part, which is what this shim
//    used to report for every kernel.
//  * Setting it takes any value from 0 up to the opt-in limit less the static shared memory (101376 less 256 =
//    101120 for a kernel with 256 bytes; 52224 for one with 48 KiB); a negative value or one past that is
//    cudaErrorInvalidValue / CUDA_ERROR_INVALID_VALUE and changes nothing. The value is the limit from then on,
//    below 48 KiB too: a launch asking for more than it (static shared memory not counted) is refused.
//  * The preferred shared memory carveout starts at -1 and takes -1 to 100; anything else is invalid.
//  * The required cluster dimensions (width, height, depth) take any value from 0 up, kept as given. The
//    cluster scheduling policy takes 0 to 2. The non-portable cluster size is kept as the runtime was given it
//    (any int) and as a flag by the driver (1 for any non-zero value).
//  * The rest are read-only: 0 to 7, 10 (cluster size must be set) and anything past 15 are invalid.
//  * State belongs to the kernel on a device (a second device starts again), and a CUkernel and the CUfunction
//    made from it share it.
//  * The kernel's thread limit (maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) is what its
//    registers and __launch_bounds__ allow, not the part's 1024: 128 registers a thread leave 512 threads, 168
//    leave 384, 254 or 255 leave 256, and __launch_bounds__(128) says 128.
//  * constSizeBytes is the module's user constant memory: the __constant__ variables, laid out in order with
//    their alignment (432 bytes for a 400-byte, a 5-byte and a 24-byte aligned to 8 array), the same for every
//    kernel of the module.
//
// Required cluster dimensions and "cluster size must be set" of a kernel the program set nothing for come from the
// kernel's own __cluster_dims__ / .explicitcluster: derived from the documentation, not checked against a card
// (an RTX 3060 has no clusters, and answers 0 for both).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu_funcattr {

// Attribute numbers, the same in cudaFuncAttribute and CUfunction_attribute (the driver's 0 to 7 are read-only
// figures the runtime reports as fields).
constexpr int kMaxThreads = 0, kSharedSize = 1, kConstSize = 2, kLocalSize = 3, kNumRegs = 4, kPtxVersion = 5,
              kBinaryVersion = 6, kCacheModeCA = 7, kMaxDynamicShared = 8, kCarveout = 9, kClusterMustBeSet = 10,
              kClusterWidth = 11, kClusterHeight = 12, kClusterDepth = 13, kNonPortable = 14, kSchedulingPolicy = 15;

// What a program has set for a kernel on one device.
struct State {
  int max_dynamic_shared = -1;               // -1: not set, the default applies
  int carveout = -1;
  std::array<int, 3> cluster{-1, -1, -1};    // -1: not set, the kernel's __cluster_dims__ applies
  int scheduling_policy = 0;
  int non_portable = 0;
};

// The most dynamic shared memory a launch may ask for, as a kernel with `static_bytes` of static shared memory
// on `p` is told it may.
inline int max_dynamic_shared(const State& st, const vgpu::DeviceProfile& p, uint64_t static_bytes) {
  if (st.max_dynamic_shared >= 0) return st.max_dynamic_shared;
  const uint64_t per_block = p.limits.shared_mem_per_block;
  return static_cast<int>(per_block > static_bytes ? per_block - static_bytes : 0);
}

// The largest value cudaFuncAttributeMaxDynamicSharedMemorySize accepts.
inline int max_settable_dynamic_shared(const vgpu::DeviceProfile& p, uint64_t static_bytes) {
  const uint64_t optin = std::max<uint64_t>(p.limits.shared_mem_per_block_optin, p.limits.shared_mem_per_block);
  return static_cast<int>(optin > static_bytes ? optin - static_bytes : 0);
}

// Applies a set. False: the value or the attribute is refused (cudaErrorInvalidValue /
// CUDA_ERROR_INVALID_VALUE) and nothing changed. `driver` selects the one place the two APIs differ: the
// driver keeps the non-portable cluster size as a flag.
inline bool set(State& st, const vgpu::DeviceProfile& p, uint64_t static_bytes, int attr, int value, bool driver) {
  switch (attr) {
    case kMaxDynamicShared:
      if (value < 0 || value > max_settable_dynamic_shared(p, static_bytes)) return false;
      st.max_dynamic_shared = value;
      return true;
    case kCarveout:
      if (value < -1 || value > 100) return false;
      st.carveout = value;
      return true;
    case kClusterWidth:
    case kClusterHeight:
    case kClusterDepth:
      if (value < 0) return false;
      st.cluster[static_cast<size_t>(attr - kClusterWidth)] = value;
      return true;
    case kNonPortable:
      st.non_portable = driver ? (value != 0 ? 1 : 0) : value;
      return true;
    case kSchedulingPolicy:
      if (value < 0 || value > 2) return false;
      st.scheduling_policy = value;
      return true;
    default:
      return false;
  }
}

// The thread limit of a kernel: the part's, the kernel's __launch_bounds__, and the most threads its registers
// leave room for (registers are handed out per warp in units of 256).
inline int max_threads(const vgpu::ptx::EntryFn& fn, const vgpu::DeviceProfile& p, uint32_t regs_per_thread) {
  uint64_t limit = p.limits.max_threads_per_block;
  const uint64_t bound = uint64_t{fn.max_ntid[0]} * std::max(1u, fn.max_ntid[1]) * std::max(1u, fn.max_ntid[2]);
  if (fn.max_ntid[0] && bound) limit = std::min(limit, bound);
  if (regs_per_thread && p.limits.registers_per_block && p.warp_size) {
    const uint64_t per_warp = (uint64_t{regs_per_thread} * p.warp_size + 255) / 256 * 256;
    const uint64_t warps = p.limits.registers_per_block / per_warp;
    limit = std::min<uint64_t>(limit, warps * p.warp_size);
  }
  return static_cast<int>(limit);
}

// What the kernel's cluster attributes read when the program has not set them.
inline int cluster_dim(const State& st, const vgpu::ptx::EntryFn& fn, size_t axis) {
  return st.cluster[axis] >= 0 ? st.cluster[axis] : static_cast<int>(fn.req_cluster[axis]);
}

}  // namespace vgpu_funcattr
