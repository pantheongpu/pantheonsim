// The structures the driver shim reads that cuda.h defines, with the sizes and offsets cuda.h gives them: the shim
// is built with no toolkit, so it declares them itself, and nvidia/tests/unit/test_driver_abi_sweep.cpp compares
// every one with the toolkit's (a static_assert per size and field) wherever a cuda.h is at hand. Pointers to the
// toolkit's opaque handles are void*; a CUmemLocation is the pair of ints it is.
#pragma once

#include <cstddef>

namespace vgpu::abi {

struct MemLocationABI {
  int type;
  int id;
};

// CUmemcpyAttributes_v1.
struct MemcpyAttributesABI {
  int srcAccessOrder;
  MemLocationABI srcLocHint, dstLocHint;
  unsigned int flags;
};

// CUmemcpy3DOperand_v1: a pointer (type 1) or an array (type 2).
struct Memcpy3DOperandABI {
  int type;
  union {
    struct {
      unsigned long long ptr;
      size_t rowLength, layerHeight;
      MemLocationABI locHint;
    } ptr;
    struct {
      void* array;
      size_t x, y, z;
    } array;
  } op;
};

// CUDA_MEMCPY3D_BATCH_OP_v1, with CUextent3D's three fields in line.
struct Memcpy3DBatchOpABI {
  Memcpy3DOperandABI src, dst;
  size_t width, height, depth;
  int srcAccessOrder;
  unsigned int flags;
};

// CUDA_MEMCPY3D_PEER_v1.
struct Memcpy3DPeerABI {
  size_t srcXInBytes, srcY, srcZ, srcLOD;
  int srcMemoryType;
  const void* srcHost;
  unsigned long long srcDevice;
  void* srcArray;
  void* srcContext;
  size_t srcPitch, srcHeight;
  size_t dstXInBytes, dstY, dstZ, dstLOD;
  int dstMemoryType;
  void* dstHost;
  unsigned long long dstDevice;
  void* dstArray;
  void* dstContext;
  size_t dstPitch, dstHeight;
  size_t WidthInBytes, Height, Depth;
};

}  // namespace vgpu::abi
