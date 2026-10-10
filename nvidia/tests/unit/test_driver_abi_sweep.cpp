// The structures the driver shim declares for itself (nvidia/src/driver_sweep_abi.hpp) are the toolkit's: the same
// size, and every field at the toolkit's offset. The shim is built with no toolkit, so this is where a layout it
// reads differently from cuda.h shows up. Compiled against the toolkit's cuda.h.
#include <cuda.h>

#include <cstddef>
#include <cstdio>

#include "driver_sweep_abi.hpp"

using namespace vgpu::abi;

#define SAME_SIZE(Mine, Theirs) static_assert(sizeof(Mine) == sizeof(Theirs), #Mine " is not " #Theirs " in size")
#define SAME_OFFSET(Mine, Theirs, field) \
  static_assert(offsetof(Mine, field) == offsetof(Theirs, field), #Mine "::" #field " is not at " #Theirs "::" #field)

SAME_SIZE(MemLocationABI, CUmemLocation);
SAME_OFFSET(MemLocationABI, CUmemLocation, type);
SAME_OFFSET(MemLocationABI, CUmemLocation, id);

#if CUDA_VERSION >= 12080
SAME_SIZE(MemcpyAttributesABI, CUmemcpyAttributes);
SAME_OFFSET(MemcpyAttributesABI, CUmemcpyAttributes, srcAccessOrder);
SAME_OFFSET(MemcpyAttributesABI, CUmemcpyAttributes, srcLocHint);
SAME_OFFSET(MemcpyAttributesABI, CUmemcpyAttributes, dstLocHint);
SAME_OFFSET(MemcpyAttributesABI, CUmemcpyAttributes, flags);

SAME_SIZE(Memcpy3DOperandABI, CUmemcpy3DOperand);
SAME_OFFSET(Memcpy3DOperandABI, CUmemcpy3DOperand, type);
SAME_OFFSET(Memcpy3DOperandABI, CUmemcpy3DOperand, op);
static_assert(sizeof(Memcpy3DBatchOpABI) == sizeof(CUDA_MEMCPY3D_BATCH_OP), "CUDA_MEMCPY3D_BATCH_OP differs in size");
static_assert(offsetof(Memcpy3DBatchOpABI, src) == offsetof(CUDA_MEMCPY3D_BATCH_OP, src), "src");
static_assert(offsetof(Memcpy3DBatchOpABI, dst) == offsetof(CUDA_MEMCPY3D_BATCH_OP, dst), "dst");
static_assert(offsetof(Memcpy3DBatchOpABI, width) == offsetof(CUDA_MEMCPY3D_BATCH_OP, extent) + offsetof(CUextent3D, width), "extent.width");
static_assert(offsetof(Memcpy3DBatchOpABI, height) == offsetof(CUDA_MEMCPY3D_BATCH_OP, extent) + offsetof(CUextent3D, height), "extent.height");
static_assert(offsetof(Memcpy3DBatchOpABI, depth) == offsetof(CUDA_MEMCPY3D_BATCH_OP, extent) + offsetof(CUextent3D, depth), "extent.depth");
static_assert(offsetof(Memcpy3DBatchOpABI, srcAccessOrder) == offsetof(CUDA_MEMCPY3D_BATCH_OP, srcAccessOrder), "srcAccessOrder");
static_assert(offsetof(Memcpy3DBatchOpABI, flags) == offsetof(CUDA_MEMCPY3D_BATCH_OP, flags), "flags");
// the two operand shapes: a pointer's fields and an array's
static_assert(offsetof(Memcpy3DOperandABI, op.ptr.ptr) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), ptr.ptr), "op.ptr.ptr");
static_assert(offsetof(Memcpy3DOperandABI, op.ptr.rowLength) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), ptr.rowLength), "op.ptr.rowLength");
static_assert(offsetof(Memcpy3DOperandABI, op.ptr.layerHeight) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), ptr.layerHeight), "op.ptr.layerHeight");
static_assert(offsetof(Memcpy3DOperandABI, op.ptr.locHint) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), ptr.locHint), "op.ptr.locHint");
static_assert(offsetof(Memcpy3DOperandABI, op.array.array) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), array.array), "op.array.array");
static_assert(offsetof(Memcpy3DOperandABI, op.array.x) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), array.offset) + offsetof(CUoffset3D, x), "op.array.x");
static_assert(offsetof(Memcpy3DOperandABI, op.array.z) == offsetof(CUmemcpy3DOperand, op) + offsetof(decltype(CUmemcpy3DOperand::op), array.offset) + offsetof(CUoffset3D, z), "op.array.z");
#endif

SAME_SIZE(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcXInBytes);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcMemoryType);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcHost);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcDevice);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcArray);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcContext);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcPitch);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, srcHeight);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstXInBytes);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstMemoryType);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstHost);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstDevice);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstArray);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstContext);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstPitch);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, dstHeight);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, WidthInBytes);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, Height);
SAME_OFFSET(Memcpy3DPeerABI, CUDA_MEMCPY3D_PEER, Depth);

int main() {
  std::printf("PASS: the driver shim's structures are cuda.h's (checked at compile time)\n");
  return 0;
}
