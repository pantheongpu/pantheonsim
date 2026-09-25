// The TMA tensor map: the 128-byte object cuTensorMapEncodeTiled writes and
// cp.async.bulk.tensor reads.
//
// CUDA documents it as opaque -- "should only be accessed through CUDA APIs
// and PTX" -- so the bytes inside are this simulator's own choice, and the
// only thing that matters is that the encoder and the interpreter agree. It
// opens with a magic value so that a map nothing encoded (an uninitialized
// __grid_constant__, a pointer to the wrong buffer) is reported as that
// instead of being read as a tensor of whatever the bytes happened to say.
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace vgpu::exec {

// The encodings of CUtensorMapDataType, CUtensorMapInterleave,
// CUtensorMapSwizzle and CUtensorMapFloatOOBfill (cuda.h).
enum class TmapType : uint8_t {
  U8 = 0, U16, U32, S32, U64, S64, F16, F32, F64, BF16, F32Ftz, TF32, TF32Ftz,
  U4x16Align8, U4x16Align16, U6x16Align16,
};
enum class TmapSwizzle : uint8_t { None = 0, B32, B64, B128, B128Atom32, B128Atom32Flip8, B128Atom64 };

struct TensorMap {
  static constexpr uint64_t kMagic = 0x3150414d54555047ull;   // "GPUTMAP1"
  uint64_t address = 0;
  uint32_t rank = 0;                        // 1..5
  TmapType type = TmapType::U8;
  uint8_t interleave = 0;
  TmapSwizzle swizzle = TmapSwizzle::None;
  uint8_t oob_nan = 0;                      // CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
  std::array<uint64_t, 5> dim{};            // elements
  std::array<uint64_t, 5> stride{};         // bytes; stride[0] is the element size
  std::array<uint32_t, 5> box{};            // elements traversed per dimension
  std::array<uint32_t, 5> elem_stride{};    // traversal step, 1..8

  static uint32_t type_bytes(TmapType t) {
    switch (t) {
      case TmapType::U8: return 1;
      case TmapType::U16: case TmapType::F16: case TmapType::BF16: return 2;
      case TmapType::U64: case TmapType::S64: case TmapType::F64: return 8;
      case TmapType::U4x16Align8: case TmapType::U4x16Align16: case TmapType::U6x16Align16: return 0;
      default: return 4;
    }
  }

  // 16 qwords: magic, a reserved word, address, a packed word of the small
  // fields, dims[5], strides[4] (the element size is implied by the type),
  // box[0..3], and box[4] with the element strides packed 4 bits each.
  //
  // Opaque does not mean untouched. CuTe clears bit 21 of the second qword
  // of a map whose global prefix is not contiguous -- a flag in NVIDIA's own
  // layout -- so that word carries nothing here. Putting the address there
  // lost bit 21 of it, and a TMA store landed 2 MiB away, in freed memory.
  void encode(void* out) const {
    uint64_t q[16] = {};
    q[0] = kMagic;
    q[2] = address;
    q[3] = uint64_t{rank} | (uint64_t{static_cast<uint8_t>(type)} << 8) | (uint64_t{interleave} << 16) |
           (uint64_t{static_cast<uint8_t>(swizzle)} << 24) | (uint64_t{oob_nan} << 32);
    for (int i = 0; i < 5; ++i) q[4 + i] = dim[i];
    for (int i = 0; i < 4; ++i) q[9 + i] = stride[i + 1];
    for (int i = 0; i < 4; ++i) q[13] |= uint64_t{box[i] & 0xFFFF} << (16 * i);
    q[14] = box[4] & 0xFFFF;
    for (int i = 0; i < 5; ++i) q[14] |= uint64_t{elem_stride[i] & 0xF} << (16 + 4 * i);
    std::memcpy(out, q, sizeof q);
  }

  // False when the bytes were not written by encode().
  bool decode(const void* in) {
    uint64_t q[16];
    std::memcpy(q, in, sizeof q);
    if (q[0] != kMagic) return false;
    address = q[2];
    rank = static_cast<uint32_t>(q[3] & 0xFF);
    type = static_cast<TmapType>((q[3] >> 8) & 0xFF);
    interleave = static_cast<uint8_t>((q[3] >> 16) & 0xFF);
    swizzle = static_cast<TmapSwizzle>((q[3] >> 24) & 0xFF);
    oob_nan = static_cast<uint8_t>((q[3] >> 32) & 0xFF);
    for (int i = 0; i < 5; ++i) dim[i] = q[4 + i];
    stride[0] = type_bytes(type);
    for (int i = 0; i < 4; ++i) stride[i + 1] = q[9 + i];
    for (int i = 0; i < 4; ++i) box[i] = static_cast<uint32_t>((q[13] >> (16 * i)) & 0xFFFF);
    box[4] = static_cast<uint32_t>(q[14] & 0xFFFF);
    for (int i = 0; i < 5; ++i) elem_stride[i] = static_cast<uint32_t>((q[14] >> (16 + 4 * i)) & 0xF);
    return rank >= 1 && rank <= 5;
  }

  // Bytes in a swizzled shared-memory row for the modes implemented here, 0
  // for none. The 128B atom variants are Blackwell's and are refused.
  static uint32_t swizzle_bytes(TmapSwizzle s) {
    switch (s) {
      case TmapSwizzle::B32: return 32;
      case TmapSwizzle::B64: return 64;
      case TmapSwizzle::B128: return 128;
      default: return 0;
    }
  }
};

// The shared-memory swizzle TMA and wgmma both use (PTX ISA 5.5.7 and
// 9.7.17.5.1.2): within each 128-byte line, the 16-byte chunk index is XORed
// with the line's index modulo the pattern -- bits 4.. XOR bits 7.. of the
// address, taking one bit for 32B, two for 64B and three for 128B.
inline uint64_t swizzle_address(uint64_t addr, uint32_t swizzle_bytes) {
  if (!swizzle_bytes) return addr;
  return addr ^ (((addr >> 7) & (swizzle_bytes / 16 - 1)) << 4);
}

// cuTensorMapEncodeTiled's checks and encoding, shared by the driver and the
// runtime (which answers cudaGetDriverEntryPoint for it without loading the
// driver library). Invalid means CUDA_ERROR_INVALID_VALUE and Unsupported
// CUDA_ERROR_NOT_SUPPORTED; `why` names the documented rule that was broken.
enum class TmapResult { Ok, Invalid, Unsupported };

inline TmapResult encode_tiled(void* tensorMap, unsigned dataType, unsigned rank, void* globalAddress,
                               const unsigned long long* globalDim, const unsigned long long* globalStrides,
                               const unsigned* boxDim, const unsigned* elementStrides, unsigned interleave,
                               unsigned swizzle, unsigned l2Promotion, unsigned oobFill, std::string* why) {
  auto bad = [&](std::string w) { *why = std::move(w); return TmapResult::Invalid; };
  auto unsupported = [&](std::string w) { *why = std::move(w) + " is not implemented"; return TmapResult::Unsupported; };
  if (!tensorMap || !globalDim || !boxDim || !elementStrides || (rank > 1 && !globalStrides))
    return bad("a required pointer is null");
  if (reinterpret_cast<uintptr_t>(tensorMap) % 64) return bad("tensorMap must be 64-byte aligned");
  if (dataType > 15) return bad("unknown tensorDataType " + std::to_string(dataType));
  if (rank == 0 || rank > 5) return bad("tensorRank must be 1 to 5");
  if (interleave > 2 || swizzle > 6 || l2Promotion > 3 || oobFill > 1)
    return bad("an enum argument is out of range");
  const auto type = static_cast<TmapType>(dataType);
  const uint32_t esize = TensorMap::type_bytes(type);
  if (esize == 0) return unsupported("the packed sub-byte types (16U4, 16U6)");
  if (interleave != 0) return unsupported("an interleaved layout (NC/8HWC8, NC/16HWC16)");
  if (swizzle > 3) return unsupported("the 128B swizzle with 32B or 64B atomicity (Blackwell)");
  const uint64_t addr = reinterpret_cast<uint64_t>(globalAddress);
  if (addr % 16) return bad("globalAddress must be 16-byte aligned");
  TensorMap m;
  m.address = addr;
  m.rank = rank;
  m.type = type;
  m.swizzle = static_cast<TmapSwizzle>(swizzle);
  const bool is_float = type == TmapType::F16 || type == TmapType::F32 || type == TmapType::F64 ||
                        type == TmapType::BF16 || type == TmapType::F32Ftz ||
                        type == TmapType::TF32 || type == TmapType::TF32Ftz;
  if (oobFill && !is_float) return bad("the NaN out-of-bounds fill needs a floating-point type");
  m.oob_nan = static_cast<uint8_t>(oobFill);
  m.stride[0] = esize;
  for (unsigned i = 0; i < rank; ++i) {
    if (globalDim[i] == 0 || globalDim[i] > (1ull << 32))
      return bad("globalDim[" + std::to_string(i) + "] must be 1 to 2^32");
    if (boxDim[i] == 0 || boxDim[i] > 256) return bad("boxDim[" + std::to_string(i) + "] must be 1 to 256");
    if (elementStrides[i] == 0 || elementStrides[i] > 8)
      return bad("elementStrides[" + std::to_string(i) + "] must be 1 to 8");
    m.dim[i] = globalDim[i];
    m.box[i] = boxDim[i];
    // With no interleave the first element stride is ignored: TMA has no
    // stride along dimension 0.
    m.elem_stride[i] = i == 0 ? 1 : elementStrides[i];
  }
  for (unsigned i = 0; i + 1 < rank; ++i) {
    if (globalStrides[i] % 16 || globalStrides[i] >= (1ull << 40))
      return bad("globalStrides[" + std::to_string(i) + "] must be a multiple of 16 below 2^40");
    m.stride[i + 1] = globalStrides[i];
  }
  if (uint64_t{boxDim[0]} * esize % 16)
    return bad("boxDim[0] times the element size must be a multiple of 16 bytes");
  const uint32_t swz = TensorMap::swizzle_bytes(m.swizzle);
  if (swz && uint64_t{boxDim[0]} * esize > swz)
    return bad("the box's inner dimension (" + std::to_string(uint64_t{boxDim[0]} * esize) +
               " bytes) is wider than the " + std::to_string(swz) + "-byte swizzle");
  m.encode(tensorMap);
  return TmapResult::Ok;
}

// cuTensorMapReplaceAddress: the same map pointed at a new base.
inline TmapResult replace_address(void* tensorMap, void* globalAddress, std::string* why) {
  TensorMap m;
  if (!tensorMap || !m.decode(tensorMap)) {
    *why = "tensorMap was not made by cuTensorMapEncode*";
    return TmapResult::Invalid;
  }
  if (reinterpret_cast<uint64_t>(globalAddress) % 16) {
    *why = "globalAddress must be 16-byte aligned";
    return TmapResult::Invalid;
  }
  m.address = reinterpret_cast<uint64_t>(globalAddress);
  m.encode(tensorMap);
  return TmapResult::Ok;
}

}  // namespace vgpu::exec
