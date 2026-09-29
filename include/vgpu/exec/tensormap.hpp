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
  // im2col mode (cuTensorMapEncodeIm2col): the tensor is NWC, NHWC or NDHWC,
  // dimension 0 the channels and the last the batch. A load walks
  // `pixels` pixels through the bounding box in W, H, D space -- from
  // lower[i] to dim + upper[i] - 1 along each, W fastest -- and takes
  // `channels` channels of each. Index 0 of the corners is W.
  bool im2col = false;
  std::array<int32_t, 3> lower{}, upper{};
  uint32_t channels = 0;                    // 1..256
  uint32_t pixels = 0;                      // 1..1024

  // Bits per bounding-box corner value, which is what the hardware keeps
  // (cuTensorMapEncodeIm2col): 16 for a 3D tensor, 8 for 4D, 5 for 5D.
  static uint32_t corner_bits(uint32_t rank) { return rank == 3 ? 16 : rank == 4 ? 8 : 5; }

  static uint32_t type_bytes(TmapType t) {
    switch (t) {
      case TmapType::U8: return 1;
      case TmapType::U16: case TmapType::F16: case TmapType::BF16: return 2;
      case TmapType::U64: case TmapType::S64: case TmapType::F64: return 8;
      case TmapType::U4x16Align8: case TmapType::U4x16Align16: case TmapType::U6x16Align16: return 0;
      default: return 4;
    }
  }

  // The layout. The tile-mode fields sit where NVIDIA's descriptor has them
  // -- tensormap.replace compiles to plain stores into it, so SASS that
  // rewrites a map in place must find each field where ptxas puts it (probed
  // with ptxas, one field and ordinal at a time):
  //   0x00       the global address
  //   0x08       bits 4-6 rank - 1, 7-10 the element type (the ISA's
  //              tensormap.replace numbering; tf32 is f32 or f32.ftz with
  //              bit 16), 11-12 the interleave, 13-14 the swizzle (none, 32,
  //              64, 128 bytes), 15 the NaN fill, 19-21 the 128-byte
  //              swizzle's atomicity (16, 32, 32 with the 8-byte flip, 64)
  //   0x0c+4i    the stride of dimension i + 1 in 16-byte units, low 32 bits;
  //              0x1c holds bits 32-35 of each, a nibble apiece
  //   0x20+4i    dimension i's extent - 1
  //   0x34       bits 3i..3i+2 dimension i's element stride (8 as 0), 24-31
  //              the box's extent in dimension 0 - 1
  //   0x38       byte i: the box's extent in dimension i + 1 - 1
  // What no replace reaches is this simulator's own: the im2col fields at
  // 0x70 and, at 0x78, a magic value, so a map nothing encoded (an
  // uninitialized __grid_constant__, a pointer to the wrong buffer) is
  // reported as that. CuTe clears bit 21 at 0x08 for a map whose global
  // prefix is not contiguous; nothing here keeps anything there.
  static uint32_t type_code(TmapType t, bool* tf32) {
    static const uint8_t codes[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 8, 7, 8, 11, 12, 13};
    *tf32 = t == TmapType::TF32 || t == TmapType::TF32Ftz;
    return codes[static_cast<uint8_t>(t) & 15];
  }
  static bool type_of_code(uint32_t code, bool tf32, TmapType* t) {
    static const TmapType types[] = {TmapType::U8,  TmapType::U16,    TmapType::U32,         TmapType::S32,
                                     TmapType::U64, TmapType::S64,    TmapType::F16,         TmapType::F32,
                                     TmapType::F32Ftz, TmapType::F64, TmapType::BF16,        TmapType::U4x16Align8,
                                     TmapType::U4x16Align16, TmapType::U6x16Align16};
    if (code > 13) return false;
    *t = types[code];
    if (tf32) {
      if (code != 7 && code != 8) return false;
      *t = code == 7 ? TmapType::TF32 : TmapType::TF32Ftz;
    }
    return true;
  }

  void encode(void* out) const {
    uint8_t b[128] = {};
    const auto put32 = [&](unsigned at, uint32_t v) { std::memcpy(b + at, &v, 4); };
    const auto put64 = [&](unsigned at, uint64_t v) { std::memcpy(b + at, &v, 8); };
    put64(0x00, address);
    bool tf32 = false;
    const uint32_t code = type_code(type, &tf32);
    static const uint8_t swz_mode[] = {0, 1, 2, 3, 3, 3, 3}, swz_atom[] = {0, 0, 0, 0, 1, 2, 3};
    const uint8_t sw = static_cast<uint8_t>(swizzle) % 7;
    put32(0x08, ((rank - 1) & 7) << 4 | code << 7 | (uint32_t{interleave} & 3) << 11 | uint32_t{swz_mode[sw]} << 13 |
                    (uint32_t{oob_nan} & 1) << 15 | (tf32 ? 1u << 16 : 0) | uint32_t{swz_atom[sw]} << 19);
    uint32_t hi = 0;
    for (int i = 0; i < 4; ++i) {
      put32(0x0c + 4 * i, static_cast<uint32_t>(stride[i + 1] >> 4));
      hi |= static_cast<uint32_t>((stride[i + 1] >> 36) & 0xF) << (4 * i);
    }
    put32(0x1c, hi);
    // Dimensions past the rank hold an extent, box and stride of 1.
    const auto less_one = [](uint64_t v) { return static_cast<uint32_t>(v ? v - 1 : 0); };
    for (int i = 0; i < 5; ++i) put32(0x20 + 4 * i, less_one(dim[i]));
    uint32_t w34 = (less_one(box[0]) & 0xFF) << 24;
    for (int i = 0; i < 5; ++i) w34 |= ((elem_stride[i] ? elem_stride[i] : 1) & 7) << (3 * i);
    put32(0x34, w34);
    uint32_t w38 = 0;
    for (int i = 1; i < 5; ++i) w38 |= (less_one(box[i]) & 0xFF) << (8 * (i - 1));
    put32(0x38, w38);
    // im2col's fields, zero for a tiled map.
    if (im2col) {
      uint64_t q = 1 | (uint64_t{(channels - 1) & 0xFF} << 8) | (uint64_t{(pixels - 1) & 0x3FF} << 16);
      const uint32_t bits = corner_bits(rank), n = rank - 2;
      const uint64_t mask = (1ull << bits) - 1;
      for (uint32_t i = 0; i < n; ++i) {
        q |= (static_cast<uint64_t>(lower[i]) & mask) << (32 + bits * i);
        q |= (static_cast<uint64_t>(upper[i]) & mask) << (32 + bits * (n + i));
      }
      put64(0x70, q);
    }
    put64(0x78, kMagic);
    std::memcpy(out, b, sizeof b);
  }

  // False when the bytes were not written by encode() (or were rewritten
  // into something no map is).
  bool decode(const void* in) {
    uint8_t b[128];
    std::memcpy(b, in, sizeof b);
    const auto get32 = [&](unsigned at) {
      uint32_t v;
      std::memcpy(&v, b + at, 4);
      return v;
    };
    const auto get64 = [&](unsigned at) {
      uint64_t v;
      std::memcpy(&v, b + at, 8);
      return v;
    };
    if (get64(0x78) != kMagic) return false;
    address = get64(0x00);
    const uint32_t w8 = get32(0x08);
    rank = ((w8 >> 4) & 7) + 1;
    if (!type_of_code((w8 >> 7) & 15, (w8 >> 16) & 1, &type)) return false;
    interleave = static_cast<uint8_t>((w8 >> 11) & 3);
    const uint32_t mode = (w8 >> 13) & 3, atom = (w8 >> 19) & 3;
    swizzle = mode < 3 ? static_cast<TmapSwizzle>(mode)
                       : atom == 0   ? TmapSwizzle::B128
                       : atom == 1 ? TmapSwizzle::B128Atom32
                       : atom == 2 ? TmapSwizzle::B128Atom32Flip8
                                   : TmapSwizzle::B128Atom64;
    oob_nan = static_cast<uint8_t>((w8 >> 15) & 1);
    const uint32_t hi = get32(0x1c);
    stride[0] = type_bytes(type);
    for (int i = 0; i < 4; ++i)
      stride[i + 1] = (uint64_t{get32(0x0c + 4 * i)} | (uint64_t{(hi >> (4 * i)) & 0xF} << 32)) << 4;
    for (int i = 0; i < 5; ++i) dim[i] = uint64_t{get32(0x20 + 4 * i)} + 1;
    const uint32_t w34 = get32(0x34), w38 = get32(0x38);
    box[0] = (w34 >> 24) + 1;
    for (int i = 1; i < 5; ++i) box[i] = ((w38 >> (8 * (i - 1))) & 0xFF) + 1;
    for (int i = 0; i < 5; ++i) {
      elem_stride[i] = (w34 >> (3 * i)) & 7;
      if (!elem_stride[i]) elem_stride[i] = 8;
    }
    const uint64_t q = get64(0x70);
    im2col = q & 1;
    lower = upper = {};
    channels = pixels = 0;
    if (im2col) {
      if (rank < 3) return false;
      channels = static_cast<uint32_t>((q >> 8) & 0xFF) + 1;
      pixels = static_cast<uint32_t>((q >> 16) & 0x3FF) + 1;
      const uint32_t bits = corner_bits(rank), n = rank - 2;
      auto field = [&](uint32_t k) {   // a bits-wide two's-complement value
        const uint64_t v = (q >> (32 + bits * k)) & ((1ull << bits) - 1);
        int32_t x = static_cast<int32_t>(v);
        if (v >> (bits - 1)) x -= int32_t{1} << bits;
        return x;
      };
      for (uint32_t i = 0; i < n; ++i) {
        lower[i] = field(i);
        upper[i] = field(n + i);
      }
    }
    return true;
  }

  // Bytes in a swizzled shared-memory row for the modes implemented here, 0
  // for none, and the chunk the swizzle moves: 16 bytes, or Blackwell's 32
  // and 64 (cuda.h: "Swizzle 32B chunks within 128B span"). The FLIP_8B
  // variant is not implemented and has no span here.
  static uint32_t swizzle_bytes(TmapSwizzle s) {
    switch (s) {
      case TmapSwizzle::B32: return 32;
      case TmapSwizzle::B64: return 64;
      case TmapSwizzle::B128:
      case TmapSwizzle::B128Atom32:
      case TmapSwizzle::B128Atom64: return 128;
      default: return 0;
    }
  }
  // The packed sub-byte types move sixteen values at a time (PTX ISA
  // 5.5.1.1, cuda.h): `global` bytes of them packed in global memory, and in
  // shared memory `shared` bytes -- the same eight for .b4x16, padded to 16
  // for .b4x16_p64 and .b6x16_p32. False for every other type.
  static bool packed_group(TmapType t, uint32_t* global, uint32_t* shared) {
    switch (t) {
      case TmapType::U4x16Align8: *global = 8; *shared = 8; return true;
      case TmapType::U4x16Align16: *global = 8; *shared = 16; return true;
      case TmapType::U6x16Align16: *global = 12; *shared = 16; return true;
      default: return false;
    }
  }
  static uint32_t swizzle_atom(TmapSwizzle s) {
    return s == TmapSwizzle::B128Atom32 ? 32 : s == TmapSwizzle::B128Atom64 ? 64 : 16;
  }
};

// The shared-memory swizzle TMA, wgmma and tcgen05 use (PTX ISA 5.5.7,
// 9.7.17.5.1.2 and 9.7.18.10.6): within each 128-byte line, the index of an
// `atom`-byte chunk is XORed with the line's index modulo the pattern -- for
// 16-byte chunks bits 4.. XOR bits 7.., one bit for 32B, two for 64B and three
// for 128B (CuTe's Swizzle<B,4,3>); for Blackwell's 32- and 64-byte chunks
// in a 128-byte span, bits 5-6 XOR bits 7-8 (Swizzle<2,5,2>) and bit 6 XOR
// bit 7 (Swizzle<1,6,1>).
inline uint64_t swizzle_address(uint64_t addr, uint32_t swizzle_bytes, uint32_t atom = 16) {
  if (!swizzle_bytes) return addr;
  const uint32_t shift = atom == 64 ? 6 : atom == 32 ? 5 : 4;
  return addr ^ (((addr >> 7) & (swizzle_bytes / atom - 1)) << shift);
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
  uint32_t group_global = 0, group_shared = 0;
  const bool packed = TensorMap::packed_group(type, &group_global, &group_shared);
  const bool padded = packed && group_shared == 16;   // 16U4_ALIGN16B, 16U6_ALIGN16B
  if (interleave != 0) return unsupported("an interleaved layout (NC/8HWC8, NC/16HWC16)");
  if (swizzle == 5) return unsupported("CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B");
  const uint64_t addr = reinterpret_cast<uint64_t>(globalAddress);
  if (addr % 16) return bad("globalAddress must be 16-byte aligned");
  // The packed types' own rules (cuda.h, cuTensorMapEncodeTiled).
  if (padded) {
    if (addr % 32) return bad("a 16U4_ALIGN16B or 16U6_ALIGN16B tensor's globalAddress must be 32-byte aligned");
    if (globalDim[0] % 128) return bad("a 16U4_ALIGN16B or 16U6_ALIGN16B tensor's globalDim[0] must be a multiple of 128");
    if (boxDim[0] != 128) return bad("a 16U4_ALIGN16B or 16U6_ALIGN16B tensor's boxDim[0] must be 128");
    for (unsigned i = 0; i + 1 < rank; ++i)
      if (globalStrides[i] % 32)
        return bad("a 16U4_ALIGN16B or 16U6_ALIGN16B tensor's strides must be multiples of 32");
    // 16U6: none, 128B, 128B_ATOM_32B, and 128B_ATOM_64B for stores;
    // 16U4_ALIGN16B: none, 128B and 128B_ATOM_32B, loads only.
    const bool ok = swizzle == 0 || swizzle == 3 || swizzle == 4 ||
                    (type == TmapType::U6x16Align16 && swizzle == 6);
    if (!ok) return bad("this swizzle mode is not one cuda.h allows for a 16U4_ALIGN16B or 16U6_ALIGN16B tensor");
  } else if (packed) {
    if (globalDim[0] % 2) return bad("a 16U4_ALIGN8B tensor's globalDim[0] must be a multiple of 2");
    // A group of sixteen straddling the tensor's edge is not described.
    if (globalDim[0] % 16) return unsupported("a 16U4_ALIGN8B tensor whose globalDim[0] is not a multiple of 16");
  }
  if (packed && oobFill) return bad("the NaN out-of-bounds fill is not available for the packed types");
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
  // The box's inner dimension in bytes: of global memory for the rule on
  // 16-byte multiples, of shared memory against the swizzle span. (A
  // 16U4_ALIGN8B box is sixteen values to eight bytes, and in groups of
  // sixteen.)
  if (packed && boxDim[0] % 16) return bad("a packed type's boxDim[0] must be a multiple of 16 values");
  const uint64_t inner_global = packed ? uint64_t{boxDim[0]} / 16 * group_global : uint64_t{boxDim[0]} * esize;
  const uint64_t inner_shared = packed ? uint64_t{boxDim[0]} / 16 * group_shared : inner_global;
  if (!padded && inner_global % 16)
    return bad("boxDim[0] times the element size must be a multiple of 16 bytes");
  const uint32_t swz = TensorMap::swizzle_bytes(m.swizzle);
  if (swz && inner_shared > swz)
    return bad("the box's inner dimension (" + std::to_string(inner_shared) +
               " bytes) is wider than the " + std::to_string(swz) + "-byte swizzle");
  m.encode(tensorMap);
  return TmapResult::Ok;
}

// cuTensorMapEncodeIm2col's checks and encoding, as for encode_tiled. The
// corners' limits and the other rules are the ones cuda.h documents.
inline TmapResult encode_im2col(void* tensorMap, unsigned dataType, unsigned rank, void* globalAddress,
                                const unsigned long long* globalDim, const unsigned long long* globalStrides,
                                const int* lowerCorner, const int* upperCorner, unsigned channelsPerPixel,
                                unsigned pixelsPerColumn, const unsigned* elementStrides, unsigned interleave,
                                unsigned swizzle, unsigned l2Promotion, unsigned oobFill, std::string* why) {
  auto bad = [&](std::string w) { *why = std::move(w); return TmapResult::Invalid; };
  auto unsupported = [&](std::string w) { *why = std::move(w) + " is not implemented"; return TmapResult::Unsupported; };
  if (!tensorMap || !globalDim || !globalStrides || !lowerCorner || !upperCorner || !elementStrides)
    return bad("a required pointer is null");
  if (reinterpret_cast<uintptr_t>(tensorMap) % 64) return bad("tensorMap must be 64-byte aligned");
  if (dataType > 15) return bad("unknown tensorDataType " + std::to_string(dataType));
  if (rank < 3 || rank > 5) return bad("an im2col tensorRank must be 3, 4 or 5");
  if (interleave > 2 || swizzle > 6 || l2Promotion > 3 || oobFill > 1)
    return bad("an enum argument is out of range");
  const auto type = static_cast<TmapType>(dataType);
  const uint32_t esize = TensorMap::type_bytes(type);
  if (esize == 0) return unsupported("the packed sub-byte types (16U4, 16U6)");
  if (interleave != 0) return unsupported("an interleaved layout (NC/8HWC8, NC/16HWC16)");
  if (swizzle == 5) return unsupported("CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B");
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
    if (elementStrides[i] == 0 || elementStrides[i] > 8)
      return bad("elementStrides[" + std::to_string(i) + "] must be 1 to 8");
    m.dim[i] = globalDim[i];
    m.elem_stride[i] = i == 0 ? 1 : elementStrides[i];
  }
  for (unsigned i = 0; i + 1 < rank; ++i) {
    if (globalStrides[i] % 16 || globalStrides[i] >= (1ull << 40))
      return bad("globalStrides[" + std::to_string(i) + "] must be a multiple of 16 below 2^40");
    m.stride[i + 1] = globalStrides[i];
  }
  const int bits = static_cast<int>(TensorMap::corner_bits(rank));
  const int lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
  for (unsigned i = 0; i + 2 < rank; ++i) {
    if (lowerCorner[i] < lo || lowerCorner[i] > hi || upperCorner[i] < lo || upperCorner[i] > hi)
      return bad("a " + std::to_string(rank) + "D tensor's box corners must be within [" +
                 std::to_string(lo) + ", " + std::to_string(hi) + "]");
    // The box runs from lower to dim + upper - 1, which must hold a pixel.
    if (static_cast<int64_t>(globalDim[i + 1]) + upperCorner[i] - lowerCorner[i] <= 0)
      return bad("the bounding box must have non-zero area");
    m.lower[i] = lowerCorner[i];
    m.upper[i] = upperCorner[i];
  }
  if (channelsPerPixel == 0 || channelsPerPixel > 256) return bad("channelsPerPixel must be 1 to 256");
  if (pixelsPerColumn == 0 || pixelsPerColumn > 1024) return bad("pixelsPerColumn must be 1 to 1024");
  const uint32_t swz = TensorMap::swizzle_bytes(m.swizzle);
  if (swz && uint64_t{channelsPerPixel} * esize > swz)
    return bad("channelsPerPixel times the element size (" + std::to_string(uint64_t{channelsPerPixel} * esize) +
               " bytes) is wider than the " + std::to_string(swz) + "-byte swizzle");
  m.im2col = true;
  m.channels = channelsPerPixel;
  m.pixels = pixelsPerColumn;
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
