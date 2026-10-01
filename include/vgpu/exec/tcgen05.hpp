// Blackwell's fifth-generation tensor core (tcgen05, sm_100a), for both
// engines: a CTA's Tensor Memory, and the operations on it -- mma, ld/st's
// fragment layouts, cp and shift -- defined with the PTX interpreter's
// arithmetic (src/exec/interpreter.cpp), called by it and by the SASS
// executor with the operands each has read.
#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::exec {

// A CTA's Tensor Memory (sm_100, PTX ISA 9.7.18.1): 128 lanes of 512 32-bit
// columns, handed out by tcgen05.alloc in power-of-two runs of at least 32
// columns. The cells are made on the first allocation, so a kernel that never
// uses it costs nothing. Real contents start undefined; these start zero, as
// shared memory does here.
struct TensorMemory {
  static constexpr uint32_t kLanes = 128, kCols = 512;
  std::vector<uint32_t> cells;                          // lane-major
  std::map<uint32_t, uint32_t> live;                    // first column -> columns
  bool exclusive = false;                               // the live one is .exclusive
  bool relinquished = false;
  // cta_group::2 allocations are made by one warp in each CTA of the pair,
  // together. The first of the two to arrive makes it in both CTAs and leaves
  // its column here for the other, which takes it instead of allocating again;
  // deallocation works the same way.
  std::deque<std::pair<uint32_t, uint32_t>> peer_alloc, peer_dealloc;
  // tcgen05.mma.ws's four B collector buffers: whether each holds a fill, and
  // of which B (its descriptor, and the instruction descriptor's B type,
  // transpose and N).
  struct Collector {
    bool valid = false;
    uint64_t desc = 0;
    uint32_t b_fields = 0;
  };
  std::array<Collector, 4> collector_b{};
  uint32_t& at(uint32_t lane, uint32_t col) { return cells[size_t{lane} * kCols + col]; }
  bool allocated(uint32_t col) const {
    auto it = live.upper_bound(col);
    if (it == live.begin()) return false;
    --it;
    return col < it->first + it->second;
  }
  bool free_run(uint32_t col, uint32_t n) const {
    for (const auto& [c, len] : live)
      if (c < col + n && col < c + len) return false;
    return col + n <= kCols;
  }
};

// What the tcgen05 operations need from the engine running them: the CTAs
// the operation works on -- one, or with .cta_group::2 the pair, even rank
// first -- their Tensor Memory and shared memory, and a way to fail with the
// instruction's place attached.
class Tcgen05Host {
 public:
  virtual ~Tcgen05Host() = default;
  virtual uint32_t ctas() const = 0;
  virtual uint32_t self() const = 0;   // this CTA's index among them
  virtual TensorMemory& tmem(uint32_t cta) = 0;
  // `bytes` (1 to 8) of CTA `cta`'s shared memory at a CTA-local offset.
  virtual uint64_t smem_load(uint32_t cta, uint64_t off, uint32_t bytes) = 0;
  [[noreturn]] virtual void fail(Err code, const std::string& why) = 0;
};

// A tcgen05 shared-memory matrix descriptor (9.7.18.4.1).
WgmmaDesc tcgen05_desc(uint64_t d, Tcgen05Host& h, bool lbo_abs_ok = false);   // bit 52 needs sm_103a

// One thread's tcgen05.mma, its operands read.
struct Tcgen05Mma {
  ptx::Tcgen05MmaKind kind = ptx::Tcgen05MmaKind::F16;
  uint32_t cta_group = 1;
  bool block_scale = false, sparse = false, ws = false, a_tmem = false;
  uint32_t scale_vec = 0;        // 0: the kind's default
  uint32_t idesc = 0;
  uint64_t a = 0;                // A's descriptor, or its Tensor Memory address
  uint64_t b = 0;                // B's descriptor
  uint32_t d = 0;                // D's Tensor Memory address
  bool accumulate = true;        // enable-input-d
  uint32_t sp_meta = 0, scale_a = 0, scale_b = 0;
  bool has_zero_mask = false;
  uint64_t zero_mask = 0;
  std::array<uint32_t, 8> disable_lanes{};
  int scale_d = -1;              // scale-input-d, -1 when absent
  ptx::Tcgen05Collector collector = ptx::Tcgen05Collector::Discard;
  uint32_t collector_buf = 0;
  // The module's target: sm_103a alone has K = 96 fp4 and absolute
  // leading-dimension descriptors (9.7.18.2.1.1, 9.7.18.4.1.1).
  int target_sm = 0;
  bool target_arch = false;   // an architecture-specific ("a") target
};
void tcgen05_mma(const Tcgen05Mma& op, Tcgen05Host& h);

// Where register j of thread t goes for a tcgen05.ld/st shape: the lane
// relative to the address's lane, and the 32-bit column relative to its
// column (figures 186-190). .16x32bx2's upper half-warp adds
// immHalfSplitoff to the column; the caller does that.
void tmem_fragment(ptx::Tcgen05Shape s, uint32_t t, uint32_t j, uint32_t* lane, uint32_t* col);

// tcgen05.cp, its operands read: the shape, .warpx4/.warpx2 (4, 2 = 02_13,
// 3 = 01_23, 0 none), the decompression (4, 6 or 0), the destination and the
// source descriptor.
struct Tcgen05Cp {
  ptx::Tcgen05CpShape shape = ptx::Tcgen05CpShape::S128x256b;
  int multicast = 0, decompress = 0;
  uint32_t taddr = 0;
  uint64_t desc = 0;
};
void tcgen05_cp(const Tcgen05Cp& op, Tcgen05Host& h);

// tcgen05.shift.down at a Tensor Memory address.
void tcgen05_shift(uint32_t taddr, Tcgen05Host& h);

}  // namespace vgpu::exec
