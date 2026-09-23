#include "vgpu/amd_exec.hpp"

#include <cmath>
#include <limits>
#include <cstring>
#include <vector>

#include "vgpu/amd_gcn.hpp"
#include "vgpu/error.hpp"

namespace vgpu::amd {
namespace {

using gcn::Inst;
using gcn::Operand;
using gcn::OperandKind;

constexpr uint32_t kSgprs = 102;      // s0 through s101
constexpr uint32_t kVgprs = 256;
constexpr uint32_t kLanes = 64;

// Where LDS sits in the one address space a flat access uses. The hardware
// puts it in an aperture the wave reads from src_shared_base; this model puts
// it below every device allocation (vgpu/memory.hpp starts those at
// 0x2000'0000'0000), so an address says for itself which memory it means.
constexpr uint64_t kSharedBase = 0x1000'0000'0000ull;
constexpr uint64_t kSharedSize = 1ull << 20;

float as_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
uint32_t as_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}
double as_double(uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, 8);
  return d;
}
uint64_t as_bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, 8);
  return b;
}
_Float16 as_half(uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return h;
}
uint16_t as_bits(_Float16 h) {
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

// The kinds of float v_cmp_class asks about, one bit each, in the ISA's order.
bool matches_class(float f, uint32_t mask) {
  const bool negative = std::signbit(f);
  uint32_t bit = 0;
  if (std::isnan(f)) bit = 1u << 1;                       // a quiet NaN; nothing here signals
  else if (std::isinf(f)) bit = negative ? 1u << 2 : 1u << 9;
  else if (f == 0) bit = negative ? 1u << 5 : 1u << 6;
  else if (std::fpclassify(f) == FP_SUBNORMAL) bit = negative ? 1u << 4 : 1u << 7;
  else bit = negative ? 1u << 3 : 1u << 8;
  return (mask & bit) != 0;
}

// One wavefront: its own scalar registers, VCC, EXEC and SCC, and 64 lanes of
// vector registers.
struct Wave {
  uint32_t sgpr[kSgprs] = {};
  uint32_t vgpr[kVgprs][kLanes] = {};
  uint64_t vcc = 0, exec = 0;
  bool scc = false;
  uint64_t pc = 0;
  bool done = false;
  bool at_barrier = false;
  uint32_t first_lane = 0;   // this wave's first work-item in the group
};

// The work-group the waves share: its LDS, and how many waves are still to
// reach the barrier.
struct Group {
  std::vector<uint8_t> lds;
  // Each work-item's private memory, one block per lane of the group: what a
  // kernel spills into when it runs out of registers.
  std::vector<uint8_t> scratch;
  uint32_t scratch_per_lane = 0;
  std::vector<Wave> waves;
};

struct Machine {
  const Dispatch& d;
  MemoryManager& mem;
  DispatchStats stats;

  // ---- Register access ----------------------------------------------------

  uint32_t sgpr(const Wave& w, uint32_t i) const {
    if (i >= kSgprs) throw Error::make(Err::Internal, "scalar register ", i, " is past the file");
    return w.sgpr[i];
  }
  void set_sgpr(Wave& w, uint32_t i, uint32_t v) {
    if (i >= kSgprs) throw Error::make(Err::Internal, "scalar register ", i, " is past the file");
    w.sgpr[i] = v;
  }
  uint64_t sgpr64(const Wave& w, uint32_t i) const { return sgpr(w, i) | static_cast<uint64_t>(sgpr(w, i + 1)) << 32; }
  void set_sgpr64(Wave& w, uint32_t i, uint64_t v) {
    set_sgpr(w, i, static_cast<uint32_t>(v));
    set_sgpr(w, i + 1, static_cast<uint32_t>(v >> 32));
  }

  // A scalar operand's value: a register, a special register, or a constant.
  uint64_t scalar(const Wave& w, const Operand& o) const {
    switch (o.kind) {
      case OperandKind::Sgpr: return o.width >= 2 ? sgpr64(w, o.index) : sgpr(w, o.index);
      case OperandKind::Vcc: return w.vcc;
      case OperandKind::Exec: return w.exec;
      case OperandKind::ExecLo: return static_cast<uint32_t>(w.exec);
      case OperandKind::ExecHi: return static_cast<uint32_t>(w.exec >> 32);
      case OperandKind::InlineFloat: return as_bits(static_cast<float>(o.fvalue));
      case OperandKind::SharedBase: return kSharedBase;
      case OperandKind::Inline:
      case OperandKind::Literal: return static_cast<uint64_t>(o.value);
      case OperandKind::M0: return 0;
      case OperandKind::Vgpr:
      case OperandKind::None: break;
    }
    throw Error::make(Err::Internal, "a scalar operand this does not read");
  }
  void write_scalar(Wave& w, const Operand& o, uint64_t v) {
    switch (o.kind) {
      case OperandKind::Sgpr:
        if (o.width >= 2) set_sgpr64(w, o.index, v);
        else set_sgpr(w, o.index, static_cast<uint32_t>(v));
        return;
      case OperandKind::Vcc: w.vcc = v; return;
      case OperandKind::Exec: w.exec = v; return;
      case OperandKind::ExecLo: w.exec = (w.exec & ~0xFFFFFFFFull) | static_cast<uint32_t>(v); return;
      case OperandKind::ExecHi: w.exec = (w.exec & 0xFFFFFFFFull) | (v << 32); return;
      default: break;
    }
    throw Error::make(Err::Internal, "a scalar destination this does not write");
  }

  // A source as one lane sees it: a vector register's lane, or the same
  // scalar value for every lane.
  uint32_t lane_src(const Wave& w, const Operand& o, uint32_t lane) const {
    if (o.kind == OperandKind::Vgpr) return w.vgpr[o.index][lane];
    return static_cast<uint32_t>(scalar(w, o));
  }
  uint64_t lane_src64(const Wave& w, const Operand& o, uint32_t lane) const {
    if (o.kind == OperandKind::Vgpr)
      return w.vgpr[o.index][lane] | static_cast<uint64_t>(w.vgpr[o.index + 1][lane]) << 32;
    // An inline constant is the number itself, so in a 64-bit instruction it
    // is that number as a double, not a float's bits with something above them.
    if (o.kind == OperandKind::InlineFloat) return as_bits(o.fvalue);
    return scalar(w, o);
  }
  // A source read as a float, with the modifiers a VOP3 source carries: the
  // absolute value first, then the negation, as the ISA applies them.
  float lane_float(const Wave& w, const Operand& o, uint32_t lane) const {
    float f = as_float(lane_src(w, o, lane));
    if (o.abs) f = std::fabs(f);
    return o.neg ? -f : f;
  }
  double lane_double(const Wave& w, const Operand& o, uint32_t lane) const {
    double d = as_double(lane_src64(w, o, lane));
    if (o.abs) d = std::fabs(d);
    return o.neg ? -d : d;
  }
  // A float result, held to [0, 1] where the instruction asked for it.
  void write_float(Wave& w, const Inst& in, uint32_t lane, float v) {
    if (in.clamp) v = std::isnan(v) ? 0.0f : std::fmin(1.0f, std::fmax(0.0f, v));
    write_lane(w, in.dst[0], lane, as_bits(v));
  }
  void write_double(Wave& w, const Inst& in, uint32_t lane, double v) {
    if (in.clamp) v = std::isnan(v) ? 0.0 : std::fmin(1.0, std::fmax(0.0, v));
    write_lane64(w, in.dst[0], lane, as_bits(v));
  }
  void write_lane(Wave& w, const Operand& o, uint32_t lane, uint32_t v) { w.vgpr[o.index][lane] = v; }
  void write_lane64(Wave& w, const Operand& o, uint32_t lane, uint64_t v) {
    w.vgpr[o.index][lane] = static_cast<uint32_t>(v);
    w.vgpr[o.index + 1][lane] = static_cast<uint32_t>(v >> 32);
  }

  // ---- The instructions ---------------------------------------------------

  void scalar_alu(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    const uint64_t a = in.src.empty() ? 0 : scalar(w, in.src[0]);
    const uint64_t b = in.src.size() > 1 ? scalar(w, in.src[1]) : 0;
    if (op == "s_getpc_b64") {
      // The address of the instruction after this one, which is what the
      // hardware gives: the program counter has already moved on.
      write_scalar(w, in.dst[0], w.pc);
    } else if (op == "s_mov_b32" || op == "s_mov_b64") {
      write_scalar(w, in.dst[0], a);
    } else if (op == "s_movk_i32") {
      write_scalar(w, in.dst[0], static_cast<uint64_t>(static_cast<int64_t>(in.simm)));
    } else if (op == "s_add_u32") {
      const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum >> 32;                           // the carry out
    } else if (op == "s_addc_u32") {
      const uint64_t sum = static_cast<uint32_t>(a) + static_cast<uint64_t>(static_cast<uint32_t>(b)) + w.scc;
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum >> 32;
    } else if (op == "s_add_i32") {
      const int64_t sum = static_cast<int32_t>(a) + static_cast<int64_t>(static_cast<int32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(sum));
      w.scc = sum != static_cast<int32_t>(sum);      // signed overflow
    } else if (op == "s_sub_i32") {
      const int64_t diff = static_cast<int32_t>(a) - static_cast<int64_t>(static_cast<int32_t>(b));
      write_scalar(w, in.dst[0], static_cast<uint32_t>(diff));
      w.scc = diff != static_cast<int32_t>(diff);
    } else if (op == "s_or_b32") {
      const uint32_t v = static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_xor_b32") {
      const uint32_t v = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_ashr_i32") {
      const uint32_t v = static_cast<uint32_t>(static_cast<int32_t>(a) >> (b & 31));
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_and_b32") {
      const uint32_t v = static_cast<uint32_t>(a) & static_cast<uint32_t>(b);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_xor_b64") {
      const uint64_t v = a ^ b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_andn2_b64") {
      const uint64_t v = a & ~b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_ff1_i32_b64") {
      // The first set bit, counting from bit 0, or -1 when there is none.
      write_scalar(w, in.dst[0], a ? static_cast<uint32_t>(__builtin_ctzll(a)) : 0xFFFFFFFFu);
    } else if (op == "s_and_b64") {
      const uint64_t v = a & b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_or_b64") {
      const uint64_t v = a | b;
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_lshl_b64") {
      const uint64_t v = a << (b & 63);
      write_scalar(w, in.dst[0], v);
      w.scc = v != 0;
    } else if (op == "s_cselect_b64") {
      // What the last comparison decided picks a source.
      write_scalar(w, in.dst[0], w.scc ? a : b);
    } else if (op == "s_and_saveexec_b64") {
      // Divergence, as the compiler writes it: keep EXEC, narrow it to the
      // lanes the condition took.
      const uint64_t saved = w.exec;
      w.exec = a & saved;
      write_scalar(w, in.dst[0], saved);
      w.scc = w.exec != 0;
    } else {
      throw Error::make(Err::Unsupported, "scalar instruction ", op, " is decoded but not implemented");
    }
  }

  // A scalar comparison: SCC is the whole result.
  void scalar_compare(Wave& w, const Inst& in) {
    const uint64_t a = scalar(w, in.src[0]), b = scalar(w, in.src[1]);
    const std::string& op = in.name;
    if (op == "s_cmp_lt_i32") w.scc = static_cast<int32_t>(a) < static_cast<int32_t>(b);
    else if (op == "s_cmp_eq_u32") w.scc = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
    else if (op == "s_cmp_lg_u64") w.scc = a != b;
    else throw Error::make(Err::Unsupported, "scalar comparison ", op, " is decoded but not implemented");
  }

  void scalar_load(Wave& w, const Inst& in) {
    const uint64_t base = scalar(w, in.src[0]) + static_cast<uint64_t>(in.offset);
    const uint32_t words = in.dst[0].width;
    for (uint32_t i = 0; i < words; ++i)
      set_sgpr(w, in.dst[0].index + i, static_cast<uint32_t>(mem.load_scalar(base + 4 * i, 4)));
  }

  static uint32_t first_active(const Wave& w) {
    return w.exec ? static_cast<uint32_t>(__builtin_ctzll(w.exec)) : 0;
  }

  // The three instructions a float division is built from. The compiler
  // emits v_div_scale to bring extreme operands into range, refines a
  // reciprocal, then v_div_fmas and v_div_fixup put the result back.
  //
  // The scaling exists because the hardware's reciprocal is approximate, and
  // this reciprocal is exact, so the sequence lands on the correctly rounded
  // quotient either way.
  void divide_step(Wave& w, const Inst& in, uint32_t lane) {
    const std::string& op = in.name;
    // The same three steps, in double precision: the ISA gives each a form of
    // its own, and the scaling is by 2^128 rather than 2^64.
    if (op.size() > 4 && op.compare(op.size() - 4, 4, "_f64") == 0) {
      if (op == "v_div_scale_f64") {
        const double value = lane_double(w, in.src[0], lane), den = lane_double(w, in.src[1], lane),
                     num = lane_double(w, in.src[2], lane);
        bool scaled = false;
        double out = value;
        // Both have to be ordinary numbers to compare their exponents:
        // ilogb of a zero is INT_MIN, and subtracting that overflows.
        if (std::isfinite(den) && std::isfinite(num) && den != 0 && num != 0) {
          const int64_t de = std::ilogb(den), ne = std::ilogb(num);
          if (ne - de >= 1022 || ne - de <= -1022) {
            out = std::ldexp(value, 128);
            scaled = true;
          }
        }
        write_lane64(w, in.dst[0], lane, as_bits(out));
        if (in.dst.size() > 1) {
          const uint64_t bit = uint64_t{1} << lane;
          write_scalar(w, in.dst[1], (scalar(w, in.dst[1]) & ~bit) | (scaled ? bit : 0));
        }
      } else if (op == "v_div_fmas_f64") {
        const double r = std::fma(lane_double(w, in.src[0], lane), lane_double(w, in.src[1], lane),
                                  lane_double(w, in.src[2], lane));
        write_lane64(w, in.dst[0], lane, as_bits((w.vcc >> lane) & 1 ? std::ldexp(r, 128) : r));
      } else {   // v_div_fixup_f64
        const double q = lane_double(w, in.src[0], lane), den = lane_double(w, in.src[1], lane),
                     num = lane_double(w, in.src[2], lane);
        double out = q;
        if (std::isnan(num) || std::isnan(den)) out = std::numeric_limits<double>::quiet_NaN();
        else if (den == 0) out = num == 0 ? std::numeric_limits<double>::quiet_NaN()
                                          : std::copysign(std::numeric_limits<double>::infinity(), num) *
                                                std::copysign(1.0, den);
        else if (std::isinf(den)) out = std::isinf(num) ? std::numeric_limits<double>::quiet_NaN()
                                                        : std::copysign(0.0, num) * std::copysign(1.0, den);
        else if (std::isinf(num)) out = std::copysign(std::numeric_limits<double>::infinity(), num) *
                                        std::copysign(1.0, den);
        write_lane64(w, in.dst[0], lane, as_bits(out));
      }
      return;
    }
    if (op == "v_div_scale_f32") {
      // src0 is the value to scale, src1 the denominator, src2 the numerator.
      const float value = lane_float(w, in.src[0], lane), den = lane_float(w, in.src[1], lane),
                  num = lane_float(w, in.src[2], lane);
      // Scale only where the quotient would otherwise overflow or flush to
      // zero: the ISA scales by 2^64, and says so in the condition register.
      bool scaled = false;
      float out = value;
      // As above: a zero has no exponent to compare.
      if (std::isfinite(den) && std::isfinite(num) && den != 0 && num != 0) {
        const int64_t de = std::ilogb(den), ne = std::ilogb(num);
        if (ne - de >= 126 || ne - de <= -126) {
          out = std::ldexp(value, 64);
          scaled = true;
        }
      }
      write_lane(w, in.dst[0], lane, as_bits(out));
      if (in.dst.size() > 1) {
        const uint64_t bit = uint64_t{1} << lane;
        write_scalar(w, in.dst[1], (scalar(w, in.dst[1]) & ~bit) | (scaled ? bit : 0));
      }
    } else if (op == "v_div_fmas_f32") {
      // A fused multiply-add, times 2^64 where the scaling said so.
      const float r = std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                               lane_float(w, in.src[2], lane));
      write_lane(w, in.dst[0], lane, as_bits((w.vcc >> lane) & 1 ? std::ldexp(r, 64) : r));
    } else {   // v_div_fixup_f32: the quotient, with the cases the sequence cannot do
      const float q = lane_float(w, in.src[0], lane), den = lane_float(w, in.src[1], lane),
                  num = lane_float(w, in.src[2], lane);
      float out = q;
      if (std::isnan(num) || std::isnan(den)) out = std::numeric_limits<float>::quiet_NaN();
      else if (den == 0) out = num == 0 ? std::numeric_limits<float>::quiet_NaN()
                                        : std::copysign(std::numeric_limits<float>::infinity(), num) *
                                              std::copysign(1.0f, den);
      else if (std::isinf(den)) out = std::isinf(num) ? std::numeric_limits<float>::quiet_NaN()
                                                     : std::copysign(0.0f, num) * std::copysign(1.0f, den);
      else if (std::isinf(num)) out = std::copysign(std::numeric_limits<float>::infinity(), num) *
                                      std::copysign(1.0f, den);
      write_lane(w, in.dst[0], lane, as_bits(out));
    }
  }

  void vector_alu(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;   // EXEC says which lanes write
      if (op == "v_mov_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane));
      } else if (op == "v_add_f32_e32") {
        write_lane(w, in.dst[0], lane,
                   as_bits(as_float(lane_src(w, in.src[0], lane)) + as_float(lane_src(w, in.src[1], lane))));
      } else if (op == "v_mul_f32_e32") {
        write_lane(w, in.dst[0], lane,
                   as_bits(as_float(lane_src(w, in.src[0], lane)) * as_float(lane_src(w, in.src[1], lane))));
      } else if (op == "v_lshlrev_b32_e32") {
        // The "rev" forms shift the second source by the first.
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 31));
      } else if (op == "v_lshrrev_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) >> (lane_src(w, in.src[0], lane) & 31));
      } else if (op == "v_ashrrev_i32_e32") {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(static_cast<int32_t>(lane_src(w, in.src[1], lane)) >>
                                         (lane_src(w, in.src[0], lane) & 31)));
      } else if (op == "v_lshlrev_b64") {
        write_lane64(w, in.dst[0], lane, lane_src64(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 63));
      } else if (op == "v_lshl_add_u32") {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 31)) +
                       lane_src(w, in.src[2], lane));
      } else if (op == "v_lshl_add_u64") {
        write_lane64(w, in.dst[0], lane,
                     (lane_src64(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 63)) +
                         lane_src64(w, in.src[2], lane));
      } else if (op == "v_ashrrev_i64") {
        write_lane64(w, in.dst[0], lane,
                     static_cast<uint64_t>(static_cast<int64_t>(lane_src64(w, in.src[1], lane)) >>
                                           (lane_src(w, in.src[0], lane) & 63)));
      } else if (op == "v_and_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) & lane_src(w, in.src[1], lane));
      } else if (op == "v_or_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) | lane_src(w, in.src[1], lane));
      } else if (op == "v_xor_b32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) ^ lane_src(w, in.src[1], lane));
      } else if (op == "v_add_u32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) + lane_src(w, in.src[1], lane));
      } else if (op == "v_sub_u32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) - lane_src(w, in.src[1], lane));
      } else if (op == "v_add3_u32") {
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[0], lane) + lane_src(w, in.src[1], lane) + lane_src(w, in.src[2], lane));
      } else if (op == "v_mul_lo_u32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane) * lane_src(w, in.src[1], lane));
      } else if (op == "v_mul_hi_i32") {
        const int64_t p = static_cast<int64_t>(static_cast<int32_t>(lane_src(w, in.src[0], lane))) *
                          static_cast<int32_t>(lane_src(w, in.src[1], lane));
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(static_cast<uint64_t>(p) >> 32));
      } else if (op == "v_bfe_u32") {
        // The bits src2 wide starting at src1.
        const uint32_t value = lane_src(w, in.src[0], lane), start = lane_src(w, in.src[1], lane) & 31,
                       width = lane_src(w, in.src[2], lane) & 31;
        write_lane(w, in.dst[0], lane, width == 0 ? 0u : (value >> start) & ((width >= 32) ? ~0u : ((1u << width) - 1)));
      } else if (op == "v_cvt_f32_i32_e32") {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(static_cast<int32_t>(lane_src(w, in.src[0], lane)))));
      } else if (op == "v_cvt_f32_u32_e32") {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(lane_src(w, in.src[0], lane))));
      } else if (op == "v_fmac_f32_e32") {
        // The destination is also the addend.
        write_lane(w, in.dst[0], lane,
                   as_bits(std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                                    as_float(w.vgpr[in.dst[0].index][lane]))));
      } else if (op == "v_fma_f32") {
        write_lane(w, in.dst[0], lane,
                   as_bits(std::fma(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane),
                                    lane_float(w, in.src[2], lane))));
      } else if (op == "v_cndmask_b32_e64") {
        // One lane's bit of the condition register picks a source.
        const uint64_t cond = scalar(w, in.src[2]);
        write_lane(w, in.dst[0], lane, lane_src(w, (cond >> lane) & 1 ? in.src[1] : in.src[0], lane));
      } else if (op == "v_sub_f32_e32") {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) - lane_float(w, in.src[1], lane));
      } else if (op == "v_min_f32_e32") {
        write_float(w, in, lane, std::fmin(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane)));
      } else if (op == "v_max_f32_e32" || op == "v_max_f32_e64") {
        write_float(w, in, lane, std::fmax(lane_float(w, in.src[0], lane), lane_float(w, in.src[1], lane)));
      } else if (op == "v_add_f32_e64") {
        write_float(w, in, lane, lane_float(w, in.src[0], lane) + lane_float(w, in.src[1], lane));
      } else if (op == "v_max_u32_e32") {
        write_lane(w, in.dst[0], lane, std::max(lane_src(w, in.src[0], lane), lane_src(w, in.src[1], lane)));
      } else if (op == "v_subrev_u32_e32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[1], lane) - lane_src(w, in.src[0], lane));
      } else if (op == "v_mul_hi_u32") {
        const uint64_t p = static_cast<uint64_t>(lane_src(w, in.src[0], lane)) * lane_src(w, in.src[1], lane);
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(p >> 32));
      } else if (op == "v_bcnt_u32_b32") {
        // The set bits of the first source, counted into the second.
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[1], lane) + static_cast<uint32_t>(__builtin_popcount(lane_src(w, in.src[0], lane))));
      } else if (op == "v_ffbh_u32_e32") {
        // The leading zeros, and -1 when there is no set bit at all.
        const uint32_t v = lane_src(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane, v ? static_cast<uint32_t>(__builtin_clz(v)) : 0xFFFFFFFFu);
      } else if (op == "v_cvt_i32_f32_e32") {
        const float f = lane_float(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(std::isnan(f)          ? 0
                                         : f <= -2147483648.0f  ? INT32_MIN
                                         : f >= 2147483648.0f   ? INT32_MAX
                                                                : static_cast<int32_t>(f)));
      } else if (op == "v_mul_u32_u24_e32") {
        // Only the low 24 bits of each source take part.
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) & 0xFFFFFF) * (lane_src(w, in.src[1], lane) & 0xFFFFFF));
      } else if (op == "v_readfirstlane_b32") {
        // The value in the first active lane, into a scalar register.
        if (lane != first_active(w)) continue;
        write_scalar(w, in.dst[0], w.vgpr[in.src[0].index][lane]);
      } else if (op == "v_cvt_u32_f32_e32") {
        const float f = lane_float(w, in.src[0], lane);
        write_lane(w, in.dst[0], lane,
                   std::isnan(f) || f <= 0 ? 0u : f >= 4294967296.0f ? 0xFFFFFFFFu : static_cast<uint32_t>(f));
      } else if (op == "v_sqrt_f32_e32") {
        write_float(w, in, lane, std::sqrt(lane_float(w, in.src[0], lane)));
      } else if (op == "v_exp_f32_e32") {
        write_float(w, in, lane, std::exp2(lane_float(w, in.src[0], lane)));
      } else if (op == "v_log_f32_e32") {
        write_float(w, in, lane, std::log2(lane_float(w, in.src[0], lane)));
      } else if (op == "v_cndmask_b32_e32") {
        const uint64_t cond = scalar(w, in.src[2]);
        write_lane(w, in.dst[0], lane, lane_src(w, (cond >> lane) & 1 ? in.src[1] : in.src[0], lane));
      } else if (op == "v_add_f64") {
        write_double(w, in, lane, lane_double(w, in.src[0], lane) + lane_double(w, in.src[1], lane));
      } else if (op == "v_mul_f64") {
        write_double(w, in, lane, lane_double(w, in.src[0], lane) * lane_double(w, in.src[1], lane));
      } else if (op == "v_fma_f64") {
        write_double(w, in, lane,
                     std::fma(lane_double(w, in.src[0], lane), lane_double(w, in.src[1], lane),
                              lane_double(w, in.src[2], lane)));
      } else if (op == "v_fmac_f64_e32") {
        write_double(w, in, lane,
                     std::fma(lane_double(w, in.src[0], lane), lane_double(w, in.src[1], lane),
                              as_double(lane_src64(w, in.dst[0], lane))));
      } else if (op == "v_rcp_f64_e32") {
        write_double(w, in, lane, 1.0 / lane_double(w, in.src[0], lane));
      } else if (op == "v_pk_fma_f16") {
        // Two halves in one register, each its own multiply-add.
        const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane),
                       c = lane_src(w, in.src[2], lane);
        const auto half_fma = [&](uint32_t shift) {
          const _Float16 x = as_half(static_cast<uint16_t>(a >> shift)), y = as_half(static_cast<uint16_t>(b >> shift)),
                         z = as_half(static_cast<uint16_t>(c >> shift));
          return as_bits(static_cast<_Float16>(static_cast<float>(x) * static_cast<float>(y) + static_cast<float>(z)));
        };
        write_lane(w, in.dst[0], lane, half_fma(0) | static_cast<uint32_t>(half_fma(16)) << 16);
      } else if (op == "v_rcp_f32_e32" || op == "v_rcp_iflag_f32_e32") {
        // The hardware's reciprocal is a table good to about one unit in the
        // last place; this is the exact one, so a program that refines it
        // (which is how the compiler divides) lands on the same answer. The
        // iflag form differs only in which exceptions it raises, and nothing
        // here raises any.
        write_float(w, in, lane, 1.0f / lane_float(w, in.src[0], lane));
      } else if (op == "v_mbcnt_lo_u32_b32" || op == "v_mbcnt_hi_u32_b32") {
        // The lanes below this one that are set in the mask, counted into the
        // second source: how a wave numbers its active lanes.
        const uint32_t mask = lane_src(w, in.src[0], lane);
        const uint32_t below = op == "v_mbcnt_lo_u32_b32"
                                   ? (lane >= 32 ? 0xFFFFFFFFu : (lane ? (1u << lane) - 1 : 0))
                                   : (lane < 32 ? 0u : (1u << (lane - 32)) - 1);
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[1], lane) + static_cast<uint32_t>(__builtin_popcount(mask & below)));
      } else if (op.rfind("v_div_", 0) == 0) {
        divide_step(w, in, lane);
      } else if (op == "v_mad_u64_u32") {
        // A 32x32 product added to a 64-bit value, with the carry out.
        const unsigned __int128 p = static_cast<unsigned __int128>(lane_src(w, in.src[0], lane)) *
                                        lane_src(w, in.src[1], lane) +
                                    lane_src64(w, in.src[2], lane);
        write_lane64(w, in.dst[0], lane, static_cast<uint64_t>(p));
        if (in.dst.size() > 1) {
          const uint64_t carry = static_cast<uint64_t>(p >> 64) ? uint64_t{1} << lane : 0;
          write_scalar(w, in.dst[1], (scalar(w, in.dst[1]) & ~(uint64_t{1} << lane)) | carry);
        }
      } else if (op == "v_readlane_b32") {
        // Reads one lane, into a scalar register: not a per-lane operation.
        if (lane != first_active(w)) continue;
        const uint32_t which = static_cast<uint32_t>(scalar(w, in.src[1])) & (kLanes - 1);
        write_scalar(w, in.dst[0], w.vgpr[in.src[0].index][which]);
      } else {
        throw Error::make(Err::Unsupported, "vector instruction ", op, " is decoded but not implemented");
      }
    }
  }

  void compare(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    uint64_t result = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;   // an inactive lane's bit reads 0
      const uint32_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
      bool set = false;
      if (op == "v_cmp_gt_i32_e32") set = static_cast<int32_t>(a) > static_cast<int32_t>(b);
      else if (op == "v_cmp_lt_i32_e32") set = static_cast<int32_t>(a) < static_cast<int32_t>(b);
      else if (op == "v_cmp_gt_u32_e32") set = a > b;
      else if (op == "v_cmp_eq_u32_e32") set = a == b;
      else if (op == "v_cmp_le_u32_e32") set = a <= b;
      else if (op == "v_cmp_ne_u32_e32") set = a != b;
      else if (op == "v_cmp_lt_f32_e64" || op == "v_cmp_lt_f32_e32")
        set = lane_float(w, in.src[0], lane) < lane_float(w, in.src[1], lane);
      else if (op == "v_cmp_gt_f32_e64" || op == "v_cmp_gt_f32_e32")
        set = lane_float(w, in.src[0], lane) > lane_float(w, in.src[1], lane);
      else if (op == "v_cmp_ge_f32_e32") set = lane_float(w, in.src[0], lane) >= lane_float(w, in.src[1], lane);
      else if (op == "v_cmp_class_f32_e32") set = matches_class(lane_float(w, in.src[0], lane), b);
      else throw Error::make(Err::Unsupported, "comparison ", op, " is decoded but not implemented");
      if (set) result |= uint64_t{1} << lane;
    }
    write_scalar(w, in.dst[0], result);
  }

  // One work-item's private memory: its own block of the group's scratch.
  uint8_t* scratch_at(Group& g, const Wave& w, uint32_t lane, uint64_t offset, uint32_t bytes) {
    const uint64_t base = uint64_t{w.first_lane + lane} * g.scratch_per_lane;
    if (offset + bytes > g.scratch_per_lane || base + offset + bytes > g.scratch.size())
      throw Error::make(Err::InvalidValue, "a scratch access at ", offset, " is past the ", g.scratch_per_lane,
                        " bytes a work-item reserved");
    return &g.scratch[base + offset];
  }

  // LDS, which the work-group shares.
  void lds_access(Wave& w, const Inst& in, Group& g) {
    const std::string& op = in.name;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint32_t addr = in.name == "ds_bpermute_b32" ? 0 : lane_src(w, in.src[0], lane);
      const auto at = [&](uint64_t offset) {
        const uint64_t a = addr + offset;
        if (a + 4 > g.lds.size())
          throw Error::make(Err::InvalidValue, "an LDS access at ", a, " is past the ", g.lds.size(),
                            " bytes the kernel reserved");
        return a;
      };
      if (op == "ds_write_b32") {
        const uint32_t v = lane_src(w, in.src[1], lane);
        std::memcpy(&g.lds[at(static_cast<uint64_t>(in.offset))], &v, 4);
      } else if (op == "ds_read_b32") {
        uint32_t v = 0;
        std::memcpy(&v, &g.lds[at(static_cast<uint64_t>(in.offset))], 4);
        write_lane(w, in.dst[0], lane, v);
      } else if (op == "ds_add_u32") {
        // An atomic add in LDS: every lane's addition lands.
        uint32_t v = 0;
        std::memcpy(&v, &g.lds[at(static_cast<uint64_t>(in.offset))], 4);
        v += lane_src(w, in.src[1], lane);
        std::memcpy(&g.lds[at(static_cast<uint64_t>(in.offset))], &v, 4);
      } else if (op == "ds_bpermute_b32") {
        // A lane reads what another lane holds: the address says which, in
        // bytes, and the source register is read across the wave.
        const uint32_t from = (lane_src(w, in.src[0], lane) >> 2) & (kLanes - 1);
        write_lane(w, in.dst[0], lane, w.vgpr[in.src[1].index][from]);
      } else if (op == "ds_read2st64_b32") {
        // Two dwords, each offset by its own count of 64 dwords.
        uint32_t v0 = 0, v1 = 0;
        std::memcpy(&v0, &g.lds[at(uint64_t{static_cast<uint32_t>(in.offset)} * 64 * 4)], 4);
        std::memcpy(&v1, &g.lds[at(uint64_t{static_cast<uint32_t>(in.offset1)} * 64 * 4)], 4);
        write_lane(w, in.dst[0], lane, v0);
        w.vgpr[in.dst[0].index + 1][lane] = v1;
      } else {
        throw Error::make(Err::Unsupported, "LDS instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // A flat address says for itself which memory it means: the shared
  // aperture is LDS, and everything else is the device's.
  void flat_access(Wave& w, const Inst& in, Group& g) {
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint64_t addr = lane_src64(w, in.src[0], lane) + static_cast<uint64_t>(in.offset);
      const bool shared = addr >= kSharedBase && addr < kSharedBase + kSharedSize;
      const uint64_t where = shared ? addr - kSharedBase : addr;
      if (shared && where + 4 > g.lds.size())
        throw Error::make(Err::InvalidValue, "a flat access reaches LDS at ", where, ", past the ", g.lds.size(),
                          " bytes the kernel reserved");
      if (in.name == "flat_store_dword") {
        const uint32_t v = lane_src(w, in.src[1], lane);
        if (shared) std::memcpy(&g.lds[where], &v, 4);
        else mem.store_scalar(where, 4, v);
      } else if (in.name == "flat_load_dword") {
        uint32_t v = 0;
        if (shared) std::memcpy(&v, &g.lds[where], 4);
        else v = static_cast<uint32_t>(mem.load_scalar(where, 4));
        write_lane(w, in.dst[0], lane, v);
      } else {
        throw Error::make(Err::Unsupported, "flat instruction ", in.name, " is decoded but not implemented");
      }
    }
  }

  // A work-item's private memory, which a kernel spills into.
  void scratch_access(Wave& w, const Inst& in, Group& g) {
    const std::string& op = in.name;
    const uint32_t words = op == "scratch_store_dwordx4"   ? 4
                           : op == "scratch_store_dwordx3" ? 3
                           : op == "scratch_store_dwordx2" ? 2
                                                           : 1;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      const uint64_t offset =
          (in.has_vaddr ? lane_src(w, in.src[0], lane) : 0) + static_cast<uint64_t>(in.offset);
      uint8_t* at = scratch_at(g, w, lane, offset, 4 * words);
      if (op.rfind("scratch_store", 0) == 0) {
        for (uint32_t k = 0; k < words; ++k) {
          const uint32_t v = w.vgpr[in.src[1].index + k][lane];
          std::memcpy(at + 4 * k, &v, 4);
        }
      } else if (op.rfind("scratch_load", 0) == 0) {
        for (uint32_t k = 0; k < words; ++k) {
          uint32_t v = 0;
          std::memcpy(&v, at + 4 * k, 4);
          w.vgpr[in.dst[0].index + k][lane] = v;
        }
      } else {
        throw Error::make(Err::Unsupported, "scratch instruction ", op, " is decoded but not implemented");
      }
    }
  }

  void global_access(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      // The address is a 64-bit one in a register pair, or a scalar base with
      // a 32-bit offset per lane.
      const uint64_t addr = (in.has_saddr ? sgpr64(w, in.saddr) + lane_src(w, in.src[0], lane)
                                          : lane_src64(w, in.src[0], lane)) +
                            static_cast<uint64_t>(static_cast<int64_t>(in.offset));
      if (op == "global_load_dword") {
        write_lane(w, in.dst[0], lane, static_cast<uint32_t>(mem.load_scalar(addr, 4)));
      } else if (op == "global_load_dwordx2") {
        write_lane64(w, in.dst[0], lane, mem.load_scalar(addr, 8));
      } else if (op == "global_store_dword") {
        mem.store_scalar(addr, 4, lane_src(w, in.src[1], lane));
      } else if (op == "global_store_dwordx2") {
        mem.store_scalar(addr, 8, lane_src64(w, in.src[1], lane));
      } else if (op == "global_atomic_and" || op == "global_atomic_or") {
        const uint32_t before = static_cast<uint32_t>(mem.load_scalar(addr, 4)), v = lane_src(w, in.src[1], lane);
        mem.store_scalar(addr, 4, op == "global_atomic_and" ? before & v : before | v);
      } else if (op == "global_atomic_cmpswap") {
        // The pair is the value to write and the one it must find.
        const uint32_t value = lane_src(w, in.src[1], lane),
                       expected = w.vgpr[in.src[1].index + 1][lane];
        if (static_cast<uint32_t>(mem.load_scalar(addr, 4)) == expected) mem.store_scalar(addr, 4, value);
      } else if (op == "global_atomic_add") {
        // Lane by lane, which is what makes it atomic: every lane's addition
        // lands, whatever order they come in.
        const uint32_t before = static_cast<uint32_t>(mem.load_scalar(addr, 4));
        mem.store_scalar(addr, 4, before + lane_src(w, in.src[1], lane));
      } else {
        throw Error::make(Err::Unsupported, "memory instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // Runs one instruction. Returns false when the wave has stopped or parked
  // at a barrier, so the group can run another wave.
  bool step(Wave& w, Group& g) {
    const CodeObject& o = *d.object;
    const Inst in = gcn::decode(o.text, w.pc - o.text_addr, w.pc);
    w.pc += in.size;
    ++stats.instructions;
    switch (in.enc) {
      case gcn::Enc::Sop1:
      case gcn::Enc::Sop2:
      case gcn::Enc::Sopk:
        scalar_alu(w, in);
        return true;
      case gcn::Enc::Sopc:
        scalar_compare(w, in);
        return true;
      case gcn::Enc::Smem:
        scalar_load(w, in);
        return true;
      case gcn::Enc::Vop1:
      case gcn::Enc::Vop2:
      case gcn::Enc::Vop3:
      case gcn::Enc::Vop3p:
        // A comparison in its long form is still a comparison: it writes a
        // mask of the lanes that passed, not a value per lane.
        if (in.name.rfind("v_cmp_", 0) == 0) compare(w, in);
        else vector_alu(w, in);
        return true;
      case gcn::Enc::Vopc:
        compare(w, in);
        return true;
      case gcn::Enc::Ds:
        lds_access(w, in, g);
        return true;
      case gcn::Enc::Flat:
        if (in.segment == Inst::Segment::Scratch) scratch_access(w, in, g);
        else if (in.segment == Inst::Segment::Flat) flat_access(w, in, g);
        else global_access(w, in);
        return true;
      case gcn::Enc::Sopp: break;
      default:
        throw Error::make(Err::Unsupported, gcn::enc_name(in.enc), " is decoded but not implemented");
    }
    // The program-flow instructions.
    if (in.name == "s_endpgm") {
      w.done = true;
      return false;
    }
    if (in.name == "s_nop" || in.name == "s_waitcnt") return true;   // nothing is out of order here
    if (in.name == "s_barrier") {
      w.at_barrier = true;
      ++stats.barriers;
      return false;
    }
    if (in.name == "s_branch") {
      w.pc = in.target;
      return true;
    }
    if (in.name == "s_cbranch_execz") {
      if (!w.exec) w.pc = in.target;
      return true;
    }
    if (in.name == "s_cbranch_execnz") {
      if (w.exec) w.pc = in.target;
      return true;
    }
    if (in.name == "s_cbranch_scc0") {
      if (!w.scc) w.pc = in.target;
      return true;
    }
    if (in.name == "s_cbranch_scc1") {
      if (w.scc) w.pc = in.target;
      return true;
    }
    throw Error::make(Err::Unsupported, "instruction ", in.name, " is decoded but not implemented");
  }
};

}  // namespace

DispatchStats execute(const Dispatch& d, MemoryManager& mem) {
  if (!d.object || !d.kernel) throw Error::make(Err::InvalidValue, "a dispatch needs a kernel");
  const Kernel& k = *d.kernel;
  if (d.wave_size != kLanes)
    throw Error::make(Err::InvalidValue, "a CDNA wavefront is ", kLanes, " lanes, not ", d.wave_size);
  const uint64_t threads = uint64_t{d.group_size[0]} * d.group_size[1] * d.group_size[2];
  if (!threads) throw Error::make(Err::InvalidValue, "a work-group has no work-items");
  if (k.max_flat_workgroup_size && threads > k.max_flat_workgroup_size)
    throw Error::make(Err::InvalidValue, "a work-group of ", threads, " work-items is past the ",
                      k.max_flat_workgroup_size, " this kernel allows");

  Machine m{d, mem, {}};
  const uint32_t waves_per_group = static_cast<uint32_t>((threads + kLanes - 1) / kLanes);
  for (uint32_t gz = 0; gz < d.groups[2]; ++gz)
    for (uint32_t gy = 0; gy < d.groups[1]; ++gy)
      for (uint32_t gx = 0; gx < d.groups[0]; ++gx) {
        Group group;
        group.lds.assign(k.group_segment, 0);
        // Each work-item's private memory. A kernel that spills says how much
        // it needs; the rest get none.
        group.scratch_per_lane = (k.private_segment + 3) & ~3u;
        group.scratch.assign(static_cast<size_t>(group.scratch_per_lane) * threads, 0);
        group.waves.resize(waves_per_group);
        for (uint32_t i = 0; i < waves_per_group; ++i) {
          Wave& w = group.waves[i];
          w.pc = k.entry;
          w.first_lane = i * kLanes;
          // The lanes this wave has of the work-group, which is short in the
          // last wave when the group is not a multiple of 64.
          const uint64_t left = threads - w.first_lane;
          w.exec = left >= kLanes ? ~uint64_t{0} : (uint64_t{1} << left) - 1;
          // What the hardware leaves in registers before the first
          // instruction: the user SGPRs the descriptor asked for, then the
          // work-group's id, and each lane's id in v0 (and v1, v2 where the
          // group has those dimensions).
          uint32_t at = 0;
          if (k.private_segment_buffer) at += 4;
          if (k.dispatch_ptr) at += 2;
          if (k.queue_ptr) at += 2;
          if (k.kernarg_segment_ptr) {
            m.set_sgpr64(w, at, d.kernarg);
            at += 2;
          }
          if (k.dispatch_id) at += 2;
          if (k.flat_scratch_init) at += 2;
          m.set_sgpr(w, at, gx);
          m.set_sgpr(w, at + 1, gy);
          m.set_sgpr(w, at + 2, gz);
          for (uint32_t lane = 0; lane < kLanes; ++lane) {
            const uint64_t flat = w.first_lane + lane;
            const uint32_t x = static_cast<uint32_t>(flat % d.group_size[0]),
                           y = static_cast<uint32_t>(flat / d.group_size[0] % d.group_size[1]),
                           z = static_cast<uint32_t>(flat / d.group_size[0] / d.group_size[1]);
            // From ABI version 5 a work-item's three ids are packed into v0,
            // ten bits each, and the kernel pulls them out; before it each id
            // had a register of its own.
            if (d.object->packed_work_item_id()) {
              w.vgpr[0][lane] = x | y << 10 | z << 20;
            } else {
              w.vgpr[0][lane] = x;
              w.vgpr[1][lane] = y;
              w.vgpr[2][lane] = z;
            }
          }
          ++m.stats.waves;
        }

        // The group's waves run until every one has stopped. A wave parked at
        // a barrier waits for the others to reach it, as the hardware makes
        // it wait.
        for (bool working = true; working;) {
          working = false;
          for (Wave& w : group.waves) {
            if (w.done || w.at_barrier) continue;
            working = true;
            while (m.step(w, group)) {
            }
          }
          if (!working) {
            // Every wave is stopped or waiting: release the barrier.
            bool any = false;
            for (Wave& w : group.waves)
              if (w.at_barrier) {
                w.at_barrier = false;
                any = working = true;
              }
            if (!any) break;
          }
        }
      }
  return m.stats;
}

}  // namespace vgpu::amd
