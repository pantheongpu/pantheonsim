#include "vgpu/amd_exec.hpp"

#include <cmath>
#include <limits>
#include <cstring>
#include <array>
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
template <typename T>
bool matches_class(T f, uint32_t mask) {
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
  // The accumulation registers: a second bank a kernel keeps values in when
  // it has more of them than the vector registers hold.
  std::vector<std::array<uint32_t, kLanes>> agpr;
  uint64_t vcc = 0, exec = 0;
  uint32_t m0 = 0;   // a lane number, where an instruction takes one from it
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
      case OperandKind::Vcc: return o.width >= 2 ? w.vcc : static_cast<uint32_t>(w.vcc);
      case OperandKind::Exec: return w.exec;
      case OperandKind::ExecLo: return static_cast<uint32_t>(w.exec);
      case OperandKind::ExecHi: return static_cast<uint32_t>(w.exec >> 32);
      case OperandKind::InlineFloat: return as_bits(static_cast<float>(o.fvalue));
      case OperandKind::SharedBase: return kSharedBase;
      case OperandKind::Inline:
      case OperandKind::Literal: return static_cast<uint64_t>(o.value);
      case OperandKind::M0: return w.m0;
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
      case OperandKind::M0: w.m0 = static_cast<uint32_t>(v); return;
      default: break;
    }
    throw Error::make(Err::Internal, "a scalar destination this does not write");
  }

  // ---- Reading a lane of another lane's register ---------------------------
  //
  // Set while a cross-lane instruction runs: the operand whose lanes were
  // shuffled, and what each lane's copy of it came to.
  const Operand* dpp_operand = nullptr;
  const std::array<uint32_t, kLanes>* dpp_values = nullptr;

  // A source as one lane sees it: a vector register's lane, or the same
  // scalar value for every lane.
  uint32_t lane_src(const Wave& w, const Operand& o, uint32_t lane) const {
    if (&o == dpp_operand) return (*dpp_values)[lane];
    if (o.kind == OperandKind::Agpr)
      return o.index < w.agpr.size() ? w.agpr[o.index][lane] : 0;   // one never written holds nothing
    const uint32_t v = o.kind == OperandKind::Vgpr ? w.vgpr[o.index][lane] : static_cast<uint32_t>(scalar(w, o));
    return o.sel == 6 ? v : selected(v, o.sel, o.sext);
  }
  // The part of a register a sub-dword instruction reads: a byte or a half of
  // it, taken into 32 bits with its sign or without.
  static uint32_t selected(uint32_t v, uint8_t sel, bool sext) {
    if (sel <= 3) {
      const uint8_t byte = static_cast<uint8_t>(v >> (8 * sel));
      return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(byte))) : byte;
    }
    if (sel <= 5) {
      const uint16_t half = static_cast<uint16_t>(sel == 4 ? v : v >> 16);
      return sext ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(half))) : half;
    }
    return v;
  }
  uint64_t lane_src64(const Wave& w, const Operand& o, uint32_t lane) const {
    if (o.kind == OperandKind::Vgpr)
      return w.vgpr[o.index][lane] | static_cast<uint64_t>(w.vgpr[o.index + 1][lane]) << 32;
    // An inline constant is the number itself, so in a 64-bit instruction it
    // is that number as a double, not a float's bits with something above them.
    if (o.kind == OperandKind::InlineFloat) return as_bits(o.fvalue);
    return scalar(w, o);
  }
  // A source read as a half: the low half of the register, or an inline
  // constant, which is the number itself rather than a float's bits.
  _Float16 lane_half(const Wave& w, const Operand& o, uint32_t lane) const {
    _Float16 h;
    if (o.kind == OperandKind::InlineFloat) {
      h = static_cast<_Float16>(o.fvalue);
    } else {
      const uint16_t bits = static_cast<uint16_t>(lane_src(w, o, lane));
      std::memcpy(&h, &bits, 2);
    }
    if (o.abs) h = h < static_cast<_Float16>(0) ? -h : h;
    return o.neg ? -h : h;
  }
  // A half result, which fills the low half of the register and zeroes the
  // high half, as every 16-bit instruction here does.
  void write_half(Wave& w, const Inst& in, uint32_t lane, _Float16 v) {
    uint16_t bits = 0;
    std::memcpy(&bits, &v, 2);
    write_lane(w, in.dst[0], lane, bits);
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
  // An accumulation register is written the first time a kernel uses one, so
  // the bank grows to what the kernel actually asked for rather than always
  // being as large as it could be.
  static std::array<uint32_t, kLanes>& acc(Wave& w, uint32_t index) {
    if (index >= kVgprs)
      throw Error::make(Err::InvalidValue, "an accumulation register numbered ", index, " is past the ", kVgprs,
                        " a wave has");
    if (w.agpr.size() <= index) w.agpr.resize(index + 1);
    return w.agpr[index];
  }
  void write_lane(Wave& w, const Operand& o, uint32_t lane, uint32_t v) {
    if (o.kind == OperandKind::Agpr) acc(w, o.index)[lane] = v;
    else w.vgpr[o.index][lane] = v;
  }
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
    } else if (op == "s_mulk_i32") {
      // The destination is also a source: it is multiplied in place.
      write_scalar(w, in.dst[0], static_cast<uint32_t>(scalar(w, in.dst[0]) * static_cast<uint32_t>(in.simm)));
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
    } else if (op == "s_mul_i32") {
      write_scalar(w, in.dst[0], static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
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
    } else if (op == "s_swappc_b64") {
      // A call: where to come back to is what the program counter already
      // holds, since it was moved past this instruction before it ran.
      const uint64_t to = scalar(w, in.src[0]);
      write_scalar(w, in.dst[0], w.pc);
      w.pc = to;
    } else if (op == "s_setpc_b64") {
      w.pc = scalar(w, in.src[0]);   // the return
    } else if (op == "s_brev_b32") {
      uint32_t v = static_cast<uint32_t>(a), r = 0;
      for (uint32_t k = 0; k < 32; ++k) r |= ((v >> k) & 1) << (31 - k);
      write_scalar(w, in.dst[0], r);
      // A bit reversal sets no condition code, as the move it is a form of
      // does not.
    } else if (op == "s_andn2_saveexec_b64") {
      // The other half of a divergence: keep EXEC, and take the lanes the
      // condition did not.
      const uint64_t saved = w.exec;
      w.exec = a & ~saved;
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
    // A counter, rather than a load. What a card returns is a clock at a
    // fixed rate; what this returns is the instructions the dispatch has
    // retired, which is this model's cycle count. It only ever goes up, which
    // is what a program timing a stretch of its own code depends on.
    if (in.name == "s_memtime" || in.name == "s_memrealtime") {
      set_sgpr(w, in.dst[0].index, static_cast<uint32_t>(stats.instructions));
      set_sgpr(w, in.dst[0].index + 1, static_cast<uint32_t>(stats.instructions >> 32));
      return;
    }
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

  // The arithmetic a 64-bit value needs, which a register pair holds: an add
  // or a subtract over one half at a time, reporting what carried out of it
  // as a mask of lanes, and the forms that read that mask back for the other
  // half. The mask goes to VCC in the short form, and to the pair the long
  // form names.
  bool carry_alu(Wave& w, const Inst& in) {
    const std::string& op = in.name;
    if (op.find("_co_u32") == std::string::npos) return false;
    const bool add = op.rfind("v_add", 0) == 0;
    const bool takes_carry = op.find("_addc_") != std::string::npos || op.find("_subb") != std::string::npos;
    const bool reversed = op.find("subbrev") != std::string::npos;   // it subtracts the first from the second
    const uint64_t carried_in = takes_carry ? scalar(w, in.src[2]) : 0;
    uint64_t carried_out = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      if (!(w.exec >> lane & 1)) continue;
      uint64_t a = lane_src(w, in.src[0], lane), b = lane_src(w, in.src[1], lane);
      if (reversed) std::swap(a, b);
      const uint64_t c = takes_carry ? (carried_in >> lane) & 1 : 0;
      const uint64_t r = add ? a + b + c : a - b - c;
      if (add ? (r >> 32) & 1 : a < b + c) carried_out |= uint64_t{1} << lane;
      write_lane(w, in.dst[0], lane, static_cast<uint32_t>(r));
    }
    write_scalar(w, in.dst[1], carried_out);
    return true;
  }

  // The same arithmetic, with the first source taken from another lane and
  // only some of the lanes written. Narrowing EXEC to those lanes is what
  // keeps the rest of them as they were, since every write here asks EXEC
  // first.
  void cross_lane_alu(Wave& w, const Inst& in) {
    std::array<uint32_t, kLanes> values{};
    const uint64_t writes = dpp_shuffle(w, in, values);
    const uint64_t saved = w.exec;
    dpp_operand = &in.src[0];
    dpp_values = &values;
    w.exec = writes;
    try {
      vector_alu(w, in);
    } catch (...) {
      w.exec = saved;
      dpp_operand = nullptr;
      dpp_values = nullptr;
      throw;
    }
    w.exec = saved;
    dpp_operand = nullptr;
    dpp_values = nullptr;
  }

  void vector_alu(Wave& w, const Inst& in) {
    // The sub-dword form of an instruction does what the short form does,
    // over the part of each register it names, and the cross-lane form does
    // it over the lanes it named, so both are the same arithmetic under the
    // name the short form has.
    const std::string as_short =
        in.sdwa   ? in.name.substr(0, in.name.size() - 5) + "_e32"
        : in.dpp  ? in.name.substr(0, in.name.size() - 4) + "_e32"
                  : std::string();
    const std::string& op = in.sdwa || in.dpp ? as_short : in.name;
    // A sub-dword instruction that writes only part of its destination is
    // refused: every one the compiler has been seen to emit writes all of it,
    // and guessing at the rest would give a wrong answer with nothing to show
    // for it.
    if (in.sdwa && (in.dst_sel != 6 || in.dst_unused != 0))
      throw Error::make(Err::Unsupported, op, " writes only part of its destination, which this does not model");
    if (carry_alu(w, in)) return;
    if (op == "v_writelane_b32") {
      // The one instruction here that names the lane it writes: a scalar
      // value into one lane of a register, whatever EXEC says.
      const uint32_t lane = static_cast<uint32_t>(scalar(w, in.src[1])) & 63;
      w.vgpr[in.dst[0].index][lane] = static_cast<uint32_t>(scalar(w, in.src[0]));
      return;
    }
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
      } else if (op == "v_add_u16_e32") {
        // 16-bit arithmetic writes the low half of the destination and zeroes
        // the high half: the compiler leaves out the mask a widening would
        // otherwise need after one of these.
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) + lane_src(w, in.src[1], lane)));
      } else if (op == "v_lshlrev_b16_e32") {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[1], lane) << (lane_src(w, in.src[0], lane) & 15)));
      } else if (op == "v_mad_legacy_u16") {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) * lane_src(w, in.src[1], lane) +
                                         lane_src(w, in.src[2], lane)));
      } else if (op == "v_accvgpr_read_b32" || op == "v_accvgpr_write_b32") {
        write_lane(w, in.dst[0], lane, lane_src(w, in.src[0], lane));
      } else if (op == "v_cvt_f64_f32_e32") {
        write_lane64(w, in.dst[0], lane, as_bits(static_cast<double>(lane_float(w, in.src[0], lane))));
      } else if (op == "v_cvt_f64_i32_e32") {
        write_lane64(w, in.dst[0], lane,
                     as_bits(static_cast<double>(static_cast<int32_t>(lane_src(w, in.src[0], lane)))));
      } else if (op == "v_cvt_f64_u32_e32") {
        write_lane64(w, in.dst[0], lane, as_bits(static_cast<double>(lane_src(w, in.src[0], lane))));
      } else if (op == "v_trunc_f64_e32") {
        write_double(w, in, lane, std::trunc(lane_double(w, in.src[0], lane)));
      } else if (op == "v_ceil_f64_e32") {
        write_double(w, in, lane, std::ceil(lane_double(w, in.src[0], lane)));
      } else if (op == "v_floor_f64_e32") {
        write_double(w, in, lane, std::floor(lane_double(w, in.src[0], lane)));
      } else if (op == "v_rndne_f64_e32") {
        write_double(w, in, lane, std::nearbyint(lane_double(w, in.src[0], lane)));
      } else if (op == "v_rsq_f64_e32") {
        // As with the reciprocal: the hardware's is a table and this is the
        // exact one, so the refinement the compiler builds around it lands on
        // the same answer either way.
        write_double(w, in, lane, 1.0 / std::sqrt(lane_double(w, in.src[0], lane)));
      } else if (op == "v_min_f64" || op == "v_max_f64") {
        const double x = lane_double(w, in.src[0], lane), y = lane_double(w, in.src[1], lane);
        write_double(w, in, lane, op == "v_min_f64" ? std::fmin(x, y) : std::fmax(x, y));
      } else if (op == "v_ldexp_f64") {
        write_double(w, in, lane,
                     std::ldexp(lane_double(w, in.src[0], lane),
                                static_cast<int32_t>(lane_src(w, in.src[1], lane))));
      } else if (op == "v_mov_b64_e32") {
        write_lane64(w, in.dst[0], lane, lane_src64(w, in.src[0], lane));
      } else if (op == "v_floor_f32_e32") {
        write_float(w, in, lane, std::floor(lane_float(w, in.src[0], lane)));
      } else if (op == "v_ceil_f32_e32") {
        write_float(w, in, lane, std::ceil(lane_float(w, in.src[0], lane)));
      } else if (op == "v_rndne_f32_e32") {
        // To the nearest, and to the even one where it falls in the middle.
        write_float(w, in, lane, std::nearbyint(lane_float(w, in.src[0], lane)));
      } else if (op == "v_sin_f32_e32" || op == "v_cos_f32_e32") {
        // The argument is in turns: a whole turn is 1.0, not 2pi.
        const double turns = static_cast<double>(lane_float(w, in.src[0], lane));
        const double radians = turns * 6.283185307179586476925286766559;
        write_float(w, in, lane,
                    static_cast<float>(op == "v_sin_f32_e32" ? std::sin(radians) : std::cos(radians)));
      } else if (op == "v_ldexp_f32") {
        write_float(w, in, lane,
                    std::ldexp(lane_float(w, in.src[0], lane),
                               static_cast<int32_t>(lane_src(w, in.src[1], lane))));
      } else if (op == "v_min3_f32" || op == "v_max3_f32") {
        const float x = lane_float(w, in.src[0], lane), y = lane_float(w, in.src[1], lane),
                    z = lane_float(w, in.src[2], lane);
        write_float(w, in, lane, op == "v_min3_f32" ? std::fmin(std::fmin(x, y), z)
                                                    : std::fmax(std::fmax(x, y), z));
      } else if (op == "v_min3_i32" || op == "v_max3_i32") {
        const int32_t x = static_cast<int32_t>(lane_src(w, in.src[0], lane)),
                      y = static_cast<int32_t>(lane_src(w, in.src[1], lane)),
                      z = static_cast<int32_t>(lane_src(w, in.src[2], lane));
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(op == "v_min3_i32" ? std::min(std::min(x, y), z)
                                                            : std::max(std::max(x, y), z)));
      } else if (op == "v_perm_b32") {
        // Four bytes chosen out of the eight the two sources make, the first
        // source holding the top four. A selector that asks for a sign or a
        // constant instead is refused: the compiler emits these to move bytes
        // about, and nothing here has been seen to ask for the rest.
        const uint64_t bytes = static_cast<uint64_t>(lane_src(w, in.src[0], lane)) << 32 |
                               lane_src(w, in.src[1], lane);
        const uint32_t sel = lane_src(w, in.src[2], lane);
        uint32_t out = 0;
        for (uint32_t k = 0; k < 4; ++k) {
          const uint32_t which = (sel >> (8 * k)) & 0xFF;
          if (which > 7)
            throw Error::make(Err::Unsupported, "v_perm_b32 asked for byte ", which,
                              ", which is a sign or a constant rather than one of the eight, and this does not "
                              "model those");
          out |= static_cast<uint32_t>((bytes >> (8 * which)) & 0xFF) << (8 * k);
        }
        write_lane(w, in.dst[0], lane, out);
      } else if (op == "v_or3_b32") {
        write_lane(w, in.dst[0], lane,
                   lane_src(w, in.src[0], lane) | lane_src(w, in.src[1], lane) | lane_src(w, in.src[2], lane));
      } else if (op == "v_add_f16_e32") {
        write_half(w, in, lane, lane_half(w, in.src[0], lane) + lane_half(w, in.src[1], lane));
      } else if (op == "v_sub_f16_e32") {
        write_half(w, in, lane, lane_half(w, in.src[0], lane) - lane_half(w, in.src[1], lane));
      } else if (op == "v_mul_f16_e32") {
        write_half(w, in, lane, lane_half(w, in.src[0], lane) * lane_half(w, in.src[1], lane));
      } else if (op == "v_fma_f16") {
        // A fused multiply-add: one rounding, which is what the C means by
        // fma and what the half the compiler folded into it expects.
        write_half(w, in, lane,
                   static_cast<_Float16>(std::fma(static_cast<float>(lane_half(w, in.src[0], lane)),
                                                  static_cast<float>(lane_half(w, in.src[1], lane)),
                                                  static_cast<float>(lane_half(w, in.src[2], lane)))));
      } else if (op == "v_cvt_f16_f32_e32") {
        write_half(w, in, lane, static_cast<_Float16>(lane_float(w, in.src[0], lane)));
      } else if (op == "v_cvt_f32_f16_e32") {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(lane_half(w, in.src[0], lane))));
      } else if (op == "v_mul_i32_i24_e32") {
        // The low 24 bits of each source, as signed numbers.
        const auto i24 = [](uint32_t v) { return static_cast<int32_t>(v << 8) >> 8; };
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(i24(lane_src(w, in.src[0], lane)) * i24(lane_src(w, in.src[1], lane))));
      } else if (op == "v_mul_lo_u16_e32") {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint16_t>(lane_src(w, in.src[0], lane) * lane_src(w, in.src[1], lane)));
      } else if (op == "v_xad_u32") {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) ^ lane_src(w, in.src[1], lane)) + lane_src(w, in.src[2], lane));
      } else if (op == "v_bfrev_b32_e32") {
        uint32_t v = lane_src(w, in.src[0], lane), r = 0;
        for (uint32_t k = 0; k < 32; ++k) r |= ((v >> k) & 1) << (31 - k);
        write_lane(w, in.dst[0], lane, r);
      } else if (op == "v_min_u32_e32") {
        write_lane(w, in.dst[0], lane, std::min(lane_src(w, in.src[0], lane), lane_src(w, in.src[1], lane)));
      } else if (op == "v_lshl_or_b32") {
        write_lane(w, in.dst[0], lane,
                   (lane_src(w, in.src[0], lane) << (lane_src(w, in.src[1], lane) & 31)) |
                       lane_src(w, in.src[2], lane));
      } else if (op == "v_trunc_f32_e32") {
        write_lane(w, in.dst[0], lane, as_bits(std::trunc(lane_float(w, in.src[0], lane))));
      } else if (op == "v_cvt_f32_f64_e32") {
        write_lane(w, in.dst[0], lane, as_bits(static_cast<float>(as_double(lane_src64(w, in.src[0], lane)))));
      } else if (op == "v_cvt_i32_f64_e32") {
        write_lane(w, in.dst[0], lane,
                   static_cast<uint32_t>(static_cast<int32_t>(as_double(lane_src64(w, in.src[0], lane)))));
      } else if (op == "v_fmamk_f32") {
        // The middle source is the constant the instruction carries.
        write_lane(w, in.dst[0], lane,
                   as_bits(std::fma(lane_float(w, in.src[0], lane), as_float(static_cast<uint32_t>(in.src[1].value)),
                                    lane_float(w, in.src[2], lane))));
      } else if (op == "v_bfe_i32") {
        // The same bits as v_bfe_u32, with the top one carried into the rest.
        const uint32_t value = lane_src(w, in.src[0], lane), start = lane_src(w, in.src[1], lane) & 31,
                       width = lane_src(w, in.src[2], lane) & 31;
        uint32_t out = 0;
        if (width != 0) {
          out = width >= 32 ? value >> start : (value >> start) & ((1u << width) - 1);
          if (width < 32 && (out >> (width - 1) & 1)) out |= ~((1u << width) - 1);
        }
        write_lane(w, in.dst[0], lane, out);
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
      } else if (op == "v_sub_f32_e32" || op == "v_sub_f32_e64") {
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
      } else if (op == "v_pk_fma_f32" || op == "v_pk_add_f32" || op == "v_pk_mul_f32") {
        // Two floats in a register pair, each its own arithmetic. Which
        // register of a pair feeds which result is op_sel and op_sel_hi: a
        // zero in op_sel_hi means one value serves both, which is what a
        // constant is.
        const auto part = [&](size_t which, bool high) {
          const Operand& o = in.src[which];
          const uint32_t pick = high ? (in.op_sel_hi >> which) & 1 : (in.op_sel >> which) & 1;
          float f = o.kind == OperandKind::Vgpr ? as_float(w.vgpr[o.index + pick][lane])
                                                : as_float(static_cast<uint32_t>(scalar(w, o)));
          return o.neg ? -f : f;
        };
        for (uint32_t half = 0; half < 2; ++half) {
          const bool high = half == 1;
          const float x = part(0, high), y = part(1, high);
          const float r = op == "v_pk_add_f32"   ? x + y
                          : op == "v_pk_mul_f32" ? x * y
                                                 : std::fma(x, y, part(2, high));
          w.vgpr[in.dst[0].index + half][lane] = as_bits(r);
        }
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
      else if (op == "v_cmp_ne_u32_e32" || op == "v_cmp_ne_u32_e64") set = a != b;
      else if (op == "v_cmp_eq_u32_e64") set = a == b;
      else if (op == "v_cmp_lt_u32_e64") set = a < b;
      else if (op == "v_cmp_ge_u32_e32" || op == "v_cmp_ge_u32_e64") set = a >= b;
      else if (op == "v_cmp_lt_u64_e32")
        set = lane_src64(w, in.src[0], lane) < lane_src64(w, in.src[1], lane);
      else if (op == "v_cmp_eq_u64_e32")
        set = lane_src64(w, in.src[0], lane) == lane_src64(w, in.src[1], lane);
      else if (op == "v_cmp_ne_u64_e32")
        set = lane_src64(w, in.src[0], lane) != lane_src64(w, in.src[1], lane);
      else if (op == "v_cmp_lt_f32_e64" || op == "v_cmp_lt_f32_e32")
        set = lane_float(w, in.src[0], lane) < lane_float(w, in.src[1], lane);
      else if (op == "v_cmp_gt_f32_e64" || op == "v_cmp_gt_f32_e32")
        set = lane_float(w, in.src[0], lane) > lane_float(w, in.src[1], lane);
      else if (op == "v_cmp_ge_f32_e32") set = lane_float(w, in.src[0], lane) >= lane_float(w, in.src[1], lane);
      else if (op == "v_cmp_class_f32_e32") set = matches_class(lane_float(w, in.src[0], lane), b);
      else if (op == "v_cmp_class_f64_e32")
        set = matches_class(lane_double(w, in.src[0], lane), lane_src(w, in.src[1], lane));
      else if (op.find("_f64_") != std::string::npos) {
        const double x = lane_double(w, in.src[0], lane), y = lane_double(w, in.src[1], lane);
        if (op == "v_cmp_lt_f64_e32") set = x < y;
        else if (op == "v_cmp_eq_f64_e32") set = x == y;
        else if (op == "v_cmp_gt_f64_e32") set = x > y;
        else if (op == "v_cmp_ge_f64_e32") set = x >= y;
        else if (op == "v_cmp_neq_f64_e32") set = !(x == y);   // a NaN is not equal to itself
        else throw Error::make(Err::Unsupported, "comparison ", op, " is decoded but not implemented");
      }
      else if (op.find("_f16_") != std::string::npos) {
        const _Float16 x = lane_half(w, in.src[0], lane), y = lane_half(w, in.src[1], lane);
        if (op == "v_cmp_lt_f16_e32") set = x < y;
        else if (op == "v_cmp_eq_f16_e32") set = x == y;
        else if (op == "v_cmp_gt_f16_e32") set = x > y;
        else if (op == "v_cmp_ge_f16_e32") set = x >= y;
        else if (op == "v_cmp_neq_f16_e32") set = !(x == y);   // a NaN is not equal to itself
        else if (op == "v_cmp_ngt_f16_e32") set = !(x > y);    // and it is not greater either
        else throw Error::make(Err::Unsupported, "comparison ", op, " is decoded but not implemented");
      }
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
      } else if (op == "ds_xor_b32" || op == "ds_max_i32") {
        uint32_t before = 0;
        std::memcpy(&before, &g.lds[at(static_cast<uint64_t>(in.offset))], 4);
        const uint32_t v = lane_src(w, in.src[1], lane);
        const uint32_t after = op == "ds_xor_b32"
                                   ? before ^ v
                                   : static_cast<uint32_t>(std::max(static_cast<int32_t>(before),
                                                                    static_cast<int32_t>(v)));
        std::memcpy(&g.lds[at(static_cast<uint64_t>(in.offset))], &after, 4);
      } else if (op == "ds_write2_b32") {
        // Two words, each at its own offset, counted in words.
        const uint32_t v0 = lane_src(w, in.src[1], lane), v1 = lane_src(w, in.src[2], lane);
        std::memcpy(&g.lds[at(uint64_t{static_cast<uint32_t>(in.offset)} * 4)], &v0, 4);
        std::memcpy(&g.lds[at(uint64_t{static_cast<uint32_t>(in.offset1)} * 4)], &v1, 4);
      } else if (op == "ds_read2_b32") {
        uint32_t v0 = 0, v1 = 0;
        std::memcpy(&v0, &g.lds[at(uint64_t{static_cast<uint32_t>(in.offset)} * 4)], 4);
        std::memcpy(&v1, &g.lds[at(uint64_t{static_cast<uint32_t>(in.offset1)} * 4)], 4);
        w.vgpr[in.dst[0].index][lane] = v0;
        w.vgpr[in.dst[0].index + 1][lane] = v1;
      } else {
        throw Error::make(Err::Unsupported, "LDS instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // What a load or store narrower than a register moves, and whether what it
  // loads keeps its sign: the name says so, and the three segments spell it
  // the same way.
  struct Narrow {
    uint32_t bytes = 0;
    bool sign = false;
  };
  static bool narrow(const std::string& op, Narrow* n) {
    const size_t at = op.rfind('_');
    if (at == std::string::npos) return false;
    const std::string what = op.substr(at + 1);
    if (what == "ubyte" || what == "byte") *n = {1, false};
    else if (what == "sbyte") *n = {1, true};
    else if (what == "ushort" || what == "short") *n = {2, false};
    else if (what == "sshort") *n = {2, true};
    else return false;
    return true;
  }

  // What a narrow load puts in the register: the bytes it read, with the sign
  // carried into the rest where the name says so.
  static uint32_t widen(uint64_t raw, const Narrow& n) {
    if (!n.sign) return static_cast<uint32_t>(raw);
    return n.bytes == 1 ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)))
                        : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
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
      Narrow n;
      if (narrow(in.name, &n)) {
        if (shared && where + n.bytes > g.lds.size())
          throw Error::make(Err::InvalidValue, "a flat access reaches LDS at ", where, ", past the ", g.lds.size(),
                            " bytes the kernel reserved");
        if (in.name.rfind("flat_store", 0) == 0) {
          const uint32_t v = lane_src(w, in.src[1], lane);
          if (shared) std::memcpy(&g.lds[where], &v, n.bytes);
          else mem.store_scalar(where, n.bytes, v);
        } else {
          uint64_t raw = 0;
          if (shared) std::memcpy(&raw, &g.lds[where], n.bytes);
          else raw = mem.load_scalar(where, n.bytes);
          write_lane(w, in.dst[0], lane, widen(raw, n));
        }
      } else if (in.name == "flat_store_dword") {
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
      Narrow n;
      if (narrow(op, &n)) {
        if (op.rfind("global_store", 0) == 0) mem.store_scalar(addr, n.bytes, lane_src(w, in.src[1], lane));
        else write_lane(w, in.dst[0], lane, widen(mem.load_scalar(addr, n.bytes), n));
      } else if (op.rfind("global_load_dword", 0) == 0) {
        // One word, or two, or four: a register each, in order.
        for (uint32_t k = 0; k < in.dst[0].width; ++k)
          w.vgpr[in.dst[0].index + k][lane] = static_cast<uint32_t>(mem.load_scalar(addr + 4 * k, 4));
      } else if (op.rfind("global_store_dword", 0) == 0) {
        for (uint32_t k = 0; k < in.src[1].width; ++k)
          mem.store_scalar(addr + 4 * k, 4, w.vgpr[in.src[1].index + k][lane]);
      } else if (op == "global_atomic_add_x2") {
        // The one that works on a pair; every other atomic here is 32-bit.
        const uint64_t before = mem.load_scalar(addr, 8);
        mem.store_scalar(addr, 8, before + lane_src64(w, in.src[1], lane));
        if (!in.dst.empty()) write_lane64(w, in.dst[0], lane, before);
      } else if (op == "global_atomic_cmpswap") {
        // The pair is the value to write and the one it must find.
        const uint32_t value = lane_src(w, in.src[1], lane),
                       expected = w.vgpr[in.src[1].index + 1][lane];
        const uint32_t before = static_cast<uint32_t>(mem.load_scalar(addr, 4));
        if (before == expected) mem.store_scalar(addr, 4, value);
        if (!in.dst.empty()) write_lane(w, in.dst[0], lane, before);
      } else if (op.rfind("global_atomic_", 0) == 0) {
        // Lane by lane, which is what makes these atomic: every lane's turn
        // lands, whatever order they come in, and each is told what it found.
        const uint32_t before = static_cast<uint32_t>(mem.load_scalar(addr, 4)), v = lane_src(w, in.src[1], lane);
        uint32_t after = 0;
        if (op == "global_atomic_add") after = before + v;
        else if (op == "global_atomic_sub") after = before - v;
        else if (op == "global_atomic_and") after = before & v;
        else if (op == "global_atomic_or") after = before | v;
        else if (op == "global_atomic_xor") after = before ^ v;
        else if (op == "global_atomic_swap") after = v;
        else if (op == "global_atomic_add_f32") after = as_bits(as_float(before) + as_float(v));
        else throw Error::make(Err::Unsupported, "memory instruction ", op, " is decoded but not implemented");
        mem.store_scalar(addr, 4, after);
        if (!in.dst.empty()) write_lane(w, in.dst[0], lane, before);
      } else {
        throw Error::make(Err::Unsupported, "memory instruction ", op, " is decoded but not implemented");
      }
    }
  }

  // Which lane a lane reads its first source from, when the instruction says
  // another lane's. A row is sixteen lanes and a bank is four of those; a
  // shift, a rotate or a mirror stays inside its row, and the two broadcasts
  // carry the last lane of a row into the rows above it. False where there is
  // no such lane.
  //
  // The direction is not a guess: the sequence a wave adds itself up with
  // shifts by one, two, four and eight and ends with the total in the last
  // lane, which only comes out if each lane reads the lane below it.
  static bool dpp_source(uint32_t ctrl, uint32_t lane, uint32_t* from) {
    const uint32_t row = lane & ~15u, in_row = lane & 15u;
    if (ctrl <= 0xFF) {   // four lanes choosing among their own four
      *from = (lane & ~3u) + ((ctrl >> (2 * (lane & 3))) & 3);
      return true;
    }
    if (ctrl >= 0x101 && ctrl <= 0x10F) {   // row_shl: the lane above
      const uint32_t n = ctrl - 0x100;
      if (in_row + n > 15) return false;
      *from = row + in_row + n;
      return true;
    }
    if (ctrl >= 0x111 && ctrl <= 0x11F) {   // row_shr: the lane below
      const uint32_t n = ctrl - 0x110;
      if (n > in_row) return false;
      *from = row + in_row - n;
      return true;
    }
    if (ctrl >= 0x121 && ctrl <= 0x12F) {   // row_ror: the same, wrapping
      const uint32_t n = ctrl - 0x120;
      *from = row + ((in_row + 16 - n) & 15);
      return true;
    }
    if (ctrl == 0x140) {   // the row reversed
      *from = row + (15 - in_row);
      return true;
    }
    if (ctrl == 0x141) {   // each half of the row reversed
      *from = (lane & ~7u) + (7 - (lane & 7));
      return true;
    }
    if (ctrl == 0x142) {   // the last lane of the row below
      if (lane < 16) return false;
      *from = row - 1;
      return true;
    }
    if (ctrl == 0x143) {   // the last lane of the half of the wave below
      if (lane < 32) return false;
      *from = (lane & ~31u) - 1;
      return true;
    }
    throw Error::make(Err::Unsupported, "a cross-lane instruction reading across the whole wave rather than "
                                        "within a row, which this decodes but does not model");
  }

  // What each lane's first source comes to, and which lanes the instruction
  // writes at all: a row and a bank mask say which, and bound_ctrl says
  // whether a lane with no lane to read gets a zero or is left alone.
  uint64_t dpp_shuffle(const Wave& w, const Inst& in, std::array<uint32_t, kLanes>& values) {
    const Operand& o = in.src[0];
    uint64_t writes = 0;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      values[lane] = 0;
      if (!(w.exec >> lane & 1)) continue;
      if (!((in.row_mask >> (lane >> 4)) & 1)) continue;
      if (!((in.bank_mask >> ((lane >> 2) & 3)) & 1)) continue;
      uint32_t from = 0;
      const bool there = dpp_source(in.dpp_ctrl, lane, &from) && ((w.exec >> from) & 1);
      if (!there && !in.bound_ctrl) continue;   // nothing to read, and nothing written
      values[lane] = there ? w.vgpr[o.index][from] : 0u;
      writes |= uint64_t{1} << lane;
    }
    return writes;
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
        else if (in.dpp) cross_lane_alu(w, in);
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

namespace {

// What the runtime tells a kernel about the grid it is part of, written into
// the kernarg segment after the kernel's own arguments. The names and offsets
// are the code object's own (its metadata lists them); the values are this
// dispatch's.
void fill_hidden_arguments(const Dispatch& d, const Kernel& k, MemoryManager& mem) {
  const uint64_t threads_x = uint64_t{d.groups[0]} * d.group_size[0];
  const uint64_t threads_y = uint64_t{d.groups[1]} * d.group_size[1];
  const uint64_t threads_z = uint64_t{d.groups[2]} * d.group_size[2];
  const uint32_t dims = d.groups[2] > 1 || d.group_size[2] > 1   ? 3
                        : d.groups[1] > 1 || d.group_size[1] > 1 ? 2
                                                                 : 1;
  for (const KernelArg& a : k.args) {
    if (!a.hidden()) continue;
    uint64_t value = 0;
    const std::string& kind = a.kind;
    if (kind == "hidden_block_count_x") value = d.groups[0];
    else if (kind == "hidden_block_count_y") value = d.groups[1];
    else if (kind == "hidden_block_count_z") value = d.groups[2];
    else if (kind == "hidden_group_size_x") value = d.group_size[0];
    else if (kind == "hidden_group_size_y") value = d.group_size[1];
    else if (kind == "hidden_group_size_z") value = d.group_size[2];
    else if (kind == "hidden_grid_size_x") value = threads_x;
    else if (kind == "hidden_grid_size_y") value = threads_y;
    else if (kind == "hidden_grid_size_z") value = threads_z;
    else if (kind == "hidden_grid_dims") value = dims;
    else if (kind == "hidden_shared_base") value = kSharedBase;
    else if (kind == "hidden_dynamic_lds_size") value = d.dynamic_lds;
    // Everything else -- the remainders of a grid that divides evenly, the
    // global offsets, the buffers a hostcall or a printf would use -- is
    // zero, and a kernel that needs one of those will say so by failing on a
    // null pointer rather than reading something made up.
    else continue;
    if (a.offset + a.size > k.kernarg_size) continue;
    for (uint32_t b = 0; b < a.size && b < 8; ++b)
      mem.store_scalar(d.kernarg + a.offset + b, 1, (value >> (8 * b)) & 0xFF);
  }
}

// The packet the hardware is given for a dispatch, which a kernel may read
// instead of its implicit arguments (the HSA kernel dispatch packet: its
// sizes, its segments, and where its arguments are).
uint64_t write_dispatch_packet(const Dispatch& d, const Kernel& k, MemoryManager& mem) {
  const uint32_t dims = d.groups[2] > 1 || d.group_size[2] > 1   ? 3
                        : d.groups[1] > 1 || d.group_size[1] > 1 ? 2
                                                                 : 1;
  std::vector<uint8_t> packet(64, 0);
  const auto put16 = [&](uint32_t at, uint16_t v) { std::memcpy(&packet[at], &v, 2); };
  const auto put32 = [&](uint32_t at, uint32_t v) { std::memcpy(&packet[at], &v, 4); };
  const auto put64 = [&](uint32_t at, uint64_t v) { std::memcpy(&packet[at], &v, 8); };
  put16(0, 2 << 0);                       // header: a kernel dispatch packet
  put16(2, static_cast<uint16_t>(dims));  // setup: how many dimensions the grid has
  put16(4, static_cast<uint16_t>(d.group_size[0]));
  put16(6, static_cast<uint16_t>(d.group_size[1]));
  put16(8, static_cast<uint16_t>(d.group_size[2]));
  put32(12, static_cast<uint32_t>(uint64_t{d.groups[0]} * d.group_size[0]));
  put32(16, static_cast<uint32_t>(uint64_t{d.groups[1]} * d.group_size[1]));
  put32(20, static_cast<uint32_t>(uint64_t{d.groups[2]} * d.group_size[2]));
  put32(24, k.private_segment);
  put32(28, k.group_segment);
  put64(32, k.entry);
  put64(40, d.kernarg);
  const uint64_t where = mem.alloc(packet.size());
  mem.write(where, packet.data(), packet.size());
  return where;
}

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
  // What the work-group's LDS comes to: what the kernel reserved, and what
  // the launch added.
  const uint64_t group_segment = uint64_t{k.group_segment} + d.dynamic_lds;
  if (group_segment > (64u << 10))
    throw Error::make(Err::InvalidValue, "a work-group asking for ", group_segment,
                      " bytes of LDS is past the 65536 a CDNA work-group has");

  // What the kernel is told about its grid, and the packet it may read it
  // from. Both are written before any wave starts.
  if (d.kernarg) fill_hidden_arguments(d, k, mem);
  uint64_t packet = 0;
  if (k.dispatch_ptr) packet = write_dispatch_packet(d, k, mem);

  Machine m{d, mem, {}};
  const uint32_t waves_per_group = static_cast<uint32_t>((threads + kLanes - 1) / kLanes);
  for (uint32_t gz = 0; gz < d.groups[2]; ++gz)
    for (uint32_t gy = 0; gy < d.groups[1]; ++gy)
      for (uint32_t gx = 0; gx < d.groups[0]; ++gx) {
        Group group;
        group.lds.assign(group_segment, 0);
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
          if (k.dispatch_ptr) {
            m.set_sgpr64(w, at, packet);
            at += 2;
          }
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
  if (packet) mem.free(packet);
  return m.stats;
}

}  // namespace vgpu::amd
