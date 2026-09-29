// Instruction semantics for the SASS executor (included into exec.cpp,
// inside its anonymous namespace). What each Instr::f[] field holds is set by
// the op's decoder in decode.cpp.

template <class F>
void each_lane(Mask m, F&& f) {
  while (m) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    f(l);
    m &= m - 1;
  }
}

uint32_t Runner::ival(Warp& w, const Operand& o, unsigned lane) {
  uint32_t v = read32(w, o, lane);
  if (o.neg) v = 0u - v;
  if (o.bnot) v = ~v;
  return v;
}

// ---- dispatch ------------------------------------------------------------------

void Runner::execute(Block& blk, Warp& w, const Instr& ins, Mask group) {
  const Mask ex = guard_mask(w, ins, group);
  bool pcs_set = false;
  try {
    switch (ins.op) {
      case Op::Unknown:
        fault(w, ins, static_cast<unsigned>(std::countr_zero(group)), "instruction not decodable: " + ins.mnemonic);
      case Op::BRA: case Op::BRX: case Op::JMP: case Op::CALL: case Op::RET: case Op::EXIT:
      case Op::BSSY: case Op::BSYNC: case Op::BREAK: case Op::WARPSYNC: case Op::BAR:
      case Op::NOP: case Op::YIELD: case Op::NANOSLEEP: case Op::BPT: case Op::DEPBAR:
      case Op::MEMBAR: case Op::ERRBAR: case Op::CCTL: case Op::BMOV:
        pcs_set = exec_control(blk, w, ins, ex, group);
        break;
      case Op::LDG: case Op::STG: case Op::LDS: case Op::STS: case Op::LDL: case Op::STL:
      case Op::LD: case Op::ST: case Op::LDC: case Op::ATOM: case Op::ATOMG: case Op::ATOMS:
      case Op::RED: case Op::LDGSTS: case Op::LDGDEPBAR: case Op::LDSM: case Op::SULD: case Op::SUST:
      case Op::TEX: case Op::TLD:
        exec_mem(blk, w, ins, ex);
        break;
      case Op::SHFL: case Op::VOTE: case Op::VOTEU: case Op::MATCH: case Op::REDUX:
        exec_warp(blk, w, ins, ex, group);
        break;
      case Op::HMMA: case Op::IMMA: case Op::BMMA: case Op::DMMA: case Op::MOVM:
        exec_mma(blk, w, ins, ex);
        break;
      case Op::FADD: case Op::FMUL: case Op::FFMA: case Op::FMNMX: case Op::FSETP: case Op::FSET:
      case Op::FSEL: case Op::FCHK: case Op::MUFU: case Op::DADD: case Op::DMUL: case Op::DFMA:
      case Op::DSETP: case Op::HADD2: case Op::HMUL2: case Op::HFMA2: case Op::HSETP2: case Op::HMNMX2:
      case Op::I2F: case Op::I2FP: case Op::F2I: case Op::F2IP: case Op::F2F: case Op::F2FP:
      case Op::FRND: case Op::I2I: case Op::I2IP:
        exec_float(blk, w, ins, ex);
        break;
      default:
        exec_int(blk, w, ins, ex);
        break;
    }
  } catch (const Fault& f) {
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what());
  } catch (const Error& e) {
    // A memory error from the device (an address out of bounds, freed).
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), e.what());
  }
  if (!pcs_set) each_lane(group, [&](unsigned l) { w.pc[l] += 16; });
  // VGPU_SASS_TRACE=<warp>: every instruction that warp runs, with the lane
  // group and the first active lane's results.
  static const int trace = [] { const char* v = std::getenv("VGPU_SASS_TRACE"); return v && *v ? std::atoi(v) : -1; }();
  if (trace >= 0 && static_cast<int>(w.index) == trace && blk.ctaid[0] == 0 && blk.ctaid[1] == 0 && blk.ctaid[2] == 0) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(ex ? ex : group));
    std::string vals;
    for (const Operand& d : ins.dst) {
      char b[48];
      if (d.kind == Kind::Reg && d.reg != kRZ) std::snprintf(b, sizeof b, " R%u=%08x", d.reg, w.reg(d.reg, l));
      else if (d.kind == Kind::UReg && d.reg < 63) std::snprintf(b, sizeof b, " UR%u=%08x", d.reg, w.ur[d.reg]);
      else if (d.kind == Kind::Pred && d.reg != kPT) std::snprintf(b, sizeof b, " P%u=%d", d.reg, (w.p[d.reg] >> l) & 1);
      else b[0] = 0;
      vals += b;
    }
    std::fprintf(stderr, "  [%08x %08x] %04llx  %-50s%s\n", group, ex, static_cast<unsigned long long>(ins.pc),
                 to_text(ins).c_str(), vals.c_str());
  }
}

// ---- integer ------------------------------------------------------------------

namespace {
uint32_t lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {
  uint32_t r = 0;
  for (int i = 0; i < 8; ++i)
    if ((lut >> i) & 1) r |= ((i & 4) ? a : ~a) & ((i & 2) ? b : ~b) & ((i & 1) ? c : ~c);
  return r;
}
bool int_cmp(unsigned cmp, int64_t a, int64_t b) {
  switch (cmp) {
    case 0: return false;
    case 1: return a < b;
    case 2: return a == b;
    case 3: return a <= b;
    case 4: return a > b;
    case 5: return a != b;
    case 6: return a >= b;
    default: return true;
  }
}
bool bool_op(unsigned op, bool a, bool b) {
  switch (op) {
    case 0: return a && b;
    case 1: return a || b;
    default: return a != b;
  }
}
uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel, unsigned mode) {
  const uint64_t x = (static_cast<uint64_t>(b) << 32) | a;
  const auto byte = [&](unsigned i) { return static_cast<uint32_t>((x >> (8 * (i & 7))) & 0xff); };
  uint32_t r = 0;
  switch (mode) {
    case 0:
      for (unsigned i = 0; i < 4; ++i) {
        const unsigned s = (sel >> (4 * i)) & 0xf;
        uint32_t v = byte(s & 7);
        if (s & 8) v = (v & 0x80) ? 0xff : 0;   // replicate the sign bit
        r |= v << (8 * i);
      }
      return r;
    case 1: {   // F4E: forward 4 extract
      const unsigned s = sel & 7;
      for (unsigned i = 0; i < 4; ++i) r |= byte(s + i) << (8 * i);
      return r;
    }
    case 2: {   // B4E: backward 4 extract
      const unsigned s = sel & 7;
      for (unsigned i = 0; i < 4; ++i) r |= byte(s + 8 - i) << (8 * i);
      return r;
    }
    case 3: {   // RC8: replicate one byte
      const uint32_t v = byte(sel & 3);
      return v * 0x01010101u;
    }
    case 4: {   // ECL: edge clamp left
      const unsigned s = sel & 3;
      for (unsigned i = 0; i < 4; ++i) r |= byte(std::max(i, s) == i ? s : i < s ? i : s) << (8 * i);
      for (unsigned i = 0; i < 4; ++i) r = (r & ~(0xffu << (8 * i))) | (byte(i > s ? s : i) << (8 * i));
      return r;
    }
    case 5: {   // ECR: edge clamp right
      const unsigned s = sel & 3;
      for (unsigned i = 0; i < 4; ++i) r |= byte(i < s ? s : i) << (8 * i);
      return r;
    }
    case 6: {   // RC16: replicate one half
      const uint32_t v = (sel & 1) ? (a >> 16) : (a & 0xffff);
      return v | (v << 16);
    }
    default: return 0;
  }
}
}  // namespace

void Runner::exec_int(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  const bool uniform = ins.guard_uniform || (!ins.dst.empty() && ins.dst[0].kind == Kind::UReg);
  // Uniform ops compute once for the warp; run them on one lane.
  if (uniform && ex) ex = Mask{1} << std::countr_zero(ex);
  switch (ins.op) {
    case Op::MOV: case Op::UMOV:
      if (ins.mnemonic == "LEPC") {   // the lane's own program counter
        each_lane(ex, [&](unsigned l) { write64(w, ins.dst[0], l, w.pc[l]); });
        return;
      }
      each_lane(ex, [&](unsigned l) { write32(w, ins.dst[0], l, read32(w, ins.src[0], l)); });
      return;
    case Op::IMAD: case Op::UIMAD: {
      const unsigned kind = ins.f[0];
      const bool sgn = ins.f[1], x = ins.f[2];
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = ival(w, ins.src[0], l), b = ival(w, ins.src[1], l);
        const bool cin = x && pred(w, ins.src[3], l);
        if (kind == 0x25) {   // WIDE: a 64-bit product plus a 64-bit addend
          const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a)) *
                                                            static_cast<int32_t>(b))
                                    : static_cast<uint64_t>(a) * b;
          uint64_t c = read64(w, ins.src[2], l);
          if (ins.src[2].neg) c = 0 - c;
          if (ins.src[2].bnot) c = ~c;
          write64(w, ins.dst[0], l, prod + c + cin);
        } else if (kind == 0x27) {   // HI: the product's high word plus the addend
          const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a)) *
                                                            static_cast<int32_t>(b))
                                    : static_cast<uint64_t>(a) * b;
          const uint32_t c = ival(w, ins.src[2], l);
          write32(w, ins.dst[0], l, static_cast<uint32_t>(prod >> 32) + c + cin);
        } else {
          const uint32_t c = ival(w, ins.src[2], l);
          const uint64_t full = static_cast<uint64_t>(a * b) + c + cin;
          write32(w, ins.dst[0], l, static_cast<uint32_t>(full));
          if (ins.dst.size() > 1) set_pred(w, ins.dst[1], l, (full >> 32) != 0);
        }
      });
      return;
    }
    case Op::IADD3: case Op::UIADD3: {
      const bool x = ins.f[0];
      each_lane(ex, [&](unsigned l) {
        // Each operand is added as its bits, a negated one as its complement
        // plus one: that is what makes the carries come out right.
        uint64_t sum = 0;
        for (int i = 0; i < 3; ++i) {
          const Operand& o = ins.src[i];
          uint32_t v = read32(w, o, l);
          if (o.neg) {
            sum += static_cast<uint32_t>(~v);
            sum += 1;
          } else {
            if (o.bnot) v = ~v;
            sum += v;
          }
        }
        if (x) sum += pred(w, ins.src[3], l) + pred(w, ins.src[4], l);
        write32(w, ins.dst[0], l, static_cast<uint32_t>(sum));
        if (!x) {
          if (ins.dst.size() > 1) set_pred(w, ins.dst[1], l, (sum >> 32) & 1);
          if (ins.dst.size() > 2) set_pred(w, ins.dst[2], l, (sum >> 33) & 1);
        }
      });
      return;
    }
    case Op::LOP3: case Op::ULOP3: {
      const bool has_p = ins.dst.size() > 1;
      const Operand& d = has_p ? ins.dst[1] : ins.dst[0];
      each_lane(ex, [&](unsigned l) {
        const uint32_t r = lop3(read32(w, ins.src[0], l), read32(w, ins.src[1], l), read32(w, ins.src[2], l), ins.f[0]);
        write32(w, d, l, r);
        if (has_p) set_pred(w, ins.dst[0], l, ins.f[1] ? r == kAll : r != 0);
      });
      return;
    }
    case Op::SHF: case Op::USHF: {
      const unsigned type = ins.f[0];   // 0 S64, 1 U64, 2 S32, 3 U32
      const bool wrap = ins.f[1], right = ins.f[2], hi = ins.f[3];
      const bool is64 = type < 2, sgn = type == 0 || type == 2;
      each_lane(ex, [&](unsigned l) {
        const uint64_t v = read32(w, ins.src[0], l) | (static_cast<uint64_t>(read32(w, ins.src[2], l)) << 32);
        uint32_t n = read32(w, ins.src[1], l);
        const uint32_t width = is64 ? 64 : 32;
        n = wrap ? n % width : std::min(n, width);
        uint64_t r;
        if (!right) r = n >= 64 ? 0 : v << n;
        else if (sgn) r = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(n, 63u));
        else r = n >= 64 ? 0 : v >> n;
        write32(w, ins.dst[0], l, static_cast<uint32_t>(hi ? r >> 32 : r));
      });
      return;
    }
    case Op::ISETP: case Op::UISETP: {
      const unsigned cmp = ins.f[0], bop = ins.f[2];
      const bool sgn = ins.f[1], exf = ins.f[3];
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l), b = read32(w, ins.src[1], l);
        const bool acc = pred(w, ins.src[2], l);
        bool r;
        if (!exf) {
          r = sgn ? int_cmp(cmp, static_cast<int32_t>(a), static_cast<int32_t>(b)) : int_cmp(cmp, a, b);
        } else {
          // The high words of a 64-bit compare; the low words' answer comes
          // in as a predicate and decides when the high words are equal.
          const bool low = pred(w, ins.src[3], l);
          const bool hi = sgn ? int_cmp(cmp, static_cast<int32_t>(a), static_cast<int32_t>(b)) : int_cmp(cmp, a, b);
          if (cmp == 2) r = a == b && low;
          else if (cmp == 5) r = a != b || low;
          else r = a == b ? low : hi;
        }
        set_pred(w, ins.dst[0], l, bool_op(bop, r, acc));
        set_pred(w, ins.dst[1], l, bool_op(bop, !r, acc));
      });
      return;
    }
    case Op::LEA: case Op::ULEA: {
      const bool hi = ins.f[0], x = ins.f[1], sx32 = ins.f[2];
      const unsigned s = ins.f[3];
      const bool has_p = ins.dst.size() > 1;
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l);
        const uint32_t an = ins.src[0].neg ? 0u - a : a;
        const uint32_t b = ival(w, ins.src[1], l);
        uint32_t shifted;
        if (!hi) {
          shifted = s >= 32 ? 0 : an << s;
        } else {
          const uint32_t top = sx32 ? static_cast<uint32_t>(static_cast<int32_t>(an) >> 31) : read32(w, ins.src[2], l);
          const uint64_t v = an | (static_cast<uint64_t>(top) << 32);
          shifted = static_cast<uint32_t>((v << s) >> 32);
        }
        const size_t pidx = (hi && !sx32) ? 4 : 3;
        const bool cin = x && pred(w, ins.src[pidx], l);
        const uint64_t sum = static_cast<uint64_t>(shifted) + b + cin;
        write32(w, ins.dst[0], l, static_cast<uint32_t>(sum));
        if (has_p) set_pred(w, ins.dst[1], l, (sum >> 32) != 0);
      });
      return;
    }
    case Op::PRMT: case Op::UPRMT:
      each_lane(ex, [&](unsigned l) {
        write32(w, ins.dst[0], l, prmt(read32(w, ins.src[0], l), read32(w, ins.src[2], l), read32(w, ins.src[1], l), ins.f[0]));
      });
      return;
    case Op::SEL: case Op::USEL:
      each_lane(ex, [&](unsigned l) {
        write32(w, ins.dst[0], l, pred(w, ins.src[2], l) ? read32(w, ins.src[0], l) : read32(w, ins.src[1], l));
      });
      return;
    case Op::IMNMX:
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l), b = read32(w, ins.src[1], l);
        const bool mn = pred(w, ins.src[2], l);
        uint32_t r;
        if (ins.f[0]) r = mn ? static_cast<uint32_t>(std::min<int32_t>(a, b)) : static_cast<uint32_t>(std::max<int32_t>(a, b));
        else r = mn ? std::min(a, b) : std::max(a, b);
        write32(w, ins.dst[0], l, r);
      });
      return;
    case Op::IABS:
      each_lane(ex, [&](unsigned l) {
        const int32_t v = static_cast<int32_t>(read32(w, ins.src[0], l));
        write32(w, ins.dst[0], l, static_cast<uint32_t>(v < 0 ? 0u - static_cast<uint32_t>(v) : static_cast<uint32_t>(v)));
      });
      return;
    case Op::POPC: case Op::UPOPC:
      each_lane(ex, [&](unsigned l) { write32(w, ins.dst[0], l, static_cast<uint32_t>(std::popcount(ival(w, ins.src[0], l)))); });
      return;
    case Op::FLO: case Op::UFLO: {
      const bool sgn = ins.f[0], sh = ins.f[1];
      const Operand& d = ins.dst.back();
      each_lane(ex, [&](unsigned l) {
        uint32_t v = ival(w, ins.src[0], l);
        if (sgn && (v >> 31)) v = ~v;   // signed: the first bit unlike the sign
        uint32_t r = v ? 31 - static_cast<uint32_t>(std::countl_zero(v)) : kAll;
        if (sh && r != kAll) r = 31 - r;
        write32(w, d, l, r);
      });
      return;
    }
    case Op::BREV:
      each_lane(ex, [&](unsigned l) {
        uint32_t v = read32(w, ins.src[0], l), r = 0;
        for (int i = 0; i < 32; ++i) r |= ((v >> i) & 1u) << (31 - i);
        write32(w, ins.dst[0], l, r);
      });
      return;
    case Op::BMSK:
      each_lane(ex, [&](unsigned l) {
        const uint32_t pos = read32(w, ins.src[0], l), width = read32(w, ins.src[1], l);
        const uint32_t p = ins.f[0] ? pos & 31 : std::min(pos, 32u), n = ins.f[0] ? width & 31 : std::min(width, 32u);
        const uint64_t mask = n >= 32 ? 0xffffffffull : ((uint64_t{1} << n) - 1);
        write32(w, ins.dst[0], l, static_cast<uint32_t>((mask << p) & 0xffffffffull));
      });
      return;
    case Op::PLOP3: case Op::UPLOP3:
      each_lane(ex, [&](unsigned l) {
        const bool a = pred(w, ins.src[0], l), b = pred(w, ins.src[1], l), c = pred(w, ins.src[2], l);
        const unsigned idx = (a << 2) | (b << 1) | static_cast<unsigned>(c);
        set_pred(w, ins.dst[0], l, (ins.f[0] >> idx) & 1);
        set_pred(w, ins.dst[1], l, (ins.f[1] >> idx) & 1);
      });
      return;
    case Op::P2R: {
      const unsigned shift = 8 * ins.f[0];
      each_lane(ex, [&](unsigned l) {
        uint32_t bits = 0;
        for (unsigned i = 0; i < 7; ++i) bits |= ((w.p[i] >> l) & 1u) << i;
        const uint32_t mask = read32(w, ins.src[2], l) << shift;
        const uint32_t base = read32(w, ins.src[1], l);
        write32(w, ins.dst[0], l, (base & ~mask) | ((bits << shift) & mask));
      });
      return;
    }
    case Op::R2P: {
      const unsigned shift = 8 * ins.f[0];
      each_lane(ex, [&](unsigned l) {
        const uint32_t v = read32(w, ins.src[0], l) >> shift, mask = read32(w, ins.src[1], l);
        for (unsigned i = 0; i < 7; ++i)
          if ((mask >> i) & 1) {
            if ((v >> i) & 1) w.p[i] |= Mask{1} << l;
            else w.p[i] &= ~(Mask{1} << l);
          }
      });
      return;
    }
    case Op::S2R: case Op::S2UR: case Op::CS2R:
      each_lane(ex, [&](unsigned l) {
        const unsigned idx = ins.f[0];
        if (ins.op == Op::CS2R) {
          const uint32_t v = idx == 0xff ? 0 : sreg(blk, w, idx, l);
          if (ins.dst[0].width == 2) write64(w, ins.dst[0], l, idx == 0xff ? 0 : sreg(blk, w, idx, l) | (static_cast<uint64_t>(sreg(blk, w, idx + 1, l)) << 32));
          else write32(w, ins.dst[0], l, v);
        } else {
          write32(w, ins.dst[0], l, sreg(blk, w, idx, l));
        }
      });
      return;
    case Op::R2UR:
      if (ex) {
        const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
        write32(w, ins.dst.back(), l, read32(w, ins.src[0], l));
      }
      return;
    case Op::ULDC: {
      if (!ex) return;
      const unsigned size = ins.f[0];
      const Operand& c = ins.src[0];
      if (size == 5) {
        write64(w, ins.dst[0], 0, read64(w, c, 0));
      } else {
        uint32_t v = read32(w, c, 0);
        switch (size) {
          case 0: v &= 0xff; break;
          case 1: v = static_cast<uint32_t>(static_cast<int8_t>(v)); break;
          case 2: v &= 0xffff; break;
          case 3: v = static_cast<uint32_t>(static_cast<int16_t>(v)); break;
          default: break;
        }
        write32(w, ins.dst[0], 0, v);
      }
      return;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet");
  }
}

// ---- float -------------------------------------------------------------------

namespace {

// Runs `f` with the host's rounding mode set to SASS's (0 RN, 1 RM, 2 RP, 3 RZ).
template <class F>
auto rounded(unsigned rnd, F&& f) {
  static const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  if (rnd == 0) return f();
  const int old = std::fegetround();
  std::fesetround(modes[rnd & 3]);
  auto r = f();
  std::fesetround(old);
  return r;
}

float ftz(float f) { return std::fpclassify(f) == FP_SUBNORMAL ? std::copysign(0.0f, f) : f; }
float sat(float f) { return f != f ? 0.0f : f < 0.0f ? 0.0f : f > 1.0f ? 1.0f : f; }

// Half and bfloat16 conversions, round to nearest even.
float half_to_f32(uint16_t h) { return static_cast<float>(std::bit_cast<_Float16>(h)); }
uint16_t f32_to_half(float f) { return std::bit_cast<uint16_t>(static_cast<_Float16>(f)); }
uint16_t f64_to_half(double d) { return std::bit_cast<uint16_t>(static_cast<_Float16>(d)); }
float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(static_cast<uint32_t>(h) << 16); }
uint16_t f32_to_bf16(float f) {
  uint32_t b = std::bit_cast<uint32_t>(f);
  if (f != f) return static_cast<uint16_t>((b >> 16) | 0x40);
  b += 0x7fff + ((b >> 16) & 1);
  return static_cast<uint16_t>(b >> 16);
}
uint16_t f64_to_bf16(double d) {
  // Round once, from double: bf16 has 8 bits of mantissa, and going through
  // float first could round twice.
  if (d != d) return 0x7fc0;
  const float f = static_cast<float>(d);
  const double back = static_cast<double>(bf16_to_f32(f32_to_bf16(f)));
  (void)back;
  return f32_to_bf16(f);
}

}  // namespace

void Runner::exec_float(Block&, Warp& w, const Instr& ins, Mask ex) {
  const auto f32src = [&](const Operand& o, unsigned l, bool flush) {
    float f = as_f32(read32(w, o, l));
    if (flush) f = ftz(f);
    if (o.abs) f = std::fabs(f);
    if (o.neg) f = -f;
    return f;
  };
  const auto f64src = [&](const Operand& o, unsigned l) {
    double d = as_f64(o.kind == Kind::FImm && o.fwidth == 64 ? static_cast<uint64_t>(o.fbits) << 32 : read64(w, o, l));
    if (o.kind == Kind::FImm && o.fwidth == 64) {
      // The immediate is the double's top 32 bits.
    }
    if (o.abs) d = std::fabs(d);
    if (o.neg) d = -d;
    return d;
  };
  switch (ins.op) {
    case Op::FADD: case Op::FMUL: case Op::FFMA: {
      const unsigned rnd = ins.f[0];
      const bool fz = ins.f[1], st = ins.f[2];
      each_lane(ex, [&](unsigned l) {
        const float a = f32src(ins.src[0], l, fz), b = f32src(ins.src[1], l, fz);
        float r;
        if (ins.op == Op::FADD) r = rounded(rnd, [&] { return a + b; });
        else if (ins.op == Op::FMUL) r = rounded(rnd, [&] { return a * b; });
        else {
          const float c = f32src(ins.src[2], l, fz);
          r = rounded(rnd, [&] { return std::fma(a, b, c); });
        }
        if (fz) r = ftz(r);
        if (st) r = sat(r);
        write32(w, ins.dst[0], l, f32_bits(r));
      });
      return;
    }
    case Op::FMNMX:
      each_lane(ex, [&](unsigned l) {
        const bool fz = ins.f[0];
        const float a = f32src(ins.src[0], l, fz), b = f32src(ins.src[1], l, fz);
        const bool mn = pred(w, ins.src[2], l);
        float r;
        if (a != a) r = b;
        else if (b != b) r = a;
        else if (a == 0 && b == 0) r = mn ? (std::signbit(a) ? a : b) : (std::signbit(a) ? b : a);
        else r = mn ? std::min(a, b) : std::max(a, b);
        write32(w, ins.dst[0], l, f32_bits(r));
      });
      return;
    case Op::MUFU: {
      const unsigned op = ins.f[0];
      each_lane(ex, [&](unsigned l) {
        const float x = ftz(f32src(ins.src[0], l, true));
        float r;
        switch (op) {
          case 0: r = std::cos(x * 6.28318530717958647692f); break;   // the argument is in turns
          case 1: r = std::sin(x * 6.28318530717958647692f); break;
          case 2: r = std::exp2(x); break;
          case 3: r = std::log2(x); break;
          case 4: r = 1.0f / x; break;
          case 5: r = 1.0f / std::sqrt(x); break;
          case 8: r = std::sqrt(x); break;
          case 9: r = std::tanh(x); break;
          case 6: case 7: {
            // RCP64H/RSQ64H: the high word of a double's reciprocal (root).
            const double d = as_f64(static_cast<uint64_t>(read32(w, ins.src[0], l)) << 32);
            const double v = op == 6 ? 1.0 / d : 1.0 / std::sqrt(d);
            write32(w, ins.dst[0], l, static_cast<uint32_t>(f64_bits(v) >> 32));
            return;
          }
          default: fault(w, ins, l, "MUFU op " + std::to_string(op));
        }
        write32(w, ins.dst[0], l, f32_bits(ftz(r)));
      });
      return;
    }
    case Op::DADD: case Op::DMUL: case Op::DFMA: {
      const unsigned rnd = ins.f[0];
      each_lane(ex, [&](unsigned l) {
        const double a = f64src(ins.src[0], l), b = f64src(ins.src[1], l);
        double r;
        if (ins.op == Op::DADD) r = rounded(rnd, [&] { return a + b; });
        else if (ins.op == Op::DMUL) r = rounded(rnd, [&] { return a * b; });
        else {
          const double c = f64src(ins.src[2], l);
          r = rounded(rnd, [&] { return std::fma(a, b, c); });
        }
        write64(w, ins.dst[0], l, f64_bits(r));
      });
      return;
    }
    case Op::I2F: case Op::I2FP: {
      const unsigned dsize = ins.op == Op::I2FP ? 2 : ins.f[0], ssize = ins.op == Op::I2FP ? 2 : ins.f[1];
      const bool sgn = ins.op == Op::I2FP ? ins.f[0] != 0 : ins.f[2] != 0;
      const unsigned rnd = ins.op == Op::I2FP ? ins.f[1] : ins.f[3];
      each_lane(ex, [&](unsigned l) {
        uint64_t v = ssize == 3 ? read64(w, ins.src[0], l) : read32(w, ins.src[0], l);
        int64_t sv;
        switch (ssize) {
          case 0: sv = sgn ? static_cast<int8_t>(v) : static_cast<int64_t>(v & 0xff); break;
          case 1: sv = sgn ? static_cast<int16_t>(v) : static_cast<int64_t>(v & 0xffff); break;
          case 2: sv = sgn ? static_cast<int32_t>(v) : static_cast<int64_t>(v & 0xffffffff); break;
          default: sv = static_cast<int64_t>(v); break;
        }
        const bool u64 = ssize == 3 && !sgn;
        if (dsize == 3) {
          const double d = rounded(rnd, [&] { return u64 ? static_cast<double>(v) : static_cast<double>(sv); });
          write64(w, ins.dst[0], l, f64_bits(d));
        } else {
          const float f = rounded(rnd, [&] { return u64 ? static_cast<float>(v) : static_cast<float>(sv); });
          if (dsize == 1) write32(w, ins.dst[0], l, f32_to_half(f));
          else write32(w, ins.dst[0], l, f32_bits(f));
        }
      });
      return;
    }
    case Op::F2I: {
      const unsigned dsize = ins.f[0], ssize = ins.f[1], rnd = ins.f[3];
      const bool sgn = ins.f[2];
      each_lane(ex, [&](unsigned l) {
        double x;
        if (ssize == 3) x = f64src(ins.src[0], l);
        else if (ssize == 1) x = half_to_f32(static_cast<uint16_t>(read32(w, ins.src[0], l)));
        else x = f32src(ins.src[0], l, ins.f[4] != 0);
        switch (rnd) {
          case 0: x = std::nearbyint(x); break;
          case 1: x = std::floor(x); break;
          case 2: x = std::ceil(x); break;
          default: x = std::trunc(x); break;
        }
        const unsigned bits = 8u << dsize;
        uint64_t r;
        if (x != x) r = 0;
        else if (sgn) {
          const double lo = -std::ldexp(1.0, static_cast<int>(bits) - 1), hi = std::ldexp(1.0, static_cast<int>(bits) - 1) - 1;
          r = static_cast<uint64_t>(static_cast<int64_t>(x < lo ? lo : x > hi ? hi : x));
        } else {
          const double hi = std::ldexp(1.0, static_cast<int>(bits)) - 1;
          r = static_cast<uint64_t>(x < 0 ? 0 : x > hi ? hi : x);
        }
        if (dsize == 3) write64(w, ins.dst[0], l, r);
        else write32(w, ins.dst[0], l, static_cast<uint32_t>(r));
      });
      return;
    }
    case Op::F2F: {
      const unsigned dsize = ins.f[0], ssize = ins.f[1], rnd = ins.f[3];
      each_lane(ex, [&](unsigned l) {
        double x;
        if (ssize == 3) x = f64src(ins.src[0], l);
        else if (ssize == 1) x = half_to_f32(static_cast<uint16_t>(read32(w, ins.src[0], l)));
        else x = f32src(ins.src[0], l, false);
        if (dsize == 3) write64(w, ins.dst[0], l, f64_bits(x));
        else if (dsize == 2) write32(w, ins.dst[0], l, f32_bits(rounded(rnd, [&] { return static_cast<float>(x); })));
        else write32(w, ins.dst[0], l, f64_to_half(x));
      });
      return;
    }
    case Op::F2FP:
      each_lane(ex, [&](unsigned l) {
        const float a = f32src(ins.src[0], l, false), b = f32src(ins.src[1], l, false);
        uint32_t r;
        if (ins.f[0]) r = (static_cast<uint32_t>(f32_to_bf16(a)) << 16) | f32_to_bf16(b);
        else r = (static_cast<uint32_t>(f32_to_half(a)) << 16) | f32_to_half(b);
        write32(w, ins.dst[0], l, r);
      });
      return;
    case Op::I2I:
      each_lane(ex, [&](unsigned l) {
        const int32_t v = static_cast<int32_t>(read32(w, ins.src[0], l));
        int32_t lo, hi;
        switch (ins.f[0]) {
          case 0: lo = 0; hi = 255; break;
          case 1: lo = -128; hi = 127; break;
          case 2: lo = 0; hi = 65535; break;
          default: lo = -32768; hi = 32767; break;
        }
        write32(w, ins.dst[0], l, static_cast<uint32_t>(std::clamp(v, lo, hi)));
      });
      return;
    case Op::HADD2: case Op::HMUL2: case Op::HFMA2: {
      const bool bf = ins.f[0], f32out = ins.f[1], fz = ins.f[2], st = ins.f[3];
      // A source's two lanes (low, high) as floats, after its swizzle.
      const auto halves = [&](const Operand& o, unsigned l, float out[2]) {
        if (o.kind == Kind::FImm) {   // split immediates: each is one half, for both lanes
          const float v = bf ? bf16_to_f32(static_cast<uint16_t>(o.fbits)) : half_to_f32(static_cast<uint16_t>(o.fbits));
          out[0] = out[1] = v;
          return;
        }
        const uint32_t v = read32(w, o, l);
        if (o.suffix == ".F32") {
          out[0] = out[1] = as_f32(v);
        } else {
          const auto conv = [&](uint16_t h) { return bf ? bf16_to_f32(h) : half_to_f32(h); };
          float lo = conv(static_cast<uint16_t>(v)), hi = conv(static_cast<uint16_t>(v >> 16));
          if (o.suffix == ".H0_H0") hi = lo;
          else if (o.suffix == ".H1_H1") lo = hi;
          out[0] = lo;
          out[1] = hi;
        }
        for (int i = 0; i < 2; ++i) {
          if (o.abs) out[i] = std::fabs(out[i]);
          if (o.neg) out[i] = -out[i];
        }
      };
      // The immediate pair: the second printed half is the low lane.
      each_lane(ex, [&](unsigned l) {
        float a[2], b[2], c[2] = {0, 0};
        halves(ins.src[0], l, a);
        size_t next = 1;
        if (ins.src.size() > next + 1 && ins.src[next].kind == Kind::FImm && ins.src[next + 1].kind == Kind::FImm) {
          const auto conv = [&](uint32_t h) { return bf ? bf16_to_f32(static_cast<uint16_t>(h)) : half_to_f32(static_cast<uint16_t>(h)); };
          b[1] = conv(ins.src[next].fbits);
          b[0] = conv(ins.src[next + 1].fbits);
          next += 2;
        } else {
          halves(ins.src[next], l, b);
          next += 1;
        }
        if (ins.op == Op::HFMA2) {
          if (ins.src.size() > next + 1 && ins.src[next].kind == Kind::FImm && ins.src[next + 1].kind == Kind::FImm) {
            const auto conv = [&](uint32_t h) { return bf ? bf16_to_f32(static_cast<uint16_t>(h)) : half_to_f32(static_cast<uint16_t>(h)); };
            c[1] = conv(ins.src[next].fbits);
            c[0] = conv(ins.src[next + 1].fbits);
          } else {
            halves(ins.src[next], l, c);
          }
        }
        double r[2];
        for (int i = 0; i < 2; ++i) {
          if (ins.op == Op::HADD2) r[i] = static_cast<double>(a[i]) + b[i];
          else if (ins.op == Op::HMUL2) r[i] = static_cast<double>(a[i]) * b[i];
          else r[i] = std::fma(static_cast<double>(a[i]), static_cast<double>(b[i]), static_cast<double>(c[i]));
          if (st) r[i] = r[i] != r[i] ? 0.0 : std::clamp(r[i], 0.0, 1.0);
        }
        if (f32out) {
          float f = static_cast<float>(r[0]);
          if (fz) f = ftz(f);
          write32(w, ins.dst[0], l, f32_bits(f));
          return;
        }
        uint16_t h[2];
        for (int i = 0; i < 2; ++i) h[i] = bf ? f64_to_bf16(r[i]) : f64_to_half(r[i]);
        write32(w, ins.dst[0], l, h[0] | (static_cast<uint32_t>(h[1]) << 16));
      });
      return;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet");
  }
}

// ---- memory ------------------------------------------------------------------

void Runner::exec_mem(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  const unsigned size = ins.f[0];
  const uint32_t bytes = size <= 1 ? 1 : size <= 3 ? 2 : size == 4 ? 4 : size == 5 ? 8 : 16;
  const auto extend = [&](uint32_t v) -> uint32_t {
    switch (size) {
      case 0: return v & 0xff;
      case 1: return static_cast<uint32_t>(static_cast<int8_t>(v));
      case 2: return v & 0xffff;
      case 3: return static_cast<uint32_t>(static_cast<int16_t>(v));
      default: return v;
    }
  };
  // The address of a memory operand for one lane: the register (64-bit for
  // .E and the .64 forms), the uniform register when there is one, and the
  // offset. Operand::text holds the printed form; its parts are re-read here.
  const auto address = [&](const Operand& a, unsigned l, bool wide, bool with_ur, unsigned ur_pos, unsigned stride) {
    uint64_t v = 0;
    const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
    if (ra != kRZ) v = wide ? (w.reg(ra, l) | (static_cast<uint64_t>(w.reg(ra + 1, l)) << 32)) : w.reg(ra, l);
    if (stride) v *= stride;
    if (with_ur) {
      const unsigned u = static_cast<unsigned>(ins.w.field(ur_pos, m_.sm >= 100 ? 8 : 6));
      if (u != (m_.sm >= 100 ? 255u : 63u)) v += w.ur[u];
    }
    return v + static_cast<uint64_t>(a.imm);
  };
  switch (ins.op) {
    case Op::LDG: case Op::LD: case Op::STG: case Op::ST: {
      const bool store = ins.op == Op::STG || ins.op == Op::ST;
      const bool generic = ins.op == Op::LD || ins.op == Op::ST;
      const bool e = ins.w.bit(72);
      const bool wide = e || ins.w.bit(90);
      const bool ur = ins.w.bit(91) && !e;
      const Operand& a = ins.src[0];   // the address (a store's data is src[1])
      each_lane(ex, [&](unsigned l) {
        const uint64_t g = address(a, l, wide, ur, store ? 64 : 32, 0);
        uint64_t off;
        const Space sp = generic ? classify(g, &off) : (off = g, Space::Global);
        if (store) {
          const Operand& d = ins.src[1];
          if (bytes <= 4) {
            const uint32_t v = read32(w, d, l);
            mem_write(blk, w, l, sp, off, &v, bytes);
          } else {
            uint32_t v[4];
            for (unsigned i = 0; i < bytes / 4; ++i) v[i] = d.reg == kRZ ? 0 : w.reg(d.reg + i, l);
            mem_write(blk, w, l, sp, off, v, bytes);
          }
        } else {
          const Operand& d = ins.dst[0];
          if (bytes <= 4) {
            uint32_t v = 0;
            mem_read(blk, w, l, sp, off, &v, bytes);
            write32(w, d, l, extend(v));
          } else {
            uint32_t v[4];
            mem_read(blk, w, l, sp, off, v, bytes);
            if (d.reg != kRZ)
              for (unsigned i = 0; i < bytes / 4; ++i) w.reg(d.reg + i, l) = v[i];
          }
        }
      });
      return;
    }
    case Op::LDS: case Op::STS: case Op::LDL: case Op::STL: {
      const bool store = ins.op == Op::STS || ins.op == Op::STL;
      const bool shared = ins.op == Op::LDS || ins.op == Op::STS;
      const unsigned strides[] = {0, 4, 8, 16};
      const unsigned stride = shared ? strides[ins.f[3]] : 0;
      const bool ur = ins.w.bit(91);
      const Operand& a = ins.src[0];   // the address (a store's data is src[1])
      each_lane(ex, [&](unsigned l) {
        const uint64_t off = static_cast<uint32_t>(address(a, l, false, ur, store ? 64 : 32, stride));
        const Space sp = shared ? Space::Shared : Space::Local;
        if (store) {
          const Operand& d = ins.src[1];
          uint32_t v[4] = {};
          for (unsigned i = 0; i < std::max(1u, bytes / 4); ++i) v[i] = d.reg == kRZ ? 0 : w.reg(d.reg + i, l);
          mem_write(blk, w, l, sp, off, v, bytes);
        } else {
          uint32_t v[4] = {};
          mem_read(blk, w, l, sp, off, v, bytes);
          const Operand& d = ins.dst[0];
          if (bytes <= 4) write32(w, d, l, extend(v[0]));
          else if (d.reg != kRZ)
            for (unsigned i = 0; i < bytes / 4; ++i) w.reg(d.reg + i, l) = v[i];
        }
      });
      return;
    }
    case Op::LDC: {
      const unsigned bank = ins.f[4];
      const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
      const int64_t off = ins.w.sfield(38, 16);
      each_lane(ex, [&](unsigned l) {
        const uint64_t o = static_cast<uint64_t>((ra == kRZ ? 0 : static_cast<int64_t>(static_cast<int32_t>(w.reg(ra, l)))) + off);
        if (bytes == 8) write64(w, ins.dst[0], l, cbank32(bank, o) | (static_cast<uint64_t>(cbank32(bank, o + 4)) << 32));
        else write32(w, ins.dst[0], l, extend(cbank32(bank, o & ~3ull) >> (8 * (o & 3))));
      });
      return;
    }
    case Op::ATOMS: case Op::ATOMG: case Op::ATOM: case Op::RED: {
      const bool shared = ins.op == Op::ATOMS;
      const bool red = ins.op == Op::RED;
      const unsigned type = ins.f[4];   // 0 U32, 1 S32, 2 U64, 3 F32, 4 F16x2, 5 S64, 6 F64
      const bool is64 = type == 2 || type == 5 || type == 6;
      const unsigned n = is64 ? 8 : 4;
      const bool cas = shared && ins.mods[0].rfind("CAS", 0) == 0;
      const bool popc_inc = shared && ins.f[2] == 1;
      const Operand& a = red ? ins.src[0] : ins.src[shared ? 0 : 0];
      // Warp-aggregated increment: one add of the active lane count.
      each_lane(ex, [&](unsigned l) {
        uint64_t addr;
        Space sp;
        if (shared) {
          const unsigned strides[] = {0, 4, 8, 16};
          addr = static_cast<uint32_t>(address(a, l, false, ins.w.bit(91) || popc_inc, 64, strides[ins.w.field(78, 2)]));
          sp = Space::Shared;
        } else {
          const bool e = ins.w.bit(72);
          const uint64_t g = address(a, l, e || ins.w.bit(90), ins.w.bit(91) && !e, 64, 0);
          sp = ins.op == Op::ATOM ? classify(g, &addr) : (addr = g, Space::Global);
        }
        std::unique_lock<std::mutex> lock;
        if (sp == Space::Global) lock = std::unique_lock<std::mutex>(atomic_lock(addr));
        uint64_t old = 0;
        mem_read(blk, w, l, sp, addr, &old, n);
        const Operand& vop = red ? ins.src[1] : ins.src[1];
        uint64_t v = popc_inc ? 1 : (is64 ? read64(w, vop, l) : read32(w, vop, l));
        uint64_t nw = old;
        if (cas) {
          const uint64_t cmp = v, swap = read32(w, ins.src[2], l);
          if (old == cmp) nw = swap;
        } else {
          switch (popc_inc ? 3 : ins.f[0]) {
            case 0:   // ADD
              if (type == 3) nw = f32_bits(as_f32(static_cast<uint32_t>(old)) + as_f32(static_cast<uint32_t>(v)));
              else if (type == 6) nw = f64_bits(as_f64(old) + as_f64(v));
              else if (type == 4) {
                const uint16_t lo = f32_to_half(half_to_f32(static_cast<uint16_t>(old)) + half_to_f32(static_cast<uint16_t>(v)));
                const uint16_t hi = f32_to_half(half_to_f32(static_cast<uint16_t>(old >> 16)) + half_to_f32(static_cast<uint16_t>(v >> 16)));
                nw = lo | (static_cast<uint32_t>(hi) << 16);
              } else nw = old + v;
              break;
            case 1:   // MIN
              if (type == 1) nw = static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(v)));
              else if (type == 5) nw = static_cast<uint64_t>(std::min(static_cast<int64_t>(old), static_cast<int64_t>(v)));
              else nw = std::min(old, v);
              break;
            case 2:   // MAX
              if (type == 1) nw = static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(v)));
              else if (type == 5) nw = static_cast<uint64_t>(std::max(static_cast<int64_t>(old), static_cast<int64_t>(v)));
              else nw = std::max(old, v);
              break;
            case 3: nw = popc_inc ? old + 1 : (static_cast<uint32_t>(old) >= static_cast<uint32_t>(v) ? 0 : old + 1); break;   // INC
            case 4: nw = (old == 0 || static_cast<uint32_t>(old) > static_cast<uint32_t>(v)) ? v : old - 1; break;             // DEC
            case 5: nw = old & v; break;
            case 6: nw = old | v; break;
            case 7: nw = old ^ v; break;
            case 8: nw = v; break;   // EXCH
            default: fault(w, ins, l, "atomic op " + std::to_string(ins.f[0]));
          }
        }
        if (!is64) nw &= 0xffffffff;
        mem_write(blk, w, l, sp, addr, &nw, n);
        if (!red) {
          const Operand& d = ins.op == Op::ATOMS ? ins.dst[0] : ins.dst[1];
          if (is64) write64(w, d, l, old);
          else write32(w, d, l, static_cast<uint32_t>(old));
          if (cas && ins.f[1] == 3) write32(w, d, l, old == v ? 1 : 0);   // CAST.SPIN: whether it stored
        }
      });
      return;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet");
  }
}

// ---- warp-wide -----------------------------------------------------------------

void Runner::exec_warp(Block&, Warp& w, const Instr& ins, Mask ex, Mask) {
  switch (ins.op) {
    case Op::SHFL: {
      // dst[0] the in-range predicate, dst[1] the value; src: value, lane, c
      // (c = clamp in bits 0-4, segment mask in 8-12), as PTX's shfl.
      uint32_t vals[32];
      for (unsigned l = 0; l < 32; ++l) vals[l] = read32(w, ins.src[0], l);
      each_lane(ex, [&](unsigned l) {
        const uint32_t b = read32(w, ins.src[1], l), c = read32(w, ins.src[2], l);
        const uint32_t clamp = c & 0x1f, segmask = (c >> 8) & 0x1f;
        const uint32_t maxl = (l & segmask) | (clamp & ~segmask), minl = l & segmask;
        uint32_t j;
        bool in;
        switch (ins.f[0]) {
          case 0: j = minl | (b & ~segmask & 0x1f); in = j <= maxl; break;
          case 1: j = l - b; in = static_cast<int32_t>(j) >= static_cast<int32_t>(maxl); break;
          case 2: j = l + b; in = j <= maxl; break;
          default: j = l ^ b; in = j <= maxl; break;
        }
        if (!in) j = l;
        write32(w, ins.dst[1], l, vals[j & 31]);
        set_pred(w, ins.dst[0], l, in);
      });
      return;
    }
    case Op::VOTE: case Op::VOTEU: {
      Mask ballot = 0;
      each_lane(ex, [&](unsigned l) {
        if (pred(w, ins.src[0], l)) ballot |= Mask{1} << l;
      });
      const bool all = (ballot & ex) == ex, any = ballot != 0, eq = all || !any;
      const bool r = ins.f[0] == 0 ? all : ins.f[0] == 1 ? any : eq;
      const Operand& pd = ins.dst.back();
      if (ins.op == Op::VOTEU) {
        if (ins.dst.size() > 1) w.ur[ins.dst[0].reg < 63 ? ins.dst[0].reg : 0] = ins.dst[0].reg < 63 ? ballot : w.ur[0];
        if (pd.reg != kPT) w.up[pd.reg] = r;
        return;
      }
      each_lane(ex, [&](unsigned l) {
        if (ins.dst.size() > 1) write32(w, ins.dst[0], l, ballot);
        set_pred(w, pd, l, r);
      });
      return;
    }
    case Op::MATCH: {
      const bool any = ins.f[0], u64 = ins.f[1];
      uint64_t v[32];
      each_lane(ex, [&](unsigned l) { v[l] = u64 ? read64(w, ins.src[0], l) : read32(w, ins.src[0], l); });
      bool all_same = true;
      const unsigned first = static_cast<unsigned>(std::countr_zero(ex ? ex : 1u));
      each_lane(ex, [&](unsigned l) { all_same = all_same && v[l] == v[first]; });
      each_lane(ex, [&](unsigned l) {
        Mask m = 0;
        each_lane(ex, [&](unsigned k) { if (v[k] == v[l]) m |= Mask{1} << k; });
        if (any) write32(w, ins.dst.back(), l, m);
        else {
          write32(w, ins.dst.back(), l, all_same ? ex : 0);
          if (ins.dst.size() > 1) set_pred(w, ins.dst[0], l, all_same);
        }
      });
      return;
    }
    case Op::REDUX: {
      const unsigned op = ins.f[0];
      const bool sgn = ins.f[1];
      uint64_t acc = op == 0 ? kAll : 0;
      int64_t sacc = 0;
      bool first = true;
      each_lane(ex, [&](unsigned l) {
        const uint32_t v = read32(w, ins.src[0], l);
        switch (op) {
          case 0: acc &= v; break;
          case 1: acc |= v; break;
          case 2: acc ^= v; break;
          case 3: acc += v; break;
          case 4:
            if (sgn) sacc = first ? static_cast<int32_t>(v) : std::min<int64_t>(sacc, static_cast<int32_t>(v));
            else acc = first ? v : std::min<uint64_t>(acc, v);
            break;
          case 5:
            if (sgn) sacc = first ? static_cast<int32_t>(v) : std::max<int64_t>(sacc, static_cast<int32_t>(v));
            else acc = first ? v : std::max<uint64_t>(acc, v);
            break;
        }
        first = false;
      });
      const uint32_t r = (op >= 4 && sgn) ? static_cast<uint32_t>(sacc) : static_cast<uint32_t>(acc);
      if (ins.dst[0].reg < 63) w.ur[ins.dst[0].reg] = r;
      return;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet");
  }
}

// ---- tensor cores --------------------------------------------------------------

void Runner::exec_mma(Block&, Warp& w, const Instr& ins, Mask) {
  fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet");
}

// ---- control -------------------------------------------------------------------

bool Runner::exec_control(Block& blk, Warp& w, const Instr& ins, Mask ex, Mask group) {
  const auto base_of = [&](unsigned l) { return w.pc[l] - ins.pc; };   // the section's code address
  const auto release_bsync = [&](unsigned x) {
    const Mask members = w.b[x] & w.alive & ~w.exited;
    Mask arrived = 0;
    for (unsigned l = 0; l < 32; ++l)
      if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::BSync && w.wait_arg[l] == x) arrived |= Mask{1} << l;
    if ((members & ~arrived) == 0 && arrived) {
      each_lane(arrived, [&](unsigned l) {
        w.waiting &= ~(Mask{1} << l);
        w.wait_kind[l] = Wait::None;
        w.pc[l] += 16;
      });
    }
  };
  const auto release_warpsync = [&] {
    // Every lane waiting at a WARPSYNC is released once every lane its mask
    // names has arrived at one or exited.
    Mask waiting_ws = 0;
    for (unsigned l = 0; l < 32; ++l)
      if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::WarpSync) waiting_ws |= Mask{1} << l;
    Mask release = 0;
    each_lane(waiting_ws, [&](unsigned l) {
      const Mask need = w.wait_arg[l] & w.alive & ~w.exited;
      if ((need & ~waiting_ws) == 0) release |= Mask{1} << l;
    });
    each_lane(release, [&](unsigned l) {
      w.waiting &= ~(Mask{1} << l);
      w.wait_kind[l] = Wait::None;
      w.pc[l] += 16;
    });
  };
  const auto release_bar = [&](unsigned id) {
    Block::Barrier& b = blk.bars[id];
    uint32_t live = 0;
    for (const Warp& ww : blk.warps) live += static_cast<uint32_t>(std::popcount(ww.alive & ~ww.exited));
    const uint32_t expected = b.expected ? b.expected : live;
    if (b.arrived < expected) return;
    for (Warp& ww : blk.warps)
      for (unsigned l = 0; l < 32; ++l)
        if (((ww.waiting >> l) & 1) && ww.wait_kind[l] == Wait::Bar && ww.wait_arg[l] == id) {
          ww.waiting &= ~(Mask{1} << l);
          ww.wait_kind[l] = Wait::None;
          ww.pc[l] += 16;
        }
    for (Warp& ww : blk.warps) {
      ww.b2r = b.red == 0 ? b.popc : b.red == 1 ? b.and_ : b.or_;
      ww.b2r_pred = b.red == 0 ? b.popc != 0 : b.red == 1 ? b.and_ : b.or_;
    }
    b = Block::Barrier{};
  };
  switch (ins.op) {
    case Op::NOP: case Op::DEPBAR: case Op::MEMBAR: case Op::ERRBAR: case Op::CCTL: case Op::NANOSLEEP:
      return false;
    case Op::YIELD:
      w.give_way = true;
      return false;
    case Op::BPT:
      if (ex) throw Error(Err::Trap, "SASS kernel " + k_.name + " executed BPT.TRAP (a trap)");
      return false;
    case Op::BRA: {
      Mask take = ex;
      if (ins.f[0] >= 2) {
        // .DIV/.CONV: branch when the warp is (not) converged on the mask the
        // uniform register gives (~URZ: every lane).
        const Operand& u = ins.src[ins.src.size() - 2];
        uint32_t want = u.reg >= (m_.sm >= 100 ? 255u : 63u) ? 0 : w.ur[u.reg];
        if (u.bnot) want = ~want;
        const Mask full = want & w.alive & ~w.exited;
        const bool converged = group == full;
        take = (ins.f[0] == 2 ? !converged : converged) ? ex : 0;
      }
      const int64_t target = ins.src.back().imm;
      each_lane(group, [&](unsigned l) {
        const uint64_t base = base_of(l);
        const uint64_t next = w.pc[l] + 16;
        w.pc[l] = ((take >> l) & 1) ? base + static_cast<uint64_t>(target) : next;
      });
      // A group that branches backwards while other lanes wait gives way now
      // and then.
      if (take && target <= static_cast<int64_t>(ins.pc) && (w.runnable() & ~group)) {
        if (++w.spins >= 64) {
          w.spins = 0;
          w.give_way = true;
        }
      }
      return true;
    }
    case Op::BRX: {
      each_lane(group, [&](unsigned l) {
        if ((ex >> l) & 1) w.pc[l] = w.pc[l] + 16 + static_cast<uint64_t>(ins.src[0].imm) + w.reg(ins.src[0].reg, l);
        else w.pc[l] += 16;
      });
      return true;
    }
    case Op::CALL: {
      const bool abs = ins.f[0], reg_rel = ins.f[1];
      bool handled = false;
      each_lane(group, [&](unsigned l) {
        if (!((ex >> l) & 1)) {
          w.pc[l] += 16;
          return;
        }
        uint64_t target;
        if (abs) {
          const unsigned r = ins.src[0].reg;
          target = w.reg(r, l) | (static_cast<uint64_t>(w.reg(r + 1, l)) << 32);
        } else if (reg_rel) {
          target = base_of(l) + static_cast<uint64_t>(ins.src[0].imm) + w.reg(ins.src[0].reg, l);
        } else {
          target = base_of(l) + static_cast<uint64_t>(ins.src[0].imm);
        }
        if (const auto b = m_.builtins.find(target); b != m_.builtins.end()) {
          handled = true;
          return;   // run once for the lanes below
        }
        w.pc[l] = target;
      });
      if (handled) {
        // A call into a function VirtualGPU provides: run it for the calling
        // lanes, then carry on after the call.
        const unsigned l0 = static_cast<unsigned>(std::countr_zero(ex));
        const unsigned r = ins.src[0].reg;
        const uint64_t target = w.reg(r, l0) | (static_cast<uint64_t>(w.reg(r + 1, l0)) << 32);
        builtin_call(blk, w, ins, ex, m_.builtins.at(target));
        each_lane(ex, [&](unsigned l) { w.pc[l] += 16; });
      }
      return true;
    }
    case Op::RET:
      each_lane(group, [&](unsigned l) {
        if ((ex >> l) & 1) w.pc[l] = base_of(l) + w.reg(ins.src[0].reg, l);
        else w.pc[l] += 16;
      });
      return true;
    case Op::EXIT:
      each_lane(group, [&](unsigned l) {
        if ((ex >> l) & 1) w.exited |= Mask{1} << l;
        else w.pc[l] += 16;
      });
      // Lanes that exit no longer hold anything up.
      for (unsigned x = 0; x < 16; ++x) release_bsync(x);
      release_warpsync();
      for (unsigned id = 0; id < 16; ++id)
        if (blk.bars[id].arrived) release_bar(id);
      return true;
    case Op::BSSY:
      if (ex) w.b[ins.dst[0].reg] = ex | (w.b[ins.dst[0].reg] & 0);
      return false;
    case Op::BREAK:
      w.b[ins.src.back().reg] &= ~ex;
      release_bsync(ins.src.back().reg);
      return false;
    case Op::BSYNC: {
      const unsigned x = ins.src[0].reg;
      each_lane(group, [&](unsigned l) {
        if ((ex >> l) & 1) {
          w.waiting |= Mask{1} << l;
          w.wait_kind[l] = Wait::BSync;
          w.wait_arg[l] = x;
        } else {
          w.pc[l] += 16;
        }
      });
      release_bsync(x);
      return true;
    }
    case Op::WARPSYNC: {
      each_lane(group, [&](unsigned l) {
        if ((ex >> l) & 1) {
          w.waiting |= Mask{1} << l;
          w.wait_kind[l] = Wait::WarpSync;
          w.wait_arg[l] = read32(w, ins.src.back(), l);
        } else {
          w.pc[l] += 16;
        }
      });
      release_warpsync();
      return true;
    }
    case Op::BAR: {
      if (ins.mnemonic == "B2R") {
        each_lane(ex, [&](unsigned l) {
          write32(w, ins.dst[0], l, w.b2r);
          set_pred(w, ins.dst[1], l, w.b2r_pred);
        });
        return false;
      }
      const unsigned mode = ins.f[0];
      const unsigned id = ins.src[0].kind == Kind::Imm ? static_cast<unsigned>(ins.src[0].imm) : read32(w, ins.src[0], 0) & 15;
      Block::Barrier& b = blk.bars[id];
      b.mode = mode;
      b.red = ins.f[1];
      each_lane(ex, [&](unsigned l) {
        ++b.arrived;
        if (mode == 2) {
          const bool p = pred(w, ins.src.back(), l);
          b.popc += p;
          b.and_ = b.and_ && p;
          b.or_ = b.or_ || p;
        }
      });
      if (mode == 1) {   // ARV: arrive without waiting
        each_lane(group, [&](unsigned l) { w.pc[l] += 16; });
        release_bar(id);
        return true;
      }
      each_lane(group, [&](unsigned l) {
        if ((ex >> l) & 1) {
          w.waiting |= Mask{1} << l;
          w.wait_kind[l] = Wait::Bar;
          w.wait_arg[l] = id;
        } else {
          w.pc[l] += 16;
        }
      });
      release_bar(id);
      return true;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet");
  }
}

// ---- functions VirtualGPU provides ------------------------------------------------

void Runner::builtin_call(Block& blk, Warp& w, const Instr& ins, Mask ex, const std::string& name) {
  // The CUDA ABI: arguments in R4, R5, R6, R7...; a result in R4 (and R5).
  if (name == "vprintf") {
    std::lock_guard<std::mutex> g(stats_mu_);
    each_lane(ex, [&](unsigned l) {
      const uint64_t fmt = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
      uint64_t args = w.reg(6, l) | (static_cast<uint64_t>(w.reg(7, l)) << 32);
      const auto read_bytes = [&](uint64_t g, void* out, uint32_t n) {
        uint64_t off;
        const Space sp = classify(g, &off);
        mem_read(blk, w, l, sp, off, out, n);
      };
      std::string f;
      for (uint64_t a = fmt;; ++a) {
        char c;
        read_bytes(a, &c, 1);
        if (!c) break;
        f += c;
      }
      std::string out;
      for (size_t i = 0; i < f.size(); ++i) {
        if (f[i] != '%') {
          out += f[i];
          continue;
        }
        size_t j = i + 1;
        while (j < f.size() && std::strchr("-+ #0123456789.*", f[j])) ++j;
        std::string lenmod;
        while (j < f.size() && std::strchr("hlLqjzt", f[j])) lenmod += f[j++];
        if (j >= f.size()) break;
        const char conv = f[j];
        std::string spec = f.substr(i, j - i + 1);
        char buf[512];
        if (conv == '%') {
          out += '%';
        } else if (conv == 's') {
          args = (args + 7) & ~7ull;
          uint64_t p = 0;
          read_bytes(args, &p, 8);
          args += 8;
          std::string str;
          for (uint64_t a = p; p; ++a) {
            char c;
            read_bytes(a, &c, 1);
            if (!c) break;
            str += c;
          }
          std::snprintf(buf, sizeof buf, spec.c_str(), str.c_str());
          out += buf;
        } else if (std::strchr("fFeEgGaA", conv)) {
          args = (args + 7) & ~7ull;
          double d = 0;
          read_bytes(args, &d, 8);
          args += 8;
          std::snprintf(buf, sizeof buf, spec.c_str(), d);
          out += buf;
        } else {
          const bool wide = lenmod.find('l') != std::string::npos || lenmod == "j" || lenmod == "z" || lenmod == "t" ||
                            conv == 'p';
          const uint32_t n = wide ? 8 : 4;
          args = (args + n - 1) & ~static_cast<uint64_t>(n - 1);
          uint64_t v = 0;
          read_bytes(args, &v, n);
          args += n;
          if (wide) std::snprintf(buf, sizeof buf, spec.c_str(), static_cast<unsigned long long>(v));
          else std::snprintf(buf, sizeof buf, spec.c_str(), static_cast<unsigned>(v));
          out += buf;
        }
        i = j;
      }
      std::fwrite(out.data(), 1, out.size(), stdout);
      w.reg(4, l) = static_cast<uint32_t>(out.size());
    });
    std::fflush(stdout);
    return;
  }
  if (name == "malloc" || name == "free") {
    static std::mutex heap_mu;
    static std::map<const MemoryManager*, std::pair<uint64_t, std::map<uint64_t, uint64_t>>> heaps;
    std::lock_guard<std::mutex> g(heap_mu);
    auto& heap = heaps[&mem_];
    each_lane(ex, [&](unsigned l) {
      const uint64_t arg = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
      if (name == "malloc") {
        uint64_t p = 0;
        if (arg && heap.first + arg <= cfg_.device_heap_bytes) {
          try {
            p = mem_.alloc(arg);
            heap.first += arg;
            heap.second[p] = arg;
          } catch (const Error&) {
            p = 0;
          }
        }
        w.reg(4, l) = static_cast<uint32_t>(p);
        w.reg(5, l) = static_cast<uint32_t>(p >> 32);
      } else if (arg) {
        const auto it = heap.second.find(arg);
        if (it == heap.second.end()) fault(w, ins, l, "device free() of a pointer this kernel's heap did not allocate");
        heap.first -= it->second;
        heap.second.erase(it);
        mem_.free(arg);
      }
    });
    return;
  }
  if (name == "__assertfail") {
    const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
    const uint64_t msg = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
    std::string text;
    for (uint64_t a = msg; msg; ++a) {
      uint64_t off;
      const Space sp = classify(a, &off);
      char c;
      mem_read(blk, w, l, sp, off, &c, 1);
      if (!c) break;
      text += c;
    }
    throw Error(Err::DeviceAssert, "device-side assertion failed: " + text);
  }
  fault(w, ins, 0, "call to " + name + ", which is not available");
}
