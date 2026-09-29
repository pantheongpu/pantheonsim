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

// The profiler's instruction classes, by SASS op.
exec::InstClass inst_class(Op op) {
  using C = exec::InstClass;
  switch (op) {
    case Op::FADD: case Op::FMUL: case Op::FFMA: case Op::FMNMX: case Op::FSETP: case Op::FSET: case Op::FSEL:
    case Op::FCHK: case Op::MUFU:
      return C::Fp32;
    case Op::DADD: case Op::DMUL: case Op::DFMA: case Op::DSETP: return C::Fp64;
    case Op::HADD2: case Op::HMUL2: case Op::HFMA2: case Op::HSETP2: case Op::HMNMX2: return C::Fp16;
    case Op::I2F: case Op::I2FP: case Op::F2I: case Op::F2IP: case Op::F2F: case Op::F2FP: case Op::FRND:
    case Op::I2I: case Op::I2IP: case Op::PRMT: case Op::UPRMT:
      return C::BitConvert;
    case Op::BRA: case Op::BRX: case Op::JMP: case Op::CALL: case Op::RET: case Op::EXIT: case Op::BSSY:
    case Op::BSYNC: case Op::BREAK: case Op::WARPSYNC: case Op::BAR: case Op::BPT: case Op::YIELD:
    case Op::NANOSLEEP:
      return C::Control;
    case Op::LDG: case Op::STG: case Op::LDS: case Op::STS: case Op::LDL: case Op::STL: case Op::LD: case Op::ST:
    case Op::LDC: case Op::ATOM: case Op::ATOMG: case Op::ATOMS: case Op::RED: case Op::LDGSTS: case Op::LDSM:
    case Op::SULD: case Op::SUST: case Op::TEX: case Op::TLD: case Op::TLD4: case Op::MEMBAR: case Op::CCTL:
      return C::Memory;
    case Op::HMMA: case Op::IMMA: case Op::BMMA: case Op::DMMA: case Op::QMMA: case Op::MOVM: return C::Tensor;
    case Op::IMAD: case Op::IADD3: case Op::LOP3: case Op::SHF: case Op::LEA: case Op::IMNMX: case Op::IABS:
    case Op::POPC: case Op::FLO: case Op::BREV: case Op::BMSK: case Op::SGXT: case Op::IDP: case Op::VABSDIFF4: case Op::ISETP:
    case Op::UIMAD: case Op::UIADD3: case Op::ULOP3: case Op::USHF: case Op::ULEA: case Op::UISETP:
    case Op::UPOPC: case Op::UFLO:
      return C::Integer;
    default: return C::Misc;
  }
}

void Runner::execute(Block& blk, Warp& w, const Instr& ins, Mask group) {
  const Mask ex = guard_mask(w, ins, group);
  {
    exec::LaunchStats& st = blk.st;
    const uint64_t lanes = static_cast<uint64_t>(std::popcount(ex));
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[static_cast<size_t>(inst_class(ins.op))] += lanes;
    if (ins.op == Op::HMMA || ins.op == Op::IMMA || ins.op == Op::BMMA || ins.op == Op::DMMA || ins.op == Op::QMMA)
      ++st.tensor_instructions;
    if (ins.op == Op::BAR && ins.mnemonic == "BAR" && ex) ++st.barriers;
    if (ins.op == Op::BRA && ex && ex != group) ++st.divergent_branches;
  }
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
      case Op::RED: case Op::LDGSTS: case Op::LDGDEPBAR: case Op::LDSM:
        exec_mem(blk, w, ins, ex);
        break;
      case Op::TEX: case Op::TLD: case Op::TLD4: case Op::SULD: case Op::SUST:
        exec_tex(blk, w, ins, ex);
        break;
      case Op::SHFL: case Op::VOTE: case Op::VOTEU: case Op::MATCH: case Op::REDUX:
        exec_warp(blk, w, ins, ex, group);
        break;
      case Op::HMMA: case Op::IMMA: case Op::BMMA: case Op::DMMA: case Op::QMMA: case Op::MOVM:
        exec_mma(blk, w, ins, ex);
        break;
      case Op::FADD: case Op::FMUL: case Op::FFMA: case Op::FMNMX: case Op::FSETP: case Op::FSET:
      case Op::FSEL: case Op::FCHK: case Op::MUFU: case Op::DADD: case Op::DMUL: case Op::DFMA:
      case Op::DSETP: case Op::HADD2: case Op::HMUL2: case Op::HFMA2: case Op::HSETP2: case Op::HMNMX2:
      case Op::I2F: case Op::I2FP: case Op::F2I: case Op::F2IP: case Op::F2F: case Op::F2FP:
      case Op::FRND: case Op::I2I: case Op::I2IP: case Op::HSET2:
        exec_float(blk, w, ins, ex);
        break;
      default:
        exec_int(blk, w, ins, ex);
        break;
    }
  } catch (const Fault& f) {
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what());
  } catch (const Error& e) {
    // An error from the device (an address out of bounds, a trap), kept as
    // what it is with the instruction's place added (once).
    if (e.message().rfind("SASS kernel ", 0) == 0) throw;
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), e.message(), e.code());
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
        if (ins.dst.size() > 1) set_pred(w, ins.dst[1], l, (sum >> 32) & 1);
        if (ins.dst.size() > 2) set_pred(w, ins.dst[2], l, (sum >> 33) & 1);
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
    case Op::IDP: {
      const bool two = ins.f[0], sa = ins.f[1], sb = ins.f[2], hi = ins.f[3];
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l), b = read32(w, ins.src[1], l);
        uint32_t acc = read32(w, ins.src[2], l);
        const auto byte = [](uint32_t v, unsigned i, bool s) {
          const uint32_t x = (v >> (8 * i)) & 0xff;
          return s ? static_cast<int32_t>(static_cast<int8_t>(x)) : static_cast<int32_t>(x);
        };
        for (unsigned i = 0; i < (two ? 2u : 4u); ++i) {
          int32_t x;
          if (two) {
            const uint32_t h = (a >> (16 * i)) & 0xffff;
            x = sa ? static_cast<int32_t>(static_cast<int16_t>(h)) : static_cast<int32_t>(h);
          } else {
            x = byte(a, i, sa);
          }
          acc += static_cast<uint32_t>(x) * static_cast<uint32_t>(byte(b, i + (two && hi ? 2 : 0), sb));
        }
        write32(w, ins.dst[0], l, acc);
      });
      return;
    }
    case Op::VABSDIFF4: {
      const bool accum = ins.f[0], sgn = ins.f[1];
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l), b = read32(w, ins.src[1], l), c = read32(w, ins.src[2], l);
        uint32_t packed = 0, sum = 0;
        for (unsigned i = 0; i < 4; ++i) {
          int32_t x = (a >> (8 * i)) & 0xff, y = (b >> (8 * i)) & 0xff;
          if (sgn) {
            x = static_cast<int8_t>(x);
            y = static_cast<int8_t>(y);
          }
          const uint32_t d = static_cast<uint32_t>(x > y ? x - y : y - x);
          packed |= (d & 0xff) << (8 * i);
          sum += d;
        }
        write32(w, ins.dst[0], l, accum ? sum + c : packed);
      });
      return;
    }
    case Op::SGXT:
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l), b = read32(w, ins.src[1], l);
        const uint32_t n = ins.f[1] ? b & 31 : std::min(b, 32u);
        uint32_t r;
        if (n == 0) r = 0;
        else if (n >= 32) r = a;
        else if (ins.f[0]) r = static_cast<uint32_t>(static_cast<int32_t>(a << (32 - n)) >> (32 - n));
        else r = a & ((1u << n) - 1);
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
      if (ins.f[1]) {   // c[bank][URa + offset]
        const uint64_t o = w.ur[c.reg] + static_cast<uint64_t>(c.imm);
        if (size == 5) write64(w, ins.dst[0], 0, cbank32(ins.f[2], o) | (static_cast<uint64_t>(cbank32(ins.f[2], o + 4)) << 32));
        else write32(w, ins.dst[0], 0, cbank32(ins.f[2], o));
        return;
      }
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
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet", Err::UnsupportedPtx);
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

// A float comparison by its 4-bit code: F, LT, EQ, LE, GT, NE, GE, NUM, NAN,
// then LT..GE again unordered (true when either side is NaN), T.
bool float_cmp(unsigned cmp, double a, double b) {
  const bool un = a != a || b != b;
  switch (cmp) {
    case 0: return false;
    case 1: return !un && a < b;
    case 2: return !un && a == b;
    case 3: return !un && a <= b;
    case 4: return !un && a > b;
    case 5: return !un && a != b;
    case 6: return !un && a >= b;
    case 7: return !un;
    case 8: return un;
    case 9: return un || a < b;
    case 10: return un || a == b;
    case 11: return un || a <= b;
    case 12: return un || a > b;
    case 13: return un || a != b;
    case 14: return un || a >= b;
    default: return true;
  }
}
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
        else if (ins.op == Op::FMUL && ins.f[4] != 4) {
          // .D8/.D4/.D2/.M2/.M4/.M8: the product times 2^(f - 4), rounded once.
          const int k = static_cast<int>(ins.f[4]) - 4;
          r = rounded(rnd, [&] { return static_cast<float>(std::ldexp(static_cast<double>(a) * b, k)); });
        } else if (ins.op == Op::FMUL) r = rounded(rnd, [&] { return a * b; });
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
        if (ins.f[1] && (a != a || b != b)) {   // .NAN: NaN in, the canonical NaN out
          write32(w, ins.dst[0], l, 0x7fffffffu);
          return;
        }
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
        if (ins.op == Op::I2F && ssize < 2) v >>= (ssize == 1 ? 16 : 8) * ins.f[4];   // .H1, .B1-.B3
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
        else if (dsize == 4) write32(w, ins.dst[0], l, f32_to_bf16(static_cast<float>(x)));
        else if (dsize == 2) write32(w, ins.dst[0], l, f32_bits(rounded(rnd, [&] { return static_cast<float>(x); })));
        else write32(w, ins.dst[0], l, f64_to_half(x));
      });
      return;
    }
    case Op::F2FP:
      if (ins.f[3] >= 6 || ins.f[5] == 4) {   // the FP8 forms
        const bool unpack = ins.f[5] == 4, sat = ins.f[6];
        each_lane(ex, [&](unsigned l) {
          if (unpack) {   // b's two bytes (low first) to two halves
            const uint32_t v = read32(w, ins.src[0], l);
            const bool e5 = ins.f[4] == 2;
            const uint16_t lo = f64_to_half(exec::fp8_value(v & 0xff, e5));
            const uint16_t hi = f64_to_half(exec::fp8_value((v >> 8) & 0xff, e5));
            write32(w, ins.dst[0], l, lo | (static_cast<uint32_t>(hi) << 16));
            return;
          }
          const bool e5 = ins.f[3] == 6;
          float a = f32src(ins.src[0], l, false), b = f32src(ins.src[1], l, false);
          if (ins.f[2]) {
            if (a < 0) a = 0;
            if (b < 0) b = 0;
          }
          uint32_t r = (exec::fp8_bits(a, e5, sat) << 8) | exec::fp8_bits(b, e5, sat);
          if (ins.f[5] == 1) r |= read32(w, ins.src[2], l) & 0xffff0000u;   // MERGE_C
          write32(w, ins.dst[0], l, r);
        });
        return;
      }
      each_lane(ex, [&](unsigned l) {
        float a = f32src(ins.src[0], l, false), b = f32src(ins.src[1], l, false);
        if (ins.f[2]) {   // .RELU
          if (a < 0) a = 0;
          if (b < 0) b = 0;
        }
        uint32_t r;
        if (ins.f[0]) r = (static_cast<uint32_t>(f32_to_bf16(a)) << 16) | f32_to_bf16(b);
        else r = (static_cast<uint32_t>(f32_to_half(a)) << 16) | f32_to_half(b);
        write32(w, ins.dst[0], l, r);
      });
      return;
    case Op::I2IP:
      each_lane(ex, [&](unsigned l) {
        const unsigned t = ins.f[0], bits = t < 2 ? 8 : 16;
        const bool sgn = t & 1;
        const int64_t lo = sgn ? -(int64_t{1} << (bits - 1)) : 0, hi = sgn ? (int64_t{1} << (bits - 1)) - 1 : (int64_t{1} << bits) - 1;
        const auto sat = [&](uint32_t v) {
          return static_cast<uint32_t>(std::clamp<int64_t>(static_cast<int32_t>(v), lo, hi)) & ((1u << bits) - 1);
        };
        uint64_t d = sat(read32(w, ins.src[1], l)) | (static_cast<uint64_t>(sat(read32(w, ins.src[0], l))) << bits);
        d |= static_cast<uint64_t>(read32(w, ins.src[2], l)) << (2 * bits);
        write32(w, ins.dst[0], l, static_cast<uint32_t>(d));
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
    case Op::FSETP: case Op::FSET: {
      const unsigned cmp = ins.f[0], bop = ins.f[1];
      const bool fz = ins.f[2];
      each_lane(ex, [&](unsigned l) {
        const float a = f32src(ins.src[0], l, fz), b = f32src(ins.src[1], l, fz);
        const bool acc = pred(w, ins.src[2], l), r = float_cmp(cmp, a, b);
        if (ins.op == Op::FSET) {
          write32(w, ins.dst[0], l, bool_op(bop, r, acc) ? f32_bits(1.0f) : 0u);
        } else {
          set_pred(w, ins.dst[0], l, bool_op(bop, r, acc));
          set_pred(w, ins.dst[1], l, bool_op(bop, !r, acc));
        }
      });
      return;
    }
    case Op::FSEL:
      each_lane(ex, [&](unsigned l) {
        const Operand& o = pred(w, ins.src[2], l) ? ins.src[0] : ins.src[1];
        uint32_t v = read32(w, o, l);
        if (o.abs) v &= 0x7fffffffu;
        if (o.neg) v ^= 0x80000000u;
        write32(w, ins.dst[0], l, v);
      });
      return;
    case Op::DSETP: {
      const unsigned cmp = ins.f[0], bop = ins.f[1];
      each_lane(ex, [&](unsigned l) {
        const double a = f64src(ins.src[0], l), b = f64src(ins.src[1], l);
        const bool acc = pred(w, ins.src[2], l);
        if (cmp == 0 || cmp == 15) {
          // MIN/MAX: whether a is the answer (a NaN loses; of two zeros MIN
          // takes the negative one), and whether both are NaN -- what fmin
          // and fmax need to pick a register pair and quiet the NaN.
          const bool an = a != a, bn = b != b;
          bool pick_a;
          if (an || bn) pick_a = !an;
          else if (a == b) pick_a = cmp == 0 ? std::signbit(a) : !std::signbit(a);
          else pick_a = cmp == 0 ? a < b : a > b;
          set_pred(w, ins.dst[0], l, bool_op(bop, pick_a, acc));
          set_pred(w, ins.dst[1], l, an && bn);
          return;
        }
        const bool r = float_cmp(cmp, a, b);
        set_pred(w, ins.dst[0], l, bool_op(bop, r, acc));
        set_pred(w, ins.dst[1], l, bool_op(bop, !r, acc));
      });
      return;
    }
    case Op::FRND: {
      const unsigned rnd = ins.f[0], dsz = ins.f[2], ssz = ins.f[3];
      const bool fz = ins.f[1];
      const auto round = [&](double x) {
        switch (rnd) {
          case 1: return std::floor(x);
          case 2: return std::ceil(x);
          case 3: return std::trunc(x);
          default: return std::nearbyint(x);   // the host mode is to-nearest-even
        }
      };
      each_lane(ex, [&](unsigned l) {
        double x;
        if (ssz == 3) x = f64src(ins.src[0], l);
        else if (ssz == 1) x = half_to_f32(static_cast<uint16_t>(read32(w, ins.src[0], l)));
        else x = f32src(ins.src[0], l, fz);
        const double r = round(x);
        if (dsz == 3) write64(w, ins.dst[0], l, f64_bits(r));
        else if (dsz == 1) write32(w, ins.dst[0], l, f64_to_half(r));
        else write32(w, ins.dst[0], l, f32_bits(fz ? ftz(static_cast<float>(r)) : static_cast<float>(r)));
      });
      return;
    }
    case Op::FCHK:
      // Whether a / b must take the division's slow path. Saying yes always
      // is exact: the slow path is correctly rounded for every input, the
      // fast one only for the inputs this check lets through.
      each_lane(ex, [&](unsigned l) { set_pred(w, ins.dst[0], l, true); });
      return;
    case Op::HSETP2: case Op::HSET2: {
      const unsigned cmp = ins.f[0], bop = ins.f[1];
      const bool bf_result = ins.f[2], bf16 = ins.f[3], fz = ins.f[4];
      const auto conv = [&](uint16_t h) {
        const float f = bf16 ? bf16_to_f32(h) : half_to_f32(h);
        return fz ? ftz(f) : f;
      };
      const auto halves = [&](const Operand& o, unsigned l, float out[2]) {
        const uint32_t v = read32(w, o, l);
        float lo = conv(static_cast<uint16_t>(v)), hi = conv(static_cast<uint16_t>(v >> 16));
        if (o.suffix == ".H0_H0") hi = lo;
        else if (o.suffix == ".H1_H1") lo = hi;
        else if (o.suffix == ".F32") lo = hi = as_f32(v);
        out[0] = lo;
        out[1] = hi;
        for (int i = 0; i < 2; ++i) {
          if (o.abs) out[i] = std::fabs(out[i]);
          if (o.neg) out[i] = -out[i];
        }
      };
      each_lane(ex, [&](unsigned l) {
        float a[2], b[2];
        halves(ins.src[0], l, a);
        size_t next;
        if (ins.src.size() > 3 && ins.src[1].kind == Kind::FImm && ins.src[2].kind == Kind::FImm) {
          b[1] = conv(static_cast<uint16_t>(ins.src[1].fbits));   // the high half prints first
          b[0] = conv(static_cast<uint16_t>(ins.src[2].fbits));
          next = 3;
        } else {
          halves(ins.src[1], l, b);
          next = 2;
        }
        const bool acc = pred(w, ins.src[next], l);
        bool r[2];
        for (int i = 0; i < 2; ++i) r[i] = bool_op(bop, float_cmp(cmp, a[i], b[i]), acc);
        if (ins.op == Op::HSETP2) {
          set_pred(w, ins.dst[0], l, r[0]);
          set_pred(w, ins.dst[1], l, r[1]);
        } else {
          const uint32_t one = bf16 ? 0x3f80u : 0x3c00u;
          uint32_t v = 0;
          for (int i = 0; i < 2; ++i)
            if (r[i]) v |= (bf_result ? one : 0xffffu) << (16 * i);
          write32(w, ins.dst[0], l, v);
        }
      });
      return;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet", Err::UnsupportedPtx);
  }
}

// ---- memory ------------------------------------------------------------------

// Counts one warp-level access: per-lane operations and bytes, the 32-byte
// sectors the lanes' addresses touch, and shared-memory bank conflicts.
void count_access(exec::LaunchStats& st, int space /* 0 global, 1 shared, 2 local */, bool store,
                  const uint64_t* addr, Mask lanes, uint32_t bytes) {
  const uint64_t n = static_cast<uint64_t>(std::popcount(lanes));
  if (!n) return;
  std::vector<uint64_t> sectors;
  each_lane(lanes, [&](unsigned l) {
    for (uint64_t s = addr[l] / 32; s <= (addr[l] + bytes - 1) / 32; ++s) sectors.push_back(s);
  });
  std::sort(sectors.begin(), sectors.end());
  const uint64_t distinct = static_cast<uint64_t>(std::unique(sectors.begin(), sectors.end()) - sectors.begin());
  switch (space) {
    case 0:
      (store ? st.global_stores : st.global_loads) += n;
      (store ? st.global_bytes_written : st.global_bytes_read) += n * bytes;
      st.global_sectors += distinct;
      ++st.global_requests;
      (store ? st.global_sectors_st : st.global_sectors_ld) += distinct;
      ++(store ? st.global_requests_st : st.global_requests_ld);
      (store ? st.global_bytes_st : st.global_bytes_ld) += n * bytes;
      break;
    case 1: {
      (store ? st.shared_stores : st.shared_loads) += n;
      (store ? st.shared_bytes_written : st.shared_bytes_read) += n * bytes;
      ++st.shared_requests;
      ++(store ? st.shared_requests_st : st.shared_requests_ld);
      // 32 banks of 4 bytes: lanes on different words of one bank take a
      // pass each; lanes on the same word share one.
      std::vector<uint64_t> words[32];
      each_lane(lanes, [&](unsigned l) {
        for (uint64_t a = addr[l] & ~3ull; a < addr[l] + bytes; a += 4) words[(a / 4) % 32].push_back(a / 4);
      });
      uint64_t passes = 1;
      for (auto& v : words) {
        std::sort(v.begin(), v.end());
        passes = std::max<uint64_t>(passes, static_cast<uint64_t>(std::unique(v.begin(), v.end()) - v.begin()));
      }
      st.shared_bank_conflicts += passes - 1;
      (store ? st.shared_bank_conflicts_st : st.shared_bank_conflicts_ld) += passes - 1;
      break;
    }
    default:
      (store ? st.local_stores : st.local_loads) += n;
      (store ? st.local_bytes_written : st.local_bytes_read) += n * bytes;
      st.local_sectors += distinct;
      ++st.local_requests;
      break;
  }
}

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
      uint64_t addrs[32] = {};
      int space = 0;
      each_lane(ex, [&](unsigned l) {
        const uint64_t g = address(a, l, wide, ur, store ? 64 : 32, 0);
        uint64_t off;
        const Space sp = generic ? classify(g, &off) : (off = g, Space::Global);
        addrs[l] = off;
        space = sp == Space::Global ? 0 : sp == Space::Shared ? 1 : 2;
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
      count_access(blk.st, space, store, addrs, ex, bytes);
      return;
    }
    case Op::LDS: case Op::STS: case Op::LDL: case Op::STL: {
      const bool store = ins.op == Op::STS || ins.op == Op::STL;
      const bool shared = ins.op == Op::LDS || ins.op == Op::STS;
      const unsigned strides[] = {0, 4, 8, 16};
      const unsigned stride = shared ? strides[ins.f[3]] : 0;
      const bool ur = ins.w.bit(91);
      const Operand& a = ins.src[0];   // the address (a store's data is src[1])
      uint64_t addrs[32] = {};
      each_lane(ex, [&](unsigned l) {
        const uint64_t off = static_cast<uint32_t>(address(a, l, false, ur, store ? 64 : 32, stride));
        addrs[l] = off;
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
      count_access(blk.st, shared ? 1 : 2, store, addrs, ex, bytes);
      return;
    }
    case Op::LDGDEPBAR:
      return;   // the copies are done when LDGSTS runs
    case Op::LDGSTS: {
      // The copy happens now; the DEPBAR that waits for it has nothing to
      // wait for.
      const bool zfill = ins.f[1];
      const bool has_p = ins.src.size() > 2;
      uint64_t gaddr[32] = {}, saddr[32] = {};
      each_lane(ex, [&](unsigned l) {
        const unsigned rs = static_cast<unsigned>(ins.w.field(16, 8));
        const uint64_t sh = static_cast<uint32_t>((rs == kRZ ? 0u : w.reg(rs, l)) + static_cast<uint32_t>(ins.src[0].imm));
        uint64_t g = address(ins.src[1], l, true, false, 0, 0);
        saddr[l] = sh;
        gaddr[l] = g;
        uint8_t buf[16] = {};
        uint32_t keep = bytes;   // bytes read; the rest are zero
        if (has_p && !pred(w, ins.src[2], l)) keep = 0;
        else if (zfill) {
          const uint32_t fill = static_cast<uint32_t>(g & (bytes - 1));
          g -= fill;
          keep = bytes - fill;
        }
        if (keep) mem_read(blk, w, l, Space::Global, g, buf, keep);
        mem_write(blk, w, l, Space::Shared, sh, buf, bytes);
      });
      count_access(blk.st, 0, false, gaddr, ex, bytes);
      count_access(blk.st, 1, true, saddr, ex, bytes);
      return;
    }
    case Op::LDSM: {
      // ldmatrix: matrix i's rows come from the addresses lanes 8i..8i+7
      // hold; lane t gets row t/4's 32-bit word t%4 (.MT88: column t/4's
      // halves from rows 2(t%4) and 2(t%4)+1).
      const unsigned nmat = ins.f[0];
      const bool trans = ins.f[1];
      uint64_t rows[32] = {};
      Mask have = 0;
      each_lane(ex, [&](unsigned l) {
        rows[l] = static_cast<uint32_t>(address(ins.src[0], l, false, ins.w.bit(91), 32, 0));
        have |= Mask{1} << l;
      });
      const auto row = [&](unsigned r) {
        if (!(have >> r & 1)) fault(w, ins, r, "LDSM: the lane holding a row address is not active");
        return rows[r];
      };
      uint32_t out[4][32] = {};
      each_lane(ex, [&](unsigned t) {
        for (unsigned i = 0; i < nmat; ++i) {
          if (!trans) {
            mem_read(blk, w, t, Space::Shared, row(8 * i + t / 4) + 4 * (t % 4), &out[i][t], 4);
          } else {
            uint16_t lo = 0, hi = 0;
            mem_read(blk, w, t, Space::Shared, row(8 * i + 2 * (t % 4)) + 2 * (t / 4), &lo, 2);
            mem_read(blk, w, t, Space::Shared, row(8 * i + 2 * (t % 4) + 1) + 2 * (t / 4), &hi, 2);
            out[i][t] = lo | (static_cast<uint32_t>(hi) << 16);
          }
        }
      });
      const unsigned d = ins.dst[0].reg;
      each_lane(ex, [&](unsigned t) {
        for (unsigned i = 0; i < nmat; ++i)
          if (d != kRZ) w.reg(d + i, t) = out[i][t];
      });
      blk.st.shared_loads += static_cast<uint64_t>(std::popcount(ex));
      blk.st.shared_bytes_read += static_cast<uint64_t>(std::popcount(ex)) * 4 * nmat;
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
      const bool cas = ins.f[5] || (shared && ins.mods[0].rfind("CAS", 0) == 0);
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
          const uint64_t cmp = v, swap = is64 ? read64(w, ins.src[2], l) : read32(w, ins.src[2], l);
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
        ++blk.st.atomics;
        blk.st.atomic_bytes += n;
        if (!red) {
          const Operand& d = ins.op == Op::ATOMS ? ins.dst[0] : ins.dst[1];
          // ATOM/ATOMG's predicate: done in hardware (a generic CAS of a
          // shared address included, so no software path is needed).
          if (ins.op != Op::ATOMS) set_pred(w, ins.dst[0], l, true);
          if (is64) write64(w, d, l, old);
          else write32(w, d, l, static_cast<uint32_t>(old));
          if (cas && ins.f[1] == 3) write32(w, d, l, old == v ? 1 : 0);   // CAST.SPIN: whether it stored
        }
      });
      return;
    }
    default:
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet", Err::UnsupportedPtx);
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
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet", Err::UnsupportedPtx);
  }
}

// ---- tensor cores --------------------------------------------------------------

// The warp-wide matrix multiply-accumulates. They are PTX's mma.sync, and
// read their registers in the fragment layouts the PTX ISA documents (9.7.16):
// lane = 4 * group + tid; A register r holds row group + 8(r % 2), columns
// from tid * per_reg + (r / 2) * 4 * per_reg; B register r rows tid * per_reg
// + 4 * per_reg * r, column group; C/D element s at row group + 8(s / 2),
// column 2 tid + s % 2. The numerics are the PTX interpreter's, which were
// checked against an RTX 3060: float products exact, summed in f32 in K
// order; f64 a chain of fused multiply-adds; integers exact, .SAT clamping
// after every 128 bits of K; 1-bit ones counting .AND/.XOR bits.
namespace {
enum class MmaT { F16, BF16, TF32, S8, U8, S4, U4, B1, F64, E4M3, E5M2 };
unsigned mma_bits(MmaT t) {
  switch (t) {
    case MmaT::F16: case MmaT::BF16: return 16;
    case MmaT::S8: case MmaT::U8: case MmaT::E4M3: case MmaT::E5M2: return 8;
    case MmaT::S4: case MmaT::U4: return 4;
    case MmaT::B1: return 1;
    case MmaT::F64: return 64;
    default: return 32;
  }
}
double mma_elem(MmaT t, uint64_t v, unsigned e) {
  switch (t) {
    case MmaT::F16: return half_to_f32(static_cast<uint16_t>(v >> (16 * e)));
    case MmaT::BF16: return bf16_to_f32(static_cast<uint16_t>(v >> (16 * e)));
    case MmaT::TF32: {
      // tf32: the top 19 bits of an f32; the rest are ignored.
      return as_f32(static_cast<uint32_t>(v) & 0xffffe000u);
    }
    case MmaT::S8: return static_cast<int8_t>(v >> (8 * e));
    case MmaT::E4M3: return exec::fp8_value(static_cast<uint32_t>(v >> (8 * e)), false);
    case MmaT::E5M2: return exec::fp8_value(static_cast<uint32_t>(v >> (8 * e)), true);
    case MmaT::U8: return static_cast<uint8_t>(v >> (8 * e));
    case MmaT::S4: return static_cast<int8_t>(static_cast<uint8_t>(((v >> (4 * e)) & 0xf) << 4)) >> 4;
    case MmaT::U4: return static_cast<double>((v >> (4 * e)) & 0xf);
    case MmaT::B1: return static_cast<double>((v >> e) & 1);
    case MmaT::F64: return as_f64(v);
  }
  return 0;
}
}  // namespace

// ---- textures and surfaces ---------------------------------------------------------
//
// The object is named by a constant-bank word (the header index, 40-53, in
// the bank at 54-58): the 64-bit handle the host created, which the shared
// sampler (vgpu/exec/texture.hpp) looks up as the PTX interpreter does.
//
// A fetch's operands are two register vectors, from Mesa NAK's encoder (MIT)
// and checked against ptxas's output: the first holds the layer index, if
// any, then the coordinates; the second the LOD or bias, the packed offsets
// (bit 76), the depth reference (bit 78). The .SCR form (bit 60) splits the
// whole list of up to four between them, two and two (1+1 for two, 2+1 for
// three). The enabled components (the mask, 72-75) come back in order, the
// first two in the register pair at 16 and the rest in the pair at 64.
//
// Surfaces take the coordinates -- x in bytes, then y, then the layer or z --
// in consecutive registers from 24; SUST's data from 32. The out-of-range
// mode (59-60) is .IGN (PTX .zero), the default (.clamp) or .TRAP.
void Runner::exec_tex(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  const Word& word = ins.w;
  const unsigned bank = static_cast<unsigned>(word.field(54, 5));
  const uint64_t hoff = 4 * word.field(40, 14);
  const uint64_t handle = cbank32(bank, hoff) | (static_cast<uint64_t>(cbank32(bank, hoff + 4)) << 32);
  const bool surface = ins.op == Op::SULD || ins.op == Op::SUST;
  const exec::TextureDesc& d =
      exec::texture_lookup(cfg_.textures, handle, surface ? exec::TexKind::Surface : exec::TexKind::Texture);
  const auto rd = [&](unsigned r, unsigned l) -> uint32_t { return r == kRZ ? 0u : w.reg(r, l); };
  const auto wr = [&](unsigned r, unsigned l, uint32_t v) {
    if (r != kRZ) w.reg(r, l) = v;
  };
  const uint64_t n = static_cast<uint64_t>(std::popcount(ex));

  if (surface) {
    const unsigned dim = static_cast<unsigned>(word.field(61, 3));   // 1D, 1D_BUFFER, 1D_ARRAY, 2D, 2D_ARRAY, 3D
    const unsigned size = static_cast<unsigned>(word.field(73, 3));  // U8, S8, U16, S16, 32, 64, 128
    if (dim > 5 || size > 6) fault(w, ins, 0, "surface access form not known");
    const uint32_t bytes = size <= 1 ? 1 : size <= 3 ? 2 : size == 4 ? 4 : size == 5 ? 8 : 16;
    static const uint8_t kOob[4] = {exec::kSurfaceZero, exec::kSurfaceClamp, exec::kSurfaceTrap, exec::kSurfaceTrap};
    const unsigned cr = static_cast<unsigned>(word.field(24, 8));
    const unsigned dr = ins.op == Op::SULD ? static_cast<unsigned>(word.field(16, 8))
                                           : static_cast<unsigned>(word.field(32, 8));
    const unsigned words = (bytes + 3) / 4;
    each_lane(ex, [&](unsigned l) {
      exec::SurfaceAccess a;
      a.layered = dim == 2 || dim == 4;
      a.dims = dim == 5 ? 3 : dim >= 3 ? 2 : 1;
      a.x = static_cast<int32_t>(rd(cr, l));
      if (a.dims > 1) a.y = static_cast<int32_t>(rd(cr + 1, l));
      if (a.dims > 2) a.z = static_cast<int32_t>(rd(cr + 2, l));
      if (a.layered) a.layer = rd(cr + a.dims, l);
      a.bytes = bytes;
      a.oob = kOob[word.field(59, 2)];
      const std::optional<uint64_t> at = exec::surface_address(d, a);
      if (ins.op == Op::SULD) {
        for (unsigned i = 0; i < words; ++i) {
          uint32_t v = 0;
          if (at) v = static_cast<uint32_t>(mem_.load_scalar(*at + 4 * i, std::min(bytes, 4u)));
          if (size == 1) v = static_cast<uint32_t>(static_cast<int8_t>(v));
          if (size == 3) v = static_cast<uint32_t>(static_cast<int16_t>(v));
          wr(dr + i, l, v);
        }
      } else if (at) {
        for (unsigned i = 0; i < words; ++i) mem_.store_scalar(*at + 4 * i, std::min(bytes, 4u), rd(dr + i, l));
      }
    });
    (ins.op == Op::SUST ? blk.st.global_stores : blk.st.global_loads) += n;
    (ins.op == Op::SUST ? blk.st.global_bytes_written : blk.st.global_bytes_read) += n * bytes;
    return;
  }

  // dim: 1D, 2D, 3D, CUBE, ARRAY_1D, ARRAY_2D, -, ARRAY_CUBE
  const unsigned dim = ins.f[0];
  if (dim == 6) fault(w, ins, 0, "texture dimension 6 is not known");
  const bool layered = dim >= 4, cube = dim == 3 || dim == 7;
  const uint32_t dims = cube ? 3 : (dim & 3) + 1;
  const unsigned lod_mode = ins.op == Op::TLD4 ? 0 : static_cast<unsigned>(word.field(87, 3));
  const bool has_lod = lod_mode == 2 || lod_mode == 3 || lod_mode == 5;
  if (lod_mode == 4 || lod_mode == 5)
    fault(w, ins, 0, "texture fetches with a LOD clamp are not implemented yet", Err::UnsupportedPtx);
  if (word.bit(76) || (ins.op == Op::TLD4 && word.bit(77)))
    fault(w, ins, 0, "texture fetches with offsets are not implemented yet", Err::UnsupportedPtx);
  if (word.bit(78)) fault(w, ins, 0, "depth-compare texture fetches are not implemented yet", Err::UnsupportedPtx);
  if (ins.op == Op::TEX && lod_mode == 2)
    fault(w, ins, 0, "texture fetches with a LOD bias are not implemented yet", Err::UnsupportedPtx);
  const unsigned n0 = (layered ? 1 : 0) + dims, n1 = has_lod ? 1 : 0;
  const unsigned total = n0 + n1;
  unsigned in0 = n0;   // components in the first vector
  if (word.bit(60)) {
    if (total > 4) fault(w, ins, 0, "a .SCR texture fetch of more than four components");
    in0 = total == 1 ? 1 : total == 2 ? 1 : 2;
  }
  const unsigned r0 = static_cast<unsigned>(word.field(24, 8)), r1 = static_cast<unsigned>(word.field(32, 8));
  const unsigned d0 = static_cast<unsigned>(word.field(16, 8)), d1 = static_cast<unsigned>(word.field(64, 8));
  const unsigned mask = static_cast<unsigned>(word.field(72, 4));
  const bool int_coords = ins.op == Op::TLD;
  each_lane(ex, [&](unsigned l) {
    uint32_t comp[4] = {};
    for (unsigned i = 0; i < total; ++i) comp[i] = i < in0 ? rd(r0 + i, l) : rd(r1 + (i - in0), l);
    exec::TexFetch f;
    // ptxas fetches a 1D texture as a 2D one at y = 0, which is what the
    // hardware makes of it (a 2D texture of height 1); the sampler's 1D
    // fetch is that already.
    f.dims = dims == 2 && !layered && d.height == 0 && d.depth == 0 ? 1 : dims;
    f.layered = layered;
    f.cube = cube;
    unsigned k = 0;
    if (layered) f.layer = comp[k++];
    for (unsigned i = 0; i < dims; ++i) f.coord[i] = comp[k++];
    f.float_coords = !int_coords;
    // The result's type is the texture's (its format and read mode), not
    // the instruction's: SASS has one TEX for every PTX destination type.
    f.float_result = d.kind == exec::ChannelKind::Float || d.read_as_normalized_float;
    if (has_lod) {
      f.explicit_lod = true;
      f.lod = int_coords ? static_cast<double>(static_cast<int32_t>(comp[k])) : static_cast<double>(std::bit_cast<float>(comp[k]));
    }
    if (ins.op == Op::TLD4) f.gather = static_cast<int>(ins.f[2]);
    uint32_t out[4];
    exec::texture_fetch(mem_, d, f, out);
    unsigned j = 0;
    for (unsigned c = 0; c < 4; ++c) {
      if (!(mask >> c & 1)) continue;
      wr(j < 2 ? d0 + j : d1 + (j - 2), l, out[c]);
      ++j;
    }
  });
  blk.st.global_loads += n;
  blk.st.global_bytes_read += n * 16;
}

void Runner::exec_mma(Block&, Warp& w, const Instr& ins, Mask ex) {
  if (ins.op == Op::MOVM) {
    // MOVM.16.MT88: transpose an 8x8 matrix of 16-bit elements across the
    // warp (movmatrix): lane 4g+t holds row g, columns 2t, 2t+1.
    uint16_t m[8][8];
    for (unsigned l = 0; l < 32; ++l) {
      const uint32_t v = read32(w, ins.src[0], l);
      m[l / 4][2 * (l % 4)] = static_cast<uint16_t>(v);
      m[l / 4][2 * (l % 4) + 1] = static_cast<uint16_t>(v >> 16);
    }
    each_lane(ex, [&](unsigned l) {
      const unsigned g = l / 4, t = l % 4;
      write32(w, ins.dst[0], l, m[2 * t][g] | (static_cast<uint32_t>(m[2 * t + 1][g]) << 16));
    });
    return;
  }
  unsigned M = 16, K = 8;
  MmaT at = MmaT::F16, bt = MmaT::F16;
  enum { AccF32, AccF16, AccS32, AccF64 } acc = AccF32;
  bool sat = false, b1_and = false;
  unsigned rnd = 0;
  switch (ins.op) {
    case Op::HMMA: {
      if (ins.f[3]) fault(w, ins, 0, "sparse HMMA is not implemented yet");
      static const unsigned ks[] = {8, 16, 4, 0};
      K = ks[ins.f[0]];
      at = bt = ins.f[2] == 1 ? MmaT::BF16 : ins.f[2] == 2 ? MmaT::TF32 : MmaT::F16;
      if (at == MmaT::TF32 && ins.f[0] == 0) K = 8;
      acc = ins.f[1] ? AccF32 : AccF16;
      break;
    }
    case Op::IMMA: {
      if (ins.f[3]) fault(w, ins, 0, "sparse IMMA is not implemented yet");
      const unsigned shape = ins.f[0];
      M = shape < 4 ? 8 : 16;
      const bool a4 = ins.f[4], b4 = ins.f[5];
      at = a4 ? (ins.f[1] ? MmaT::S4 : MmaT::U4) : (ins.f[1] ? MmaT::S8 : MmaT::U8);
      bt = b4 ? (ins.f[2] ? MmaT::S4 : MmaT::U4) : (ins.f[2] ? MmaT::S8 : MmaT::U8);
      static const unsigned ks[] = {16, 0, 32, 0, 16, 32, 64, 128};
      K = ks[shape];
      acc = AccS32;
      sat = ins.f[6];
      break;
    }
    case Op::QMMA:   // m16n8k32, FP8 A and B, products and sums in f32
      K = 32;
      at = ins.f[1] ? MmaT::E5M2 : MmaT::E4M3;
      bt = ins.f[2] ? MmaT::E5M2 : MmaT::E4M3;
      acc = ins.f[0] ? AccF32 : AccF16;
      break;
    case Op::DMMA:
      M = 8;
      K = 4;
      at = bt = MmaT::F64;
      acc = AccF64;
      rnd = ins.f[0];
      break;
    case Op::BMMA: {
      static const unsigned ms[] = {8, 16, 16, 0}, ks[] = {128, 128, 256, 0};
      M = ms[ins.f[0]];
      K = ks[ins.f[0]];
      at = bt = MmaT::B1;
      acc = AccS32;
      b1_and = ins.f[1];
      break;
    }
    default: break;
  }
  constexpr unsigned kN = 8;
  const unsigned bits = mma_bits(at);
  const bool wide = at == MmaT::F64;
  const unsigned per_reg = wide ? 1 : 32 / bits;
  const unsigned a_regs = M * K / (32 * per_reg), b_regs = K * kN / (32 * per_reg);
  const unsigned c_elems = M * kN / 32;
  const unsigned rw = wide ? 2 : 1;   // registers per element register
  const unsigned ra = ins.src[0].reg, rb = ins.src[1].reg, rc = ins.src[2].reg, rd = ins.dst[0].reg;
  const auto reg = [&](unsigned base, unsigned i, unsigned l) -> uint64_t {
    if (base == kRZ) return 0;
    const uint64_t lo = w.reg(base + i * rw, l);
    return wide ? lo | (static_cast<uint64_t>(w.reg(base + i * rw + 1, l)) << 32) : lo;
  };
  std::vector<double> A(M * K, 0.0), B(K * kN, 0.0);
  double C[16 * kN] = {};
  for (unsigned r = 0; r < a_regs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, t = l % 4, row = g + (r % 2) * 8, col0 = t * per_reg + (r / 2) * (per_reg * 4);
      const uint64_t v = reg(ra, r, l);
      for (unsigned e = 0; e < per_reg; ++e) A[row * K + col0 + e] = mma_elem(at, v, e);
    }
  for (unsigned r = 0; r < b_regs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, t = l % 4, row0 = t * per_reg + r * (per_reg * 4);
      const uint64_t v = reg(rb, r, l);
      for (unsigned e = 0; e < per_reg; ++e) B[(row0 + e) * kN + g] = mma_elem(bt, v, e);
    }
  const auto cd = [&](unsigned l, unsigned slot) { return (l / 4 + (slot / 2) * 8) * kN + (l % 4) * 2 + slot % 2; };
  for (unsigned l = 0; l < 32; ++l)
    for (unsigned s = 0; s < c_elems; ++s) {
      double v;
      if (acc == AccF16) {
        const uint32_t word = rc == kRZ ? 0 : w.reg(rc + s / 2, l);
        v = half_to_f32(static_cast<uint16_t>(word >> (16 * (s % 2))));
      } else if (acc == AccF64) {
        v = as_f64(rc == kRZ ? 0 : (w.reg(rc + 2 * s, l) | (static_cast<uint64_t>(w.reg(rc + 2 * s + 1, l)) << 32)));
      } else if (acc == AccS32) {
        v = static_cast<int32_t>(rc == kRZ ? 0 : w.reg(rc + s, l));
      } else {
        v = as_f32(rc == kRZ ? 0 : w.reg(rc + s, l));
      }
      C[cd(l, s)] = v;
    }
  double D[16 * kN] = {};
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < kN; ++j) {
      if (at == MmaT::B1) {
        int64_t a = static_cast<int64_t>(C[i * kN + j]);
        for (unsigned k = 0; k < K; ++k) {
          const bool x = A[i * K + k] != 0, y = B[k * kN + j] != 0;
          a += b1_and ? (x && y) : (x != y);
        }
        D[i * kN + j] = static_cast<int32_t>(a);
      } else if (acc == AccS32) {
        const unsigned block = sat ? 128 / bits : K;
        int64_t a = static_cast<int64_t>(C[i * kN + j]);
        for (unsigned k0 = 0; k0 < K; k0 += block) {
          for (unsigned k = k0; k < k0 + block && k < K; ++k)
            a += static_cast<int64_t>(A[i * K + k]) * static_cast<int64_t>(B[k * kN + j]);
          if (sat) a = std::clamp<int64_t>(a, INT32_MIN, INT32_MAX);
        }
        D[i * kN + j] = static_cast<int32_t>(a);
      } else if (acc == AccF64) {
        D[i * kN + j] = rounded(rnd, [&] {
          double a = C[i * kN + j];
          for (unsigned k = 0; k < K; ++k) a = std::fma(A[i * K + k], B[k * kN + j], a);
          return a;
        });
      } else {
        float a = static_cast<float>(C[i * kN + j]);
        for (unsigned k = 0; k < K; ++k) {
          const float prod = static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * kN + j]);
          a = a + prod;
        }
        D[i * kN + j] = a;
      }
    }
  each_lane(ex, [&](unsigned l) {
    if (rd == kRZ) return;
    for (unsigned s = 0; s < c_elems; ++s) {
      const double v = D[cd(l, s)];
      if (acc == AccF16) {
        uint32_t& word = w.reg(rd + s / 2, l);
        const uint16_t h = f64_to_half(v);
        word = (s % 2) ? (word & 0xffff) | (static_cast<uint32_t>(h) << 16) : (word & 0xffff0000u) | h;
      } else if (acc == AccF64) {
        const uint64_t b = f64_bits(v);
        w.reg(rd + 2 * s, l) = static_cast<uint32_t>(b);
        w.reg(rd + 2 * s + 1, l) = static_cast<uint32_t>(b >> 32);
      } else if (acc == AccS32) {
        w.reg(rd + s, l) = static_cast<uint32_t>(static_cast<int32_t>(v));
      } else {
        w.reg(rd + s, l) = f32_bits(static_cast<float>(v));
      }
    }
  });
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
    case Op::BMOV:   // a convergence barrier's thread mask, to or from a register
      if (ins.f[0]) {
        const unsigned bar = ins.src[0].reg & 15;
        const Mask m = w.b[bar];
        each_lane(ex, [&](unsigned l) { write32(w, ins.dst[0], l, static_cast<uint32_t>(m)); });
        if (ins.f[1] && ex) w.b[bar] = 0;   // .CLEAR
      } else if (ex) {
        const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
        w.b[ins.dst[0].reg & 15] = static_cast<Mask>(read32(w, ins.src[0], l));
      }
      each_lane(group, [&](unsigned l) { w.pc[l] += 16; });
      return true;
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
        if ((ex >> l) & 1)
          w.pc[l] = w.pc[l] + 16 + static_cast<uint64_t>(ins.src[0].imm) +
                    (ins.f[0] ? w.ur[ins.src[0].reg] : w.reg(ins.src[0].reg, l));
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
      fault(w, ins, 0, std::string("SASS: ") + op_name(ins.op) + " is not implemented yet", Err::UnsupportedPtx);
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
