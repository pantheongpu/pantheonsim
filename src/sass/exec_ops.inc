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
// Whether the executor runs an instruction: what a cubin must hold for the
// runtime to prefer it over the fatbin's PTX (sass::unsupported).
// SR_PM0-SR_PM7 (0x64, 0x66, ... 0x72, each with its high half after it),
// which ptxas reads for %pm0-%pm7 and %pm0_64-%pm7_64.
bool reads_pm_counter(const Instr& ins) {
  return (ins.op == Op::S2R || ins.op == Op::S2UR || ins.op == Op::CS2R) && ins.f[0] >= 0x64 &&
         ins.f[0] <= 0x73;
}

bool executes(const Instr& ins) {
  // The performance-monitor counters are refused, as the PTX path refuses
  // their registers by name: so a fatbin falls back to its PTX, which says so.
  if (reads_pm_counter(ins)) return false;
  switch (ins.op) {
    case Op::HMMA: case Op::IMMA:
      return true;
    case Op::F2FP:
      // A narrow form (merging into C, or unpacking a narrow source) whose
      // format the decoder did not recognise would run as some other one.
      if (ins.f[5] == 1 && ins.f[3] < 10) return false;
      if (ins.f[5] == 4 && ins.f[4] != 0 && ins.f[4] < 10) return false;
      return true;
    case Op::LDSM: {
      // ldmatrix's forms: 16-bit 8x8, either way; packed 4- or 6-bit 8x16;
      // 8-bit or packed 16x16, transposed. Anything else is not one.
      const unsigned fmt = ins.f[2], shape = ins.f[3];
      if (shape == 0 || shape == 2) return fmt == 0;
      if (shape == 5) return fmt == 1 || fmt == 2;
      if (shape == 9) return fmt != 0;
      return false;
    }
    case Op::TEX: case Op::TLD: case Op::TLD4: {
      const Word& w = ins.w;
      const bool bw = ins.sm >= 100;   // Blackwell moves the LOD's low bit to 59 and the offsets to 56-57
      const unsigned lod = ins.op == Op::TLD4 ? 0
                           : bw               ? static_cast<unsigned>(w.field(59, 1) | (w.field(87, 2) << 1))
                                              : static_cast<unsigned>(w.field(87, 3));
      if (lod == 4 || lod == 5 || (ins.op == Op::TEX && lod == 2)) return false;   // LOD clamps, bias
      const bool offsets = bw ? w.field(56, 2) != 0 : (w.bit(76) || (ins.op == Op::TLD4 && w.bit(77)));
      return !offsets && !w.bit(78);   // offsets, depth compare
    }
    case Op::BMSK: case Op::BREV: case Op::CS2R: case Op::FLO: case Op::IABS: case Op::IADD3: case Op::IDP:
    case Op::IMAD: case Op::IMNMX: case Op::ISETP: case Op::LEA: case Op::LOP3: case Op::MOV: case Op::P2R:
    case Op::PLOP3: case Op::POPC: case Op::PRMT: case Op::R2P: case Op::S2R: case Op::S2UR:
    case Op::SEL: case Op::SGXT: case Op::SHF: case Op::UFLO: case Op::UIADD3: case Op::UIMAD: case Op::UISETP:
    case Op::ULDC: case Op::ULEA: case Op::ULOP3: case Op::UMOV: case Op::UPLOP3: case Op::UPOPC: case Op::UPRMT:
    case Op::USEL: case Op::USHF: case Op::VABSDIFF4: case Op::VIADD: case Op::VIMNMX: case Op::VIADDMNMX:
    case Op::VIMNMX3: case Op::IADD: case Op::UP2UR: case Op::ELECT: case Op::USETMAXREG: case Op::WARPGROUP:
    case Op::FMNMX3: case Op::STSM: case Op::RPCMOV: case Op::HGMMA: case Op::IGMMA: case Op::BGMMA: case Op::QGMMA:
      return true;
    case Op::UTMALDG:
      return !ins.f[6];   // not im2col::w or ::w::128, which the PTX engine does not run either
    case Op::SYNCS: case Op::ARRIVES: case Op::UTMASTG: case Op::UTMAREDG: case Op::UBLKCP:
    case Op::UBLKRED: case Op::LDTM: case Op::STTM: case Op::UTCMMA: case Op::UTCBAR: case Op::UTCCP:
    case Op::UTCSHIFT: case Op::UTCATOMSWS: case Op::STAS: case Op::REDAS: case Op::UGETNEXTWORKID:
    case Op::DADD: case Op::DFMA: case Op::DMUL: case Op::DSETP: case Op::F2F: case Op::F2I:
    case Op::FADD: case Op::FCHK: case Op::FFMA: case Op::FMNMX: case Op::FMUL: case Op::FRND: case Op::FSEL:
    case Op::FSET: case Op::FSETP: case Op::HADD2: case Op::HFMA2: case Op::HMUL2: case Op::HSET2: case Op::HSETP2:
    case Op::I2F: case Op::I2FP: case Op::I2I: case Op::I2IP: case Op::MUFU:
    case Op::ATOM: case Op::ATOMG: case Op::ATOMS: case Op::LD: case Op::LDC: case Op::LDG: case Op::LDGDEPBAR:
    case Op::LDGSTS: case Op::LDL: case Op::LDS: case Op::RED: case Op::ST: case Op::STG:
    case Op::STL: case Op::STS:
    case Op::MATCH: case Op::REDUX: case Op::SHFL: case Op::VOTE: case Op::VOTEU: case Op::MOVM:
      return true;
    case Op::BMMA: case Op::DMMA: case Op::QMMA: case Op::SULD: case Op::SUST:
    case Op::BAR: case Op::BMOV: case Op::BPT: case Op::BRA: case Op::BREAK: case Op::BRX: case Op::BSSY:
    case Op::BSYNC: case Op::CALL: case Op::CCTL: case Op::DEPBAR: case Op::ERRBAR: case Op::EXIT: case Op::MEMBAR:
    case Op::NANOSLEEP: case Op::NOP: case Op::RET: case Op::YIELD:
      return true;
    case Op::WARPSYNC: case Op::ENDCOLLECTIVE: case Op::UCGABAR:
      return true;
    case Op::R2UR:
      return true;
    default:
      return false;
  }
}

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
    case Op::HMMA: case Op::IMMA: case Op::BMMA: case Op::DMMA: case Op::QMMA: case Op::MOVM:
    case Op::HGMMA: case Op::IGMMA: case Op::BGMMA: case Op::QGMMA:
      return C::Tensor;
    case Op::IMAD: case Op::IADD3: case Op::LOP3: case Op::SHF: case Op::LEA: case Op::IMNMX: case Op::IABS:
    case Op::VIADD: case Op::VIMNMX: case Op::VIADDMNMX: case Op::VIMNMX3:
    case Op::POPC: case Op::FLO: case Op::BREV: case Op::BMSK: case Op::SGXT: case Op::IDP: case Op::VABSDIFF4: case Op::ISETP:
    case Op::UIMAD: case Op::UIADD3: case Op::ULOP3: case Op::USHF: case Op::ULEA: case Op::UISETP:
    case Op::UPOPC: case Op::UFLO:
      return C::Integer;
    default: return C::Misc;
  }
}

void Runner::execute(Block& blk, Warp& w, const Instr& ins, Mask group, bool advance) {
  const Mask ex = guard_mask(w, ins, group);
  {
    exec::LaunchStats& st = blk.st;
    const uint64_t lanes = static_cast<uint64_t>(std::popcount(ex));
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[static_cast<size_t>(inst_class(ins.op))] += lanes;
    if (ins.op == Op::HMMA || ins.op == Op::IMMA || ins.op == Op::BMMA || ins.op == Op::DMMA || ins.op == Op::QMMA ||
        ins.op == Op::HGMMA || ins.op == Op::IGMMA || ins.op == Op::BGMMA || ins.op == Op::QGMMA)
      ++st.tensor_instructions;
    if (ins.op == Op::BAR && ins.mnemonic == "BAR" && ex) ++st.barriers;
    if (ins.op == Op::BRA && ex && ex != group) ++st.divergent_branches;
  }
  bool pcs_set = false;
  try {
    if (!ex && straight(ins)) {
      // predicated off on every lane: an ALU or memory instruction does nothing
    } else if (fast_alu(w, ins, ex)) {
      // done: the fast path computed exactly what the general one does
    } else switch (ins.op) {
      case Op::Unknown:
        fault(w, ins, static_cast<unsigned>(std::countr_zero(group)), "instruction not decodable: " + ins.mnemonic);
      case Op::BRA: case Op::BRX: case Op::JMP: case Op::CALL: case Op::RET: case Op::EXIT:
      case Op::BSSY: case Op::BSYNC: case Op::BREAK: case Op::WARPSYNC: case Op::BAR:
      case Op::NOP: case Op::YIELD: case Op::NANOSLEEP: case Op::BPT: case Op::DEPBAR:
      case Op::MEMBAR: case Op::ERRBAR: case Op::CCTL: case Op::BMOV: case Op::ENDCOLLECTIVE: case Op::UCGABAR:
        pcs_set = exec_control(blk, w, ins, ex, group);
        break;
      case Op::LDG: case Op::STG: case Op::LDS: case Op::STS: case Op::LDL: case Op::STL:
      case Op::LD: case Op::ST: case Op::LDC: case Op::ATOM: case Op::ATOMG: case Op::ATOMS:
      case Op::RED: case Op::LDGSTS: case Op::LDGDEPBAR: case Op::LDSM: case Op::STSM:
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
      case Op::HGMMA: case Op::IGMMA: case Op::BGMMA: case Op::QGMMA:
        exec_gmma(blk, w, ins, ex);
        break;
      case Op::SYNCS: case Op::ARRIVES:
        exec_syncs(blk, w, ins, ex);
        break;
      case Op::UTMALDG: case Op::UTMASTG: case Op::UTMAREDG: case Op::UBLKCP: case Op::UBLKRED:
        exec_tma(blk, w, ins, ex);
        break;
      case Op::LDTM: case Op::STTM: case Op::UTCMMA: case Op::UTCBAR: case Op::UTCCP: case Op::UTCSHIFT:
      case Op::UTCATOMSWS:
        exec_tcgen05(blk, w, ins, ex);
        break;
      case Op::STAS: case Op::REDAS:
        exec_async_store(blk, w, ins, ex);
        break;
      case Op::UGETNEXTWORKID:
        exec_clc(blk, w, ins, ex);
        break;
      case Op::FADD: case Op::FMUL: case Op::FFMA: case Op::FMNMX: case Op::FSETP: case Op::FSET:
      case Op::FSEL: case Op::FCHK: case Op::MUFU: case Op::DADD: case Op::DMUL: case Op::DFMA:
      case Op::DSETP: case Op::HADD2: case Op::HMUL2: case Op::HFMA2: case Op::HSETP2: case Op::HMNMX2:
      case Op::I2F: case Op::I2FP: case Op::F2I: case Op::F2IP: case Op::F2F: case Op::F2FP:
      case Op::FRND: case Op::I2I: case Op::I2IP: case Op::HSET2: case Op::FMNMX3:
        exec_float(blk, w, ins, ex);
        break;
      default:
        exec_int(blk, w, ins, ex);
        break;
    }
  } catch (const Fault& f) {
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what());
  } catch (const std::out_of_range& f) {   // a register past the warp's file
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what(), Err::Internal);
  } catch (const Error& e) {
    // An error from the device (an address out of bounds, a trap), kept as
    // what it is with the instruction's place added (once).
    if (e.message().rfind("SASS kernel ", 0) == 0) throw;
    fault(w, ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), e.message(), e.code());
  }
  if (!pcs_set && advance) each_lane(group, [&](unsigned l) { w.pc[l] += 16; });
  // VGPU_SASS_TRACE=<warp>: every instruction that warp of block (0,0,0)
  // runs, with the lane group and the first active lane's results;
  // VGPU_SASS_TRACE_KERNEL=<text> keeps it to kernels whose name holds it.
  static const int trace = [] { const char* v = std::getenv("VGPU_SASS_TRACE"); return v && *v ? std::atoi(v) : -1; }();
  static const char* const trace_kernel = std::getenv("VGPU_SASS_TRACE_KERNEL");
  if (trace >= 0 && static_cast<int>(w.index) == trace && blk.ctaid[0] == 0 && blk.ctaid[1] == 0 && blk.ctaid[2] == 0 &&
      (!trace_kernel || !*trace_kernel || k_.name.find(trace_kernel) != std::string::npos)) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(ex ? ex : group));
    const unsigned urz = m_.sm >= 100 ? 255u : 63u;
    std::string vals;
    for (const Operand& d : ins.dst)
      for (unsigned i = 0; i < std::max(1u, static_cast<unsigned>(d.width)); ++i) {
        char b[48];
        const unsigned n = d.reg + i;
        if (d.kind == Kind::Reg && d.reg != kRZ && n < w.nregs) std::snprintf(b, sizeof b, " R%u=%08x", n, w.reg(n, l));
        else if (d.kind == Kind::UReg && n < urz) std::snprintf(b, sizeof b, " UR%u=%08x", n, w.ur[n]);
        else if (d.kind == Kind::Pred && d.reg != kPT && i == 0) std::snprintf(b, sizeof b, " P%u=%d", d.reg, (w.p[d.reg] >> l) & 1);
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
template <class T>
bool int_cmp(unsigned cmp, T a, T b) {
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
      if (ins.f[1]) {   // MOV.64
        each_lane(ex, [&](unsigned l) { write64(w, ins.dst[0], l, read64(w, ins.src[0], l)); });
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
          const unsigned __int128 full = static_cast<unsigned __int128>(prod) + c + cin;
          write64(w, ins.dst[0], l, static_cast<uint64_t>(full));
          // The carry out of the 64-bit sum, which the next IMAD.WIDE.X of a
          // 64-bit multiply (ptxas's division by a constant) adds in.
          if (ins.dst.size() > 1) set_pred(w, ins.dst[1], l, (full >> 64) != 0);
        } else if (kind == 0x27) {
          // HI: the high word of the product plus a 64-bit addend -- c is a
          // register pair (ptxas's division by a constant puts n in the high
          // half to add it there).
          const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a)) *
                                                            static_cast<int32_t>(b))
                                    : static_cast<uint64_t>(a) * b;
          uint64_t c = ins.src[2].kind == Kind::Reg || ins.src[2].kind == Kind::UReg
                           ? read64(w, ins.src[2], l)
                           : static_cast<uint64_t>(sgn ? static_cast<int64_t>(static_cast<int32_t>(ival(w, ins.src[2], l)))
                                                       : ival(w, ins.src[2], l));
          if (ins.src[2].neg && (ins.src[2].kind == Kind::Reg || ins.src[2].kind == Kind::UReg)) c = 0 - c;
          // .X's carry comes in at bit 0 of the 64-bit sum, not into the
          // high word: hi(a*b + c + carry) (mad.hi.u64's last step is
          // IMAD.HI.U32.X, which was one too high when the low half did not
          // carry).
          const unsigned __int128 sum = static_cast<unsigned __int128>(prod) + c + cin;
          write32(w, ins.dst[0], l, static_cast<uint32_t>(sum >> 32));
          if (ins.dst.size() > 1) set_pred(w, ins.dst[1], l, (sum >> 64) != 0);
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
      if (ins.f[1]) {   // UIADD3.64
        each_lane(ex, [&](unsigned l) {
          uint64_t sum = 0;
          for (int i = 0; i < 3; ++i) {
            uint64_t v = read64(w, ins.src[i], l);
            if (ins.src[i].neg) v = 0 - v;
            if (ins.src[i].bnot) v = ~v;
            sum += v;
          }
          write64(w, ins.dst[0], l, sum);
        });
        return;
      }
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
        // The two carry-outs count the overflow (0 to 2) between them, each
        // worth one to the IADD3.X that adds both: a - b - c overflows twice.
        const uint64_t over = sum >> 32;
        if (ins.dst.size() > 1) set_pred(w, ins.dst[1], l, over >= 1);
        if (ins.dst.size() > 2) set_pred(w, ins.dst[2], l, over >= 2);
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
      if (ins.f[4]) {   // sm_120's 64-bit compare
        each_lane(ex, [&](unsigned l) {
          const uint64_t a = read64(w, ins.src[0], l), b = read64(w, ins.src[1], l);
          const bool acc = pred(w, ins.src[2], l);
          const bool r = sgn ? int_cmp(cmp, static_cast<int64_t>(a), static_cast<int64_t>(b)) : int_cmp(cmp, a, b);
          set_pred(w, ins.dst[0], l, bool_op(bop, r, acc));
          set_pred(w, ins.dst[1], l, bool_op(bop, !r, acc));
        });
        return;
      }
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
        uint64_t sum = static_cast<uint64_t>(shifted) + b + cin;
        // LEA Rd, P, -Ra, Rb: the adder takes ~(Ra << s) + 1, so negating 0
        // carries out -- the carry ptxas feeds a 64-bit negate's high word.
        if (!hi && ins.src[0].neg)
          sum = static_cast<uint64_t>(static_cast<uint32_t>(~(s >= 32 ? 0 : a << s))) + 1 + b + cin;
        // LEA Rd, P, Ra, -Rb: likewise the adder takes ~Rb + 1, so a zero Rb carries out. ival() has already
        // negated b (0 stays 0), which loses that carry, and the carry feeds the next LEA.HI.X / IADD3.X of a
        // 64-bit index (im2col's `h_im * width - pad`); with pad 0 the high word came out 0xffffffff.
        if (ins.src[1].neg && b == 0) sum += uint64_t{1} << 32;
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
        const bool p = pred(w, ins.src[2], l);
        if (ins.f[0]) write64(w, ins.dst[0], l, p ? read64(w, ins.src[0], l) : read64(w, ins.src[1], l));   // SEL.64
        else write32(w, ins.dst[0], l, p ? read32(w, ins.src[0], l) : read32(w, ins.src[1], l));
      });
      return;
    case Op::IMNMX:
      if (ins.f[1]) {   // sm_120's 64-bit form: preds out, the register third
        each_lane(ex, [&](unsigned l) {
          const uint64_t a = read64(w, ins.src[0], l), b = read64(w, ins.src[1], l);
          const bool mn = pred(w, ins.src[2], l);
          uint64_t r;
          if (ins.f[0]) {
            const int64_t x = static_cast<int64_t>(a), y = static_cast<int64_t>(b);
            r = static_cast<uint64_t>(mn ? std::min(x, y) : std::max(x, y));
          } else {
            r = mn ? std::min(a, b) : std::max(a, b);
          }
          write64(w, ins.dst[2], l, r);
        });
        return;
      }
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
    case Op::VIADD:
      each_lane(ex, [&](unsigned l) {
        const uint32_t a = read32(w, ins.src[0], l), b = ival(w, ins.src[1], l);
        uint32_t r;
        if (ins.f[0] == 1 && ins.f[1]) {   // .S32.ISAT: saturating
          r = static_cast<uint32_t>(std::clamp<int64_t>(int64_t{static_cast<int32_t>(a)} + static_cast<int32_t>(b), INT32_MIN, INT32_MAX));
        } else if (ins.f[0] == 2) {   // per byte, wrapping; a negated b is negated byte by byte
          const uint32_t braw = read32(w, ins.src[1], l);
          r = 0;
          for (int k = 0; k < 4; ++k) {
            const uint32_t bk = (braw >> (8 * k)) & 0xff;
            r |= ((((a >> (8 * k)) & 0xff) + (ins.src[1].neg ? 0u - bk : bk)) & 0xff) << (8 * k);
          }
        } else {
          r = a + b;
        }
        write32(w, ins.dst[0], l, r);
      });
      return;
    case Op::IADD:   // sm_120: a + b, 32 or 64 bits
      each_lane(ex, [&](unsigned l) {
        if (ins.f[0]) {
          uint64_t a = read64(w, ins.src[0], l), b = read64(w, ins.src[1], l);
          if (ins.src[0].neg) a = 0 - a;
          if (ins.src[1].neg) b = 0 - b;
          write64(w, ins.dst[0], l, a + b);
        } else {
          write32(w, ins.dst[0], l, ival(w, ins.src[0], l) + ival(w, ins.src[1], l));
        }
      });
      return;
    case Op::UP2UR:   // the uniform predicates (masked) into a register
      if (ex) {
        uint32_t bits = 0;
        for (unsigned i = 0; i < 7; ++i) bits |= (w.up[i] ? 1u : 0u) << i;
        const uint32_t mask = static_cast<uint32_t>(ins.src[2].imm);
        write32(w, ins.dst[0], 0, (read32(w, ins.src[1], 0) & ~mask) | (bits & mask));
      }
      return;
    case Op::VIMNMX: case Op::VIADDMNMX: case Op::VIMNMX3: {
      const bool sgn = ins.f[0], s16x2 = ins.f[1];
      const unsigned n = ins.op == Op::VIMNMX ? 2 : 3;
      const auto pick = [&](uint32_t x, uint32_t y, bool mn) -> uint32_t {
        if (sgn) {
          const int32_t a = static_cast<int32_t>(x), b = static_cast<int32_t>(y);
          return static_cast<uint32_t>(mn ? std::min(a, b) : std::max(a, b));
        }
        return mn ? std::min(x, y) : std::max(x, y);
      };
      each_lane(ex, [&](unsigned l) {
        const bool mn = pred(w, ins.src[n], l);
        uint32_t a = read32(w, ins.src[0], l);
        const uint32_t b = ival(w, ins.src[1], l);
        const auto relu = [&](uint32_t r) { return ins.f[2] && static_cast<int32_t>(r) < 0 ? 0u : r; };
        if (ins.op == Op::VIADDMNMX) {
          write32(w, ins.dst[0], l, relu(pick(a + b, read32(w, ins.src[2], l), mn)));
        } else if (!s16x2) {
          const uint32_t r = pick(a, b, mn);
          write32(w, ins.dst[0], l, relu(ins.op == Op::VIMNMX ? r : pick(r, read32(w, ins.src[2], l), mn)));
        } else {   // per 16-bit half or per byte, signed or not
          const uint32_t c = ins.op == Op::VIMNMX3 ? read32(w, ins.src[2], l) : b;
          const unsigned bits = ins.f[1] == 2 ? 8 : 16, lanes = 32 / bits;
          const uint32_t mask = (1u << bits) - 1;
          uint32_t r = 0;
          for (unsigned h = 0; h < lanes; ++h) {
            const auto part = [&](uint32_t v) {
              const uint32_t u = (v >> (bits * h)) & mask;
              return sgn ? static_cast<int32_t>(u << (32 - bits)) >> (32 - bits) : static_cast<int32_t>(u);
            };
            const int32_t m = mn ? std::min({part(a), part(b), part(c)}) : std::max({part(a), part(b), part(c)});
            r |= (static_cast<uint32_t>(m) & mask) << (bits * h);
          }
          write32(w, ins.dst[0], l, relu(r));
        }
      });
      return;
    }
    case Op::ELECT: {   // elect.sync: the lowest thread with the predicate leads
      Mask cand = 0;
      each_lane(ex, [&](unsigned l) {
        if (pred(w, ins.src[0], l)) cand |= Mask{1} << l;
      });
      const unsigned leader = cand ? static_cast<unsigned>(std::countr_zero(cand)) : 32;
      each_lane(ex, [&](unsigned l) { set_pred(w, ins.dst[0], l, l == leader); });
      if (ex && leader < 32) write32(w, ins.dst[1], static_cast<unsigned>(std::countr_zero(ex)), leader);
      return;
    }
    case Op::USETMAXREG: {
      // setmaxnreg: a warpgroup gives registers back (.DEALLOC) or takes more
      // (.TRY_ALLOC), past the count the kernel was compiled for -- a
      // warp-specialized GEMM's consumers go from 168 to 232 while its
      // producer drops to 40. The allocation always succeeds; the warp's file
      // grows to the count, keeping what it holds, and a release leaves it be.
      const unsigned n = static_cast<unsigned>(ins.src[0].imm);
      if (ins.mods[0] == "TRY_ALLOC" && n > w.nregs) {
        w.nregs = std::min(n, 256u);
        w.r.resize(static_cast<size_t>(w.nregs) * 32, 0);
      }
      if (!ins.dst.empty() && ex) set_pred(w, ins.dst[0], static_cast<unsigned>(std::countr_zero(ex)), true);
      return;
    }
    case Op::WARPGROUP:   // the warpgroup MMAs complete as they issue: nothing to fence or wait for
      return;
    case Op::RPCMOV:   // RPCMOV.32 to or from the return-address register
      each_lane(ex, [&](unsigned l) {
        const unsigned sh = ins.f[1] ? 32 : 0;
        if (ins.f[0]) write32(w, ins.dst[0], l, static_cast<uint32_t>(w.rpc[l] >> sh));
        else w.rpc[l] = (w.rpc[l] & ~(uint64_t{0xffffffff} << sh)) | (static_cast<uint64_t>(read32(w, ins.src[0], l)) << sh);
      });
      return;
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
        const auto in = [&](const Operand& o) { return ins.f[2] ? (read32(w, o, l) >> 31) != 0 : pred(w, o, l); };
        const bool a = in(ins.src[0]), b = in(ins.src[1]), c = in(ins.src[2]);
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
      if (reads_pm_counter(ins))
        throw Error(Err::UnsupportedPtx, "SASS kernel " + k_.name + " reads " + ins.src[0].text +
                                             ", a performance-monitor counter; VirtualGPU has no timing "
                                             "model, so it has nothing to count");
      each_lane(ex, [&](unsigned l) {
        const unsigned idx = ins.f[0];
        if (ins.op == Op::CS2R) {
          const uint32_t v = idx == 0xff ? 0 : sreg(blk, w, idx, l);
          if (ins.dst[0].width == 2) write64(w, ins.dst[0], l, idx == 0xff ? 0 : sreg(blk, w, idx, l) | (static_cast<uint64_t>(sreg(blk, w, idx + 1, l)) << 32));
          else write32(w, ins.dst[0], l, v);
        } else if (ins.mnemonic == "CS2UR" && ins.f[3]) {   // a 64-bit special register
          write64(w, ins.dst[0], l, sreg(blk, w, idx, l) | (static_cast<uint64_t>(sreg(blk, w, idx + 1, l)) << 32));
        } else {
          write32(w, ins.dst[0], l, sreg(blk, w, idx, l));
        }
      });
      return;
    case Op::R2UR:
      // R2UR [Pd,] URd, Ra: the value of the first active lane into a
      // uniform register, and in each active lane's Pd whether its own value
      // differs from it (.OR: or'd into Pd). ptxas's "waterfall" for a
      // uniform-only instruction fed per-lane values (a TMA store's
      // coordinates): R2UR, then R2UR.OR for each further value, runs the
      // instruction once, and goes round again while any lane's Pd is set --
      // which ends only if Pd marks the lanes still to do.
      if (ex) {
        const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
        const uint32_t v = read32(w, ins.src[0], l);
        write32(w, ins.dst.back(), l, v);
        if (ins.f[0]) {
          const bool accumulate = ins.w.bit(84);
          each_lane(ex, [&](unsigned k) {
            const bool differs = read32(w, ins.src[0], k) != v;
            set_pred(w, ins.dst[0], k, differs || (accumulate && pred(w, ins.dst[0], k)));
          });
        }
      }
      return;
    case Op::ULDC: {
      if (!ex) return;
      const unsigned size = ins.f[0];
      const Operand& c = ins.src[0];
      if (ins.f[1]) {   // c[bank][URa + offset] (ULDC's indexed form; LDCU)
        const uint64_t o = (c.reg == static_cast<unsigned>(m_.sm >= 100 ? 255 : 63) ? 0 : w.ur[c.reg]) +
                           static_cast<uint64_t>(c.imm);
        const unsigned words = size == 6 ? 4 : size == 5 ? 2 : 1;
        if (words > 1) {
          Operand d = ins.dst[0];
          for (unsigned i = 0; i < words; ++i, ++d.reg) write32(w, d, 0, cbank32(ins.f[2], o + 4 * i));
          return;
        }
        uint32_t v = cbank32(ins.f[2], o & ~3ull) >> (8 * (o & 3));
        switch (size) {
          case 0: v &= 0xff; break;
          case 1: v = static_cast<uint32_t>(static_cast<int8_t>(v)); break;
          case 2: v &= 0xffff; break;
          case 3: v = static_cast<uint32_t>(static_cast<int16_t>(v)); break;
          default: break;
        }
        write32(w, ins.dst[0], 0, v);
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

// ---- fast paths -----------------------------------------------------------------
//
// The instructions compiled kernels spend their time in -- IADD3, IMAD, LOP3,
// MOV and the f32 FFMA/FADD/FMUL -- in their plain forms: register, RZ,
// immediate or constant sources, a register destination, no carries, no
// rounding mode, no flush or saturation. Each source is resolved once per
// instruction (a register's 32 lanes, or one value for every lane) instead of
// once per lane, and the operation is decided once. Every other form returns
// false and takes the general path, which computes the same bits; the SASS
// suite runs both through every program.

bool Runner::fast_alu(Warp& w, const Instr& ins, Mask ex) {
  switch (ins.op) {
    case Op::IADD3: case Op::IMAD: case Op::LOP3: case Op::MOV: case Op::ISETP: case Op::FSETP:
    case Op::FSEL: case Op::FFMA: case Op::FADD: case Op::FMUL: case Op::LEA: case Op::SHF: break;
    default: return false;
  }
  if (ins.dst.empty()) return false;
  // Predicates as the warp holds them, one bit a lane: a source (PT, !Pn,
  // and a uniform one as all or nothing), and a destination written for the
  // executing lanes only. A uniform destination takes the general path.
  const auto pmask = [&](const Operand& o) -> Mask {
    Mask v;
    if (o.kind == Kind::UPred) v = (o.reg == kPT || w.up[o.reg]) ? kAll : 0;
    else v = o.reg == kPT ? kAll : w.p[o.reg];
    return o.neg ? ~v : v;
  };
  const auto pdst_ok = [&](const Operand& o) { return o.kind == Kind::Pred; };
  const auto setp = [&](const Operand& o, Mask v) {
    if (o.reg != kPT) w.p[o.reg] = (w.p[o.reg] & ~ex) | (v & ex);
  };
  const auto combine = [](unsigned bop, Mask r, Mask acc) -> Mask {
    return bop == 0 ? (r & acc) : bop == 1 ? (r | acc) : (r ^ acc);
  };
  if (ins.op == Op::ISETP) {
    const unsigned cmp = ins.f[0], bop = ins.f[2];
    const bool sgn = ins.f[1];
    if (ins.f[3] || ins.f[4] || ins.dst.size() < 2 || !pdst_ok(ins.dst[0]) || !pdst_ok(ins.dst[1]) || ins.src.size() < 3)
      return false;
    for (int i = 0; i < 2; ++i) {
      const Operand& o = ins.src[i];
      if (!(o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::CBank || o.kind == Kind::UReg)) return false;
    }
    Mask r = 0;
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      const uint32_t a = read32(w, ins.src[0], l), b = read32(w, ins.src[1], l);
      if (sgn ? int_cmp(cmp, static_cast<int32_t>(a), static_cast<int32_t>(b)) : int_cmp(cmp, a, b)) r |= Mask{1} << l;
    }
    const Mask acc = pmask(ins.src[2]);
    setp(ins.dst[0], combine(bop, r, acc));
    setp(ins.dst[1], combine(bop, ~r, acc));
    return true;
  }
  if (ins.op == Op::FSETP) {
    const unsigned cmp = ins.f[0], bop = ins.f[1];
    const bool fz = ins.f[2];
    if (ins.dst.size() < 2 || !pdst_ok(ins.dst[0]) || !pdst_ok(ins.dst[1]) || ins.src.size() < 3) return false;
    for (int i = 0; i < 2; ++i) {
      const Operand& o = ins.src[i];
      if (!(o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::FImm || o.kind == Kind::CBank ||
            o.kind == Kind::UReg) || o.bnot)
        return false;
    }
    const auto fv = [&](const Operand& o, unsigned l) {
      float f = as_f32(read32(w, o, l));
      if (fz) f = ftz(f);
      if (o.abs) f = std::fabs(f);
      if (o.neg) f = -f;
      return f;
    };
    Mask r = 0;
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      if (float_cmp(cmp, fv(ins.src[0], l), fv(ins.src[1], l))) r |= Mask{1} << l;
    }
    const Mask acc = pmask(ins.src[2]);
    setp(ins.dst[0], combine(bop, r, acc));
    setp(ins.dst[1], combine(bop, ~r, acc));
    return true;
  }
  if (ins.dst[0].kind != Kind::Reg && !(ins.op == Op::LOP3 && ins.dst.size() == 2)) return false;
  const auto plain = [&](const Operand& o) {
    return o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::FImm || o.kind == Kind::CBank ||
           o.kind == Kind::UReg;
  };
  // Each source as 32 lanes: the register's own, or a copy -- a value
  // broadcast, or the register with its modifier applied -- so the loops
  // below have no branches in them.
  alignas(64) uint32_t buf[3][32];
  const auto lanes = [&](unsigned i, const Operand& o, auto modify) -> const uint32_t* {
    if (o.kind == Kind::Reg && o.reg != kRZ && !modify.active) return &w.reg(o.reg, 0);
    if (o.kind == Kind::Reg) {
      const uint32_t* r = o.reg == kRZ ? nullptr : &w.reg(o.reg, 0);
      for (unsigned l = 0; l < 32; ++l) buf[i][l] = modify(r ? r[l] : 0u);
    } else {
      const uint32_t v = modify(read32(w, o, 0));   // the same for every lane
      for (unsigned l = 0; l < 32; ++l) buf[i][l] = v;
    }
    return buf[i];
  };
  struct Same {
    bool active = false;
    uint32_t operator()(uint32_t v) const { return v; }
  };
  struct IntMod {   // -x as 0 - x, ~x, as ival applies them
    bool neg, bnot, active;
    uint32_t operator()(uint32_t v) const { if (neg) v = 0u - v; if (bnot) v = ~v; return v; }
  };
  struct FMod {     // |x| and -x on an f32's bits
    bool abs, neg, active;
    uint32_t operator()(uint32_t v) const { if (abs) v &= 0x7fffffffu; if (neg) v ^= 0x80000000u; return v; }
  };
  alignas(64) uint32_t r[32];
  const auto store = [&](const Operand& dst) {
    if (dst.reg == kRZ) return;
    uint32_t* d = &w.reg(dst.reg, 0);
    if (ex == kAll) {
      std::memcpy(d, r, sizeof r);
    } else {
      for (unsigned l = 0; l < 32; ++l)
        if ((ex >> l) & 1) d[l] = r[l];
    }
  };
  switch (ins.op) {
    case Op::MOV: {
      if (ins.mnemonic != "MOV" || ins.f[1] || ins.src.empty() || !plain(ins.src[0])) return false;
      const uint32_t* a = lanes(0, ins.src[0], Same{});
      for (unsigned l = 0; l < 32; ++l) r[l] = a[l];
      store(ins.dst[0]);
      return true;
    }
    case Op::IADD3: {
      if (ins.f[0] || ins.f[1] || ins.src.size() < 3) return false;
      for (size_t i = 1; i < ins.dst.size(); ++i)   // carry-outs other than PT
        if (!(ins.dst[i].kind == Kind::Pred && ins.dst[i].reg == kPT)) return false;
      const uint32_t* s[3];
      for (unsigned i = 0; i < 3; ++i) {
        const Operand& o = ins.src[i];
        if (!plain(o)) return false;
        // As its bits are added: a negated source as its complement plus one
        // (the one added below), a ~ one as its complement.
        s[i] = lanes(i, o, IntMod{false, o.neg || o.bnot, o.neg || o.bnot});
      }
      const uint32_t add = (ins.src[0].neg ? 1u : 0u) + (ins.src[1].neg ? 1u : 0u) + (ins.src[2].neg ? 1u : 0u);
      for (unsigned l = 0; l < 32; ++l) r[l] = s[0][l] + s[1][l] + s[2][l] + add;
      store(ins.dst[0]);
      return true;
    }
    case Op::IMAD: {
      const unsigned kind = ins.f[0];
      if (kind == 0x25 || kind == 0x27 || ins.f[2] || ins.dst.size() > 1 || ins.src.size() < 3) return false;
      // IMAD.MOV (RZ * RZ + c) and the like: a zero factor leaves the addend.
      const auto zero = [](const Operand& o) {
        return !o.bnot && ((o.kind == Kind::Reg && o.reg == kRZ) || (o.kind == Kind::Imm && o.imm == 0));
      };
      if ((zero(ins.src[0]) || zero(ins.src[1])) && plain(ins.src[2])) {
        const Operand& o = ins.src[2];
        const uint32_t* c = lanes(2, o, IntMod{o.neg, o.bnot, o.neg || o.bnot});
        for (unsigned l = 0; l < 32; ++l) r[l] = c[l];
        store(ins.dst[0]);
        return true;
      }
      const uint32_t* s[3];
      for (unsigned i = 0; i < 3; ++i) {
        const Operand& o = ins.src[i];
        if (!plain(o)) return false;
        s[i] = lanes(i, o, IntMod{o.neg, o.bnot, o.neg || o.bnot});
      }
      for (unsigned l = 0; l < 32; ++l) r[l] = s[0][l] * s[1][l] + s[2][l];
      store(ins.dst[0]);
      return true;
    }
    case Op::LOP3: {
      const bool has_p = ins.dst.size() > 1;
      if (has_p && (ins.dst.size() != 2 || !pdst_ok(ins.dst[0]) || ins.dst[1].kind != Kind::Reg)) return false;
      if (ins.src.size() < 3) return false;
      const uint32_t* s[3];
      for (unsigned i = 0; i < 3; ++i) {
        if (!plain(ins.src[i])) return false;
        s[i] = lanes(i, ins.src[i], Same{});
      }
      const unsigned lut = ins.f[0];
      for (unsigned l = 0; l < 32; ++l) r[l] = 0;
      for (unsigned t = 0; t < 8; ++t) {   // each minterm of the table, over every lane
        if (!((lut >> t) & 1)) continue;
        const uint32_t ma = (t & 4) ? 0u : ~0u, mb = (t & 2) ? 0u : ~0u, mc = (t & 1) ? 0u : ~0u;
        for (unsigned l = 0; l < 32; ++l) r[l] |= (s[0][l] ^ ma) & (s[1][l] ^ mb) & (s[2][l] ^ mc);
      }
      if (has_p) {
        Mask pm = 0;
        for (unsigned l = 0; l < 32; ++l)
          if (ins.f[1] ? r[l] == kAll : r[l] != 0) pm |= Mask{1} << l;
        setp(ins.dst[0], pm);
      }
      store(has_p ? ins.dst[1] : ins.dst[0]);
      return true;
    }
    case Op::LEA: {   // (a << s) + b, with .HI, .SX32, .X and the carry out as the general rule has them
      const bool hi = ins.f[0], x = ins.f[1], sx32 = ins.f[2];
      const unsigned sh = ins.f[3];
      const bool three = hi && !sx32;   // the third source is the high word
      const size_t pidx = three ? 4 : 3;
      if (ins.src.size() < (x ? pidx + 1 : pidx) || ins.dst.size() > 2) return false;
      if (ins.dst.size() == 2 && !pdst_ok(ins.dst[1])) return false;
      if (!plain(ins.src[0]) || !plain(ins.src[1]) || ins.src[0].neg || ins.src[0].bnot) return false;
      if (three && (!plain(ins.src[2]) || ins.src[2].neg || ins.src[2].bnot)) return false;
      const Mask cin = x ? pmask(ins.src[pidx]) : 0;
      const uint32_t* a = lanes(0, ins.src[0], Same{});
      const uint32_t* b = lanes(1, ins.src[1], IntMod{ins.src[1].neg, ins.src[1].bnot, ins.src[1].neg || ins.src[1].bnot});
      const uint32_t* top = three ? lanes(2, ins.src[2], Same{}) : nullptr;
      Mask carry = 0;
      for (unsigned l = 0; l < 32; ++l) {
        uint32_t shifted;
        if (!hi) {
          shifted = sh >= 32 ? 0 : a[l] << sh;
        } else {
          const uint32_t t = three ? top[l] : static_cast<uint32_t>(static_cast<int32_t>(a[l]) >> 31);
          shifted = static_cast<uint32_t>(((a[l] | (static_cast<uint64_t>(t) << 32)) << sh) >> 32);
        }
        uint64_t sum = static_cast<uint64_t>(shifted) + b[l] + ((cin >> l) & 1);
        if (ins.src[1].neg && b[l] == 0) sum += uint64_t{1} << 32;   // -0 is ~0 + 1: it carries out (see the general rule)
        r[l] = static_cast<uint32_t>(sum);
        carry |= static_cast<Mask>(sum >> 32 != 0) << l;
      }
      store(ins.dst[0]);
      if (ins.dst.size() == 2) setp(ins.dst[1], carry);
      return true;
    }
    case Op::SHF: {   // a funnel shift of (src2:src0) by src1, as the general rule computes it
      if (ins.dst.size() != 1 || ins.src.size() < 3) return false;
      for (int i = 0; i < 3; ++i)
        if (!plain(ins.src[i]) || ins.src[i].neg || ins.src[i].bnot) return false;
      const unsigned type = ins.f[0];   // 0 S64, 1 U64, 2 S32, 3 U32
      const bool wrap = ins.f[1], right = ins.f[2], hi = ins.f[3];
      const bool is64 = type < 2, sgn = type == 0 || type == 2;
      const uint32_t width = is64 ? 64 : 32;
      const uint32_t* lo = lanes(0, ins.src[0], Same{});
      const uint32_t* n = lanes(1, ins.src[1], Same{});
      const uint32_t* top = lanes(2, ins.src[2], Same{});
      for (unsigned l = 0; l < 32; ++l) {
        const uint64_t v = lo[l] | (static_cast<uint64_t>(top[l]) << 32);
        const uint32_t k = wrap ? n[l] % width : std::min(n[l], width);
        uint64_t x;
        if (!right) x = k >= 64 ? 0 : v << k;
        else if (sgn) x = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(k, 63u));
        else x = k >= 64 ? 0 : v >> k;
        r[l] = static_cast<uint32_t>(hi ? x >> 32 : x);
      }
      store(ins.dst[0]);
      return true;
    }
    case Op::FSEL: {
      if (ins.src.size() < 3 || ins.dst.size() != 1) return false;
      if (!plain(ins.src[0]) || !plain(ins.src[1]) || ins.src[0].bnot || ins.src[1].bnot) return false;
      const uint32_t* a = lanes(0, ins.src[0], FMod{ins.src[0].abs, ins.src[0].neg, ins.src[0].abs || ins.src[0].neg});
      const uint32_t* b = lanes(1, ins.src[1], FMod{ins.src[1].abs, ins.src[1].neg, ins.src[1].abs || ins.src[1].neg});
      const Mask p = pmask(ins.src[2]);
      for (unsigned l = 0; l < 32; ++l) r[l] = ((p >> l) & 1) ? a[l] : b[l];
      store(ins.dst[0]);
      return true;
    }
    case Op::FFMA: case Op::FADD: case Op::FMUL: {
      const unsigned nsrc = ins.op == Op::FFMA ? 3 : 2;
      if (ins.f[0] || ins.f[1] || ins.f[2] || (ins.op == Op::FMUL && ins.f[4] != 4) || ins.src.size() < nsrc)
        return false;
      const uint32_t* s[3];
      for (unsigned i = 0; i < nsrc; ++i) {
        const Operand& o = ins.src[i];
        if (!plain(o) || o.bnot) return false;
        s[i] = lanes(i, o, FMod{o.abs, o.neg, o.abs || o.neg});
      }
      const auto out = [](float v) { return v != v ? 0x7fffffffu : f32_bits(v); };
      if (ins.op == Op::FFMA)
        for (unsigned l = 0; l < 32; ++l) r[l] = out(std::fma(as_f32(s[0][l]), as_f32(s[1][l]), as_f32(s[2][l])));
      else if (ins.op == Op::FADD)
        for (unsigned l = 0; l < 32; ++l) r[l] = out(as_f32(s[0][l]) + as_f32(s[1][l]));
      else
        for (unsigned l = 0; l < 32; ++l) r[l] = out(as_f32(s[0][l]) * as_f32(s[1][l]));
      store(ins.dst[0]);
      return true;
    }
    default: return false;
  }
}

void Runner::exec_float(Block&, Warp& w, const Instr& ins, Mask ex) {
  // sm_120's uniform float ops compute once for the warp: on one lane, or
  // one in place (UI2FP UR4, UR4) would convert its own result again.
  if (ins.guard_uniform && ex) ex = Mask{1} << std::countr_zero(ex);
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
        // A NaN result is the card's canonical one, whatever made it (inf *
        // 0 here is 0xffc00000 on the host), as the PTX engine's canon32.
        write32(w, ins.dst[0], l, r != r ? 0x7fffffffu : f32_bits(r));
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
        // Computed in double and rounded once, as the PTX engine computes
        // the .approx functions, so the two engines agree to the bit.
        const double x = ftz(f32src(ins.src[0], l, true));
        float r;
        switch (op) {
          case 0: r = static_cast<float>(std::cos(x * 6.28318530717958647692)); break;   // the argument is in turns
          case 1: r = static_cast<float>(std::sin(x * 6.28318530717958647692)); break;
          case 2: r = static_cast<float>(std::exp2(x)); break;
          case 3: r = static_cast<float>(std::log2(x)); break;
          case 4: r = static_cast<float>(1.0 / x); break;
          case 5: r = static_cast<float>(1.0 / std::sqrt(x)); break;
          case 8: r = static_cast<float>(std::sqrt(x)); break;
          case 9: r = static_cast<float>(std::tanh(x)); break;
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
          else if (dsize == 4) write32(w, ins.dst[0], l, f32_to_bf16(f));
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
      if (ins.f[5] == 4 && ins.f[4] == 0) {   // PACK_B: F32 to TF32 (to nearest, ties away)
        each_lane(ex, [&](unsigned l) {
          uint32_t v = read32(w, ins.src[0], l);
          if ((v & 0x7f800000u) != 0x7f800000u) v = (v + 0x1000u) & 0xffffe000u;
          write32(w, ins.dst[0], l, v);
        });
        return;
      }
      if (ins.f[3] >= 6 || ins.f[5] == 4) {   // the narrow forms: FP8, FP6, FP4, UE8M0
        // The format, as the decoder names it (10 E4M3, 11 E5M2, 12 E3M2,
        // 13 E2M3, 14 E2M1, 15 UE8M0); before sm_100 only 6 (E5M2) and 7.
        const bool unpack = ins.f[5] == 4, sat = ins.f[6], relu = ins.f[2];
        unsigned fmt = unpack ? ins.f[4] : ins.f[3];
        if (fmt == 7 || (unpack && fmt == 3)) fmt = 10;
        else if (fmt == 6 || (unpack && fmt == 2)) fmt = 11;
        const unsigned width = fmt == 14 ? 4 : 8;   // bits a value takes in the pair
        // One value of the format, as the PTX engine's cvt reads and writes it
        // (vgpu/exec/numerics.hpp).
        const auto decode = [&](uint32_t code) -> double {
          switch (fmt) {
            case 10: return exec::fp8_value(code, false);
            case 11: return exec::fp8_value(code, true);
            case 12: return exec::mx_float_value(code & 0x3F, 3, 2, 3);
            case 13: return exec::mx_float_value(code & 0x3F, 2, 3, 1);
            case 14: return exec::mx_float_value(code & 0xF, 2, 1, 1);
            default: return code == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, int(code) - 127);
          }
        };
        const auto encode = [&](double v, unsigned l) -> uint32_t {
          if (relu && v < 0) v = 0.0;
          switch (fmt) {
            case 10: return exec::fp8_bits(v, false, sat);
            case 11: return exec::fp8_bits(v, true, sat);
            case 12: return exec::small_float_bits(v, 3, 2, 3);
            case 13: return exec::small_float_bits(v, 2, 3, 1);
            case 14: return exec::small_float_bits(v, 2, 1, 1);
            default: {
              const uint32_t c = exec::ue8m0_bits(v, ins.f[1] == 2, sat);   // .RZ, or .RP
              if (c == ~0u) fault(w, ins, l, "F2FP to UE8M0 of a negative value, which the ISA does not define", Err::UnsupportedPtx);
              return c;
            }
          }
        };
        each_lane(ex, [&](unsigned l) {
          if (unpack) {   // b's pair (low first, or its upper half's) to two halves
            const uint32_t v = read32(w, ins.src[0], l) >> (ins.f[7] ? 16 : 0);
            uint32_t r = 0;
            for (unsigned h = 0; h < 2; ++h) {
              double x = decode((v >> (width * h)) & ((1u << width) - 1));
              uint32_t half;
              if (relu && std::isnan(x)) {
                half = 0x7FFF;   // canonical NaN
              } else {
                if (relu && x < 0) x = 0.0;
                half = ins.f[0] ? exec::bf16_bits(x) : exec::f16_bits(x);
              }
              r |= half << (16 * h);
            }
            write32(w, ins.dst[0], l, r);
            return;
          }
          const double a = f32src(ins.src[0], l, false), b = f32src(ins.src[1], l, false);
          uint32_t r = (encode(a, l) << width) | encode(b, l);
          // MERGE_C: C moves up past the new pair, which is how ptxas builds a
          // word of pairs -- each conversion's C the one before it (four FP4
          // bytes, two FP6/FP8 halves).
          if (ins.f[5] == 1) r |= read32(w, ins.src[2], l) << (2 * width);
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
        // A NaN result is the canonical 0x7fff, as the PTX engine writes it.
        uint16_t h[2];
        for (int i = 0; i < 2; ++i) h[i] = r[i] != r[i] ? uint16_t{0x7fff} : bf ? f64_to_bf16(r[i]) : f64_to_half(r[i]);
        write32(w, ins.dst[0], l, h[0] | (static_cast<uint32_t>(h[1]) << 16));
      });
      return;
    }
    case Op::FMNMX3:   // the min (p) or max of three, NaN losing unless .NAN
      each_lane(ex, [&](unsigned l) {
        const bool fz = ins.f[0], mn = pred(w, ins.src[3], l);
        float r = f32src(ins.src[0], l, fz);
        bool nan = r != r;
        for (int i = 1; i < 3; ++i) {
          const float b = f32src(ins.src[i], l, fz);
          nan = nan || b != b;
          if (r != r) r = b;
          else if (b != b) continue;
          else if (r == 0 && b == 0) r = mn ? (std::signbit(r) ? r : b) : (std::signbit(r) ? b : r);
          else r = mn ? std::min(r, b) : std::max(r, b);
        }
        write32(w, ins.dst[0], l, ins.f[1] && nan ? 0x7fffffffu : f32_bits(r));
      });
      return;
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
  const uint32_t bytes = size <= 1 ? 1 : size <= 3 ? 2 : size == 4 ? 4 : size == 5 ? 8 : size == 8 ? 32 : 16;
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
    // Turing's global forms (a register beside a 64-bit uniform base).
    if (m_.sm < 80 && !stride && ins.w.bit(72) && ins.op != Op::LDS && ins.op != Op::STS && ins.op != Op::LDL &&
        ins.op != Op::STL && ins.op != Op::ATOMS) {
      const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
      if (ra != kRZ) v = ins.w.bit(90) || !ins.w.bit(91) ? (w.reg(ra, l) | (static_cast<uint64_t>(w.reg(ra + 1, l)) << 32))
                                                         : w.reg(ra, l);
      if (ins.w.bit(91)) {
        const unsigned u = static_cast<unsigned>(ins.w.field(ur_pos, 6));
        if (u != 63) v += w.ur[u] | (static_cast<uint64_t>(w.ur[u + 1]) << 32);
      }
      return v + static_cast<uint64_t>(a.imm);
    }
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
        space = sp == Space::Global || sp == Space::Param ? 0 : sp == Space::Shared ? 1 : 2;
        // A 32-byte access (ENL2.256) moves two register quads: a store's low
        // 16 bytes from the first printed, a load's into the second.
        const auto quad = [&](const Operand& d, unsigned i) { return i < 4 ? d.reg + i : 0; };
        if (store) {
          const Operand& d = ins.src[1];
          if (bytes <= 4) {
            const uint32_t v = read32(w, d, l);
            mem_write(blk, w, l, sp, off, &v, bytes);
          } else {
            uint32_t v[8];
            for (unsigned i = 0; i < bytes / 4; ++i) {
              const Operand& q = i < 4 ? d : ins.src[2];
              const unsigned r = quad(q, i % 4);
              v[i] = q.reg == kRZ ? 0 : w.reg(r, l);
            }
            mem_write(blk, w, l, sp, off, v, bytes);
          }
        } else {
          const Operand& d = ins.dst[0];
          if (bytes <= 4) {
            uint32_t v = 0;
            mem_read(blk, w, l, sp, off, &v, bytes);
            write32(w, d, l, extend(v));
          } else {
            uint32_t v[8];
            mem_read(blk, w, l, sp, off, v, bytes);
            for (unsigned i = 0; i < bytes / 4; ++i) {
              const Operand& q = bytes == 32 ? (i < 4 ? ins.dst[1] : d) : d;
              if (q.reg != kRZ) w.reg(quad(q, i % 4), l) = v[i];
            }
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
      if (ins.f[3] == 5 || ins.f[3] == 9) {
        // The 8-bit forms, as the PTX engine's ldmatrix (exec/ldmatrix.hpp):
        // .M816 eight rows of sixteen values, .MT1616 sixteen transposed,
        // matrix i's rows from lanes R*i.., each row unpacked to a byte a
        // value (.U4x16P64TO8, .U6x16P32TO8) or taken as it is (.8).
        const unsigned R = ins.f[3] == 9 ? 16 : 8, regs = R / 8;
        exec::LdmRow fmt;
        fmt.bits = ins.f[2] == 1 ? 4 : ins.f[2] == 2 ? 6 : 8;
        const unsigned d = ins.dst[0].reg;
        // Rows are read by the lane that names them, as LDSM's own reads are.
        for (unsigned i = 0; i < nmat; ++i) {
          uint8_t tile[16][16] = {};
          for (unsigned r = 0; r < R; ++r) {
            const unsigned src = R * i + r;
            uint8_t raw[16] = {};
            mem_read(blk, w, src, Space::Shared, row(src), raw, fmt.bytes());
            exec::ldm_unpack_row(raw, fmt, tile[r]);
          }
          for (unsigned rr = 0; rr < regs; ++rr)
            each_lane(ex, [&](unsigned t) {
              if (d != kRZ) w.reg(d + i * regs + rr, t) = exec::ldm_word(tile, t, rr, trans);
            });
        }
        blk.st.shared_loads += static_cast<uint64_t>(std::popcount(ex));
        blk.st.shared_bytes_read += static_cast<uint64_t>(std::popcount(ex)) * 4 * nmat * regs;
        return;
      }
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
    case Op::STSM: {
      // stmatrix: LDSM's layout stored -- matrix i's rows at the addresses
      // lanes 8i..8i+7 hold, lane t's register i the row t/4's word t%4
      // (.MT88: column t/4 of rows 2(t%4) and 2(t%4)+1).
      const unsigned nmat = ins.f[0];
      const bool trans = ins.f[1];
      uint64_t rows[32] = {};
      Mask have = 0;
      each_lane(ex, [&](unsigned l) {
        rows[l] = static_cast<uint32_t>(address(ins.src[0], l, false, ins.w.bit(91), 64, 0));
        have |= Mask{1} << l;
      });
      const auto row = [&](unsigned r) {
        if (!(have >> r & 1)) fault(w, ins, r, "STSM: the lane holding a row address is not active");
        return rows[r];
      };
      const unsigned d = ins.src[1].reg;
      each_lane(ex, [&](unsigned t) {
        for (unsigned i = 0; i < nmat; ++i) {
          const uint32_t v = d == kRZ ? 0 : w.reg(d + i, t);
          if (!trans) {
            mem_write(blk, w, t, Space::Shared, row(8 * i + t / 4) + 4 * (t % 4), &v, 4);
          } else {
            const uint16_t lo = static_cast<uint16_t>(v), hi = static_cast<uint16_t>(v >> 16);
            mem_write(blk, w, t, Space::Shared, row(8 * i + 2 * (t % 4)) + 2 * (t / 4), &lo, 2);
            mem_write(blk, w, t, Space::Shared, row(8 * i + 2 * (t % 4) + 1) + 2 * (t / 4), &hi, 2);
          }
        }
      });
      blk.st.shared_stores += static_cast<uint64_t>(std::popcount(ex));
      blk.st.shared_bytes_written += static_cast<uint64_t>(std::popcount(ex)) * 4 * nmat;
      return;
    }
    case Op::LDC: {
      const unsigned bank = ins.f[4];
      const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
      const int64_t off = ins.w.sfield(38, 16);
      each_lane(ex, [&](unsigned l) {
        // A bank is 64 KiB, and the address wraps in it: parameters past
        // 4 KiB (CUDA 12.1+) reach 0x8000 and beyond, which the 16-bit
        // offset holds as a negative number. ptxas relies on the wrap:
        // sm_100's c[0x0][-0x7fa8] (RZ) is 0x8058, and c[0x0][R7+-0x7f84]
        // with R7 = -4*i is parameter byte 0x807c - 4*i.
        const uint64_t o = static_cast<uint64_t>((ra == kRZ ? 0 : static_cast<int64_t>(static_cast<int32_t>(w.reg(ra, l)))) + off) & 0xffff;
        if (bytes == 8) write64(w, ins.dst[0], l, cbank32(bank, o) | (static_cast<uint64_t>(cbank32(bank, o + 4)) << 32));
        else write32(w, ins.dst[0], l, extend(cbank32(bank, o & ~3ull) >> (8 * (o & 3))));
      });
      return;
    }
    case Op::ATOMS: case Op::ATOMG: case Op::ATOM: case Op::RED: {
      if (ins.op == Op::ATOMS && ins.f[2] == 2) {   // ATOMS.ARRIVE.64: an mbarrier
        exec_syncs(blk, w, ins, ex);
        return;
      }
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
        const Operand& vop = red ? ins.src[1] : ins.src[1];
        const uint64_t v = popc_inc ? 1 : (is64 ? read64(w, vop, l) : read32(w, vop, l));
        const uint64_t swap = cas ? (is64 ? read64(w, ins.src[2], l) : read32(w, ins.src[2], l)) : 0;
        // The value the atomic stores, given the one it finds.
        const auto compute = [&](const uint64_t old) -> uint64_t {
        uint64_t nw = old;
        if (cas) {
          if (old == v) nw = swap;
        } else {
          switch (popc_inc ? 3 : ins.f[0]) {
            case 0:   // ADD, by the PTX engine's rules (an RTX 3060's)
              if (type == 3) {
                // .ftz on inputs and result; a NaN result is the canonical one.
                const auto fz = [](float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; };
                const float r = fz(fz(as_f32(static_cast<uint32_t>(old))) + fz(as_f32(static_cast<uint32_t>(v))));
                nw = r != r ? 0x7fffffffu : f32_bits(r);
              } else if (type == 6) {
                // A NaN comes through as it is: the one added, else the one in memory.
                const double x = as_f64(old), y = as_f64(v);
                nw = std::isnan(y) ? v : std::isnan(x) ? old : f64_bits(x + y);
              } else if (type == 4 || type == 7) {   // F16x2, BF16x2: each half added once in double
                const ptx::Type ht{type == 7 ? ptx::Type::Kind::BF : ptx::Type::Kind::F, 16};
                const uint64_t lo = exec::reduce_value(ptx::AtomOp::Add, ht, old & 0xffff, v & 0xffff);
                const uint64_t hi = exec::reduce_value(ptx::AtomOp::Add, ht, (old >> 16) & 0xffff, (v >> 16) & 0xffff);
                nw = lo | (hi << 16);
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
        return nw;
        };
        // Memory the host maps -- managed, pinned, or shared with another
        // process through CUDA IPC -- takes the CPU's own atomic, which host
        // code and other processes' kernels respect; the stripe lock is this
        // process's alone (vgpu/exec/host_atomic.hpp).
        uint64_t old = 0;
        uint8_t* host = sp == Space::Global && addr % n == 0 ? mem_.host_address(addr, n) : nullptr;
        if (host && reinterpret_cast<uintptr_t>(host) % n == 0) {
          old = exec::host_atomic_rmw(host, n, compute);
        } else {
          mem_read(blk, w, l, sp, addr, &old, n);
          const uint64_t nw = compute(old);
          mem_write(blk, w, l, sp, addr, &nw, n);
        }
        ++blk.st.atomics;
        blk.st.atomic_bytes += n;
        if (!red) {
          const Operand& d = ins.op == Op::ATOMS ? ins.dst[0] : ins.dst[1];
          // ATOM/ATOMG's predicate: done in hardware (a generic CAS of a
          // shared address included, so no software path is needed).
          if (ins.op != Op::ATOMS) set_pred(w, ins.dst[0], l, true);
          if (is64) write64(w, d, l, old);
          else write32(w, d, l, static_cast<uint32_t>(old));
          // CAST.SPIN: whether it stored (ATOMS keeps the mode in f[1], ATOM in f[6]).
          // The .64 form's flag is the whole register pair: ptxas's loop
          // after it (sm_86's atomicAdd of a double in shared memory, as
          // CUDA Samples' jacobiCudaGraphs does) tests Rd == 1 and Rd+1 == 0,
          // so leaving the old value's high word in Rd+1 spun forever.
          if (cas && (ins.op == Op::ATOMS ? ins.f[1] == 3 : ins.f[6] != 0)) {
            if (d.kind == Kind::Pred) set_pred(w, d, l, old == v);   // sm_100's ATOMS.CAST.SPIN P
            else if (is64) write64(w, d, l, old == v ? 1 : 0);
            else write32(w, d, l, old == v ? 1 : 0);
          }
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
      if (ex) write32(w, ins.dst[0], static_cast<unsigned>(std::countr_zero(ex)), r);
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
// E3M2, E2M3 and E2M1 each in a byte (QMMA, mma.sync .kind::f8f6f4: the
// FP6 in bits 0-5, FP4 in 2-5); E2M1P packed two to a byte (OMMA).
enum class MmaT { F16, BF16, TF32, S8, U8, S4, U4, B1, F64, E4M3, E5M2, E3M2, E2M3, E2M1, E2M1P };
unsigned mma_bits(MmaT t) {
  switch (t) {
    case MmaT::F16: case MmaT::BF16: return 16;
    case MmaT::S8: case MmaT::U8: case MmaT::E4M3: case MmaT::E5M2: return 8;
    case MmaT::E3M2: case MmaT::E2M3: case MmaT::E2M1: return 8;
    case MmaT::S4: case MmaT::U4: case MmaT::E2M1P: return 4;
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
    case MmaT::E3M2: return exec::mx_float_value(static_cast<uint32_t>(v >> (8 * e)) & 0x3F, 3, 2, 3);
    case MmaT::E2M3: return exec::mx_float_value(static_cast<uint32_t>(v >> (8 * e)) & 0x3F, 2, 3, 1);
    case MmaT::E2M1: return exec::mx_float_value(static_cast<uint32_t>(v >> (8 * e + 2)) & 0xF, 2, 1, 1);
    case MmaT::E2M1P: return exec::mx_float_value(static_cast<uint32_t>(v >> (4 * e)) & 0xF, 2, 1, 1);
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
  if (!ex) return;   // no thread runs it: its object need not exist
  const Word& word = ins.w;
  uint64_t handle;
  if (ins.f[6]) {   // a uniform register: the handle's low word (the high one is zero)
    handle = w.ur[ins.f[7]];
  } else {
    const unsigned bank = static_cast<unsigned>(word.field(54, 5));
    const uint64_t hoff = 4 * word.field(40, 14);
    handle = cbank32(bank, hoff) | (static_cast<uint64_t>(cbank32(bank, hoff + 4)) << 32);
  }
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
  // The LOD mode (87-89; from sm_100 split, its low bit at 59). A fetch's
  // (TLD) is explicit, so sm_100's 0 means .LZ.
  unsigned lod_mode = ins.op == Op::TLD4 ? 0
                      : m_.sm >= 100    ? static_cast<unsigned>(word.field(59, 1) | (word.field(87, 2) << 1))
                                        : static_cast<unsigned>(word.field(87, 3));
  if (ins.op == Op::TLD && m_.sm >= 100 && lod_mode == 0) lod_mode = 1;
  const bool has_lod = lod_mode == 2 || lod_mode == 3 || lod_mode == 5;
  if (lod_mode == 4 || lod_mode == 5)
    fault(w, ins, 0, "texture fetches with a LOD clamp are not implemented yet", Err::UnsupportedPtx);
  if (m_.sm >= 100 ? word.field(56, 2) != 0 : (word.bit(76) || (ins.op == Op::TLD4 && word.bit(77))))
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
    uint32_t comp[5] = {};   // a cube array's layer, x, y, z and LOD
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

// ---- warpgroup MMA (HGMMA, IGMMA, QGMMA, BGMMA) ----------------------------------
//
// The PTX engine's wgmma, on SASS operands. Four consecutive warps form a
// warpgroup computing a 64xNxK product; warp r of the group holds rows
// 16r..16r+15 of A (when in registers) and D, and all of B, so each warp
// computes its rows when it reaches the instruction. The shared-memory
// operands are read once for the group, by the first warp to issue its n-th
// MMA -- a warp that ran ahead could otherwise release a stage and see it
// refilled before its peers read it -- and the others take their share.
void Runner::exec_gmma(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  if (ex != (w.alive & ~w.exited))
    fault(w, ins, 0, "a warpgroup MMA runs on the whole warp (.sync.aligned), and some threads are off",
          Err::UnsupportedPtx);
  if ((w.index / 4 + 1) * 4 > blk.warps.size())
    fault(w, ins, 0, "a warpgroup MMA needs four whole warps", Err::UnsupportedPtx);
  const uint32_t N = ins.f[0], K = ins.f[1], tb = ins.f[2], flags = ins.f[5];
  const bool reg_a = ins.f[3], sparse = ins.f[4], sat = flags & 16;
  enum class T { F16, BF16, TF32, E4M3, E5M2, S8, U8, B1 };
  T at, bt;
  bool acc32 = true, int_form = false, b1_and = false;
  switch (ins.op) {
    case Op::HGMMA:
      at = bt = (tb & 0x20) ? T::TF32 : (tb & 0x10) ? T::BF16 : T::F16;
      acc32 = tb & 0x08;
      break;
    case Op::IGMMA:
      at = (tb & 0x10) ? T::S8 : T::U8;
      bt = (tb & 0x400) ? T::S8 : T::U8;   // bit 82
      int_form = true;
      break;
    case Op::QGMMA:
      at = (tb & 0x10) ? T::E5M2 : T::E4M3;
      bt = (tb & 0x20) ? T::E5M2 : T::E4M3;
      acc32 = tb & 0x08;
      break;
    default:   // BGMMA
      at = bt = T::B1;
      int_form = true;
      b1_and = tb & 0x40;
      break;
  }
  const auto bytes = [](T t) { return t == T::TF32 ? 4u : (t == T::F16 || t == T::BF16) ? 2u : 1u; };
  const auto value = [](T t, uint64_t bits) -> double {
    switch (t) {
      case T::F16: return half_to_f32(static_cast<uint16_t>(bits));
      case T::BF16: return bf16_to_f32(static_cast<uint16_t>(bits));
      case T::TF32: return as_f32(static_cast<uint32_t>(bits) & 0xffffe000u);
      case T::E4M3: return exec::fp8_value(static_cast<uint32_t>(bits) & 0xff, false);
      case T::E5M2: return exec::fp8_value(static_cast<uint32_t>(bits) & 0xff, true);
      case T::S8: return static_cast<int8_t>(bits);
      case T::U8: return static_cast<double>(bits & 0xff);
      case T::B1: return static_cast<double>(bits & 1);
    }
    return 0;
  };
  const uint32_t ea = bytes(at), eb = bytes(bt), abits = at == T::B1 ? 1 : 8 * ea;
  const double sa = (flags & 4) ? -1 : 1, sb = (flags & 8) ? -1 : 1;
  const bool a_kmajor = !(flags & 1), b_kmajor = !(flags & 2);
  const uint32_t Kp = sparse ? K / 2 : K;
  const uint32_t stored = abits == 32 ? 1 : 2;
  uint32_t meta[32] = {};
  const unsigned sel = ins.f[7] >> 8;
  if (sparse) {
    const unsigned mr = ins.f[7] & 0xff;
    for (unsigned l = 0; l < 32; ++l) meta[l] = mr == kRZ ? 0 : w.reg(mr, l);
  }
  const unsigned desc_ur = ins.src[reg_a ? 1 : 0].reg;
  const auto desc = [&](unsigned u) {
    return exec::decode_wgmma_desc(w.ur[u] | (static_cast<uint64_t>(w.ur[u + 1]) << 32));
  };
  const auto smem = [&](const exec::WgmmaDesc& d, bool k_major, T t, uint32_t eb_, uint32_t mn, uint32_t k) {
    if (t == T::B1) {   // K-major only, eight to a byte
      const uint64_t at_ = exec::wgmma_smem_offset(d, true, 1, mn, k / 8);
      if (at_ >= blk.shared.size()) throw Fault("a warpgroup MMA reads past the block's shared memory");
      return static_cast<double>((blk.shared[at_] >> (k % 8)) & 1);
    }
    const uint64_t at_ = exec::wgmma_smem_offset(d, k_major, eb_, mn, k) & 0xffffff;
    if (at_ + eb_ > blk.shared.size()) throw Fault("a warpgroup MMA reads past the block's shared memory");
    uint64_t v = 0;
    std::memcpy(&v, &blk.shared[at_], eb_);
    return value(t, v);
  };
  const uint32_t row0 = (w.index % 4) * 16;
  // This warp's 16 x K slice of A.
  std::vector<double> A(size_t{16} * K, 0.0);
  if (reg_a) {
    // mma.m16n8kK's A fragment (mma.sp's, sparse).
    const unsigned ra = ins.src[0].reg, per_reg = 32 / abits, cpr = per_reg / stored;
    for (unsigned r = 0; r < 4; ++r)
      for (unsigned l = 0; l < 32; ++l) {
        const unsigned g = l / 4, t = l % 4, row = g + (r % 2) * 8, col0 = t * per_reg + (r / 2) * (per_reg * 4);
        const uint32_t v = ra == kRZ ? 0 : w.reg(ra + r, l);
        for (unsigned e = 0; e < per_reg; ++e) {
          const unsigned col = sparse ? exec::mma_sparse_column(abits, K, sel, meta, row,
                                                                t * cpr + (r / 2) * 4 * cpr + e / stored, e % stored)
                                      : col0 + e;
          if (col < K) A[row * K + col] = sa * value(at, v >> (abits * e));
        }
      }
  }
  // The shared-memory operands, read once for the warpgroup.
  auto [it, first] = blk.gmma.try_emplace({w.index / 4, w.gmma_issued++});
  GmmaSnapshot& snap = it->second;
  if (first) {
    if (!reg_a) {
      const exec::WgmmaDesc d = desc(desc_ur);
      snap.A.assign(size_t{64} * Kp, 0.0);
      for (uint32_t r = 0; r < 64; ++r)
        for (uint32_t k = 0; k < Kp; ++k) snap.A[r * Kp + k] = sa * smem(d, a_kmajor, at, ea, r, k);
    }
    const exec::WgmmaDesc d = desc(desc_ur + 2);
    snap.B.assign(size_t{K} * N, 0.0);
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k) snap.B[size_t{k} * N + n] = sb * smem(d, b_kmajor, bt, eb, n, k);
  }
  if (!reg_a) {
    if (snap.A.empty()) fault(w, ins, 0, "the warps of a warpgroup disagree on where A is", Err::UnsupportedPtx);
    for (uint32_t r = 0; r < 16; ++r)
      for (uint32_t j = 0; j < Kp; ++j) {
        const uint32_t col = sparse ? exec::mma_sparse_column(abits, K, sel, meta, r, j / stored, j % stored) : j;
        if (col < K) A[r * K + col] = snap.A[size_t{row0 + r} * Kp + j];
      }
  }
  const std::vector<double>& B = snap.B;
  // scale-d: stored as 7 - UPn (and a not); false means D = A*B.
  const unsigned up = ins.f[6] & 7;
  const bool keep = (up == 7 ? true : w.up[up]) != ((ins.f[6] & 8) != 0);
  const unsigned rd = ins.dst[0].reg, rc = ins.src[reg_a ? 2 : 1].reg;
  const bool accumulate = keep && rc != kRZ;
  const uint32_t elems = N / 2;   // accumulator elements per lane
  each_lane(ex, [&](unsigned l) {
    std::vector<uint32_t> out(acc32 || int_form ? elems : (elems + 1) / 2);
    for (size_t i = 0; i < out.size(); ++i) out[i] = rc == kRZ ? 0 : w.reg(rc + static_cast<unsigned>(i), l);
    for (uint32_t e = 0; e < elems; ++e) {
      const uint32_t g = l / 4, t = l % 4, j = e / 4, s = e % 4;
      const uint32_t row = g + 8 * (s / 2), col = 8 * j + 2 * t + (s % 2);
      if (int_form) {
        int64_t acc = accumulate ? static_cast<int32_t>(out[e]) : 0;
        for (uint32_t k = 0; k < K; ++k)
          acc += at == T::B1 ? int64_t{b1_and ? (A[row * K + k] != 0 && B[size_t{k} * N + col] != 0)
                                              : ((A[row * K + k] != 0) != (B[size_t{k} * N + col] != 0))}
                             : static_cast<int64_t>(A[row * K + k]) * static_cast<int64_t>(B[size_t{k} * N + col]);
        if (sat) acc = std::clamp<int64_t>(acc, INT32_MIN, INT32_MAX);
        out[e] = static_cast<uint32_t>(static_cast<int32_t>(acc));
        continue;
      }
      // Products exact in f32, summed in f32 in K order, as the PTX engine does.
      const uint32_t shift = acc32 ? 0 : 16 * (e % 2);
      uint32_t& slot = out[acc32 ? e : e / 2];
      float acc = 0.0f;
      if (accumulate) acc = acc32 ? as_f32(slot) : half_to_f32(static_cast<uint16_t>(slot >> shift));
      for (uint32_t k = 0; k < K; ++k)
        acc += static_cast<float>(A[row * K + k]) * static_cast<float>(B[size_t{k} * N + col]);
      if (acc32) slot = f32_bits(acc);
      else slot = (slot & ~(0xffffu << shift)) | (static_cast<uint32_t>(f64_to_half(acc)) << shift);
    }
    for (size_t i = 0; i < out.size(); ++i)
      if (rd != kRZ) w.reg(rd + static_cast<unsigned>(i), l) = out[i];
  });
  snap.taken |= static_cast<uint8_t>(1u << (w.index % 4));
  if (snap.taken == 0xF) blk.gmma.erase(it);
}

void Runner::exec_mma(Block&, Warp& w, const Instr& ins, Mask ex) {
  if (ins.op == Op::MOVM) {
    if (ins.f[0]) {
      // MOVM.U4TO8.M832 Rd, Ra (sm_100): the eight 4-bit fields of Ra, each
      // widened to a byte -- fields 0-3 into Rd, 4-7 into Rd+1. ptxas uses it
      // to run b1 MMA as u8 IMMA (one bit per nibble, the four bit planes in
      // turn); the same per-lane widening of A and B is what makes those
      // products come out, which mma_forms and wmma_types check exactly.
      each_lane(ex, [&](unsigned l) {
        const uint32_t v = read32(w, ins.src[0], l);
        uint32_t out[2] = {};
        for (unsigned i = 0; i < 8; ++i) out[i / 4] |= ((v >> (4 * i)) & 0xfu) << (8 * (i % 4));
        if (ins.dst[0].reg != kRZ) {
          w.reg(ins.dst[0].reg, l) = out[0];
          w.reg(ins.dst[0].reg + 1, l) = out[1];
        }
      });
      return;
    }
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
      static const unsigned ks[] = {8, 16, 4, 0};
      static const unsigned sp_ks[] = {8, 16, 0, 32};   // .SP: 1688, 16816, 16832
      K = ins.f[3] ? sp_ks[ins.f[0]] : ks[ins.f[0]];
      at = bt = ins.f[2] == 1 ? MmaT::BF16 : ins.f[2] == 2 ? MmaT::TF32 : MmaT::F16;
      if (at == MmaT::TF32 && ins.f[0] == 0) K = 8;
      acc = ins.f[1] ? AccF32 : AccF16;
      break;
    }
    case Op::IMMA: {
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
    case Op::QMMA: {   // m16n8k32 (OMMA and .SP: k64), products and sums in f32
      static const MmaT types[] = {MmaT::E4M3, MmaT::E5M2, MmaT::E4M3, MmaT::E3M2, MmaT::E2M3,
                                   MmaT::E2M1, MmaT::E4M3, MmaT::E4M3, MmaT::E2M1P};
      const bool scaled = ins.mnemonic == "OMMA" || (ins.mods.size() && ins.mods[0] == "SF");
      K = scaled && (ins.f[3] || ins.f[4]) ? 64 : 32;
      at = types[std::min(ins.f[1], 8u)];
      bt = types[std::min(ins.f[2], 8u)];
      acc = ins.f[0] ? AccF32 : AccF16;
      break;
    }
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
  // The block-scaled forms (QMMA.SF, OMMA.SF): one factor per row of A and
  // column of B for each K/vec block, picked as the PTX engine picks them
  // (row i of A from lane 4(i % 8) + 2 tid-a + i / 8, column j of B from
  // lane 4j + tid-b, byte bid + block), UE8M0 2^(v - 127) or unsigned UE4M3.
  const bool block_scaled = ins.op == Op::QMMA && !ins.mods.empty() && ins.mods[0] == "SF";
  unsigned scale_vec = 1;
  double SA[16][4] = {}, SB[8][4] = {};
  if (block_scaled) {
    const bool sp_form = ins.f[3];
    scale_vec = ins.f[5];
    const Operand& sel_op = ins.src[5];
    const uint32_t sel = sel_op.reg >= (m_.sm >= 100 ? 255u : 63u) ? 0 : w.ur[sel_op.reg];
    const unsigned bida = sel & 3, tida = sel >> 2 & 1, bidb = sel >> 3 & 3, tidb = sel >> 5 & 3;
    const unsigned rsa = ins.src[3].reg + (sp_form ? 1 : 0), rsb = ins.src[4].reg;
    const auto factor = [&](uint32_t byte) {
      if (ins.f[6]) return exec::fp8_value(byte & 0x7F, false);
      return byte == 0xFF ? std::numeric_limits<double>::quiet_NaN() : std::ldexp(1.0, static_cast<int>(byte) - 127);
    };
    const auto lane_reg = [&](unsigned r, unsigned l) -> uint32_t { return r == kRZ ? 0 : w.reg(r, l); };
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned b = 0; b < scale_vec; ++b)
        SA[i][b] = factor(lane_reg(rsa, 4 * (i % 8) + 2 * tida + i / 8) >> (8 * (bida + b)) & 0xFF);
    for (unsigned j = 0; j < 8; ++j)
      for (unsigned b = 0; b < scale_vec; ++b)
        SB[j][b] = factor(lane_reg(rsb, 4 * j + tidb) >> (8 * (bidb + b)) & 0xFF);
  }
  const bool sparse = (ins.op == Op::HMMA || ins.op == Op::IMMA || block_scaled) && ins.f[3];
  if (sparse) {
    // Structured-sparse A (mma.sp): each row holds half its K columns, in
    // chunks placed by the metadata register's nibbles -- the PTX engine's
    // hardware-checked rule (exec::mma_sparse_column).
    uint32_t meta[32];
    const Operand& mo = ins.src[3];
    for (unsigned l = 0; l < 32; ++l) meta[l] = mo.reg == kRZ ? 0 : w.reg(mo.reg, l);
    const bool tf32 = at == MmaT::TF32, int4 = bits == 4;
    const unsigned chunk_elems = tf32 ? 2 : int4 ? 8 : 4, stored = chunk_elems / 2, cpr = per_reg / stored;
    for (unsigned r = 0; r < a_regs / 2; ++r)
      for (unsigned l = 0; l < 32; ++l) {
        const unsigned g = l / 4, t = l % 4, row = g + (r % 2) * 8;
        const uint64_t v = reg(ra, r, l);
        for (unsigned e = 0; e < per_reg; ++e) {
          const unsigned chunk = t * cpr + (r / 2) * 4 * cpr + e / stored;
          const unsigned col = exec::mma_sparse_column(bits, K, ins.f[7], meta, row, chunk, e % stored);
          if (col < K) A[row * K + col] = mma_elem(at, v, e);
        }
      }
  } else {
    for (unsigned r = 0; r < a_regs; ++r)
      for (unsigned l = 0; l < 32; ++l) {
        const unsigned g = l / 4, t = l % 4, row = g + (r % 2) * 8, col0 = t * per_reg + (r / 2) * (per_reg * 4);
        const uint64_t v = reg(ra, r, l);
        for (unsigned e = 0; e < per_reg; ++e) A[row * K + col0 + e] = mma_elem(at, v, e);
      }
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
        const unsigned sblock = K / scale_vec;
        for (unsigned k = 0; k < K; ++k) {
          const float prod = block_scaled ? static_cast<float>(A[i * K + k] * SA[i][k / sblock]) *
                                                static_cast<float>(B[k * kN + j] * SB[j][k / sblock])
                                          : static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * kN + j]);
          a = a + prod;
        }
        D[i * kN + j] = a;
      }
    }
  each_lane(ex, [&](unsigned l) {
    if (rd == kRZ) return;
    for (unsigned s = 0; s < c_elems; ++s) {
      const double v = D[cd(l, s)];
      // NaN results are the canonical ones, as the PTX engine writes them.
      if (acc == AccF16) {
        uint32_t& word = w.reg(rd + s / 2, l);
        const uint16_t h = v != v ? uint16_t{0x7fff} : f64_to_half(v);
        word = (s % 2) ? (word & 0xffff) | (static_cast<uint32_t>(h) << 16) : (word & 0xffff0000u) | h;
      } else if (acc == AccF64) {
        const uint64_t b = f64_bits(v);
        w.reg(rd + 2 * s, l) = static_cast<uint32_t>(b);
        w.reg(rd + 2 * s + 1, l) = static_cast<uint32_t>(b >> 32);
      } else if (acc == AccS32) {
        w.reg(rd + s, l) = static_cast<uint32_t>(static_cast<int32_t>(v));
      } else {
        w.reg(rd + s, l) = v != v ? 0x7fffffffu : f32_bits(static_cast<float>(v));
      }
    }
  });
}

// ---- control -------------------------------------------------------------------

bool Runner::exec_control(Block& blk, Warp& w, const Instr& ins, Mask ex, Mask group) {
  const auto base_of = [&](unsigned l) { return w.pc[l] - ins.pc; };   // the section's code address
  // A condition predicate beside the guard (sm_90's BRA Pn and EXIT Pn,
  // BREAK Pn, sm_100's BRA.U UPn): the lanes where both hold.
  const auto with_condition = [&](Mask m) {
    for (const Operand& o : ins.src)
      if (o.kind == Kind::Pred || o.kind == Kind::UPred)
        each_lane(m, [&](unsigned l) {
          if (!pred(w, o, l)) m &= ~(Mask{1} << l);
        });
    return m;
  };
  const auto release_bsync = [&](unsigned x) {
    const Mask members = w.b[x] & w.alive & ~w.exited;
    Mask arrived = 0;
    for (unsigned l = 0; l < 32; ++l)
      if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::BSync && w.wait_arg[l] == x) arrived |= Mask{1} << l;
    // A callee that uses the same Bx as its caller saves the caller's mask with BMOV.CLEAR and puts it
    // back on return, so while it runs the barrier holds only the callee's own lanes. Lanes the caller
    // has parked at its BSYNC Bx are waiting on the saved mask, not on this one, and the callee's BSYNC
    // must not take them with it: released early, they ran on, and the caller's BSYNC then waited for
    // lanes that had already left (CUB's reduce kernels calling GooFit's functions through pointers).
    if (members) arrived &= members;
    if ((members & ~arrived) == 0 && arrived) {
      each_lane(arrived, [&](unsigned l) {
        w.waiting &= ~(Mask{1} << l);
        w.wait_kind[l] = Wait::None;
        w.pc[l] += 16;
      });
      w.b[x] &= ~arrived;   // they have converged: the barrier is free for its next BSSY
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
    // A .COLLECTIVE warpsync runs what follows it (up to ENDCOLLECTIVE) once
    // for all its lanes. Lanes that reach it from different places in the
    // code would each run their own copy here, so that is refused rather
    // than answered per path.
    if (release) {
      const uint64_t at = w.pc[std::countr_zero(release)];
      bool collective = false, apart = false;
      each_lane(release, [&](unsigned l) {
        const Instr& at_ins = fetch(w.pc[l]);
        collective = collective || (at_ins.op == Op::WARPSYNC && at_ins.f[1]);
        apart = apart || w.pc[l] != at;
      });
      if (collective && apart)
        throw Fault("WARPSYNC.COLLECTIVE reached from different code paths of one warp is not implemented");
    }
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
    case Op::CCTL:
      if (ins.f[7] == 0xaa) {
        // QSPC.{G,L,S,D} Pd, Rd, [addr] (isspacep): whether a generic address
        // is in that space's window -- D, a cluster's shared memory, being
        // the shared window with the CTA's rank above bit 24. The address is
        // a register (pair with .E), a uniform register (pair), or both.
        const unsigned urz = m_.sm >= 100 ? 255u : 63u;
        const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
        const bool wide = ins.w.bit(72);
        const unsigned ub = m_.sm >= 100 ? 8 : 6;
        const unsigned u = ins.w.bit(11) ? static_cast<unsigned>(ins.w.field(32, ub)) : urz;
        each_lane(ex, [&](unsigned l) {
          uint64_t a = 0;
          if (ra != kRZ) a = wide ? (w.reg(ra, l) | (static_cast<uint64_t>(w.reg(ra + 1, l)) << 32)) : w.reg(ra, l);
          if (u != urz) a += wide ? (w.ur[u] | (static_cast<uint64_t>(w.ur[u + 1]) << 32)) : w.ur[u];
          a += static_cast<uint64_t>(ins.src[0].imm);
          uint64_t off;
          const Space sp = classify(a, &off);
          const bool in = ins.f[0] == 0 ? sp == Space::Global : ins.f[0] == 1 ? sp == Space::Local : sp == Space::Shared;
          set_pred(w, ins.dst[0], l, in);
          if (ins.dst.size() > 1 && ins.dst[1].reg != kRZ) write32(w, ins.dst[1], l, in ? static_cast<uint32_t>(off) : 0);
        });
      }
      return false;
    case Op::NOP: case Op::DEPBAR: case Op::MEMBAR: case Op::ERRBAR: case Op::NANOSLEEP:
    case Op::ENDCOLLECTIVE:
      return false;
    case Op::YIELD:
      // A hint ptxas puts in loops, spin-waits among them (sm_80's mbarrier
      // waits poll shared memory): other lane groups, and other warps, go
      // first.
      w.give_way = true;
      w.yield = true;
      return false;
    case Op::BPT:
      if (ex) throw Error(Err::Trap, "SASS kernel " + k_.name + " executed BPT.TRAP (a trap)");
      return false;
    case Op::BRA: {
      Mask take = with_condition(ex);
      if (ins.f[1] && take) take = group;   // .U.ANY: all of them when any
      if (ins.f[0] >= 2) {
        // .DIV/.CONV: branch when the warp is (not) converged on the mask the
        // uniform register gives (~URZ: every lane). Turing's carry none and
        // mean every lane.
        uint32_t want = ~0u;
        if (ins.src.size() >= 2 && ins.src[ins.src.size() - 2].kind == Kind::UReg) {
          const Operand& u = ins.src[ins.src.size() - 2];
          want = u.reg >= (m_.sm >= 100 ? 255u : 63u) ? 0 : w.ur[u.reg];
          if (u.bnot) want = ~want;
        }
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
      const bool abs = ins.f[0], reg_rel = ins.f[1], abs_imm = ins.f[2];
      bool handled = false;
      const auto target_of = [&](unsigned l) -> uint64_t {
        if (abs_imm) return static_cast<uint64_t>(ins.src[0].imm);
        if (abs) {
          const unsigned r = ins.src[0].reg;
          return w.reg(r, l) | (static_cast<uint64_t>(w.reg(r + 1, l)) << 32);
        }
        if (reg_rel) return base_of(l) + static_cast<uint64_t>(ins.src[0].imm) + w.reg(ins.src[0].reg, l);
        return base_of(l) + static_cast<uint64_t>(ins.src[0].imm);
      };
      each_lane(group, [&](unsigned l) {
        if (!((ex >> l) & 1)) {
          w.pc[l] += 16;
          return;
        }
        const uint64_t target = target_of(l);
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
        builtin_call(blk, w, ins, ex, m_.builtins.at(target_of(l0)));
        each_lane(ex, [&](unsigned l) { w.pc[l] += 16; });
      }
      return true;
    }
    case Op::RET:
      // The return address is a register pair: a section-relative offset
      // (MOV R20, 0xd0 before a CALL.REL inside one section), or an absolute
      // code address a relocation wrote (a -G build, where every function
      // has its own section and the caller passes its own address plus the
      // return point's offset).
      each_lane(group, [&](unsigned l) {
        if (!((ex >> l) & 1)) {
          w.pc[l] += 16;
          return;
        }
        const unsigned r = ins.src[0].reg;
        const uint32_t lo = w.reg(r, l), hi = r + 1 < w.nregs ? w.reg(r + 1, l) : 0;
        w.pc[l] = hi == static_cast<uint32_t>(kCodeBase >> 32) ? (uint64_t{hi} << 32 | lo) : base_of(l) + lo;
      });
      return true;
    case Op::EXIT: {
      const Mask go = with_condition(ex);   // EXIT Pn: the lanes where both hold
      each_lane(group, [&](unsigned l) {
        if ((go >> l) & 1) w.exited |= Mask{1} << l;
        else w.pc[l] += 16;
      });
      // Lanes that exit no longer hold anything up.
      for (unsigned x = 0; x < 16; ++x) release_bsync(x);
      release_warpsync();
      for (unsigned id = 0; id < 16; ++id)
        if (blk.bars[id].arrived) release_bar(id);
      cluster_barrier_check(blk);
      return true;
    }
    case Op::UCGABAR: {   // barrier.cluster: UCGABAR_ARV arrives, UCGABAR_WAIT waits for the phase
      Cluster& cl = *blk.cluster;
      if (ins.f[0] == 0) {
        each_lane(ex, [&](unsigned l) { w.cga_target[l] = cl.phase + 1; });
        w.cga_arrived |= ex;
        cluster_barrier_check(blk);
        return false;
      }
      each_lane(group, [&](unsigned l) {
        if ((ex >> l & 1) && cl.phase < w.cga_target[l]) {
          w.waiting |= Mask{1} << l;
          w.wait_kind[l] = Wait::Cluster;
        } else {
          w.pc[l] += 16;
        }
      });
      return true;
    }
    case Op::BSSY:
      // The lanes join the barrier. Sibling paths of one divergent branch
      // can each set up the same Bx for their own reconvergence point (a
      // -G build's switch over function pointers, every case with its own
      // BSSY B6 around its call); run interleaved, an overwrite let the
      // last one's lanes alone count, and the others' BSYNC then waited
      // for lanes that had already left. They converge together instead,
      // and BSYNC takes the lanes it releases back out.
      w.b[ins.dst[0].reg] |= ex;
      return false;
    case Op::BREAK:   // BREAK Pn, Bx: the lanes where both hold leave Bx
      w.b[ins.src.back().reg] &= ~with_condition(ex);
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
          // The mask: .ALL, or the first value operand (after a
          // predicate, before a .COLLECTIVE target).
          const Operand* m = nullptr;
          for (const Operand& o : ins.src)
            if (!m && (o.kind == Kind::Reg || o.kind == Kind::UReg || o.kind == Kind::Imm || o.kind == Kind::CBank))
              m = &o;
          w.wait_arg[l] = ins.f[0] || !m ? ~Mask{0} : read32(w, *m, l);
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
      // A thread count (bar.sync 7, 256): only the warps that arrive take part,
      // and the barrier completes at that many threads -- not at every live
      // thread of the block, which held CUTLASS's consumer warpgroups at their
      // own barrier while its producer warps waited on them. As the PTX engine
      // counts it, from the ISA: a warp's arrival is all of its threads.
      if (const uint32_t count = ins.f[2]) {
        if (count % 32)
          throw Error(Err::InvalidValue, "a barrier's thread count must be a non-zero multiple of the warp size (32); got " +
                                             std::to_string(count));
        b.expected = count;
        if (ex) b.arrived += 32;
      }
      each_lane(ex, [&](unsigned l) {
        if (!ins.f[2]) ++b.arrived;
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
  if (device_runtime_call(blk, w, ins, ex, name)) return;
  if (name == "vprintf") {   // R4:R5 the format, R6:R7 the argument buffer
    each_lane(ex, [&](unsigned l) {
      const uint64_t fmt = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
      const uint64_t args = w.reg(6, l) | (static_cast<uint64_t>(w.reg(7, l)) << 32);
      const auto cstring = [&](uint64_t a) {
        std::string str;
        for (;; ++a) {
          uint64_t off;
          const Space sp = classify(a, &off);
          char c;
          mem_read(blk, w, l, sp, off, &c, 1);
          if (!c) break;
          str += c;
        }
        return str;
      };
      uint64_t cursor = 0;
      const auto fetch = [&](uint32_t size) -> uint64_t {
        cursor = (cursor + size - 1) / size * size;   // natural alignment in the buffer
        if (args == 0)
          throw Error(Err::InvalidValue, "printf format consumes arguments but no argument buffer was passed");
        uint64_t off, v = 0;
        const Space sp = classify(args + cursor, &off);
        mem_read(blk, w, l, sp, off, &v, size);
        cursor += size;
        return v;
      };
      const std::string out = exec::format_device_printf(
          cstring(fmt), fetch, cstring, [](Err e, const char* why) { throw Error(e, why); });
      exec::emit_device_printf(out);
      w.reg(4, l) = static_cast<uint32_t>(out.size());
    });
    return;
  }
  // The device runtime's per-thread error, through the two driver entry
  // points its library (linked in by -rdc builds) is built on, as the PTX
  // engine's exec_device_get_last_error: GetLastError reads it without
  // clearing it, SetLastError writes it.
  if (name == "__cuda_syscall_cnpv2GetLastError") {
    each_lane(ex, [&](unsigned l) { w.reg(4, l) = w.device_error[l]; });
    return;
  }
  if (name == "__cuda_syscall_cnpv2SetLastError") {
    each_lane(ex, [&](unsigned l) {
      w.device_error[l] = w.reg(4, l);
      w.reg(4, l) = 0;
    });
    return;
  }
  if (name == "cudaGraphLaunch") {
    // (graphExec, stream) from a kernel, as the PTX engine's
    // exec_device_graph_launch: the runtime that launched this kernel's graph
    // checks and queues it, and a launch from outside any graph, or of a
    // handle that is no device graph, faults as on an RTX 3060.
    each_lane(ex, [&](unsigned l) {
      const uint64_t g = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
      const uint64_t s = w.reg(6, l) | (static_cast<uint64_t>(w.reg(7, l)) << 32);
      int rc = -1;
      if (cfg_.graph_launcher) rc = cfg_.graph_launcher->launch(g, s);
      if (rc < 0) {
        fault(w, ins, l,
              cfg_.graph_launcher ? std::string("device-side cudaGraphLaunch of a handle that is not a graph "
                                                "instantiated for device launch")
                                  : std::string("device-side cudaGraphLaunch in a kernel that is not running in a graph"),
              Err::InvalidPointer);
        return;
      }
      w.reg(4, l) = static_cast<uint32_t>(rc);
    });
    return;
  }
  if (name == "cudaGraphSetConditional") {
    // (handle, value): sets a conditional handle's value, which the graph's
    // conditional node reads once the kernel is done -- as the PTX engine's
    // exec_graph_set_conditional does. Outside a graph, or with a handle no
    // graph made, an RTX 3060 faults with an illegal address; racing calls
    // are undefined, and here the last lane to run wins.
    each_lane(ex, [&](unsigned l) {
      const uint64_t h = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
      if (exec::GraphConditionals* table = cfg_.conditionals) {
        std::lock_guard<std::mutex> guard(table->mu);
        if (const auto it = table->values.find(h); it != table->values.end()) {
          it->second = w.reg(6, l);
          return;
        }
      }
      fault(w, ins, l,
            cfg_.conditionals ? "cudaGraphSetConditional with a handle no graph created (" + std::to_string(h) + ")"
                              : std::string("cudaGraphSetConditional in a kernel that is not running in a graph"),
            Err::InvalidPointer);
    });
    return;
  }
  if (name == "malloc" || name == "free") {
    // The device's heap, shared with the PTX engine (MemoryManager::
    // heap_alloc): each thread allocates on its own, and out of heap is null.
    each_lane(ex, [&](unsigned l) {
      const uint64_t arg = w.reg(4, l) | (static_cast<uint64_t>(w.reg(5, l)) << 32);
      if (name == "malloc") {
        const uint64_t p = mem_.heap_alloc(arg, cfg_.device_heap_bytes);
        w.reg(4, l) = static_cast<uint32_t>(p);
        w.reg(5, l) = static_cast<uint32_t>(p >> 32);
      } else if (arg && !mem_.heap_free(arg)) {   // free(nullptr) is a no-op
        fault(w, ins, l, "device free() of a pointer the device heap did not allocate, or already freed",
              Err::InvalidFree);
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
  fault(w, ins, static_cast<unsigned>(std::countr_zero(ex)),
        "call to " + name + ", which the module does not define and VirtualGPU does not provide",
        Err::UnsupportedPtx);
}

const size_t* Runner::kernel_at(uint64_t code_addr) {
  std::call_once(kernel_by_code_once_, [&] {
    for (size_t i = 0; i < m_.cubin.kernels.size(); ++i)
      if (const auto c = m_.code_index.find(m_.cubin.kernels[i].text_section); c != m_.code_index.end())
        kernel_by_code_[m_.code[c->second].base] = i;
  });
  const auto it = kernel_by_code_.find(code_addr);
  return it == kernel_by_code_.end() ? nullptr : &it->second;
}

// The device runtime's public entry points (module.cpp, kBuiltins), with
// the arguments where the CUDA ABI puts them -- a 64-bit one in a register
// pair, a dim3 in three registers -- as ptxas passes them at the call sites
// it compiles (cudaGetParameterBufferV2: the kernel in R4:R5, the grid in
// R6-R8, the block in R9-R11, the shared bytes in R12). The same behaviour as
// the PTX engine's (src/exec/interpreter.cpp, exec_get_parameter_buffer and
// the rest). False for any other name.
bool Runner::device_runtime_call(Block& blk, Warp& w, const Instr& ins, Mask ex, const std::string& name) {
  const auto arg64 = [&](unsigned r, unsigned l) {
    return w.reg(r, l) | (static_cast<uint64_t>(w.reg(r + 1, l)) << 32);
  };
  const auto ret = [&](unsigned l, uint32_t err) {   // a cudaError_t, kept when it is an error
    w.reg(4, l) = err;
    if (err) w.device_error[l] = err;
  };
  const auto store = [&](unsigned l, uint64_t addr, uint64_t v, uint32_t n) {
    // Usually a local variable's address (int d; cudaGetDevice(&d)): a
    // generic store.
    uint64_t off;
    const Space sp = classify(addr, &off);
    mem_write(blk, w, l, sp, off, &v, n);
  };
  constexpr uint32_t kSuccess = 0, kInvalidValue = 1, kInvalidConfiguration = 9;
  if (name == "__cudaCDP2GetParameterBufferV2" || name == "cudaGetParameterBufferV2") {
    // (func, gridDim, blockDim, sharedMem): a buffer in device memory, laid
    // out as the kernel's parameters are, for the calling thread to fill.
    // Null once too many launches are pending, as on hardware.
    if (!dl_)
      fault(w, ins, static_cast<unsigned>(std::countr_zero(ex)), "a device-side launch outside a launch that can run it",
            Err::Unsupported);
    each_lane(ex, [&](unsigned l) {
      const uint64_t f = arg64(4, l);
      const size_t* k = kernel_at(f);
      if (!k) {
        char b[24];
        std::snprintf(b, sizeof b, "%llx", static_cast<unsigned long long>(f));
        fault(w, ins, l, std::string("a device-side launch names 0x") + b + ", which is not a kernel of this module",
              Err::InvalidValue);
      }
      Child c;
      c.kernel = *k;
      for (unsigned i = 0; i < 3; ++i) {
        c.grid[i] = w.reg(6 + i, l);
        c.block[i] = w.reg(9 + i, l);
      }
      c.shared = w.reg(12, l);
      c.size = param_bytes(m_.cubin.kernels[*k]);
      uint64_t buf = 0;
      {
        std::lock_guard<std::mutex> guard(dl_->mu);
        if (dl_->by_buffer.size() + dl_->queue.size() < kMaxPendingLaunches) {
          buf = c.buffer = mem_.alloc(std::max<uint64_t>(c.size, 16));
          dl_->by_buffer.emplace(buf, c);
        }
      }
      w.reg(4, l) = static_cast<uint32_t>(buf);
      w.reg(5, l) = static_cast<uint32_t>(buf >> 32);
    });
    return true;
  }
  if (name == "__cudaCDP2LaunchDeviceV2" || name == "__cudaCDP2LaunchDeviceV2_ptsz" || name == "cudaLaunchDeviceV2" ||
      name == "cudaLaunchDeviceV2_ptsz") {
    // (parameterBuffer, stream): the child is queued to run after this
    // grid. The stream does not order it: every device-side stream's
    // launches may run then (cudaStreamTailLaunch must), and children run
    // in the order issued.
    if (!dl_)
      fault(w, ins, static_cast<unsigned>(std::countr_zero(ex)), "a device-side launch outside a launch that can run it",
            Err::Unsupported);
    const uint64_t block_linear =
        (uint64_t{blk.ctaid[2]} * cfg_.grid[1] + blk.ctaid[1]) * cfg_.grid[0] + blk.ctaid[0];
    each_lane(ex, [&](unsigned l) {
      const uint64_t b = arg64(4, l);
      std::lock_guard<std::mutex> guard(dl_->mu);
      const auto it = dl_->by_buffer.find(b);
      if (it == dl_->by_buffer.end()) {
        ret(l, kInvalidValue);   // not a buffer cudaGetParameterBuffer handed out
        return;
      }
      Child c = it->second;
      dl_->by_buffer.erase(it);
      const uint64_t threads = uint64_t{c.block[0]} * c.block[1] * c.block[2];
      if (!c.grid[0] || !c.grid[1] || !c.grid[2] || !threads ||
          (profile_.limits.max_threads_per_block && threads > profile_.limits.max_threads_per_block)) {
        mem_.free(c.buffer);
        ret(l, kInvalidConfiguration);
        return;
      }
      c.order = (block_linear << 24) | (dl_->per_block[block_linear]++ & 0xFFFFFF);
      dl_->queue.push_back(c);
      ret(l, kSuccess);
    });
    return true;
  }
  if (name == "__cudaCDP2GetLastError" || name == "__cudaCDP2PeekAtLastError") {
    const bool clear = name == "__cudaCDP2GetLastError";
    each_lane(ex, [&](unsigned l) {
      w.reg(4, l) = w.device_error[l];
      if (clear) w.device_error[l] = 0;
    });
    return true;
  }
  if (name == "__cudaCDP2GetDevice" || name == "__cudaCDP2GetDeviceCount") {
    // (int*): the answer the host gets, written where the pointer points.
    const int v = name == "__cudaCDP2GetDevice" ? cfg_.device_ordinal : cfg_.device_count;
    each_lane(ex, [&](unsigned l) {
      const uint64_t p = arg64(4, l);
      if (!p) return ret(l, kInvalidValue);
      store(l, p, static_cast<uint32_t>(v), 4);
      ret(l, kSuccess);
    });
    return true;
  }
  if (name == "__cudaCDP2StreamCreateWithFlags" || name == "__cudaCDP2EventCreateWithFlags") {
    // (handle*, flags): a non-null handle, distinct per call; the flags are
    // accepted as they come (child grids run in launch order, which every
    // ordering these can ask for already satisfies).
    static std::atomic<uint64_t> next{1};
    constexpr uint64_t kHandleBase = 0x5654'4750'0000'0000ull;   // "VTGP", as the PTX engine's: never an address
    each_lane(ex, [&](unsigned l) {
      const uint64_t p = arg64(4, l);
      if (!p) return ret(l, kInvalidValue);
      store(l, p, kHandleBase + next.fetch_add(1, std::memory_order_relaxed), 8);
      ret(l, kSuccess);
    });
    return true;
  }
  if (name == "__cudaCDP2StreamDestroy" || name == "__cudaCDP2EventDestroy" || name == "__cudaCDP2EventRecord" ||
      name == "__cudaCDP2EventRecord_ptsz" || name == "__cudaCDP2EventRecordWithFlags" ||
      name == "__cudaCDP2EventRecordWithFlags_ptsz" || name == "__cudaCDP2StreamWaitEvent" ||
      name == "__cudaCDP2StreamWaitEvent_ptsz") {
    // Nothing left to do: children already run one after another.
    each_lane(ex, [&](unsigned l) { ret(l, kSuccess); });
    return true;
  }
  return false;
}

// The cluster barrier completes its phase when every thread of the cluster
// that has not exited has arrived (barrier.cluster; exited threads count as
// arrived), and releases the lanes waiting for it.
void Runner::cluster_barrier_check(Block& blk) {
  Cluster& cl = *blk.cluster;
  bool any = false;
  for (Block* b : cl.blocks)
    for (const Warp& w : b->warps) {
      if ((w.alive & ~w.exited) & ~w.cga_arrived) return;
      any = any || w.cga_arrived;
    }
  if (!any) return;
  ++cl.phase;
  for (Block* b : cl.blocks)
    for (Warp& w : b->warps) {
      w.cga_arrived = 0;
      for (unsigned l = 0; l < 32; ++l)
        if ((w.waiting >> l & 1) && w.wait_kind[l] == Wait::Cluster && w.cga_target[l] <= cl.phase) {
          w.waiting &= ~(Mask{1} << l);
          w.wait_kind[l] = Wait::None;
          w.pc[l] += 16;
        }
    }
}

// ---- mbarriers (SYNCS, ATOMS.ARRIVE, ARRIVES) ----------------------------------
//
// From sm_90 the mbarrier is the 64-bit word ptxas lays out for SYNCS
// (decode.cpp, dec_syncs_*): the low word holds 0x100000 - count at bits
// 1-20, the high word counts the phase's arrivals up from that value at bits
// 11-30 and holds the phase at bit 31, so the arrival completing the phase
// carries into it. Kernels do arithmetic on the words they get back
// (pending_count, test_wait's token), so the word is kept exactly; what it has
// no room for is in Block::mbar. Each lane's operation is its own atomic,
// applied in lane order.

// The barrier's shared offset from a lane's [Ra + URb + offset] (Ra at 24,
// URb at `ur_at`) or, with ra_at < 0, [URa + offset] (URa at 24). The CTA's
// rank in its cluster rides in bits 24 up.
uint32_t Runner::mbar_offset(Block& blk, Warp& w, const Instr& ins, unsigned l, bool ur_only, Block** owner) {
  const Word& word = ins.w;
  const unsigned ub = m_.sm >= 100 ? 8 : 6, urz = m_.sm >= 100 ? 255u : 63u;
  const auto ureg = [&](unsigned u) -> uint64_t { return u >= urz ? 0 : w.ur[u]; };
  uint64_t a;
  if (ur_only) {
    a = ureg(static_cast<unsigned>(word.field(24, ub)));
  } else {
    const unsigned ra = static_cast<unsigned>(word.field(24, 8));
    a = (ra == kRZ ? 0 : w.reg(ra, l)) + ureg(static_cast<unsigned>(word.field(64, ub)));
  }
  a += static_cast<uint64_t>(word.sfield(40, 24));
  uint32_t off;
  Block& b = shared_block(blk, static_cast<uint32_t>(a), &off);
  // Arrivals may be on another block's barrier (mbarrier.arrive's
  // .shared::cluster); initializing, waiting and invalidating are for a
  // block's own.
  if (!owner && &b != &blk) throw Fault("an mbarrier of another block of the cluster, used by other than an arrive");
  if (owner) *owner = &b;
  if (off % 8) throw Fault("an mbarrier must be 8-byte aligned");
  if (off + 8 > b.shared.size()) throw Fault("an mbarrier past the block's shared memory");
  return off;
}

// One thread's arrive on an sm_90 mbarrier: `arrivals` arrivals, `tx` more
// transaction bytes due (negative: completed), `drop` leaving later phases.
// Returns the word before.
uint64_t Runner::mbar_arrive(Block& blk, uint32_t off, uint32_t arrivals, int64_t tx, bool drop) {
  uint64_t old;
  std::memcpy(&old, &blk.shared[off], 8);
  Block::MbarSide& side = blk.mbar[off];
  uint32_t lo = static_cast<uint32_t>(old), hi = static_cast<uint32_t>(old >> 32);
  uint32_t reload = (lo >> 1) & 0xfffff;   // 0x100000 - the expected count
  if (drop) reload = std::min<uint32_t>(reload + arrivals, 0x100000);   // fewer from the next phase on
  uint32_t remaining = side.arrived_all ? 0 : 0x100000 - ((hi >> 11) & 0xfffff);
  remaining -= std::min(arrivals, remaining);
  side.tx += tx;
  uint32_t phase = hi >> 31, count;
  if (remaining == 0 && side.tx == 0) {
    phase ^= 1;
    count = reload;
    side.arrived_all = false;
  } else {
    side.arrived_all = remaining == 0;
    count = remaining == 0 ? 0 : 0x100000 - remaining;
  }
  lo = (lo & ~(0xfffffu << 1)) | (reload << 1);
  hi = (hi & 0x7ffu) | (count << 11) | (phase << 31);
  const uint64_t now = lo | (static_cast<uint64_t>(hi) << 32);
  std::memcpy(&blk.shared[off], &now, 8);
  return old;
}

// One thread's arrive on an sm_80 mbarrier (ATOMS.ARRIVE.64): the high word
// counts arrivals up to 0x80000000, carrying into the phase at bit 31, and
// then starts again from the low word + 0x40000000 (init stores 0x40000000
// - count there, and arrive_drop raises it). Returns the word before.
uint64_t Runner::mbar_arrive80(Block& blk, uint32_t off) {
  uint64_t old;
  std::memcpy(&old, &blk.shared[off], 8);
  const uint32_t lo = static_cast<uint32_t>(old);
  uint32_t hi = static_cast<uint32_t>(old >> 32);
  const uint32_t count = (hi & 0x7fffffffu) + 1;
  if (count >= 0x80000000u) hi = ((hi ^ 0x80000000u) & 0x80000000u) | ((lo + 0x40000000u) & 0x7fffffffu);
  else hi = (hi & 0x80000000u) | count;
  const uint64_t now = lo | (static_cast<uint64_t>(hi) << 32);
  std::memcpy(&blk.shared[off], &now, 8);
  return old;
}

void Runner::exec_syncs(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  const unsigned urz = m_.sm >= 100 ? 255u : 63u;
  const auto ureg = [&](unsigned u) -> uint64_t { return u >= urz ? 0 : w.ur[u]; };
  const auto put_pair = [&](unsigned rd, unsigned l, uint64_t v) {
    if (rd == kRZ) return;
    w.reg(rd, l) = static_cast<uint32_t>(v);
    w.reg(rd + 1, l) = static_cast<uint32_t>(v >> 32);
  };
  if (ins.op == Op::ATOMS) {   // ATOMS.ARRIVE.64 Rd, [addr]: sm_80's arrive
    each_lane(ex, [&](unsigned l) { put_pair(ins.dst[0].reg, l, mbar_arrive80(blk, mbar_offset(blk, w, ins, l, false, nullptr))); });
    return;
  }
  if (ins.op == Op::ARRIVES) {   // cp.async's arrive; the copies have landed already (LDGSTS)
    each_lane(ex, [&](unsigned l) {
      const uint32_t off = mbar_offset(blk, w, ins, l, false, nullptr);
      if (ins.f[0] == 0) mbar_arrive80(blk, off);
      else if (ins.f[0] == 1) mbar_arrive(blk, off, 0, -1, false);   // .TRANSCNT: A0T1's transaction
      else mbar_arrive(blk, off, 1, 0, false);                        // .ARVCNT
    });
    return;
  }
  switch (ins.f[0]) {
    case 0: {   // EXCH.64 URd, [URa + offset], URb: mbarrier.init
      if (!ex) return;
      const unsigned d = ins.dst[0].reg, b = ins.src[1].reg;
      const uint32_t off = mbar_offset(blk, w, ins, static_cast<unsigned>(std::countr_zero(ex)), true, nullptr);
      uint64_t old;
      std::memcpy(&old, &blk.shared[off], 8);
      const uint64_t v = ureg(b) | (ureg(b + 1) << 32);
      std::memcpy(&blk.shared[off], &v, 8);
      blk.mbar.erase(off);
      if (d < urz) {
        w.ur[d] = static_cast<uint32_t>(old);
        w.ur[d + 1] = static_cast<uint32_t>(old >> 32);
      }
      return;
    }
    case 1: {   // ARRIVE: see dec_syncs_arrive for the modes
      const unsigned mode = ins.f[1];
      const bool drop = ins.f[2];
      const unsigned rb = ins.src[1].reg;
      each_lane(ex, [&](unsigned l) {
        const uint32_t b = rb == kRZ ? 0 : w.reg(rb, l);
        uint32_t arrivals = 0;
        int64_t tx = 0;
        switch (mode) {
          case 0: arrivals = 1; tx = b; break;              // arrive once, expect b bytes
          case 1: arrivals = 1; break;                      // .A1T0
          case 2: tx = 1; break;                            // .A0T1: one more due (cp.async's)
          case 3: tx = b; break;                            // .A0TR: expect b bytes
          case 4: tx = -static_cast<int64_t>(b); break;     // .A0TX: b bytes complete
          case 5: arrivals = b; break;                      // .ART0: arrive b times
          default: throw Fault("SYNCS.ARRIVE mode " + std::to_string(mode) + " is not implemented");
        }
        Block* at = nullptr;
        const uint32_t off = mbar_offset(blk, w, ins, l, false, &at);
        put_pair(ins.dst[0].reg, l, mbar_arrive(*at, off, arrivals, tx, drop));
      });
      return;
    }
    case 2: {   // PHASECHK Pd, [addr], Rb: has the phase of Rb's bit 31 completed?
      const unsigned rb = ins.src[1].reg;
      bool all = true;
      each_lane(ex, [&](unsigned l) {
        const uint32_t off = mbar_offset(blk, w, ins, l, false, nullptr);
        uint64_t v;
        std::memcpy(&v, &blk.shared[off], 8);
        const bool done = static_cast<uint32_t>(v >> 63) != ((rb == kRZ ? 0 : w.reg(rb, l)) >> 31);
        set_pred(w, ins.dst[0], l, done);
        all = all && done;
      });
      if (!all) w.yield = true;   // the arrivals it waits for are other warps'
      return;
    }
    case 3:   // CCTL.IV / .IVALL: mbarrier.inval
      if (ins.src.empty()) {
        if (ex) blk.mbar.clear();
      } else {
        each_lane(ex, [&](unsigned l) { blk.mbar.erase(mbar_offset(blk, w, ins, l, false, nullptr)); });
      }
      return;
    default:
      throw Fault("SYNCS form not implemented");
  }
}

// ---- the tensor memory accelerator (UTMALDG, UTMASTG, UTMAREDG, UBLKCP, UBLKRED) --
//
// Each is issued once for the warp (its operands are uniform registers, and
// ptxas elects one thread to issue it). A copy happens when issued: a load's
// data is in shared memory and its bytes complete on its mbarrier right
// away, which a kernel that waits on the barrier cannot tell from later; a
// store or reduction has reached global memory before the kernel's bulk-group
// wait (DEPBAR.LE SB0) looks. The element walk is the PTX engine's
// (exec/tma.hpp). Operands, from decode.cpp: [URa] at 32 -- a load's shared
// destination, its mbarrier at URa+1 and the coordinates from URa+2; a store's
// shared source and the coordinates from URa+1 -- and [URb] at 24, the tensor
// map's address. The uniform register at 64 holds the im2col offsets (low 16
// bits: one 16-bit offset in 3D, two 8-bit ones in 4D, three 5-bit ones in
// 5D) and the multicast CTA mask (high 16 bits).
namespace {
ptx::AtomOp sass_atom_op(uint32_t op) {
  static const ptx::AtomOp ops[] = {ptx::AtomOp::Add, ptx::AtomOp::Min, ptx::AtomOp::Max, ptx::AtomOp::Inc,
                                    ptx::AtomOp::Dec, ptx::AtomOp::And, ptx::AtomOp::Or,  ptx::AtomOp::Xor};
  return ops[op & 7];
}
}  // namespace


// Block `rank` of blk's cluster.
Block& Runner::cluster_block(Block& blk, uint32_t rank) {
  uint32_t off;
  return shared_block(blk, uint64_t{rank} << 24, &off);
}

void Runner::exec_tma(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  if (!ex) return;
  const Word& word = ins.w;
  const unsigned ub = m_.sm >= 100 ? 8 : 6, urz = m_.sm >= 100 ? 255u : 63u;
  const auto urv = [&](unsigned u) -> uint32_t { return u >= urz ? 0 : w.ur[u]; };
  const auto ur64 = [&](unsigned u) -> uint64_t { return urv(u) | (static_cast<uint64_t>(urv(u + 1)) << 32); };
  const unsigned ura = static_cast<unsigned>(word.field(32, ub)), urb = static_cast<unsigned>(word.field(24, ub)),
                 urc = static_cast<unsigned>(word.field(64, ub));
  const unsigned lane = static_cast<unsigned>(std::countr_zero(ex));
  const auto gread = [&](uint64_t a, void* out, uint32_t n) { mem_read(blk, w, lane, Space::Global, a, out, n); };
  const auto gwrite = [&](uint64_t a, const void* in, uint32_t n) { mem_write(blk, w, lane, Space::Global, a, in, n); };
  const auto greduce = [&](uint64_t a, const uint8_t* src, uint32_t es, const ptx::Type& ty, ptx::AtomOp op) {
    uint64_t b = 0;
    std::memcpy(&b, src, es);
    std::lock_guard<std::mutex> g(atomic_lock(a));
    const uint64_t old = mem_.load_scalar(a, es);
    mem_.store_scalar(a, es, exec::reduce_value(op, ty, old, b));
    ++blk.st.atomics;
    blk.st.atomic_bytes += es;
  };
  const auto shared_range = [&](Block& b, uint32_t off, uint64_t n, const char* what) {
    if (off + n > b.shared.size())
      throw Fault(std::string(what) + " reaches shared offset " + std::to_string(off + n) + ", past the block's " +
                  std::to_string(b.shared.size()) + " bytes");
  };

  if (ins.op == Op::UBLKCP || ins.op == Op::UBLKRED) {
    const unsigned dir = ins.f[0];   // 1 .S.G, 2 .G.S, 3 .S.S
    // The size in 16-byte units in the low half of its register, and a
    // multicast's CTA mask in the high half: ptxas shifts a byte count right
    // by 4 and PRMTs the mask in (my own probes, every direction).
    const uint32_t size_reg = urv(urc);
    const uint32_t n = (size_reg & 0xffff) * 16;
    std::vector<uint8_t> data(n);
    if (dir == 1) {
      const uint64_t g = ur64(urb);
      if (g % 16) throw Fault("a bulk copy's global address must be 16-byte aligned");
      gread(g, data.data(), n);
      blk.st.global_loads += 1;
      blk.st.global_bytes_read += n;
    } else {
      uint32_t off;
      Block& src = shared_block(blk, urv(urb), &off);
      if (&src != &blk) throw Fault("a bulk copy's source must be this block's shared memory");
      shared_range(src, off, n, "a bulk copy's source");
      std::memcpy(data.data(), &src.shared[off], n);
    }
    if (dir == 2) {   // to global
      const uint64_t g = ur64(ura);
      if (g % 16) throw Fault("a bulk copy's global address must be 16-byte aligned");
      if (ins.op == Op::UBLKCP) {
        gwrite(g, data.data(), n);
        blk.st.global_stores += 1;
        blk.st.global_bytes_written += n;
      } else {
        static const ptx::Type types[16] = {{ptx::Type::Kind::U, 32}, {ptx::Type::Kind::S, 32}, {ptx::Type::Kind::U, 64},
                                            {ptx::Type::Kind::S, 64}, {ptx::Type::Kind::F, 16}, {ptx::Type::Kind::F, 32},
                                            {},                       {ptx::Type::Kind::F, 64}, {ptx::Type::Kind::BF, 16}};
        const ptx::Type& ty = types[ins.f[3] & 15];
        if (!ty.bits) throw Fault("UBLKRED type " + std::to_string(ins.f[3]) + " is not known");
        for (uint32_t o = 0; o < n; o += ty.bytes()) greduce(g + o, data.data() + o, ty.bytes(), ty, sass_atom_op(ins.f[2]));
      }
      return;
    }
    // Into shared memory, completing on the destination's mbarrier.
    const uint32_t dst_addr = urv(ura), bar_addr = urv(ura + 1);
    const uint32_t mask = ins.f[1] ? size_reg >> 16 : 0;
    const auto deliver = [&](Block& to, uint32_t off, uint32_t bar_off) {
      shared_range(to, off, n, "a bulk copy");
      if (ins.op == Op::UBLKRED) {   // .S.S: an integer reduction into shared memory
        static const ptx::Type types[3] = {{ptx::Type::Kind::U, 32}, {ptx::Type::Kind::S, 32}, {ptx::Type::Kind::U, 64}};
        if (ins.f[3] > 2) throw Fault("UBLKRED.S.S type " + std::to_string(ins.f[3]) + " is not known");
        const ptx::Type& ty = types[ins.f[3]];
        for (uint32_t o = 0; o < n; o += ty.bytes()) {
          uint64_t old = 0, b = 0;
          std::memcpy(&old, &to.shared[off + o], ty.bytes());
          std::memcpy(&b, &data[o], ty.bytes());
          const uint64_t r = exec::reduce_value(sass_atom_op(ins.f[2]), ty, old, b);
          std::memcpy(&to.shared[off + o], &r, ty.bytes());
        }
      } else {
        std::memcpy(&to.shared[off], data.data(), n);
      }
      mbar_arrive(to, bar_off, 0, -static_cast<int64_t>(n), false);
    };
    uint32_t off, bar_off;
    Block& to = shared_block(blk, dst_addr, &off);
    Block& bar = shared_block(blk, bar_addr, &bar_off);
    if (&bar != &to) throw Fault("a bulk copy's mbarrier must be in the block its data goes to");
    if (off % 16) throw Fault("a bulk copy's shared address must be 16-byte aligned");
    if (!ins.f[1]) {
      deliver(to, off, bar_off);
    } else {
      // Multicast: the same data at the same offset into every block the
      // mask names, each completing on its own barrier at the barrier's
      // offset.
      for (uint32_t r = 0; r < 16; ++r)
        if (mask >> r & 1) deliver(cluster_block(blk, r), off, bar_off);
    }
    return;
  }

  // Tensor copies.
  const bool load = ins.op == Op::UTMALDG, reduce = ins.op == Op::UTMAREDG;
  const uint32_t dims = ins.f[0];
  const bool im2col = ins.f[1], multicast = ins.f[2], four_rows = ins.f[4], pair = ins.f[5];
  if (ins.f[6]) throw Error(Err::UnsupportedPtx, "im2col::w tensor copies are not implemented");
  const uint64_t tmap_at = ur64(urb);
  if (tmap_at % 64) throw Error(Err::MisalignedAccess, "a tensor map must be 64-byte aligned");
  uint64_t q[16];
  {
    uint64_t off;
    const Space sp = classify(tmap_at, &off);
    if (sp != Space::Global && sp != Space::Param) throw Fault("a tensor map in shared or local memory");
    mem_read(blk, w, lane, sp, off, q, sizeof q);
  }
  exec::TensorMap map;
  if (!map.decode(q))
    throw Error(Err::InvalidValue,
                "the tensor map this copy names was not made by cuTensorMapEncodeTiled (or was overwritten); its first "
                "bytes are not the encoder's");
  // The coordinates; .tile::gather4/scatter4's are the column and four rows.
  int64_t coords[5] = {};
  const unsigned first = load ? ura + 2 : ura + 1;
  for (uint32_t d = 0; d < (four_rows ? 5u : dims) && d < 5; ++d) coords[d] = static_cast<int32_t>(urv(first + d));
  exec::TmaBox box;
  if (auto bad = exec::tma_box(map, dims, coords, four_rows, im2col, load, reduce, &box)) throw Error(bad->code, bad->what);
  // A load's im2col offsets (or multicast mask); a store's im2col form has
  // none (.im2col_no_offs).
  const uint32_t extra = ((im2col && load) || multicast) ? urv(urc) : 0;
  std::array<int64_t, 3> offsets{};
  if (im2col) {
    const uint32_t bits = exec::TensorMap::corner_bits(map.rank);
    for (uint32_t i = 0; i + 2 < map.rank && i < 3; ++i) offsets[i] = (extra >> (bits * i)) & ((1u << bits) - 1);
  }
  const uint32_t swz = exec::TensorMap::swizzle_bytes(map.swizzle), atom = exec::TensorMap::swizzle_atom(map.swizzle);
  uint32_t smem;
  Block& home = shared_block(blk, urv(ura), &smem);
  if (smem % 16) throw Error(Err::MisalignedAccess, "a bulk copy's shared-memory address must be 16-byte aligned");
  if (load) {
    // The box, read from global memory into its shared-memory layout.
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> runs;   // (shared offset, bytes)
    runs.reserve(box.total);
    exec::tma_walk(map, box, coords + 1, offsets, four_rows, [&](uint64_t e, uint64_t gaddr, bool inside) {
      const uint32_t soff = static_cast<uint32_t>(exec::swizzle_address(smem + e * box.es, swz, atom));
      std::vector<uint8_t> v(box.packed ? box.gs : box.es, 0);
      if (inside) gread(gaddr, v.data(), static_cast<uint32_t>(v.size()));
      else if (map.oob_nan && !box.packed)
        throw Error(Err::UnsupportedPtx,
                    "the box crosses the tensor's edge and the map asks for the NaN out-of-bounds fill, whose value is "
                    "not documented; only zero fill is implemented");
      runs.emplace_back(soff, std::move(v));
    });
    blk.st.global_loads += 1;
    blk.st.global_bytes_read += box.total * box.gs;
    // The barrier counts every byte of the box, the zero-filled ones too (for
    // the packed types, the packed bytes).
    uint32_t bar_off;
    Block& bar = shared_block(blk, urv(ura + 1), &bar_off);
    // .2CTA lets the barrier be in the destination's peer CTA instead, which
    // CUTLASS's 2-SM kernels use to count both CTAs' tiles on the leader's.
    if (&bar != &home && !(pair && (bar.rank ^ 1) == home.rank))
      throw Fault(pair ? "a .2CTA tensor copy's mbarrier must be in the block its data goes to or that block's peer"
                       : "a tensor copy's mbarrier must be in the block its data goes to");
    const auto deliver = [&](Block& to) {
      for (const auto& [soff, v] : runs) {
        shared_range(to, soff, v.size(), "a tensor copy");
        std::memcpy(&to.shared[soff], v.data(), v.size());
      }
      // Each destination's signal goes to its own barrier, or with .2CTA to
      // whichever CTA of its pair has the barrier's rank parity.
      Block& b = pair ? cluster_block(blk, (to.rank & ~1u) | (bar.rank & 1u)) : to;
      mbar_arrive(b, bar_off, 0, -static_cast<int64_t>(box.total * box.gs), false);
    };
    if (!multicast) {
      deliver(home);
    } else {
      // The same box at the same offset into every block the mask names,
      // each completing on its own barrier at the barrier's offset.
      const uint32_t mask = extra >> 16;
      if (!mask) throw Error(Err::InvalidValue, "a multicast tensor copy whose CTA mask names no block");
      for (uint32_t r = 0; r < 16; ++r)
        if (mask >> r & 1) deliver(cluster_block(blk, r));
    }
    return;
  }
  // A store or reduction: the elements inside the tensor, from this block's
  // shared memory.
  if (&home != &blk) throw Fault("a tensor store's source must be this block's shared memory");
  std::optional<ptx::Type> red_ty;
  if (reduce) {
    red_ty = exec::tensor_reduce_type(map.type, sass_atom_op(ins.f[3]));
    if (!red_ty)
      throw Error(Err::UnsupportedPtx,
                  "a tensor reduction with no form of this operation for the tensor map's element type (9.7.10.28.5.4)");
  }
  uint64_t stored = 0;
  exec::tma_walk(map, box, coords + 1, offsets, four_rows, [&](uint64_t e, uint64_t gaddr, bool inside) {
    if (!inside) return;
    const uint32_t soff = static_cast<uint32_t>(exec::swizzle_address(smem + e * box.es, swz, atom));
    shared_range(blk, soff, box.es, "a tensor store");
    const uint8_t* src = &blk.shared[soff];
    if (box.packed) {
      uint8_t out[16] = {};
      if (map.type == exec::TmapType::U6x16Align16) {
        for (int i = 0; i < 16; ++i)
          for (int b = 0; b < 6; ++b)
            if (src[i] >> b & 1) out[(6 * i + b) / 8] |= static_cast<uint8_t>(1u << ((6 * i + b) % 8));
      } else {
        std::memcpy(out, src, box.gs);
      }
      gwrite(gaddr, out, box.gs);
    } else if (reduce) {
      greduce(gaddr, src, box.es, *red_ty, sass_atom_op(ins.f[3]));
    } else {
      gwrite(gaddr, src, box.es);
    }
    stored += box.gs;
  });
  if (!reduce) {
    blk.st.global_stores += 1;
    blk.st.global_bytes_written += stored;
  }
}

// ---- Blackwell's tensor core (tcgen05: LDTM, STTM, UTC*) ------------------------
//
// The operations themselves are the PTX engine's (exec/tcgen05.hpp); this
// reads their operands from the uniform registers and gives them the CTAs
// they work on -- this one, or with .2CTA the pair of the cluster whose
// ranks differ in the last bit, even first. The UTC* instructions are the
// uniform datapath's: issued once for the warp, however many lanes are
// active (for one per thread, ptxas loops over the threads with ELECT).
namespace {
class SassTcHost final : public exec::Tcgen05Host {
 public:
  SassTcHost(std::vector<Block*> cs, Block* me) : cs_(std::move(cs)) {
    for (uint32_t i = 0; i < cs_.size(); ++i)
      if (cs_[i] == me) me_ = i;
  }
  uint32_t ctas() const override { return static_cast<uint32_t>(cs_.size()); }
  uint32_t self() const override { return me_; }
  exec::TensorMemory& tmem(uint32_t c) override { return cs_[c]->tmem; }
  uint64_t smem_load(uint32_t c, uint64_t off, uint32_t bytes) override {
    const std::vector<uint8_t>& sh = cs_[c]->shared;
    if (off + bytes > sh.size())
      throw Fault("a tensor-core operand reaches shared offset " + std::to_string(off + bytes) + ", past the block's " +
                  std::to_string(sh.size()) + " bytes");
    uint64_t v = 0;
    std::memcpy(&v, &sh[off], bytes);
    return v;
  }
  [[noreturn]] void fail(Err code, const std::string& why) override { throw Error(code, why); }

 private:
  std::vector<Block*> cs_;
  uint32_t me_ = 0;
};
}  // namespace

void Runner::exec_tcgen05(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  if (!ex) return;
  const Word& word = ins.w;
  const unsigned urz = 255;
  const auto urv = [&](unsigned u) -> uint32_t { return u >= urz ? 0 : w.ur[u]; };
  const auto ur64 = [&](unsigned u) -> uint64_t { return urv(u) | (static_cast<uint64_t>(urv(u + 1)) << 32); };
  const auto u = [&](unsigned pos) { return static_cast<unsigned>(word.field(pos, 8)); };
  // The CTAs of an operation of .cta_group::n.
  const auto group = [&](bool two) -> std::vector<Block*> {
    if (!two) return {&blk};
    const Cluster& cl = *blk.cluster;
    if ((blk.rank | 1) >= cl.blocks.size())
      throw Error(Err::InvalidValue, ".2CTA works on a CTA pair -- two CTAs of a cluster whose ranks differ in the "
                                     "last bit -- and this CTA (rank " + std::to_string(blk.rank) + " of a " +
                                         std::to_string(cl.blocks.size()) + "-CTA cluster) has no peer");
    return {cl.blocks[blk.rank & ~1u], cl.blocks[blk.rank | 1u]};
  };
  switch (ins.op) {
    case Op::UTCATOMSWS: {
      // tcgen05.alloc/dealloc's allocator: a map of 16 granules of 32
      // columns. FIND_AND_SET.ALIGN UPd, URd, URb claims the lowest run of
      // URb granules aligned to its size that is free in every CTA of the
      // operation, and puts its first granule in URd (the kernel stores
      // granule * 32 as the address); AND URd, URb keeps only the granules
      // URb names. With .2CTA the first warp of the pair to get there acts
      // for both, as in the PTX engine.
      const bool find = ins.f[0], two = word.bit(85);
      std::vector<Block*> cs = group(two);
      exec::TensorMemory& mine = blk.tmem;
      if (find) {
        const uint32_t n = urv(u(32)), cols = n * 32;
        const unsigned up = ins.dst[0].reg, d = ins.dst[1].reg;
        if (!n || n > 16 || (n & (n - 1))) throw Error(Err::InvalidValue, "UTCATOMSWS.FIND_AND_SET of " + std::to_string(n) + " granules");
        uint32_t col = 0;
        bool ok = false;
        if (two && !mine.peer_alloc.empty()) {
          col = mine.peer_alloc.front().first;
          mine.peer_alloc.pop_front();
          ok = true;
        } else {
          for (col = 0; col + cols <= exec::TensorMemory::kCols; col += cols) {
            ok = true;
            for (Block* b : cs) ok = ok && b->tmem.free_run(col, cols);
            if (ok) break;
          }
          if (ok)
            for (Block* b : cs) {
              if (b->tmem.cells.empty()) b->tmem.cells.assign(size_t{exec::TensorMemory::kLanes} * exec::TensorMemory::kCols, 0);
              b->tmem.live[col] = cols;
              if (b != &blk) b->tmem.peer_alloc.emplace_back(col, cols);
            }
        }
        if (up < 7) w.up[up] = ok;
        if (d < urz) w.ur[d] = ok ? col / 32 : 0;
        if (!ok) w.yield = true;   // it retries until a peer warp deallocates
        return;
      }
      const uint32_t keep = urv(u(32));
      if (two && !mine.peer_dealloc.empty()) {
        mine.peer_dealloc.pop_front();
        return;
      }
      for (Block* b : cs) {
        for (auto it = b->tmem.live.begin(); it != b->tmem.live.end();) {
          const uint32_t g0 = it->first / 32, gn = it->second / 32;
          const uint32_t bits = ((1u << gn) - 1) << g0;
          if (!(keep & bits)) it = b->tmem.live.erase(it);
          else ++it;
        }
        if (two && b != &blk) b->tmem.peer_dealloc.emplace_back(0, 0);
      }
      return;
    }
    case Op::LDTM: case Op::STTM: {
      // tcgen05.ld/st: warp w of a warpgroup reaches lanes 32(w%4)-32(w%4)+31.
      static const ptx::Tcgen05Shape shapes[] = {ptx::Tcgen05Shape::S16x128b, ptx::Tcgen05Shape::S16x256b,
                                                 ptx::Tcgen05Shape::S32x32b, ptx::Tcgen05Shape::S16x64b,
                                                 ptx::Tcgen05Shape::S16x32bx2, ptx::Tcgen05Shape::S16x32bx2};
      static const uint32_t per_x1[] = {2, 4, 1, 1, 1, 1};
      const uint32_t shape = ins.f[0], nregs = per_x1[shape] * ins.f[1], width = ins.f[2] ? 2 : 1;
      const bool ld = ins.op == Op::LDTM;
      const Operand& tm = ld ? ins.src[0] : ins.src[0];
      const uint32_t taddr = urv(tm.reg) + static_cast<uint32_t>(tm.imm);
      const uint32_t lane0 = taddr >> 16, col0 = taddr & 0xFFFF, quarter = w.index % 4;
      const unsigned r0 = ld ? ins.dst[0].reg : ins.src[1].reg;
      exec::TensorMemory& t = blk.tmem;
      for (uint32_t j = 0; j < nregs; ++j)
        each_lane(ex, [&](unsigned th) {
          if (shape >= 4 && (th / 16) != shape - 4) return;   // this half of 16x32bx2's threads
          uint32_t dl, dc;
          exec::tmem_fragment(shapes[shape], th, j, &dl, &dc);
          const uint32_t lane = lane0 + dl, col = col0 + dc * width;
          if (lane / 32 != quarter || lane >= exec::TensorMemory::kLanes)
            throw Error(Err::InvalidValue, std::string(ld ? "LDTM" : "STTM") + " reaches Tensor Memory lane " +
                                               std::to_string(lane) + ", and warp " + std::to_string(quarter) +
                                               " of its warpgroup may only reach lanes " + std::to_string(32 * quarter) +
                                               "-" + std::to_string(32 * quarter + 31));
          for (uint32_t c = col; c < col + width; ++c)
            if (c >= exec::TensorMemory::kCols || !t.allocated(c))
              throw Error(Err::OutOfBounds, std::string(ld ? "LDTM" : "STTM") + " reaches Tensor Memory column " +
                                                std::to_string(c) + ", which this CTA has not allocated");
          if (ld) {
            if (r0 + j == kRZ) return;
            w.reg(r0 + j, th) = width == 2 ? (t.at(lane, col) & 0xFFFF) | (t.at(lane, col + 1) << 16) : t.at(lane, col);
          } else {
            const uint32_t v = r0 == kRZ ? 0 : w.reg(r0 + j, th);
            if (width == 2) {
              t.at(lane, col) = v & 0xFFFF;
              t.at(lane, col + 1) = v >> 16;
            } else {
              t.at(lane, col) = v;
            }
          }
        });
      return;
    }
    case Op::UTCBAR: {
      // tcgen05.commit: an arrive, count 1, on the barrier -- or with
      // .MULTICAST on the barrier at the same offset in every CTA the mask
      // names -- once the MMAs before it are done, which they are.
      const uint32_t at = urv(u(24));
      uint32_t off;
      Block& owner = shared_block(blk, at, &off);
      if (!ins.f[0]) {
        mbar_arrive(owner, off, 1, 0, false);
        return;
      }
      const uint32_t mask = urv(u(64)) & 0xFFFF;
      for (uint32_t r = 0; r < 16; ++r)
        if (mask >> r & 1) mbar_arrive(cluster_block(blk, r), off, 1, 0, false);
      return;
    }
    case Op::UTCCP: {
      static const ptx::Tcgen05CpShape shapes[] = {ptx::Tcgen05CpShape::S128x256b, ptx::Tcgen05CpShape::S128x256b,
                                                   ptx::Tcgen05CpShape::S4x256b,   ptx::Tcgen05CpShape::S128x128b,
                                                   ptx::Tcgen05CpShape::S64x128b,  ptx::Tcgen05CpShape::S64x128b,
                                                   ptx::Tcgen05CpShape::S32x128b,  ptx::Tcgen05CpShape::S32x128b};
      static const int multicast[] = {0, 0, 0, 0, 2, 3, 4, 4};
      const uint32_t shape = ins.f[0] & 7;
      if (shape == 1 || shape == 7) throw Error(Err::UnsupportedPtx, "UTCCP shape " + std::to_string(shape));
      exec::Tcgen05Cp cp;
      cp.shape = shapes[shape];
      cp.multicast = multicast[shape];
      cp.decompress = ins.f[2] == 1 ? 4 : ins.f[2] == 2 ? 6 : 0;
      cp.taddr = urv(ins.src[0].reg) + static_cast<uint32_t>(ins.src[0].imm);
      cp.desc = ur64(ins.src[1].reg);
      SassTcHost h(group(ins.f[1]), &blk);
      exec::tcgen05_cp(cp, h);
      return;
    }
    case Op::UTCSHIFT: {
      const uint32_t taddr = urv(ins.src[0].reg) + static_cast<uint32_t>(ins.src[0].imm);
      SassTcHost h(group(ins.f[1]), &blk);
      exec::tcgen05_shift(taddr, h);
      return;
    }
    case Op::UTCMMA: {
      exec::Tcgen05Mma m;
      // A cubin that uses K = 96 or absolute descriptors was built for
      // sm_103a; the 'a' is the only form such SASS exists in.
      m.target_sm = static_cast<int>(m_.sm);
      m.target_arch = true;
      const uint32_t p = u(40), x = u(48);
      m.idesc = urv(p + 1);
      m.cta_group = ins.f[1] ? 2 : 1;
      m.a_tmem = ins.f[2];
      m.ws = ins.f[4];
      m.sparse = m.idesc >> 2 & 1;
      switch (ins.f[0]) {
        case 0: m.kind = (m.idesc >> 7 & 7) == 2 ? ptx::Tcgen05MmaKind::TF32 : ptx::Tcgen05MmaKind::F16; break;
        case 1: m.kind = ins.f[3] ? ptx::Tcgen05MmaKind::MXF8F6F4 : ptx::Tcgen05MmaKind::F8F6F4; break;
        case 2: m.kind = ptx::Tcgen05MmaKind::I8; break;
        default: m.kind = ptx::Tcgen05MmaKind::MXF4NVF4; break;
      }
      m.block_scale = ins.f[3];
      m.scale_vec = ins.f[0] == 3 ? (ins.f[7] ? 4 : 2) : ins.f[3] ? 1 : 0;
      m.a = m.a_tmem ? urv(u(24)) : ur64(u(24));
      m.b = ur64(u(32));
      m.d = urv(u(64));
      if (m.sparse) m.sp_meta = urv(p);
      if (ins.f[3]) {
        m.scale_a = urv(x);
        m.scale_b = urv(x + 1);
      } else if (x != urz) {
        if (m.ws) {
          m.has_zero_mask = true;
          m.zero_mask = ur64(x);
        } else {
          for (uint32_t i = 0; i < (m.cta_group == 2 ? 8u : 4u); ++i) m.disable_lanes[i] = urv(x + i);
        }
      }
      const unsigned upi = static_cast<unsigned>(word.field(87, 3));
      m.accumulate = (upi == 7 ? true : w.up[upi]) != word.bit(90);
      m.scale_d = ins.f[5] ? static_cast<int>(ins.f[5]) : -1;
      if (m.ws) {
        // B's collector: 79-80 the buffer, 81 keep, 82 reuse -- fill, use,
        // lastuse and discard.
        const uint32_t c = ins.f[6];
        m.collector_buf = c & 3;
        const bool keep = c >> 2 & 1, reuse = c >> 3 & 1;
        m.collector = keep && !reuse   ? ptx::Tcgen05Collector::Fill
                      : keep && reuse  ? ptx::Tcgen05Collector::Use
                      : reuse          ? ptx::Tcgen05Collector::LastUse
                                       : ptx::Tcgen05Collector::Discard;
      }
      SassTcHost h(group(ins.f[1]), &blk);
      exec::tcgen05_mma(m, h);
      return;
    }
    default:
      throw Fault("tcgen05 form not implemented");
  }
}

// ---- stores into another block that complete on its mbarrier; cluster launch control --

// STAS / REDAS [Ra.64], Rb (st.async, red.async): Ra is the shared::cluster
// address, Ra+1 its mbarrier's -- which must be in the same block -- and the
// store's (or reduction's) bytes complete on that barrier.
void Runner::exec_async_store(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  const bool red = ins.op == Op::REDAS;
  const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8)), rb = ins.src[1].reg;
  const int64_t off = ins.w.sfield(40, 24);
  const uint32_t t = ins.f[0];
  const uint32_t bytes = red ? (t == 2 ? 8 : 4) : (t == 0 ? 4 : t == 1 ? 8 : 16);
  each_lane(ex, [&](unsigned l) {
    const uint32_t at = w.reg(ra, l) + static_cast<uint32_t>(off), bar = w.reg(ra + 1, l);
    uint32_t doff, boff;
    Block& to = shared_block(blk, at, &doff);
    Block& owner = shared_block(blk, bar, &boff);
    if (&owner != &to) throw Fault("st.async's mbarrier must be in the block its data goes to");
    if (doff % bytes) throw Error(Err::MisalignedAccess, "st.async to a misaligned shared address");
    if (uint64_t{doff} + bytes > to.shared.size()) throw Fault("st.async past the block's shared memory");
    uint32_t v[4] = {};
    for (uint32_t i = 0; i < bytes / 4; ++i) v[i] = rb == kRZ ? 0 : w.reg(rb + i, l);
    if (red) {
      static const ptx::Type types[3] = {{ptx::Type::Kind::U, 32}, {ptx::Type::Kind::S, 32}, {ptx::Type::Kind::U, 64}};
      if (t > 2) throw Fault("REDAS type " + std::to_string(t));
      static const ptx::AtomOp ops[] = {ptx::AtomOp::Add, ptx::AtomOp::Min, ptx::AtomOp::Max, ptx::AtomOp::Inc,
                                        ptx::AtomOp::Dec, ptx::AtomOp::And, ptx::AtomOp::Or,  ptx::AtomOp::Xor};
      uint64_t old = 0, b = 0;
      std::memcpy(&old, &to.shared[doff], bytes);
      std::memcpy(&b, v, bytes);
      const uint64_t r = exec::reduce_value(ops[ins.f[1] & 7], types[t], old, b);
      std::memcpy(&to.shared[doff], &r, bytes);
    } else {
      std::memcpy(&to.shared[doff], v, bytes);
    }
    mbar_arrive(owner, boff, 0, -static_cast<int64_t>(bytes), false);
  });
}

// UGETNEXTWORKID.SELFCAST/.BROADCAST [URa], [URa+1]
// (clusterlaunchcontrol.try_cancel): take the next block (or cluster) that has
// not started -- it then never runs -- and answer with its first CTA, in the
// 16 bytes ptxas's query_cancel reads: x in word 0, y and z in word 1's
// halves, and word 2's bit 0 set when a unit was taken. The answer lands at
// the same offsets in this block, or with .BROADCAST in every block of the
// cluster, completing 16 bytes on the barrier beside it.
void Runner::exec_clc(Block& blk, Warp& w, const Instr& ins, Mask ex) {
  if (!ex) return;
  const unsigned ura = static_cast<unsigned>(ins.w.field(24, 8));
  const auto urv = [&](unsigned u) -> uint32_t { return u >= 255 ? 0 : w.ur[u]; };
  uint32_t aoff, boff;
  Block& at = shared_block(blk, urv(ura), &aoff);
  Block& bar = shared_block(blk, urv(ura + 1), &boff);
  if (&at != &blk || &bar != &blk)
    throw Fault("clusterlaunchcontrol.try_cancel's response and barrier are in this CTA's shared memory");
  if (aoff % 16) throw Error(Err::MisalignedAccess, "clusterlaunchcontrol.try_cancel's response must be 16-byte aligned");
  uint32_t resp[4] = {};
  uint64_t u = next_unit_.load();
  while (u < units_ && !next_unit_.compare_exchange_weak(u, u + 1)) {
  }
  if (u < units_) {
    uint64_t x, y, z;
    if (clustered_) {
      const uint64_t ncx = cfg_.grid[0] / cshape_[0], ncy = cfg_.grid[1] / cshape_[1];
      x = (u % ncx) * cshape_[0];
      y = ((u / ncx) % ncy) * cshape_[1];
      z = (u / (ncx * ncy)) * cshape_[2];
    } else {
      x = u % cfg_.grid[0];
      y = (u / cfg_.grid[0]) % cfg_.grid[1];
      z = u / (uint64_t{cfg_.grid[0]} * cfg_.grid[1]);
    }
    resp[0] = static_cast<uint32_t>(x);
    resp[1] = static_cast<uint32_t>(y & 0xFFFF) | static_cast<uint32_t>(z << 16);
    resp[2] = 1;
  }
  const auto deliver = [&](Block& to) {
    if (uint64_t{aoff} + 16 > to.shared.size()) throw Fault("clusterlaunchcontrol's response past shared memory");
    std::memcpy(&to.shared[aoff], resp, 16);
    mbar_arrive(to, boff, 0, -16, false);
  };
  if (!ins.f[0]) deliver(blk);
  else
    for (Block* b : blk.cluster->blocks) deliver(*b);
}
