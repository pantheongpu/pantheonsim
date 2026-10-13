// The tier's handlers and its run loop (included into exec.cpp after exec_ops.inc, inside its anonymous namespace,
// because they work on the executor's own Warp and Block). The analysis is src/sass/tier.cpp; what the tier is, and
// the rule it keeps, is src/sass/tier.hpp.
//
// A handler computes, for the lanes `ex`, what Runner::fast_alu (exec_ops.inc) computes for the same form -- the
// same SIMD lane kernels (vgpu/simd/simd.hpp) wherever fast_alu uses one, the same scalar loop where it has one -- and
// writes the same registers and predicates. Counting and the guard are the run loop's, as Runner::execute does them.

// What a handler needs of the warp and the launch.
struct TCtx {
  Warp* w;
  Block* blk;
  Runner* r;
  uint32_t* R;                 // the warp's register file: register n is R + 32 * n
  const uint8_t* b0;           // constant bank 0 of the launch
  uint64_t b0n;
  const tier::Row* pool;       // the program's constant rows
  const simd::Kernels* sk;
};

// ---- operands ---------------------------------------------------------------------------------

inline uint32_t tier_mod(bool flt, uint8_t mod, uint32_t v) {
  if (flt) {   // |x| then -x on the bits
    if (mod & tier::kAbs) v &= 0x7fffffffu;
    if (mod & tier::kNeg) v ^= 0x80000000u;
  } else {     // -x as 0 - x, then ~x
    if (mod & tier::kNeg) v = 0u - v;
    if (mod & tier::kNot) v = ~v;
  }
  return v;
}

// A word of constant bank 0, as Runner::cbank32 reads bank 0 (zero past its end).
inline uint32_t tier_b0(const TCtx& c, uint64_t off) {
  if (off + 4 > c.b0n) return 0;
  uint32_t v;
  std::memcpy(&v, c.b0 + off, 4);
  return v;
}

// A source that has to be worked out when the instruction runs, as 32 lanes in `tmp`.
template <bool Flt>
[[gnu::noinline]] const uint32_t* tier_dyn(const TCtx& c, const tier::Src& s, uint32_t* tmp) {
  switch (s.dyn) {
    case tier::Dyn::Reg: {
      const uint32_t* r = c.R + s.off * 32u;
      for (unsigned l = 0; l < 32; ++l) tmp[l] = tier_mod(Flt, s.mod, r[l]);
      return tmp;
    }
    case tier::Dyn::Bank0: {
      const uint32_t v = tier_mod(Flt, s.mod, tier_b0(c, s.off));
      for (unsigned l = 0; l < 32; ++l) tmp[l] = v;
      return tmp;
    }
    case tier::Dyn::UReg: {
      const uint32_t v = tier_mod(Flt, s.mod, c.w->ur[s.off]);
      for (unsigned l = 0; l < 32; ++l) tmp[l] = v;
      return tmp;
    }
  }
  return tmp;
}

template <bool Flt>
inline const uint32_t* tier_src(const TCtx& c, const tier::Src& s, uint32_t* tmp) {
  if (s.kind == tier::SrcKind::Row) return c.R + s.off;
  if (s.kind == tier::SrcKind::Pool) return c.pool[s.off].v;
  return tier_dyn<Flt>(c, s, tmp);
}

// A 64-bit source of a double op as two rows. A register pair or a pool pair is read in place; a constant-bank or
// uniform-register value is the same for every lane.
[[gnu::noinline]] void tier_dyn64(const TCtx& c, const tier::Src& s, uint32_t* lo, uint32_t* hi) {
  uint64_t bits;
  if (s.dyn == tier::Dyn::Bank0) bits = tier_b0(c, s.off) | (static_cast<uint64_t>(tier_b0(c, uint64_t{s.off} + 4)) << 32);
  else bits = c.w->ur[s.off] | (static_cast<uint64_t>(c.w->ur[s.off + 1]) << 32);
  if (s.mod & tier::kAbs) bits &= 0x7fffffffffffffffull;
  if (s.mod & tier::kNeg) bits ^= 0x8000000000000000ull;
  for (unsigned l = 0; l < 32; ++l) {
    lo[l] = static_cast<uint32_t>(bits);
    hi[l] = static_cast<uint32_t>(bits >> 32);
  }
}

// A predicate source as the lane mask fast_alu's pmask gives.
inline Mask tier_pmask(const TCtx& c, const tier::Src& p) {
  Mask v;
  if (p.dyn == tier::Dyn::UReg) v = (p.off == kPT || c.w->up[p.off]) ? kAll : 0;
  else v = p.off == kPT ? kAll : c.w->p[p.off];
  return (p.mod & tier::kNeg) ? ~v : v;
}

// A predicate destination, written for the executing lanes only (fast_alu's setp).
inline void tier_setp(const TCtx& c, unsigned reg, Mask ex, Mask v) {
  if (reg != kPT) c.w->p[reg] = (c.w->p[reg] & ~ex) | (v & ex);
}

inline Mask tier_combine(unsigned bop, Mask r, Mask acc) { return bop == 0 ? (r & acc) : bop == 1 ? (r | acc) : (r ^ acc); }

// A result row to the destination register, for the executing lanes (fast_alu's store).
inline void tier_store(const TCtx& c, uint16_t dst, Mask ex, const uint32_t* r) {
  if (dst == tier::kNoReg) return;
  uint32_t* d = c.R + dst * 32u;
  if (ex == kAll) {
    std::memmove(d, r, 32 * sizeof(uint32_t));
  } else {
    for (unsigned l = 0; l < 32; ++l)
      if ((ex >> l) & 1) d[l] = r[l];
  }
}

// ---- handlers ---------------------------------------------------------------------------------

#define TIER_HANDLER(name) static void name(void* vc, const tier::MOp& op, Mask ex)
#define TIER_PRELUDE      \
  const TCtx& c = *static_cast<const TCtx*>(vc); \
  (void)c; (void)ex;

TIER_HANDLER(h_mov) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32];
  tier_store(c, op.dst, ex, tier_src<false>(c, op.s[0], t));
}

TIER_HANDLER(h_iadd3) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  c.sk->iadd3(r, a, b, d, op.a);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_imad) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  c.sk->imad(r, a, b, d);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_lop3) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  c.sk->lop3(r, a, b, d, op.a);
  if (op.aux & 1) {
    Mask pm = 0;
    for (unsigned l = 0; l < 32; ++l)
      if ((op.aux & 2) ? r[l] == kAll : r[l] != 0) pm |= Mask{1} << l;
    if (!(op.aux & 2)) pm |= tier_pmask(c, op.p);   // ORed with the predicate operand, as the general path does
    tier_setp(c, op.dst2, ex, pm);
  }
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_lea) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const bool hi = op.aux & 1, x = op.aux & 2, three = op.aux & 4, bneg = op.aux & 8;
  const unsigned sh = op.a;
  const Mask cin = x ? tier_pmask(c, op.p) : 0;
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const uint32_t* top = three ? tier_src<false>(c, op.s[2], t[2]) : nullptr;
  Mask carry = 0;
  for (unsigned l = 0; l < 32; ++l) {
    uint32_t shifted;
    if (!hi) {
      shifted = sh >= 32 ? 0 : a[l] << sh;
    } else {
      const uint32_t tp = three ? top[l] : static_cast<uint32_t>(static_cast<int32_t>(a[l]) >> 31);
      shifted = static_cast<uint32_t>(((a[l] | (static_cast<uint64_t>(tp) << 32)) << sh) >> 32);
    }
    uint64_t sum = static_cast<uint64_t>(shifted) + b[l] + ((cin >> l) & 1);
    if (bneg && b[l] == 0) sum += uint64_t{1} << 32;   // -0 is ~0 + 1: it carries out (see the general rule)
    r[l] = static_cast<uint32_t>(sum);
    carry |= static_cast<Mask>(sum >> 32 != 0) << l;
  }
  tier_store(c, op.dst, ex, r);
  if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, carry);
}

TIER_HANDLER(h_shf) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const unsigned type = op.a;   // 0 S64, 1 U64, 2 S32, 3 U32
  const bool wrap = op.aux & 1, right = op.aux & 2, hi = op.aux & 4;
  const bool is64 = type < 2, sgn = type == 0 || type == 2;
  const uint32_t width = is64 ? 64 : 32;
  const uint32_t* lo = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* n = tier_src<false>(c, op.s[1], t[1]);
  const uint32_t* top = tier_src<false>(c, op.s[2], t[2]);
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t v = lo[l] | (static_cast<uint64_t>(top[l]) << 32);
    const uint32_t k = wrap ? n[l] % width : std::min(n[l], width);
    uint64_t x;
    if (!right) x = k >= 64 ? 0 : v << k;
    else if (sgn) x = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(k, 63u));
    else x = k >= 64 ? 0 : v >> k;
    r[l] = static_cast<uint32_t>(hi ? x >> 32 : x);
  }
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_isetp) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32];
  static const unsigned kIntCmp[] = {0, 2, 0, 3, 4, 1, 5};   // simd::IntCmp of cmp 1..6: Lt Eq Le Gt Ne Ge
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const Mask r = c.sk->icmp_mask(kIntCmp[op.a], op.c != 0, a, b, 32) & ex;
  const Mask acc = tier_pmask(c, op.p);
  tier_setp(c, op.dst, ex, tier_combine(op.b, r, acc));
  tier_setp(c, op.dst2, ex, tier_combine(op.b, ~r, acc));
}

TIER_HANDLER(h_fsetp) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32];
  const unsigned cmp = op.a, bop = op.b;
  const bool fz = op.c != 0;
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const auto fv = [&](uint32_t bits, bool abs_, bool neg_) {
    float f = as_f32(bits);
    if (fz) f = ftz(f);
    if (abs_) f = std::fabs(f);
    if (neg_) f = -f;
    return f;
  };
  Mask r = 0;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    if (float_cmp(cmp, fv(a[l], op.aux & 1, op.aux & 2), fv(b[l], op.aux & 4, op.aux & 8))) r |= Mask{1} << l;
  }
  const Mask acc = tier_pmask(c, op.p);
  tier_setp(c, op.dst, ex, tier_combine(bop, r, acc));
  tier_setp(c, op.dst2, ex, tier_combine(bop, ~r, acc));
}

TIER_HANDLER(h_fsel) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t* a = tier_src<true>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<true>(c, op.s[1], t[1]);
  const Mask p = tier_pmask(c, op.p);
  for (unsigned l = 0; l < 32; ++l) r[l] = ((p >> l) & 1) ? a[l] : b[l];
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_ffma) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<true>(c, op.s[0], t[0]), *b = tier_src<true>(c, op.s[1], t[1]),
                 *d = tier_src<true>(c, op.s[2], t[2]);
  c.sk->ffma(r, a, b, d);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_fadd) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t *a = tier_src<true>(c, op.s[0], t[0]), *b = tier_src<true>(c, op.s[1], t[1]);
  c.sk->fadd(r, a, b);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_fmul) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t *a = tier_src<true>(c, op.s[0], t[0]), *b = tier_src<true>(c, op.s[1], t[1]);
  c.sk->fmul(r, a, b);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_dop) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;   // nothing is written to RZ
  alignas(32) uint32_t tlo[3][32], thi[3][32];
  const uint32_t* lo[3];
  const uint32_t* hi[3];
  for (unsigned i = 0; i < 3; ++i) {
    const tier::Src& s = op.s[i];
    if (s.kind == tier::SrcKind::Row) {
      lo[i] = c.R + s.off;
      hi[i] = c.R + s.off + 32;
    } else if (s.kind == tier::SrcKind::Pool) {
      lo[i] = c.pool[s.off].v;
      hi[i] = c.pool[s.off + 1].v;
    } else {
      tier_dyn64(c, s, tlo[i], thi[i]);
      lo[i] = tlo[i];
      hi[i] = thi[i];
    }
  }
  uint32_t sink[32];   // R254's partner is RZ: its half of the result goes nowhere
  uint32_t* dlo = c.R + op.dst * 32u;
  uint32_t* dhi = op.dst + 1 == kRZ ? sink : c.R + (op.dst + 1u) * 32u;
  c.sk->dop_m(op.a, op.b, dlo, dhi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ex);
}

tier::MOp::Fn tier_handler_for(tier::Form f) {
  using F = tier::Form;
  switch (f) {
    case F::Mov: return &h_mov;
    case F::Iadd3: return &h_iadd3;
    case F::Imad: return &h_imad;
    case F::Lop3: return &h_lop3;
    case F::Lea: return &h_lea;
    case F::Shf: return &h_shf;
    case F::Isetp: return &h_isetp;
    case F::Fsetp: return &h_fsetp;
    case F::Fsel: return &h_fsel;
    case F::Ffma: return &h_ffma;
    case F::Fadd: return &h_fadd;
    case F::Fmul: return &h_fmul;
    case F::Dop: return &h_dop;
    default: return nullptr;
  }
}

// ---- statistics (VGPU_TIER_STATS=1) -------------------------------------------------------------
//
// How much of a run the tier executed: warp instructions run by a handler, straight-line ones run by the interpreter
// (an unsupported form), and the rest (control flow, barriers, warp-wide ops: they end a straight-line run and are
// the interpreter's by design), per op. Printed to stderr when the process exits. Nothing is counted otherwise.

struct TierStats {
  static constexpr size_t kOps = 512;
  std::atomic<uint64_t> native[kOps]{};
  std::mutex fb_mu;
  std::map<std::string, uint64_t> fallback;   // by mnemonic and modifiers, which tell the forms apart
  std::atomic<uint64_t> total{0};   // every warp instruction of the launches (LaunchStats::instructions)
  ~TierStats() {
    uint64_t nat = 0, fb = 0;
    for (size_t i = 0; i < kOps; ++i) nat += native[i].load();
    for (const auto& [k, v] : fallback) fb += v;
    if (!nat && !fb) return;
    const uint64_t all = total.load();
    std::fprintf(stderr, "tier: %llu of %llu warp instructions ran in the tier (%.1f%%); %llu straight-line ones in the interpreter\n",
                 (unsigned long long)nat, (unsigned long long)all, all ? 100.0 * double(nat) / double(all) : 0.0,
                 (unsigned long long)fb);
    {
      std::vector<std::pair<uint64_t, size_t>> v;
      for (size_t i = 0; i < kOps; ++i)
        if (native[i].load()) v.emplace_back(native[i].load(), i);
      std::sort(v.rbegin(), v.rend());
      std::string line = "tier native:";
      for (size_t i = 0; i < v.size() && i < 24; ++i)
        line += std::string(" ") + op_name(static_cast<Op>(v[i].second)) + "=" + std::to_string(v[i].first);
      std::fprintf(stderr, "%s\n", line.c_str());
    }
    {
      std::vector<std::pair<uint64_t, std::string>> v;
      for (const auto& [k, n] : fallback) v.emplace_back(n, k);
      std::sort(v.rbegin(), v.rend());
      std::string line = "tier interpreter (straight):";
      for (size_t i = 0; i < v.size() && i < 24; ++i) line += " " + v[i].second + "=" + std::to_string(v[i].first);
      std::fprintf(stderr, "%s\n", line.c_str());
    }
  }
};

TierStats& tier_stats() {
  static TierStats s;
  return s;
}

void tier_note_fallback(const Instr& ins) {
  std::string key = ins.mnemonic;
  for (const std::string& m : ins.mods) key += "." + m;
  TierStats& s = tier_stats();
  std::lock_guard<std::mutex> g(s.fb_mu);
  ++s.fallback[key];
}

void tier_add_total(uint64_t n) { tier_stats().total.fetch_add(n, std::memory_order_relaxed); }

// ---- the run loop -------------------------------------------------------------------------------

// Runner::step_warp's straight-line run, for a code section with a tier: the same loop -- one step counted per
// instruction, at most 256 straight instructions, then the one that ends the run -- with a handler in place of
// Runner::execute where the form has one. Called with the group and the first address already chosen, and with no
// forced lane mask (BMOV.PQUAD MACTIVE), which execute() would apply to the guard.
bool Runner::tier_run(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp) {
  TCtx c{&w, &blk, this, w.r.data(), bank0_.data(), bank0_.size(), tp.pool.data(), &simd::kernels()};
  exec::LaunchStats& st = blk.st;
  const tier::MOp* ops = tp.ops.data();
  const size_t count = code.count;
  const size_t first = static_cast<size_t>((pick - code.base) / 16);
  size_t idx = first;
  const unsigned glanes = popc(group);
  const bool regs_ok = tp.max_reg <= w.nregs;
  const tier::MOp* cur = &ops[idx];
  try {
    for (unsigned n = 0;; ++n) {
      if (++w.steps > cfg_.max_steps)
        throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": a warp ran " + std::to_string(cfg_.max_steps) +
                                      " instructions (an infinite loop?)");
      const tier::MOp& op = ops[idx];
      cur = &op;
      if (n < 256 && idx + 1 < count && op.straight) {
        if (op.fn && (regs_ok || op.maxreg <= w.nregs)) {
          const Mask ex = op.unguarded ? group : guard_mask(w, *op.ins, group);
          const unsigned lanes = ex == group ? glanes : popc(ex);
          ++st.instructions;
          st.thread_instructions += lanes;
          st.inst_by_class[op.cls] += lanes;
          if (ex) op.fn(&c, op, ex);
          if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(op.ins->op)].fetch_add(1, std::memory_order_relaxed);
        } else {
          execute(blk, w, *op.ins, group, /*advance=*/false);
          if (tier_stats_) [[unlikely]] tier_note_fallback(*op.ins);
        }
        ++idx;
        continue;
      }
      break;
    }
  } catch (const Fault& f) {
    fault(w, *cur->ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what());
  } catch (const std::out_of_range& f) {   // a register past the warp's file
    fault(w, *cur->ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what(), Err::Internal);
  } catch (const Error& e) {
    if (e.message().rfind("SASS kernel ", 0) == 0) throw;
    fault(w, *cur->ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), e.message(), e.code());
  }
  if (idx != first) {
    const uint64_t pc = code.base + 16 * static_cast<uint64_t>(idx);
    for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = pc;
  }
  execute(blk, w, *ops[idx].ins, group);
  return true;
}
