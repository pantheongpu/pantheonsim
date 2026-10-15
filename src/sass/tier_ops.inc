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
// With `fold_float` the |x| and -x modifiers of a double operand are applied to the bits; without, they are left to the
// caller (an integer operand's -x and ~x are applied lane by lane).
[[gnu::noinline]] void tier_dyn64(const TCtx& c, const tier::Src& s, uint32_t* lo, uint32_t* hi, bool fold_float) {
  uint64_t bits;
  if (s.dyn == tier::Dyn::Bank0) bits = tier_b0(c, s.off) | (static_cast<uint64_t>(tier_b0(c, uint64_t{s.off} + 4)) << 32);
  else bits = c.w->ur[s.off] | (static_cast<uint64_t>(c.w->ur[s.off + 1]) << 32);
  if (fold_float) {
    if (s.mod & tier::kAbs) bits &= 0x7fffffffffffffffull;
    if (s.mod & tier::kNeg) bits ^= 0x8000000000000000ull;
  }
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
  if (ex == kAll) std::memmove(d, r, 32 * sizeof(uint32_t));
  else c.sk->blend_m(d, r, ex);   // the executing lanes only, without a branch per lane
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
  switch (op.a) {   // the tables compiled code uses most, as the plain expressions they are
    case 0xc0: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] & b[l]; break;
    case 0xfc: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] | b[l]; break;
    case 0x3c: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] ^ b[l]; break;
    case 0x96: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] ^ b[l] ^ d[l]; break;
    case 0x80: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] & b[l] & d[l]; break;
    case 0xfe: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] | b[l] | d[l]; break;
    case 0x33: for (unsigned l = 0; l < 32; ++l) r[l] = ~b[l]; break;
    case 0x0f: for (unsigned l = 0; l < 32; ++l) r[l] = ~a[l]; break;
    case 0x30: for (unsigned l = 0; l < 32; ++l) r[l] = a[l] & ~b[l]; break;
    case 0x0c: for (unsigned l = 0; l < 32; ++l) r[l] = ~a[l] & b[l]; break;
    default: c.sk->lop3(r, a, b, d, op.a);
  }
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
  if (!hi && !x && !bneg) {   // the usual one, (a << s) + b: two vector operations, and the carry out is a wrap
    for (unsigned l = 0; l < 32; ++l) r[l] = (sh >= 32 ? 0 : a[l] << sh) + b[l];
    if (op.dst2 != tier::kNoReg)
      for (unsigned l = 0; l < 32; ++l) carry |= static_cast<Mask>(r[l] < (sh >= 32 ? 0 : a[l] << sh)) << l;
    tier_store(c, op.dst, ex, r);
    if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, carry);
    return;
  }
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
  const uint32_t* lo = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* n = tier_src<false>(c, op.s[1], t[1]);
  const uint32_t* top = tier_src<false>(c, op.s[2], t[2]);
  c.sk->shf(r, lo, n, top, op.a, op.aux & 1, op.aux & 2, op.aux & 4);   // type, wrap, right, hi
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
  if (op.aux == 0) {   // no |x| or -x: the SIMD comparison (the one the interpreter's fast path uses)
    r = c.sk->fcmp_mask(cmp, fz, a, b) & ex;
  } else {
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      if (float_cmp(cmp, fv(a[l], op.aux & 1, op.aux & 2), fv(b[l], op.aux & 4, op.aux & 8))) r |= Mask{1} << l;
    }
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
      tier_dyn64(c, s, tlo[i], thi[i], true);
      lo[i] = tlo[i];
      hi[i] = thi[i];
    }
  }
  uint32_t sink[32];   // R254's partner is RZ: its half of the result goes nowhere
  uint32_t* dlo = c.R + op.dst * 32u;
  uint32_t* dhi = op.dst + 1 == kRZ ? sink : c.R + (op.dst + 1u) * 32u;
  c.sk->dop_m(op.a, op.b, dlo, dhi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ex);
}

// A 64-bit integer source as two rows (see src64i in tier.cpp).
inline void tier_src64(const TCtx& c, const tier::Src& s, uint32_t* tlo, uint32_t* thi, const uint32_t*& lo, const uint32_t*& hi) {
  if (s.kind == tier::SrcKind::Row) {
    lo = c.R + s.off;
    hi = c.R + s.off + 32;
  } else if (s.kind == tier::SrcKind::Pool) {
    lo = c.pool[s.off].v;
    hi = c.pool[s.off + 1].v;
  } else {
    tier_dyn64(c, s, tlo, thi, false);
    lo = tlo;
    hi = thi;
  }
}

TIER_HANDLER(h_iadd3x) {   // exec_int's IADD3: carries out to two predicates, carries in from two
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  const bool x = op.aux & 1;
  uint32_t over1, over2;
  c.sk->iadd3x(r, a, b, d, op.a, x ? static_cast<uint32_t>(tier_pmask(c, op.p)) : 0u,
               x ? static_cast<uint32_t>(tier_pmask(c, op.p2)) : 0u, &over1, &over2);
  tier_store(c, op.dst, ex, r);
  if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, over1);
  if (op.c != tier::kNoReg) tier_setp(c, op.c, ex, over2);
}

TIER_HANDLER(h_imad32) {   // IMAD with a carry in and/or out
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  const Mask cin = (op.aux & 1) ? tier_pmask(c, op.p) : 0;
  Mask carry = 0;
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t full = static_cast<uint64_t>(a[l] * b[l]) + d[l] + ((cin >> l) & 1);
    r[l] = static_cast<uint32_t>(full);
    carry |= static_cast<Mask>((full >> 32) != 0) << l;
  }
  tier_store(c, op.dst, ex, r);
  if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, carry);
}

TIER_HANDLER(h_imad_wide) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], tlo[32], thi[32], rlo[32], rhi[32], mlo[32], mhi[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]);
  const uint32_t *clo, *chi;
  tier_src64(c, op.s[2], tlo, thi, clo, chi);
  const uint8_t cmod = op.s[2].mod;
  if (cmod) {   // -x and ~x of the 64-bit addend, lane by lane, into rows of their own
    for (unsigned l = 0; l < 32; ++l) {
      uint64_t cc = clo[l] | (static_cast<uint64_t>(chi[l]) << 32);
      if (cmod & tier::kNeg) cc = 0 - cc;
      if (cmod & tier::kNot) cc = ~cc;
      mlo[l] = static_cast<uint32_t>(cc);
      mhi[l] = static_cast<uint32_t>(cc >> 32);
    }
    clo = mlo;
    chi = mhi;
  }
  uint32_t carry;
  c.sk->imad_wide(rlo, rhi, a, b, clo, chi, (op.aux & 2) != 0, (op.aux & 1) ? static_cast<uint32_t>(tier_pmask(c, op.p)) : 0u, &carry);
  if (op.dst != tier::kNoReg) {   // write64: R254's partner is RZ
    tier_store(c, op.dst, ex, rlo);
    if (op.dst + 1u != kRZ) tier_store(c, static_cast<uint16_t>(op.dst + 1), ex, rhi);
  }
  if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, carry);
}

TIER_HANDLER(h_imad_hi) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], tlo[32], thi[32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]);
  const bool sgn = op.aux & 2, pair = op.aux & 4;
  const uint32_t *clo, *chi = nullptr;
  if (pair) tier_src64(c, op.s[2], tlo, thi, clo, chi);
  else clo = tier_src<false>(c, op.s[2], t[2]);
  const bool neg = op.s[2].mod & tier::kNeg;
  const Mask cin = (op.aux & 1) ? tier_pmask(c, op.p) : 0;
  Mask carry = 0;
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a[l])) * static_cast<int32_t>(b[l]))
                              : static_cast<uint64_t>(a[l]) * b[l];
    uint64_t cc;
    if (pair) {
      cc = clo[l] | (static_cast<uint64_t>(chi[l]) << 32);
      if (neg) cc = 0 - cc;
    } else {
      cc = static_cast<uint64_t>(sgn ? static_cast<int64_t>(static_cast<int32_t>(clo[l])) : clo[l]);
    }
    const unsigned __int128 sum = static_cast<unsigned __int128>(prod) + cc + ((cin >> l) & 1);
    r[l] = static_cast<uint32_t>(sum >> 32);
    carry |= static_cast<Mask>((sum >> 64) != 0) << l;
  }
  tier_store(c, op.dst, ex, r);
  if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, carry);
}

TIER_HANDLER(h_isetp_ex) {   // the high words of a 64-bit compare; the low words' answer arrives as a predicate
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32];
  static const unsigned kIntCmp[] = {0, 2, 0, 3, 4, 1, 5};
  const unsigned cmp = op.a;
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const Mask eq = c.sk->icmp_mask(0, false, a, b, 32), hi = c.sk->icmp_mask(kIntCmp[cmp], op.c != 0, a, b, 32);
  const Mask low = tier_pmask(c, op.p2);
  const Mask r = cmp == 2 ? (eq & low) : cmp == 5 ? (~eq | low) : ((eq & low) | (~eq & hi));
  const Mask acc = tier_pmask(c, op.p);
  tier_setp(c, op.dst, ex, tier_combine(op.b, r, acc));
  tier_setp(c, op.dst2, ex, tier_combine(op.b, ~r, acc));
}

TIER_HANDLER(h_popc) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], r[32];
  c.sk->lane_unary(0, r, tier_src<false>(c, op.s[0], t));   // the number of set bits
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_i2fp) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], r[32];
  c.sk->i2f(op.c != 0, r, tier_src<false>(c, op.s[0], t));
  tier_store(c, op.dst, ex, r);
}

// ---- memory -------------------------------------------------------------------------------------

inline uint32_t tier_extend(unsigned size, uint32_t v) {   // a sub-word load widens as exec_mem's extend does
  switch (size) {
    case 0: return v & 0xff;
    case 1: return static_cast<uint32_t>(static_cast<int8_t>(v));
    case 2: return v & 0xffff;
    case 3: return static_cast<uint32_t>(static_cast<int16_t>(v));
    default: return v;
  }
}

// Global loads and stores of 1, 2, 4 or 8 bytes, as exec_mem does them: a coalesced run in one call, or the lanes'
// accesses together when the memory manager can make them as one-by-one would; anything else (a constant-window
// address, a misaligned or faulting access, an armed fault) is exec_mem's own, which has not been started.
template <bool Store>
static void h_global(void* vc, const tier::MOp& op, Mask ex) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Warp& w = *c.w;
  Block& blk = *c.blk;
  const unsigned bytes = op.a;
  const unsigned ra = op.s[0].off;
  const uint32_t* blo = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint32_t* bhi = (ra != kRZ && (op.aux & 1)) ? c.R + (ra + 1u) * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? w.ur[op.s[1].off] : 0;
  const uint64_t imm = static_cast<uint64_t>(op.imm);
  uint64_t addrs[32] = {};
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint64_t v = 0;
    if (blo) {
      v = blo[l];
      if (bhi) v |= static_cast<uint64_t>(bhi[l]) << 32;
    }
    addrs[l] = v + urv + imm;
  }
  constexpr uint64_t kWinEnd = kConstWindow + (uint64_t{1} << 32);
  const uint32_t* dreg = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  MemoryManager& mem = c.r->memory();
  const auto data = [&](unsigned l, uint32_t* v) {   // a store's data for one lane
    if (bytes <= 4) {
      v[0] = dreg ? dreg[l] : 0;
    } else {
      v[0] = dreg ? dreg[l] : 0;
      v[1] = dreg ? dreg[32 + l] : 0;
    }
  };
  if (ex && ((ex + (ex & -ex)) & ex) == 0) {
    const unsigned first = static_cast<unsigned>(std::countr_zero(ex));
    const unsigned count = popc(ex);
    const uint64_t g0 = addrs[first];
    bool run = !(g0 >= kConstWindow && g0 < kWinEnd);
    for (unsigned i = 0; run && i < count; ++i) run = addrs[first + i] == g0 + uint64_t{i} * bytes;
    if (run && (g0 + uint64_t{count} * bytes - 1 < kConstWindow || g0 >= kWinEnd)) {
      uint8_t buf[32 * 8];
      bool done = false;
      if (Store) {
        for (unsigned i = 0; i < count; ++i) {
          uint32_t v[2];
          data(first + i, v);
          std::memcpy(buf + i * bytes, v, bytes);
        }
        done = mem.store_run(g0, buf, uint64_t{count} * bytes, bytes);
      } else {
        done = mem.load_run(g0, buf, uint64_t{count} * bytes, bytes);
        if (done && dreg) {
          uint32_t* d = c.R + op.dst * 32u;
          for (unsigned i = 0; i < count; ++i) {
            if (bytes <= 4) {
              uint32_t v = 0;
              std::memcpy(&v, buf + i * bytes, bytes);
              d[first + i] = tier_extend(op.b, v);
            } else {
              uint32_t v[2];
              std::memcpy(v, buf + i * 8, 8);
              d[first + i] = v[0];
              d[32 + first + i] = v[1];
            }
          }
        }
      }
      if (done) {
        count_run(blk.st, Store, g0, first, count, ex, bytes);
        return;
      }
    }
  }
  // The lanes' accesses together (MemoryManager::load_batch / store_batch).
  uint64_t fa[32], fv[32];
  unsigned lane_of[32], cnt = 0;
  bool ok = ex != 0;
  for (Mask m = ex; m && ok; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    if (addrs[l] >= kConstWindow && addrs[l] < kWinEnd) ok = false;
    fa[cnt] = addrs[l];
    lane_of[cnt++] = l;
  }
  if (ok) {
    bool done;
    if (Store) {
      for (unsigned i = 0; i < cnt; ++i) {
        uint32_t v[2];
        data(lane_of[i], v);
        fv[i] = bytes == 8 ? (static_cast<uint64_t>(v[1]) << 32 | v[0]) : bytes == 4 ? v[0] : (v[0] & ((1u << (8 * bytes)) - 1));
      }
      done = mem.store_batch(fa, cnt, bytes, fv);
    } else {
      done = mem.load_batch(fa, cnt, bytes, fv);
      if (done && dreg) {
        uint32_t* d = c.R + op.dst * 32u;
        for (unsigned i = 0; i < cnt; ++i) {
          if (bytes <= 4) {
            d[lane_of[i]] = tier_extend(op.b, static_cast<uint32_t>(fv[i]));
          } else {
            d[lane_of[i]] = static_cast<uint32_t>(fv[i]);
            d[32 + lane_of[i]] = static_cast<uint32_t>(fv[i] >> 32);
          }
        }
      }
    }
    if (done) {
      count_access(blk.st, 0, Store, addrs, ex, bytes);
      return;
    }
  }
  c.r->tier_exec_mem(blk, w, *op.ins, ex);
}

// Shared memory: the interpreter's fast case -- every active lane inside the block's own shared memory -- moved
// straight between the register file and the block's array; anything else is exec_mem's.
template <bool Store>
static void h_shared(void* vc, const tier::MOp& op, Mask ex) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Warp& w = *c.w;
  Block& blk = *c.blk;
  const unsigned bytes = op.a, stride = op.c;
  const unsigned ra = op.s[0].off;
  const uint32_t* base = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? w.ur[op.s[1].off] : 0;
  const uint64_t imm = static_cast<uint64_t>(op.imm);
  uint64_t addrs[32] = {};
  const uint64_t ssize = blk.shared.size();
  bool inside = true;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint64_t v = base ? base[l] : 0;
    if (stride) v *= stride;
    const uint64_t off = static_cast<uint32_t>(v + urv + imm);
    addrs[l] = off;
    inside = inside && (off >> 24) == blk.rank && (off & 0xffffff) + bytes <= ssize;
  }
  if (!inside) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);
    return;
  }
  uint8_t* sh = blk.shared.data();
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  const unsigned words = std::max(1u, bytes / 4);
  // (The widths the instruction can have move with copies of a size known at compile time: see store_lane_bytes.)
  const bool moved = with_lane_width(bytes, [&](auto width) {
    constexpr unsigned B = decltype(width)::value;
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* at = sh + (addrs[l] & 0xffffff);
      if (Store) {
        store_lane_bytes<B>(at, [&](unsigned i) -> uint32_t { return d ? d[32 * i + l] : 0u; });
      } else if (d) {
        if constexpr (B <= 4) d[l] = tier_extend(op.b, load_lane_word<B>(at));
        else
          for (unsigned i = 0; i < B / 4; ++i) d[32 * i + l] = load_lane_word<B>(at, i);
      }
    }
  });
  if (!moved) {
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* at = sh + (addrs[l] & 0xffffff);
      if (Store) {
        uint32_t v[4] = {};
        for (unsigned i = 0; i < words; ++i) v[i] = d ? d[32 * i + l] : 0;
        std::memcpy(at, v, bytes);
      } else {
        uint32_t v[4] = {};
        std::memcpy(v, at, bytes);
        if (d) {
          if (bytes <= 4) d[l] = tier_extend(op.b, v[0]);
          else
            for (unsigned i = 0; i < words; ++i) d[32 * i + l] = v[i];
        }
      }
    }
  }
  count_access(blk.st, 1, Store, addrs, ex, bytes);
}

// Local memory: each lane's window of the warp's array, when every active lane's access is inside the thread's frame.
template <bool Store>
static void h_local(void* vc, const tier::MOp& op, Mask ex) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Warp& w = *c.w;
  Block& blk = *c.blk;
  const unsigned bytes = op.a;
  const unsigned ra = op.s[0].off;
  const uint32_t* base = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? w.ur[op.s[1].off] : 0;
  const uint64_t imm = static_cast<uint64_t>(op.imm);
  uint64_t addrs[32] = {};
  bool inside = true;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t off = static_cast<uint32_t>((base ? base[l] : 0) + urv + imm);
    addrs[l] = off;
    inside = inside && off + bytes <= w.local_size;
  }
  if (!inside) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);   // the lane that faults, and the words of the fault, are its
    return;
  }
  if (w.local.empty()) w.local.assign(size_t{w.local_size} * 32, 0);
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  const unsigned words = std::max(1u, bytes / 4);
  const bool moved = with_lane_width(bytes, [&](auto width) {
    constexpr unsigned B = decltype(width)::value;
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* at = &w.local[size_t{l} * w.local_size + addrs[l]];
      if (Store) {
        store_lane_bytes<B>(at, [&](unsigned i) -> uint32_t { return d ? d[32 * i + l] : 0u; });
      } else if (d) {
        if constexpr (B <= 4) d[l] = tier_extend(op.b, load_lane_word<B>(at));
        else
          for (unsigned i = 0; i < B / 4; ++i) d[32 * i + l] = load_lane_word<B>(at, i);
      }
    }
  });
  if (!moved) {
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* at = &w.local[size_t{l} * w.local_size + addrs[l]];
      if (Store) {
        uint32_t v[4] = {};
        for (unsigned i = 0; i < words; ++i) v[i] = d ? d[32 * i + l] : 0;
        std::memcpy(at, v, bytes);
      } else {
        uint32_t v[4] = {};
        std::memcpy(v, at, bytes);
        if (d) {
          if (bytes <= 4) d[l] = tier_extend(op.b, v[0]);
          else
            for (unsigned i = 0; i < words; ++i) d[32 * i + l] = v[i];
        }
      }
    }
  }
  count_access(blk.st, 2, Store, addrs, ex, bytes);
}

// PLOP3: for each lane, the table bit that the three predicates select, into two predicates (exec_int).
TIER_HANDLER(h_plop3) {
  TIER_PRELUDE
  const Mask a = tier_pmask(c, op.s[0]), b = tier_pmask(c, op.s[1]), d = tier_pmask(c, op.s[2]);
  Mask r0 = 0, r1 = 0;
  for (unsigned idx = 0; idx < 8; ++idx) {
    const Mask term = ((idx & 4) ? a : ~a) & ((idx & 2) ? b : ~b) & ((idx & 1) ? d : ~d);
    if ((op.a >> idx) & 1) r0 |= term;
    if ((op.b >> idx) & 1) r1 |= term;
  }
  tier_setp(c, op.dst, ex, r0);
  tier_setp(c, op.dst2, ex, r1);
}

TIER_HANDLER(h_s2r) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;
  uint32_t* d = c.R + op.dst * 32u;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    d[l] = c.r->tier_sreg(*c.blk, *c.w, op.a, l);
  }
}

TIER_HANDLER(h_sel) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const Mask p = tier_pmask(c, op.p);
  for (unsigned l = 0; l < 32; ++l) r[l] = ((p >> l) & 1) ? a[l] : b[l];
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_prmt) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *sel = tier_src<false>(c, op.s[1], t[1]),
                 *bb = tier_src<false>(c, op.s[2], t[2]);
  if (op.a == 0) c.sk->prmt0(r, a, bb, sel);
  else
    for (unsigned l = 0; l < 32; ++l) r[l] = prmt(a[l], bb[l], sel[l], op.a);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_imnmx) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const Mask mn = tier_pmask(c, op.p);
  for (unsigned l = 0; l < 32; ++l) {
    const bool m = (mn >> l) & 1;
    if (op.c) r[l] = m ? static_cast<uint32_t>(std::min<int32_t>(a[l], b[l])) : static_cast<uint32_t>(std::max<int32_t>(a[l], b[l]));
    else r[l] = m ? std::min(a[l], b[l]) : std::max(a[l], b[l]);
  }
  tier_store(c, op.dst, ex, r);
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
    case F::Plop3: return &h_plop3;
    case F::S2r: return &h_s2r;
    case F::Sel: return &h_sel;
    case F::Prmt: return &h_prmt;
    case F::Imnmx: return &h_imnmx;
    case F::Iadd3x: return &h_iadd3x;
    case F::Imad32: return &h_imad32;
    case F::ImadWide: return &h_imad_wide;
    case F::ImadHi: return &h_imad_hi;
    case F::IsetpEx: return &h_isetp_ex;
    case F::Popc: return &h_popc;
    case F::I2fp: return &h_i2fp;
    case F::Ldg: return &h_global<false>;
    case F::Stg: return &h_global<true>;
    case F::Lds: return &h_shared<false>;
    case F::Sts: return &h_shared<true>;
    case F::Ldl: return &h_local<false>;
    case F::Stl: return &h_local<true>;
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
  std::map<std::string, uint64_t> ended;      // the instructions that end a run (control flow, waits, warp-wide ops), run by execute
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
    const auto list = [&](const char* what, const std::map<std::string, uint64_t>& m) {
      std::vector<std::pair<uint64_t, std::string>> v;
      for (const auto& [k, n] : m) v.emplace_back(n, k);
      std::sort(v.rbegin(), v.rend());
      std::string line = std::string("tier ") + what + ":";
      for (size_t i = 0; i < v.size() && i < 24; ++i) line += " " + v[i].second + "=" + std::to_string(v[i].first);
      std::fprintf(stderr, "%s\n", line.c_str());
    };
    list("interpreter (straight)", fallback);
    list("interpreter (ends a run)", ended);
  }
};

TierStats& tier_stats() {
  static TierStats s;
  return s;
}

void tier_note_fallback(const Instr& ins, bool ends_run = false) {
  std::string key = ins.mnemonic;
  for (const std::string& m : ins.mods) key += "." + m;
  TierStats& s = tier_stats();
  std::lock_guard<std::mutex> g(s.fb_mu);
  ++(ends_run ? s.ended : s.fallback)[key];
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
      // step_warp's store gate: a warp's first store to global memory ends its turn, once, so that the block's other warps
      // issue the loads their prologues make before it lands (the comment there says why). The tier must stop where
      // the interpreter does, or a block's warps see each other's stores in another order.
      if (w.store_gate && (ops[idx].ins->op == Op::STG || ops[idx].ins->op == Op::ST)) {
        w.store_gate = false;
        if (blk.warps.size() > 1) {
          if (idx != first) {
            const uint64_t pc = code.base + 16 * static_cast<uint64_t>(idx);
            for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = pc;
          }
          w.yield = true;
          return true;
        }
      }
      if (++w.steps > cfg_.max_steps)
        throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": a warp ran " + std::to_string(cfg_.max_steps) +
                                      " instructions (an infinite loop?)");
      const tier::MOp& op = ops[idx];
      cur = &op;
      if (n < 256 && idx + 1 < count && op.straight) {
        if ((op.fn && (regs_ok || op.maxreg <= w.nregs)) || op.never) {
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
  const tier::MOp& last = ops[idx];
  if (last.form == tier::Form::Bra) {
    // exec_control's BRA for a fixed target: the lanes that take it go there, the others on to the next instruction;
    // a group that branches backwards while other lanes wait gives way now and then. Counted as execute() counts it.
    const Mask ex = last.unguarded ? group : guard_mask(w, *last.ins, group);
    const unsigned lanes = ex == group ? glanes : popc(ex);
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[last.cls] += lanes;
    if (ex && ex != group) ++st.divergent_branches;
    const uint64_t here = code.base + 16 * static_cast<uint64_t>(idx);
    const uint64_t target = code.base + static_cast<uint64_t>(last.imm);
    for (Mask m = group; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      w.pc[l] = ((ex >> l) & 1) ? target : here + 16;
    }
    if (ex && last.imm <= static_cast<int64_t>(last.ins->pc) && (w.runnable() & ~group)) {
      if (++w.spins >= 64) {
        w.spins = 0;
        w.give_way = true;
      }
    }
    if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::BRA)].fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  if (last.fn && !last.straight && (regs_ok || last.maxreg <= w.nregs)) {
    // PLOP3 and S2R: a handler, and the next address (execute() advances the group when nothing set the counters).
    const Mask ex = last.unguarded ? group : guard_mask(w, *last.ins, group);
    const unsigned lanes = ex == group ? glanes : popc(ex);
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[last.cls] += lanes;
    if (ex) last.fn(&c, last, ex);
    const uint64_t next = code.base + 16 * static_cast<uint64_t>(idx) + 16;
    for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = next;
    if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(last.ins->op)].fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  if (last.form == tier::Form::Bssy || last.form == tier::Form::Nop) {
    // These leave the program counters to execute()'s advance: every lane of the group goes on to the next address.
    const Mask ex = last.unguarded ? group : guard_mask(w, *last.ins, group);
    const unsigned lanes = ex == group ? glanes : popc(ex);
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[last.cls] += lanes;
    if (last.form == tier::Form::Bssy) {
      // The lanes join the barrier, in nesting order (see exec_control's BSSY).
      w.b[last.dst] |= ex;
      w.bssy_stamp[last.dst & 15] = ++blk.bssy_clock;
    }
    const uint64_t next = code.base + 16 * static_cast<uint64_t>(idx) + 16;
    for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = next;
    if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(last.ins->op)].fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  if (tier_stats_) [[unlikely]] tier_note_fallback(*last.ins, true);
  if (idx != first) {
    const uint64_t pc = code.base + 16 * static_cast<uint64_t>(idx);
    for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = pc;
  }
  execute(blk, w, *ops[idx].ins, group);
  return true;
}
