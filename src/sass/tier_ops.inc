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
    case tier::Dyn::BankN: {
      const uint32_t v = tier_mod(Flt, s.mod, c.r->tier_cbank32(s.pad, s.off));
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
  else if (s.dyn == tier::Dyn::BankN)
    bits = c.r->tier_cbank32(s.pad, s.off) | (static_cast<uint64_t>(c.r->tier_cbank32(s.pad, uint64_t{s.off} + 4)) << 32);
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

// A row of 32 words moved without a call into libc (the 128 bytes were a memmove call: 15% of the integer workload). All eight
// vectors are read before any is written, so a destination that is the source itself is no problem.
[[gnu::always_inline]] inline void tier_copy_row(uint32_t* d, const uint32_t* s) {
#if defined(__SSE2__)
  __m128i v[8];
  for (unsigned i = 0; i < 8; ++i) v[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s) + i);
  for (unsigned i = 0; i < 8; ++i) _mm_storeu_si128(reinterpret_cast<__m128i*>(d) + i, v[i]);
#else
  std::memmove(d, s, 32 * sizeof(uint32_t));
#endif
}

inline uint32_t tier_extend(unsigned size, uint32_t v) {   // a sub-word load widens as exec_mem's extend does
  switch (size) {
    case 0: return v & 0xff;
    case 1: return static_cast<uint32_t>(static_cast<int8_t>(v));
    case 2: return v & 0xffff;
    case 3: return static_cast<uint32_t>(static_cast<int16_t>(v));
    default: return v;
  }
}

// A result row to the destination register, for the executing lanes (fast_alu's store).
[[gnu::always_inline]] inline void tier_store(const TCtx& c, uint16_t dst, Mask ex, const uint32_t* r) {
  if (dst == tier::kNoReg) return;
  uint32_t* d = c.R + dst * 32u;
  if (ex == kAll) tier_copy_row(d, r);
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
  c.sk->lop3(r, a, b, d, op.a);   // (the vector kernel, which has the usual tables as expressions: faster than loops the compiler makes for the baseline x86-64)
  if (op.aux & 1) {
    // The lanes whose result is not zero, or is all ones (.PAND): the compare kernel, as fast_alu does it.
    Mask pm = static_cast<Mask>(c.sk->icmp_mask((op.aux & 2) ? 0 : 1, false, r, (op.aux & 2) ? kOnes32.data() : kZeros32.data(), 32));
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
  if (op.aux != 0) {   // |x| or -x on a source: the interpreter's own code (Runner::fast_alu's FSETP)
    c.r->tier_float_op(*c.blk, *c.w, *op.ins, ex);
    return;
  }
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const Mask r = c.sk->fcmp_mask(cmp, fz, a, b) & ex;   // the SIMD comparison the interpreter's fast path uses
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
  tier_copy_row(r, b);               // b where the predicate is clear, a where it holds: the blend kernel (fast_wide's SEL)
  c.sk->blend_m(r, a, p);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_ffma) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], r[32];
  const uint32_t *a = tier_src<true>(c, op.s[0], t[0]), *b = tier_src<true>(c, op.s[1], t[1]),
                 *d = tier_src<true>(c, op.s[2], t[2]);
  if (!op.a && !op.b) {
    c.sk->ffma(r, a, b, d);
  } else if (op.b) {   // a directed rounding mode: the host's, around the kernel
    RoundingScope scope(op.b);
    c.sk->ffma_x(r, a, b, d, op.a);
  } else {
    c.sk->ffma_x(r, a, b, d, op.a);
  }
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_fadd) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t *a = tier_src<true>(c, op.s[0], t[0]), *b = tier_src<true>(c, op.s[1], t[1]);
  if (!op.a && !op.b) {
    c.sk->fadd(r, a, b);
  } else if (op.b) {   // a directed rounding mode: the host's, around the kernel
    RoundingScope scope(op.b);
    c.sk->fadd_x(r, a, b, op.a);
  } else {
    c.sk->fadd_x(r, a, b, op.a);
  }
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_fmul) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t *a = tier_src<true>(c, op.s[0], t[0]), *b = tier_src<true>(c, op.s[1], t[1]);
  if (!op.a && !op.b) {
    c.sk->fmul(r, a, b);
  } else if (op.b) {   // a directed rounding mode: the host's, around the kernel
    RoundingScope scope(op.b);
    c.sk->fmul_x(r, a, b, op.a);
  } else {
    c.sk->fmul_x(r, a, b, op.a);
  }
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
      tier_dyn64(c, s, tlo[i], thi[i], false);   // (the modifiers are the kernel's, as for a register: a NaN operand is kept as it is)
      lo[i] = tlo[i];
      hi[i] = thi[i];
    }
  }
  uint32_t sink[32];   // R254's partner is RZ: its half of the result goes nowhere
  uint32_t* dlo = c.R + op.dst * 32u;
  uint32_t* dhi = op.dst + 1 == kRZ ? sink : c.R + (op.dst + 1u) * 32u;
  if (op.c) {   // a directed rounding mode: the host's, around the kernel (the NaN rule is the same)
    RoundingScope scope(op.c);
    c.sk->dop_m(op.a, op.b, dlo, dhi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ex);
  } else {
    c.sk->dop_m(op.a, op.b, dlo, dhi, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ex);
  }
}

// A double operand as its two rows (low words, high words), |x| and -x applied to the bits.
inline void tier_pair(const TCtx& c, const tier::Src& s, uint32_t* tlo, uint32_t* thi, const uint32_t*& lo, const uint32_t*& hi) {
  if (s.kind == tier::SrcKind::Row) {
    lo = c.R + s.off;
    hi = c.R + s.off + 32;
  } else if (s.kind == tier::SrcKind::Pool) {
    lo = c.pool[s.off].v;
    hi = c.pool[s.off + 1].v;
  } else {
    tier_dyn64(c, s, tlo, thi, true);
    lo = tlo;
    hi = thi;
  }
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

// ---- the float forms of fast_float ---------------------------------------------------------------

TIER_HANDLER(h_fmnmx) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t* a = tier_src<true>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<true>(c, op.s[1], t[1]);
  c.sk->fmnmx(r, a, b, static_cast<uint32_t>(tier_pmask(c, op.p)), op.a != 0, op.b != 0);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_mufu) {
  TIER_PRELUDE
  const unsigned fn = op.a;
  if (fn != 4 && fn != 5 && fn != 8) {   // the libm functions (SIN, COS, EX2, LG2, TANH): the interpreter's own code
    c.r->tier_float_op(*c.blk, *c.w, *op.ins, ex);
    return;
  }
  alignas(32) uint32_t t[32], r[32];
  c.sk->mufu(fn, r, tier_src<true>(c, op.s[0], t));   // RCP, RSQ, SQRT: the kernel the interpreter's fast path uses
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_f2i) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], r[32];
  c.sk->f2i(op.a != 0, op.b, op.c != 0, r, tier_src<true>(c, op.s[0], t));
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_iabs) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], r[32];
  const uint32_t* a = tier_src<false>(c, op.s[0], t);
  for (unsigned l = 0; l < 32; ++l) {
    const int32_t v = static_cast<int32_t>(a[l]);
    r[l] = static_cast<uint32_t>(v < 0 ? 0u - static_cast<uint32_t>(v) : static_cast<uint32_t>(v));
  }
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_i2f) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], r[32];
  const uint32_t* src = tier_src<false>(c, op.s[0], t);
  // (A directed rounding mode: the host's, around the same kernel, as Runner::fast_float does it.)
  std::optional<RoundingScope> scope;
  if (op.aux) scope.emplace(op.aux);
  if (op.a < 2) {
    // A byte or a half of the register (.B1-.B3, .H1 move it down), sign- or zero-extended: it converts as a signed 32-bit integer.
    alignas(32) uint32_t ext[32];
    const unsigned shift = op.b;
    const bool sgn = op.c != 0;
    for (unsigned l = 0; l < 32; ++l) {
      const uint32_t v = shift >= 32 ? 0u : src[l] >> shift;
      ext[l] = op.a == 0 ? (sgn ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(v))) : (v & 0xffu))
                         : (sgn ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(v))) : (v & 0xffffu));
    }
    c.sk->i2f(true, r, ext);
  } else {
    c.sk->i2f(op.c != 0, r, src);
  }
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_f2fp) {   // F2FP.PACK_AB: the interpreter's own code (the NaN, .RELU and rounding rules are fast_float's)
  TIER_PRELUDE
  c.r->tier_float_op(*c.blk, *c.w, *op.ins, ex);
}

TIER_HANDLER(h_unary) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], r[32];
  c.sk->lane_unary(op.a, r, tier_src<false>(c, op.s[0], t));
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_dsetp) {
  TIER_PRELUDE
  alignas(32) uint32_t tlo[2][32], thi[2][32];
  const uint32_t *lo[2], *hi[2];
  for (unsigned i = 0; i < 2; ++i) tier_pair(c, op.s[i], tlo[i], thi[i], lo[i], hi[i]);
  uint32_t res, both_nan;
  c.sk->dcmp(op.a, op.b, lo[0], hi[0], lo[1], hi[1], &res, &both_nan);
  const Mask acc = tier_pmask(c, op.p);
  tier_setp(c, op.dst, ex, tier_combine(op.c, res, acc));
  if (op.a == 0 || op.a == 15) tier_setp(c, op.dst2, ex, both_nan);
  else tier_setp(c, op.dst2, ex, tier_combine(op.c, ~res, acc));
}

// A packed-half source: the register's halves swapped or copied (mode 1: .H0_H0, 2: .H1_H1) and |.|, - on both (opbits 4, 8).
inline const uint32_t* tier_pk(const TCtx& c, const tier::Src& s, unsigned opbits, uint32_t* t, uint32_t* tmp) {
  const uint32_t* v = tier_src<false>(c, s, t);
  if (!opbits) return v;
  const unsigned mode = opbits & 3;
  const uint32_t absm = (opbits & 4) ? 0x7fff7fffu : ~0u, negm = (opbits & 8) ? 0x80008000u : 0u;
  for (unsigned l = 0; l < 32; ++l) {
    uint32_t x = v[l];
    if (mode == 1) x = (x & 0xffffu) * 0x10001u;
    else if (mode == 2) x = (x >> 16) * 0x10001u;
    tmp[l] = (x & absm) ^ negm;
  }
  return tmp;
}

TIER_HANDLER(h_hop2) {
  TIER_PRELUDE
  if (op.aux & 1) {   // HADD2.F32 (the half to float conversion): the interpreter's own code
    c.r->tier_float_op(*c.blk, *c.w, *op.ins, ex);
    return;
  }
  alignas(32) uint32_t t[3][32], u[3][32];
  const uint32_t* a = tier_pk(c, op.s[0], op.b & 15, t[0], u[0]);
  const uint32_t* b = tier_pk(c, op.s[1], (op.b >> 4) & 15, t[1], u[1]);
  if (op.dst == tier::kNoReg) return;
  const uint32_t* d3 = tier_pk(c, op.s[2], (op.b >> 8) & 15, t[2], u[2]);
  uint32_t* d = c.R + op.dst * 32u;
  if (op.a == 0) c.sk->hfma2_m(d, a, d3, b, ex, 32, 2);        // a * 1 + b (s[2] holds the ones)
  else c.sk->hfma2_m(d, a, b, d3, ex, 32, 2);                  // a * b + (-0), or a * b + c
}

// LDC: a word of a constant bank at a register + offset, the address wrapping in the 64 KiB bank (exec_mem's LDC).
TIER_HANDLER(h_ldc) {
  TIER_PRELUDE
  const unsigned bank = op.a, size = op.b, ra = op.c;
  const uint32_t* base = ra != kRZ ? c.R + ra * 32u : nullptr;
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  const bool wide = size == 5;   // mem_bytes(size) == 8
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t o = static_cast<uint64_t>((base ? static_cast<int64_t>(static_cast<int32_t>(base[l])) : 0) + op.imm) & 0xffff;
    if (wide) {
      const uint32_t lo = c.r->tier_cbank32(bank, o), hi = c.r->tier_cbank32(bank, o + 4);
      if (d) {
        d[l] = lo;
        if (op.dst + 1u != kRZ) d[32 + l] = hi;
      }
    } else {
      const uint32_t v = tier_extend(size, c.r->tier_cbank32(bank, o & ~3ull) >> (8 * (o & 3)));
      if (d) d[l] = v;
    }
  }
}

// ---- the uniform datapath ------------------------------------------------------------------------
//
// exec_int computes these once for the warp, on the first lane that runs the instruction (the operands of a uniform instruction are
// the same in every lane), and writes a UR or a UP; a predicate result may also go to a lane's own P, as bit `l`.

inline uint32_t tier_sc(const TCtx& c, const tier::Src& s, unsigned l) {   // a source as lane l reads it
  if (s.kind == tier::SrcKind::Row) return c.R[s.off + l];
  if (s.kind == tier::SrcKind::Pool) return c.pool[s.off].v[0];
  switch (s.dyn) {
    case tier::Dyn::Reg: return c.R[s.off * 32u + l];
    case tier::Dyn::Bank0: return tier_b0(c, s.off);
    case tier::Dyn::BankN: return c.r->tier_cbank32(s.pad, s.off);
    default: return c.w->ur[s.off];
  }
}
inline void tier_wur(const TCtx& c, uint16_t reg, uint32_t v) { if (reg != tier::kNoReg) c.w->ur[reg] = v; }
// A predicate destination (set_pred): a uniform UP, or bit l of a P; PT is not written.
inline void tier_wp1(const TCtx& c, bool uniform, unsigned reg, unsigned l, bool v) {
  if (reg == kPT) return;
  if (uniform) c.w->up[reg] = v;
  else if (v) c.w->p[reg] |= Mask{1} << l;
  else c.w->p[reg] &= ~(Mask{1} << l);
}
inline bool tier_pb(const TCtx& c, const tier::Src& p, unsigned l) { return ((tier_pmask(c, p) >> l) & 1) != 0; }
inline uint32_t tier_ival(uint32_t v, bool neg, bool bnot) {   // exec_int's ival: -x, then ~x
  if (neg) v = 0u - v;
  if (bnot) v = ~v;
  return v;
}

TIER_HANDLER(h_umov) {
  TIER_PRELUDE
  tier_wur(c, op.dst, tier_sc(c, op.s[0], static_cast<unsigned>(std::countr_zero(ex))));
}

// ULDC: a word of a constant bank, or a narrow field of one, or two words (exec_int's non-indexed ULDC).
TIER_HANDLER(h_uldc) {
  TIER_PRELUDE
  const unsigned bank = op.a, size = op.b;
  const uint64_t off = static_cast<uint64_t>(op.imm);
  if (size == 5) {
    const uint32_t lo = c.r->tier_cbank32(bank, off), hi = c.r->tier_cbank32(bank, off + 4);
    if (op.dst != tier::kNoReg) {
      c.w->ur[op.dst] = lo;
      if (op.dst + 1u < op.c) c.w->ur[op.dst + 1] = hi;   // (the partner of the last UR is URZ: dropped)
    }
    return;
  }
  // A narrow load takes its bytes out of the aligned word: c[0x3][0x59] is a byte of the word at 0x58.
  const uint32_t v = size < 4 ? c.r->tier_cbank32(bank, off & ~3ull) >> (8 * (off & 3)) : c.r->tier_cbank32(bank, off);
  tier_wur(c, op.dst, tier_extend(size, v));
}

TIER_HANDLER(h_uiadd3) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  uint64_t sum = 0;
  for (unsigned i = 0; i < 3; ++i) {
    uint32_t v = tier_sc(c, op.s[i], l);
    if ((op.b >> (2 * i)) & 1) {   // a negated source is added as its complement and a one
      sum += static_cast<uint32_t>(~v);
      sum += 1;
    } else {
      if ((op.b >> (2 * i + 1)) & 1) v = ~v;
      sum += v;
    }
  }
  if (op.aux & 1) sum += static_cast<uint64_t>(tier_pb(c, op.p, l)) + static_cast<uint64_t>(tier_pb(c, op.p2, l));
  tier_wur(c, op.dst, static_cast<uint32_t>(sum));
  const uint64_t over = sum >> 32;
  if (op.dst2 != tier::kNoReg) tier_wp1(c, op.aux & 2, op.dst2, l, over >= 1);
  if (op.a != tier::kNoReg) tier_wp1(c, op.aux & 4, op.a, l, over >= 2);
}

TIER_HANDLER(h_uimad) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  const uint32_t a = tier_ival(tier_sc(c, op.s[0], l), op.b & 1, op.b & 2), b = tier_ival(tier_sc(c, op.s[1], l), op.b & 4, op.b & 8),
                 d = tier_ival(tier_sc(c, op.s[2], l), op.b & 16, op.b & 32);
  const bool cin = (op.aux & 1) && tier_pb(c, op.p, l);
  const uint64_t full = static_cast<uint64_t>(a * b) + d + cin;
  tier_wur(c, op.dst, static_cast<uint32_t>(full));
  if (op.dst2 != tier::kNoReg) tier_wp1(c, op.aux & 2, op.dst2, l, (full >> 32) != 0);
}

TIER_HANDLER(h_uisetp) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  const uint32_t a = tier_sc(c, op.s[0], l), b = tier_sc(c, op.s[1], l);
  const bool acc = tier_pb(c, op.p, l);
  const bool r = op.c ? int_cmp(op.a, static_cast<int32_t>(a), static_cast<int32_t>(b)) : int_cmp(op.a, a, b);
  tier_wp1(c, true, op.dst, l, bool_op(op.b, r, acc));
  tier_wp1(c, true, op.dst2, l, bool_op(op.b, !r, acc));
}

TIER_HANDLER(h_ulea) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  const bool hi = op.aux & 1, x = op.aux & 2, sx32 = op.aux & 4, has_p = op.aux & 8;
  const unsigned s = op.a;
  const bool a_neg = op.b & 1, b_neg = op.b & 2, b_not = op.b & 4;
  const uint32_t a = tier_sc(c, op.s[0], l);
  const uint32_t an = a_neg ? 0u - a : a;
  const uint32_t b = tier_ival(tier_sc(c, op.s[1], l), b_neg, b_not);
  uint32_t shifted;
  if (!hi) {
    shifted = s >= 32 ? 0 : an << s;
  } else {
    const uint32_t top = sx32 ? static_cast<uint32_t>(static_cast<int32_t>(an) >> 31) : tier_sc(c, op.s[2], l);
    const uint64_t v = an | (static_cast<uint64_t>(top) << 32);
    shifted = s >= 64 ? 0 : static_cast<uint32_t>((v << s) >> 32);
  }
  const bool cin = x && tier_pb(c, op.p, l);
  uint64_t sum = static_cast<uint64_t>(shifted) + b + cin;
  // LEA Rd, P, -Ra, Rb: the adder takes ~(Ra << s) + 1, so negating 0 carries out.
  if (!hi && a_neg) sum = static_cast<uint64_t>(static_cast<uint32_t>(~(s >= 32 ? 0 : a << s))) + 1 + b + cin;
  // LEA Rd, P, Ra, -Rb: likewise a zero Rb carries out.
  if (b_neg && b == 0) sum += uint64_t{1} << 32;
  tier_wur(c, op.dst, static_cast<uint32_t>(sum));
  if (has_p) tier_wp1(c, op.aux & 16, op.dst2, l, (sum >> 32) != 0);
}

TIER_HANDLER(h_ulop3) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  tier_wur(c, op.dst, lop3(tier_sc(c, op.s[0], l), tier_sc(c, op.s[1], l), tier_sc(c, op.s[2], l), op.a));
}

TIER_HANDLER(h_ushf) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  const unsigned type = op.a;   // 0 S64, 1 U64, 2 S32, 3 U32
  const bool wrap = op.aux & 1, right = op.aux & 2, hi = op.aux & 4;
  const bool is64 = type < 2, sgn = type == 0 || type == 2;
  const uint64_t v = tier_sc(c, op.s[0], l) | (static_cast<uint64_t>(tier_sc(c, op.s[2], l)) << 32);
  uint32_t n = tier_sc(c, op.s[1], l);
  const uint32_t width = is64 ? 64 : 32;
  n = wrap ? n % width : std::min(n, width);
  uint64_t r;
  if (!right) r = n >= 64 ? 0 : v << n;
  else if (sgn) r = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(n, 63u));
  else r = n >= 64 ? 0 : v >> n;
  tier_wur(c, op.dst, static_cast<uint32_t>(hi ? r >> 32 : r));
}

TIER_HANDLER(h_usel) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  tier_wur(c, op.dst, tier_pb(c, op.p, l) ? tier_sc(c, op.s[0], l) : tier_sc(c, op.s[1], l));
}

TIER_HANDLER(h_uunary) {
  TIER_PRELUDE
  const unsigned l = static_cast<unsigned>(std::countr_zero(ex));
  uint32_t v = tier_ival(tier_sc(c, op.s[0], l), op.b & 1, op.b & 2);
  if (op.a == 0) {
    tier_wur(c, op.dst, popc(v));
    return;
  }
  const bool sgn = (op.a - 1) & 1, sh = (op.a - 1) & 2;   // FLO: the highest set bit (of the complement for a negative value, if signed)
  if (sgn && (v >> 31)) v = ~v;
  uint32_t r = v ? 31 - static_cast<uint32_t>(std::countl_zero(v)) : kAll;
  if (sh && r != kAll) r = 31 - r;
  tier_wur(c, op.dst, r);
}

// ---- memory -------------------------------------------------------------------------------------


// `bytes` (1, 2, 4, 8 or 16) between a lane's register words and memory, with sizes the compiler can see: a memcpy of a size known
// only at run time was a call into libc for every lane (10 to 15% of the memory workloads).
inline void tier_put(uint8_t* at, const uint32_t* v, unsigned bytes) {
  switch (bytes) {
    case 1: *at = static_cast<uint8_t>(v[0]); break;
    case 2: { const uint16_t x = static_cast<uint16_t>(v[0]); std::memcpy(at, &x, 2); break; }
    case 4: std::memcpy(at, v, 4); break;
    case 8: std::memcpy(at, v, 8); break;
    default: std::memcpy(at, v, 16); break;
  }
}
inline void tier_get(uint32_t* v, const uint8_t* at, unsigned bytes) {   // v is zero already
  switch (bytes) {
    case 1: v[0] = *at; break;
    case 2: { uint16_t x; std::memcpy(&x, at, 2); v[0] = x; break; }
    case 4: std::memcpy(v, at, 4); break;
    case 8: std::memcpy(v, at, 8); break;
    default: std::memcpy(v, at, 16); break;
  }
}

// A 16-byte global access (LDG.E.128, STG.E.128), as exec_mem's lane loop makes it: each active lane, in order, one 16-byte read or
// write of memory (the memory manager's bulk operation; not the atomic pieces a narrower access is), four registers a lane. The
// lane loop's decode of the instruction's fields at every execution is what this saves. An address in the constant window is exec_mem's.
template <bool Store>
static void tier_global16(void* vc, const tier::MOp& op, Mask ex) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Warp& w = *c.w;
  if constexpr (Store) w.idle_yields = 0;   // a store is progress (Runner::mem_write, Warp::idle_yields)
  Block& blk = *c.blk;
  const unsigned ra = op.s[0].off;
  const uint32_t* blo = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint32_t* bhi = (ra != kRZ && (op.aux & 1)) ? c.R + (ra + 1u) * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? w.ur[op.s[1].off] : 0;
  const uint64_t imm = static_cast<uint64_t>(op.imm);
  uint64_t addrs[32] = {};
  constexpr uint64_t kWinEnd = kConstWindow + (uint64_t{1} << 32);
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint64_t v = 0;
    if (blo) {
      v = blo[l];
      if (bhi) v |= static_cast<uint64_t>(bhi[l]) << 32;
    }
    addrs[l] = v + urv + imm;
    if (addrs[l] >= kConstWindow && addrs[l] < kWinEnd) {   // nothing is done yet: the interpreter runs the whole instruction
      c.r->tier_exec_mem(blk, w, *op.ins, ex);
      return;
    }
  }
  MemoryManager& mem = c.r->memory();
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint32_t v[4];
    if (Store) {
      for (unsigned k = 0; k < 4; ++k) v[k] = d ? d[32 * k + l] : 0;
      mem.write(addrs[l], v, 16);
    } else {
      mem.read(addrs[l], v, 16);
      if (d)
        for (unsigned k = 0; k < 4; ++k) d[32 * k + l] = v[k];
    }
  }
  count_access(blk.st, 0, Store, addrs, ex, 16);
}

// Global loads and stores of 1, 2, 4 or 8 bytes, as exec_mem does them: a coalesced run in one call, or the lanes'
// accesses together when the memory manager can make them as one-by-one would; anything else (a constant-window
// address, a misaligned or faulting access, an armed fault) is exec_mem's own, which has not been started.
template <bool Store>
static void h_global(void* vc, const tier::MOp& op, Mask ex) {
  if constexpr (Store) static_cast<const TCtx*>(vc)->w->idle_yields = 0;   // a store is progress (Runner::mem_write)
  if (op.a > 8) {
    tier_global16<Store>(vc, op, ex);
    return;
  }
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
    uint64_t differ = 0;   // (no short circuit: the lanes' addresses against the run, ORed)
    for (unsigned i = 0; i < count; ++i) differ |= addrs[first + i] ^ (g0 + uint64_t{i} * bytes);
    const bool run = differ == 0 && !(g0 >= kConstWindow && g0 < kWinEnd);
    if (run && (g0 + uint64_t{count} * bytes - 1 < kConstWindow || g0 >= kWinEnd)) {
      uint8_t buf[32 * 8];
      bool done = false;
      if (bytes == 4 && dreg) {
        // Words of consecutive lanes are consecutive words of the register's row: moved straight between it and memory (a load that
        // declines has written nothing; the lanes' addresses were read before).
        uint32_t* row = c.R + op.dst * 32u + first;
        done = Store ? mem.store_run(g0, row, uint64_t{count} * 4, 4) : mem.load_run(g0, row, uint64_t{count} * 4, 4);
      } else if (Store) {
        for (unsigned i = 0; i < count; ++i) {
          uint32_t v[4] = {};
          data(first + i, v);
          tier_put(buf + i * bytes, v, bytes);
        }
        done = mem.store_run(g0, buf, uint64_t{count} * bytes, bytes);
      } else {
        done = mem.load_run(g0, buf, uint64_t{count} * bytes, bytes);
        if (done && dreg) {
          uint32_t* d = c.R + op.dst * 32u;
          for (unsigned i = 0; i < count; ++i) {
            if (bytes <= 4) {
              uint32_t v[4] = {};
              tier_get(v, buf + i * bytes, bytes);
              d[first + i] = tier_extend(op.b, v[0]);
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
            d[lane_of[i]] = bytes == 4 ? static_cast<uint32_t>(fv[i]) : tier_extend(op.b, static_cast<uint32_t>(fv[i]));
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
  if constexpr (Store) w.idle_yields = 0;   // a store is progress (Runner::mem_write)
  Block& blk = *c.blk;
  const unsigned bytes = op.a, stride = op.c;
  const unsigned ra = op.s[0].off;
  const uint32_t* base = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? w.ur[op.s[1].off] : 0;
  const uint64_t imm = static_cast<uint64_t>(op.imm);
  uint64_t addrs[32] = {};
  const uint64_t ssize = blk.shared.size();
  exec::SharedWindowCheck window;   // (as exec_mem: see window_check.hpp)
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint64_t v = base ? base[l] : 0;
    if (stride) v *= stride;
    const uint64_t off = static_cast<uint32_t>(v + urv + imm);
    addrs[l] = off;
    window.add(off, blk.rank);
  }
  const bool inside = window.inside(ssize, bytes, ex != 0);
  if (!inside) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);
    return;
  }
  uint8_t* sh = blk.shared.data();
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  if (bytes == 4 && ex == kAll) {
    // A whole warp's words: the 32 consecutive words of an array indexed by the thread (the commonest shared access), or one word
    // every lane names. The same bytes move as lane by lane (a word every lane stores is the last lane's); the lanes are on one pass
    // through the banks, as the general count finds (a word that is not aligned spreads over one more bank).
    const uint64_t o0 = addrs[0] & 0xffffff;
    uint64_t seq = 0, same = 0;
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t o = addrs[l] & 0xffffff;
      seq |= o ^ (o0 + 4 * l);
      same |= o ^ o0;
    }
    if ((seq == 0 && (o0 & 3) == 0) || same == 0) {
      uint8_t* at = sh + o0;
      if (seq == 0) {
        if (Store) {
          if (d) {
#if defined(__SSE2__)
            __m128i v[8];
            for (unsigned i = 0; i < 8; ++i) v[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d) + i);
            for (unsigned i = 0; i < 8; ++i) _mm_storeu_si128(reinterpret_cast<__m128i*>(at) + i, v[i]);
#else
            std::memcpy(at, d, 128);
#endif
          } else {
            std::memset(at, 0, 128);
          }
        } else if (d) {
          tier_copy_row(d, reinterpret_cast<const uint32_t*>(at));   // (a copy of any alignment: unaligned vector loads)
        }
      } else if (Store) {
        const uint32_t v = d ? d[31] : 0u;
        std::memcpy(at, &v, 4);
      } else if (d) {
        uint32_t v;
        std::memcpy(&v, at, 4);
        for (unsigned l = 0; l < 32; ++l) d[l] = v;
      }
      count_shared(blk.st, Store, 32, 4, 1);
      return;
    }
  }
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
  exec::LocalWindowCheck window;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t off = static_cast<uint32_t>((base ? base[l] : 0) + urv + imm);
    addrs[l] = off;
    window.add(off);
  }
  const bool inside = window.inside(w.local_size, bytes, ex != 0);
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

TIER_HANDLER(h_cs2r) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;
  uint32_t* d = c.R + op.dst * 32u;
  const unsigned idx = op.a;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint32_t lo = idx == 0xff ? 0 : c.r->tier_sreg(*c.blk, *c.w, idx, l);
    if (op.b) {
      const uint32_t hi = idx == 0xff ? 0 : c.r->tier_sreg(*c.blk, *c.w, idx + 1, l);
      d[l] = lo;
      if (op.dst + 1u != kRZ) d[32 + l] = hi;
    } else {
      d[l] = lo;
    }
  }
}

TIER_HANDLER(h_sel) {
  TIER_PRELUDE
  alignas(32) uint32_t t[2][32], r[32];
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const Mask p = tier_pmask(c, op.p);
  tier_copy_row(r, b);               // b where the predicate is clear, a where it holds: the blend kernel (fast_wide's SEL)
  c.sk->blend_m(r, a, p);
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
  const uint64_t p = tier_pmask(c, op.p);
  // (The predicate chooses the minimum per lane: fast_wide's IMNMX, the same two kernel calls.)
  c.sk->ibin_m(static_cast<unsigned>(simd::IntBin::Min), op.c != 0, r, a, b, p, 32);
  c.sk->ibin_m(static_cast<unsigned>(simd::IntBin::Max), op.c != 0, r, a, b, ~p & 0xffffffffull, 32);
  tier_store(c, op.dst, ex, r);
}

// SHFL: the value of another lane of the warp, picked by the lane operand and the clamp (fast path of exec_warp: the three sources as
// 32 lanes, the answer worked out for all of them, then written for the executing lanes). A lane that is not running the
// shuffle gives 0.
// Whether all 32 words of a row are the same, and then that word.
inline bool tier_row_uniform(const uint32_t* r, uint32_t& v) {
#if defined(__SSE2__)
  const __m128i first = _mm_set1_epi32(static_cast<int>(r[0]));
  __m128i eq = _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r)), first);
  for (unsigned i = 1; i < 8; ++i) eq = _mm_and_si128(eq, _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r) + i), first));
  v = r[0];
  return _mm_movemask_epi8(eq) == 0xffff;
#else
  v = r[0];
  for (unsigned l = 1; l < 32; ++l)
    if (r[l] != v) return false;
  return true;
#endif
}

// The shuffle `sh` of the 32 words at vp into out, for the whole warp.
inline void tier_shfl_move(const tier::ShflShape& sh, const uint32_t* vp, uint32_t* out) {
  using tier::ShflShape;
  switch (sh.kind) {
    case ShflShape::None:
    case ShflShape::Ident: tier_copy_row(out, vp); return;
    case ShflShape::Splat: {
      const uint32_t v = vp[sh.x];
      for (unsigned l = 0; l < 32; ++l) out[l] = v;
      return;
    }
    case ShflShape::Butterfly: {
#if defined(__SSE2__)
      const unsigned vx = sh.x >> 2;   // (the vector the four lanes come from, then the order within it)
      const __m128i* in = reinterpret_cast<const __m128i*>(vp);
      __m128i* o = reinterpret_cast<__m128i*>(out);
      const auto run = [&](auto imm) {
        __m128i v[8];
        for (unsigned i = 0; i < 8; ++i) v[i] = _mm_shuffle_epi32(_mm_loadu_si128(in + (i ^ vx)), decltype(imm)::value);
        for (unsigned i = 0; i < 8; ++i) _mm_storeu_si128(o + i, v[i]);
      };
      switch (sh.x & 3) {
        case 0: run(std::integral_constant<int, _MM_SHUFFLE(3, 2, 1, 0)>{}); break;
        case 1: run(std::integral_constant<int, _MM_SHUFFLE(2, 3, 0, 1)>{}); break;
        case 2: run(std::integral_constant<int, _MM_SHUFFLE(1, 0, 3, 2)>{}); break;
        default: run(std::integral_constant<int, _MM_SHUFFLE(0, 1, 2, 3)>{}); break;
      }
#else
      for (unsigned l = 0; l < 32; ++l) out[l] = vp[l ^ sh.x];
#endif
      return;
    }
    case ShflShape::Down:   // lanes 0 .. n-1 read the lane x above, the others their own
      for (unsigned l = 0; l < 32; ++l) out[l] = ((sh.in >> l) & 1) ? vp[(l + sh.x) & 31] : vp[l];
      return;
    default:   // Up
      for (unsigned l = 0; l < 32; ++l) out[l] = ((sh.in >> l) & 1) ? vp[(l - sh.x) & 31] : vp[l];
      return;
  }
}

TIER_HANDLER(h_shfl) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], out[32];
  const uint32_t* vp = tier_src<false>(c, op.s[0], t[0]);
  Mask in_range = 0;
  const uint32_t *bp = nullptr, *cp = nullptr;
  if (!(op.aux & 1)) {
    bp = tier_src<false>(c, op.s[1], t[1]);
    cp = tier_src<false>(c, op.s[2], t[2]);
  }
  if (ex == kAll) {   // a full warp whose lane and clamp are the same for all its lanes: the data moves in one of a few shapes
    tier::ShflShape sh;
    if (op.aux & 1) {   // (constants: worked out by the analysis)
      sh.kind = static_cast<tier::ShflShape::Kind>(op.c & 0xff);
      sh.x = op.c >> 8;
      sh.in = op.b;
    } else {
      uint32_t b, cc;
      if (tier_row_uniform(bp, b) && tier_row_uniform(cp, cc)) tier::shfl_shape(op.a, b, cc, sh);
    }
    if (sh.kind != tier::ShflShape::None) {
      tier_shfl_move(sh, vp, out);
      tier_store(c, op.dst, kAll, out);
      tier_setp(c, op.dst2, kAll, sh.in);
      return;
    }
  }
  if (op.aux & 1) {   // the lane and the clamp are constants: the source lane of each lane and the in-range lanes were worked out by the analysis
    const uint32_t* j = c.pool[op.s[1].off].v;
    if (ex == kAll) {
      for (unsigned l = 0; l < 32; ++l) out[l] = vp[j[l]];
    } else {
      for (unsigned l = 0; l < 32; ++l) out[l] = ((ex >> j[l]) & 1) ? vp[j[l]] : 0;   // (a source lane that is not running gives 0)
    }
    tier_store(c, op.dst, ex, out);
    tier_setp(c, op.dst2, ex, op.b);
    return;
  }
  // (the mode picked once, not per lane)
  const auto lanes = [&](auto pick) {
    for (unsigned l = 0; l < 32; ++l) {
      const uint32_t b = bp[l], cc = cp[l];
      const uint32_t clamp = cc & 0x1f, segmask = (cc >> 8) & 0x1f;
      const uint32_t maxl = (l & segmask) | (clamp & ~segmask), minl = l & segmask;
      uint32_t j;
      bool in;
      pick(l, b, segmask, maxl, minl, j, in);
      if (!in) j = l;
      out[l] = ((ex >> (j & 31)) & 1) ? vp[j & 31] : 0;   // (a source lane that is not running this shuffle gives 0)
      in_range |= static_cast<Mask>(in) << l;
    }
  };
  switch (op.a) {
    case 0: lanes([](unsigned, uint32_t b, uint32_t segmask, uint32_t maxl, uint32_t minl, uint32_t& j, bool& in) {
              j = minl | (b & ~segmask & 0x1f); in = j <= maxl; }); break;
    case 1: lanes([](unsigned l, uint32_t b, uint32_t, uint32_t maxl, uint32_t, uint32_t& j, bool& in) {
              j = l - b; in = static_cast<int32_t>(j) >= static_cast<int32_t>(maxl); }); break;
    case 2: lanes([](unsigned l, uint32_t b, uint32_t, uint32_t maxl, uint32_t, uint32_t& j, bool& in) {
              j = l + b; in = j <= maxl; }); break;
    default: lanes([](unsigned l, uint32_t b, uint32_t, uint32_t maxl, uint32_t, uint32_t& j, bool& in) {
              j = l ^ b; in = j <= maxl; }); break;
  }
  tier_store(c, op.dst, ex, out);
  tier_setp(c, op.dst2, ex, in_range);
}

// VOTE.ALL/ANY/EQ: the predicate of the executing lanes as a ballot, and the verdict.
TIER_HANDLER(h_vote) {
  TIER_PRELUDE
  const Mask ballot = tier_pmask(c, op.p) & ex;
  const bool all = (ballot & ex) == ex, any = ballot != 0, eq = all || !any;
  const bool r = op.a == 0 ? all : op.a == 1 ? any : eq;
  if (op.dst != tier::kNoReg) {
    uint32_t* d = c.R + op.dst * 32u;
    for (Mask m = ex; m; m &= m - 1) d[std::countr_zero(m)] = ballot;
  }
  tier_setp(c, op.dst2, ex, r ? kAll : 0);
}

// MATCH.ANY / MATCH.ALL (exec_warp): the lanes' values are read first, then written, so a destination that is the source is no problem.
// ANY gives every executing lane the mask of the executing lanes that hold the same value (lanes of a class share it, so it is worked out
// once for each class); ALL gives every executing lane the executing mask when all hold one value and 0 otherwise, and the predicate says which.
TIER_HANDLER(h_match) {
  TIER_PRELUDE
  const uint32_t* lo = tier_src<false>(c, op.s[0], nullptr);
  const uint32_t* hi = (op.aux & 2) ? tier_src<false>(c, op.s[1], nullptr) : nullptr;
  uint64_t v[32];
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    v[l] = lo[l] | (hi ? static_cast<uint64_t>(hi[l]) << 32 : 0);
  }
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  if (op.aux & 1) {
    for (Mask todo = ex; todo;) {
      const uint64_t x = v[std::countr_zero(todo)];
      Mask cls = 0;
      for (Mask m = ex; m; m &= m - 1) {
        const unsigned k = static_cast<unsigned>(std::countr_zero(m));
        cls |= static_cast<Mask>(v[k] == x) << k;
      }
      if (d)
        for (Mask m = cls; m; m &= m - 1) d[std::countr_zero(m)] = cls;
      todo &= ~cls;
    }
  } else {
    const uint64_t first = v[std::countr_zero(ex)];
    bool same = true;
    for (Mask m = ex; m; m &= m - 1) same = same && v[std::countr_zero(m)] == first;
    if (d)
      for (Mask m = ex; m; m &= m - 1) d[std::countr_zero(m)] = same ? ex : 0;
    tier_setp(c, op.dst2, ex, same ? kAll : 0);
  }
}

// LDSM (ldmatrix of 16-bit elements, exec_mem): matrix i's row r is at the address of lane 8i + r; lane t gets row t/4's 32-bit word t%4
// (.MT88: the halves of column t/4 from rows 2(t%4) and 2(t%4)+1). A whole warp, every row inside the block's own shared memory: the words
// move straight into the registers. A partial warp (a lane that holds a row address may be missing: the interpreter reports it), an address in
// another block's window and a row that would end past the array are exec_mem's.
// Atomics and reductions: exec_mem's own code (a warp's global atomics are taken together there), without execute()'s probe of the
// interpreter's fast path and its dispatch. The run loop counts the instruction and moves the group on, as execute() does.
TIER_HANDLER(h_atom) {
  TIER_PRELUDE
  c.r->tier_exec_mem(*c.blk, *c.w, *op.ins, ex);
}

TIER_HANDLER(h_ldsm) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Block& blk = *c.blk;
  const unsigned nmat = op.a;
  uint32_t rows[32];
  const unsigned ra = op.s[0].off;
  const uint32_t* base = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? c.w->ur[op.s[1].off] : 0;
  const uint64_t imm = static_cast<uint64_t>(op.imm);
  exec::SharedWindowCheck window;
  for (unsigned l = 0; l < 32; ++l) {
    rows[l] = static_cast<uint32_t>((base ? base[l] : 0u) + urv + imm);
    window.add(rows[l], blk.rank);
  }
  if (ex != kAll || !window.inside(blk.shared.size(), 16, true)) {
    c.r->tier_exec_mem(blk, *c.w, *op.ins, ex);
    return;
  }
  const uint8_t* sh = blk.shared.data();
  uint32_t out[4][32];
  for (unsigned i = 0; i < nmat; ++i)
    for (unsigned t = 0; t < 32; ++t) {
      if (!op.b) {
        std::memcpy(&out[i][t], sh + (rows[8 * i + t / 4] & 0xffffff) + 4 * (t % 4), 4);
      } else {
        uint16_t lo, hi;
        std::memcpy(&lo, sh + (rows[8 * i + 2 * (t % 4)] & 0xffffff) + 2 * (t / 4), 2);
        std::memcpy(&hi, sh + (rows[8 * i + 2 * (t % 4) + 1] & 0xffffff) + 2 * (t / 4), 2);
        out[i][t] = lo | (static_cast<uint32_t>(hi) << 16);
      }
    }
  if (op.dst != tier::kNoReg)
    for (unsigned i = 0; i < nmat; ++i) std::memcpy(c.R + (op.dst + i) * 32u, out[i], 32 * sizeof(uint32_t));
  blk.st.shared_loads += 32;
  blk.st.shared_bytes_read += 32u * 4 * nmat;
}

// IMMA, the dense integer forms (exec_mma): the 8- or 4-bit elements of A (M x K, row-major) and of B (K x 8, kept column by column so that
// a dot product runs along contiguous elements) read out of the fragment registers as exec_mma lays them out, each product summed with the
// accumulator in 64 bits and the result cut to 32 -- the same as a wrapping 32-bit sum -- or, under .SAT, clamped to the s32 range after every
// 128 bits of K. The destination is written for the executing lanes.
TIER_HANDLER(h_imma) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;   // (an RZ destination: the interpreter computes and throws it away)
  const unsigned M = op.a, K = op.b;
  const bool sa = op.aux & 1, sb = op.aux & 2, sat = op.aux & 4, four = op.aux & 8;
  const unsigned per_reg = four ? 8 : 4, a_regs = M * K / (32 * per_reg), b_regs = K * 8 / (32 * per_reg), c_regs = M * 8 / 32;
  const auto elem = [&](uint32_t v, unsigned e, bool sgn) -> int16_t {
    if (four) {
      const unsigned x = (v >> (4 * e)) & 0xfu;
      return sgn ? static_cast<int16_t>(static_cast<int8_t>(static_cast<uint8_t>(x << 4)) >> 4) : static_cast<int16_t>(x);
    }
    const unsigned x = (v >> (8 * e)) & 0xffu;
    return sgn ? static_cast<int16_t>(static_cast<int8_t>(x)) : static_cast<int16_t>(x);
  };
  const auto word = [&](uint32_t base, unsigned i, unsigned l) -> uint32_t { return base == kRZ ? 0u : c.R[(base + i) * 32u + l]; };
  alignas(32) int16_t A[16 * 128], Bt[8 * 128];
  for (unsigned r = 0; r < a_regs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, t = l % 4, row = g + (r % 2) * 8, col0 = t * per_reg + (r / 2) * (per_reg * 4);
      const uint32_t v = word(op.s[0].off, r, l);
      for (unsigned e = 0; e < per_reg; ++e) A[row * K + col0 + e] = elem(v, e, sa);
    }
  for (unsigned r = 0; r < b_regs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, t = l % 4, row0 = t * per_reg + r * (per_reg * 4);
      const uint32_t v = word(op.s[1].off, r, l);
      for (unsigned e = 0; e < per_reg; ++e) Bt[g * K + row0 + e] = elem(v, e, sb);
    }
  const auto cd = [](unsigned l, unsigned slot) { return (l / 4 + (slot / 2) * 8) * 8 + (l % 4) * 2 + slot % 2; };
  int32_t C[16 * 8], D[16 * 8];
  for (unsigned l = 0; l < 32; ++l)
    for (unsigned s = 0; s < c_regs; ++s) C[cd(l, s)] = static_cast<int32_t>(word(op.s[2].off, s, l));
  const unsigned block = sat ? (four ? 32u : 16u) : K;   // 128 bits of K under .SAT
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < 8; ++j) {
      const int16_t* a = A + i * K;
      const int16_t* b = Bt + j * K;
      if (!sat) {
        int32_t acc = 0;
        for (unsigned k = 0; k < K; ++k) acc += static_cast<int32_t>(a[k]) * static_cast<int32_t>(b[k]);   // (|product| < 2^16, K <= 128: no overflow)
        D[i * 8 + j] = static_cast<int32_t>(static_cast<uint32_t>(C[i * 8 + j]) + static_cast<uint32_t>(acc));
      } else {
        int64_t acc = C[i * 8 + j];
        for (unsigned k0 = 0; k0 < K; k0 += block) {
          for (unsigned k = k0; k < k0 + block && k < K; ++k) acc += static_cast<int64_t>(a[k]) * static_cast<int64_t>(b[k]);
          acc = std::clamp<int64_t>(acc, INT32_MIN, INT32_MAX);
        }
        D[i * 8 + j] = static_cast<int32_t>(acc);
      }
    }
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    for (unsigned s = 0; s < c_regs; ++s) c.R[(op.dst + s) * 32u + l] = static_cast<uint32_t>(D[cd(l, s)]);
  }
}

// HMMA (the dense f16, bf16 and tf32 forms): one SIMD kernel over the warp's fragments, as exec_mma gives them to it, the
// answer written to the executing lanes (RZ as an operand is a row of zeros, as a source accumulator a zero accumulator).
TIER_HANDLER(h_hmma) {   // exec_mma's kernel call: the same kernel, and the same choice of the sum (hmma_kernel_kind: VGPU_TC_EXACT, sm_86)
  TIER_PRELUDE
  static const uint32_t kZero[4 * 32] = {};
  const unsigned ra = op.s[0].off, rb = op.s[1].off, rc = op.s[2].off;
  const unsigned c_regs = op.c ? 2 : 4;
  alignas(32) uint32_t out[4 * 32];
  c.sk->hmma(c.r->hmma_kernel_kind(op.a), op.b, op.c != 0, ra == kRZ ? kZero : c.R + ra * 32u, rb == kRZ ? kZero : c.R + rb * 32u,
             rc == kRZ ? nullptr : c.R + rc * 32u, out);
  if (op.dst != tier::kNoReg)
    for (unsigned s = 0; s < c_regs; ++s) tier_store(c, static_cast<uint16_t>(op.dst + s), ex, out + s * 32);
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
    case F::Cs2r: return &h_cs2r;
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
    case F::Fmnmx: return &h_fmnmx;
    case F::Mufu: return &h_mufu;
    case F::F2i: return &h_f2i;
    case F::I2f: return &h_i2f;
    case F::F2fp: return &h_f2fp;
    case F::Dsetp: return &h_dsetp;
    case F::Hop2: return &h_hop2;
    case F::Unary: return &h_unary;
    case F::Iabs: return &h_iabs;
    case F::Ldc: return &h_ldc;
    case F::Shfl: return &h_shfl;
    case F::Vote: return &h_vote;
    case F::Hmma: return &h_hmma;
    case F::Match: return &h_match;
    case F::Imma: return &h_imma;
    case F::Atom: return &h_atom;
    case F::Ldsm: return &h_ldsm;
    case F::Umov: return &h_umov;
    case F::Uldc: return &h_uldc;
    case F::Uiadd3: return &h_uiadd3;
    case F::Uimad: return &h_uimad;
    case F::Uisetp: return &h_uisetp;
    case F::Ulea: return &h_ulea;
    case F::Ulop3: return &h_ulop3;
    case F::Ushf: return &h_ushf;
    case F::Uunary: return &h_uunary;
    case F::Usel: return &h_usel;
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
  std::map<std::string, std::vector<std::string>> samples;   // VGPU_TIER_STATS=2: up to three disassembled examples of each
  std::atomic<uint64_t> total{0};   // every warp instruction of the launches (LaunchStats::instructions)
  std::atomic<uint64_t> passes{0}, chained{0};   // runs of straight instructions the tier made, and how many of them it chained on to
  ~TierStats() {
    uint64_t nat = 0, fb = 0;
    for (size_t i = 0; i < kOps; ++i) nat += native[i].load();
    for (const auto& [k, v] : fallback) fb += v;
    if (!nat && !fb) return;
    // (VGPU_TIER_STATS_FILE: appended to that file, for a test that must not have them among a program's own output)
    std::FILE* out = stderr;
    if (const char* f = std::getenv("VGPU_TIER_STATS_FILE"); f && *f)
      if (std::FILE* g = std::fopen(f, "a")) out = g;
    const uint64_t all = total.load();
    // (VGPU_TIER_STATS=2 lists every mnemonic, 1 the top 24 of each list)
    const size_t shown = [] { const char* v = std::getenv("VGPU_TIER_STATS"); return v && v[0] == '2' ? size_t{1000} : size_t{24}; }();
    std::fprintf(out, "tier: %llu of %llu warp instructions ran in the tier (%.1f%%); %llu straight-line ones in the interpreter\n",
                 (unsigned long long)nat, (unsigned long long)all, all ? 100.0 * double(nat) / double(all) : 0.0,
                 (unsigned long long)fb);
    {
      std::vector<std::pair<uint64_t, size_t>> v;
      for (size_t i = 0; i < kOps; ++i)
        if (native[i].load()) v.emplace_back(native[i].load(), i);
      std::sort(v.rbegin(), v.rend());
      std::string line = "tier native:";
      for (size_t i = 0; i < v.size() && i < shown; ++i)
        line += std::string(" ") + op_name(static_cast<Op>(v[i].second)) + "=" + std::to_string(v[i].first);
      std::fprintf(out, "%s\n", line.c_str());
    }
    const auto list = [&](const char* what, const std::map<std::string, uint64_t>& m) {
      std::vector<std::pair<uint64_t, std::string>> v;
      for (const auto& [k, n] : m) v.emplace_back(n, k);
      std::sort(v.rbegin(), v.rend());
      std::string line = std::string("tier ") + what + ":";
      for (size_t i = 0; i < v.size() && i < shown; ++i) line += " " + v[i].second + "=" + std::to_string(v[i].first);
      std::fprintf(out, "%s\n", line.c_str());
    };
    std::fprintf(out, "tier runs: %llu runs of straight instructions, %llu of them started by the tier's chain instead of a call of the scheduler\n",
                 (unsigned long long)passes.load(), (unsigned long long)chained.load());
    list("interpreter (straight)", fallback);
    list("interpreter (ends a run)", ended);
    for (const auto& [k, v] : samples) {
      std::string line = "tier sample " + k + ":";
      for (const std::string& t : v) line += "  [" + t + "]";
      std::fprintf(out, "%s\n", line.c_str());
    }
    if (out != stderr) std::fclose(out);
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
  static const bool detail = [] { const char* v = std::getenv("VGPU_TIER_STATS"); return v && v[0] == '2'; }();
  if (detail) {
    std::vector<std::string>& v = s.samples[key];
    const std::string text = to_text(ins);
    if (v.size() < 3 && std::find(v.begin(), v.end(), text) == v.end()) v.push_back(text);
  }
}

void tier_add_total(uint64_t n) { tier_stats().total.fetch_add(n, std::memory_order_relaxed); }

// ---- the run loop -------------------------------------------------------------------------------

// What Runner::step_warp's straight-line loop does besides executing an instruction, which this loop must do the same way
// (each is tested by nvidia/tests/e2e/sass_tier_steps.cu, run both ways by run_sass_tier.sh, unless it says otherwise):
//   1. The store gate: a warp's first STG or ST ends its turn, once, if the block has other warps (the pcs of the group are
//      set to the store when it is not the first instruction of the run).
//   2. The step count: every instruction, the one that ends a run too, is a step; one past max_steps is the ExecLimit error.
//   3. At most 256 straight instructions, then one more; the last instruction of a section ends a run.
//   4. One run is one call of step_warp, and the caller counts calls (a warp's turn is 4096 of them). A chain below stands
//      for several calls and says how many (Warp::extra), never more than the turn has left (Warp::budget).
//   5. Which lanes execute: the guard, @!PT; the counters (instructions, thread instructions, per class, divergent branches;
//      tensor issues and barriers are counted by execute(), which runs those instructions).
//   6. A fault names the instruction, the first lane of the group, and the text of the cause; a register past the warp's file
//      is the interpreter's (regs_ok) and so is its out_of_range.
//   7. The lanes' program counters are set once, where the run stops (to the stopping instruction if the run was longer than
//      one); an instruction that is not straight sets its own (BRA) or moves the group on (execute's advance).
//   8. BRA's spin rule: a taken backward branch while other lanes wait counts, 64 of them make the group give way.
//   9. BSSY's barrier mask and its nesting stamp.
//  10. Not done here, because the interpreter's path is used instead: BMOV.PQUAD MACTIVE's forced mask (step_warp does not
//      enter the tier then), the instruction trace (VGPU_SASS_TRACE turns the tier off), a register past the file.
//
// Chaining. step_warp's work between two calls is: release the lanes of a WARPSYNC set that wait their turn (ws_held), pick the
// group at the lowest address, honour a slice a YIELD gave and a give-way, find the code section. When none of the first and the last
// applies (no ws_held, no slice, no give-way, no forced mask) and every runnable lane stands at one address inside this section,
// that work is: group = the runnable lanes, pick = their address, nothing else changed. Then the next call would start exactly
// where this loop is, and the loop goes on instead of returning, crediting the caller with the call it saved. When the runnable lanes
// stand at several addresses the call picks the group at the lowest (step_warp's own rule, without a slice or a give-way), and the loop
// does that too, on to the group's first instruction if it is in this section.
// guard_mask, inlined into the tier's run loop (the interpreter's copy is a call, which its own loop does not mind).
[[gnu::always_inline]] inline Mask tier_guard(const Warp& w, const Instr& ins, Mask group) {
  if (ins.guard == kPT && !ins.guard_not) return group;
  Mask g;
  if (ins.guard_uniform) g = (ins.guard == kPT || w.up[ins.guard]) ? kAll : 0;
  else g = ins.guard == kPT ? kAll : w.p[ins.guard];
  if (ins.guard_not) g = ~g;
  return group & g;
}

// The lanes of `group` get program counter `v` (a whole warp: 32 stores the compiler turns into vector ones).
inline void tier_set_pcs(Warp& w, Mask group, uint64_t v) {
  if (group == kAll) {
    for (unsigned l = 0; l < 32; ++l) w.pc[l] = v;
  } else {
    for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = v;
  }
}

// The lanes of `ex` start to wait with `kind` on `arg`.
inline void tier_wait(Warp& w, Mask ex, Wait kind, unsigned arg) {
#if defined(__SSE2__)
  if (ex == kAll) {
    const __m128i k = _mm_set1_epi8(static_cast<char>(kind)), a = _mm_set1_epi32(static_cast<int>(arg));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[0]), k);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[16]), k);
    for (unsigned i = 0; i < 32; i += 4) _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_arg[i]), a);
    w.waiting |= ex;
    return;
  }
#endif
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    w.wait_kind[l] = kind;
    w.wait_arg[l] = arg;
  }
  w.waiting |= ex;
}

// The lanes of `w` that wait with `kind` on `arg` (the two arrays compared a vector at a time instead of looked at lane by lane).
inline Mask tier_waiting_on(const Warp& w, Wait kind, unsigned arg) {
  Mask arrived = w.waiting;
  if (!arrived) return 0;
#if defined(__SSE2__)
  const __m128i k = _mm_set1_epi8(static_cast<char>(kind));
  const uint32_t kinds = static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.wait_kind[0])), k))) |
                         static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.wait_kind[16])), k))) << 16;
  const __m128i want = _mm_set1_epi32(static_cast<int>(arg));
  uint32_t args = 0;
  for (unsigned i = 0; i < 8; ++i)
    args |= static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(
                _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.wait_arg[4 * i])), want)))) << (4 * i);
  return arrived & kinds & args;
#else
  Mask match = 0;
  for (Mask m = arrived; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    if (w.wait_kind[l] == kind && w.wait_arg[l] == arg) match |= Mask{1} << l;
  }
  return match;
#endif
}

// The lanes of `m` (waiting) stop waiting and go on past the instruction they wait at.
inline void tier_wake(Warp& w, Mask m) {
  if (!m) return;
  w.waiting &= ~m;
#if defined(__SSE2__)
  if (m == kAll) {   // a whole warp: the kinds cleared and the 32 addresses moved on a vector at a time
    const __m128i zero = _mm_setzero_si128(), step = _mm_set1_epi64x(16);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[0]), zero);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[16]), zero);
    for (unsigned i = 0; i < 32; i += 2)
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.pc[i]), _mm_add_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.pc[i])), step));
    return;
  }
#endif
  for (; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    w.wait_kind[l] = Wait::None;
    w.pc[l] += 16;
  }
}

// BSYNC's release (exec_control's release_bsync): once every lane of barrier Bx that is still a thread of the block waits at a
// BSYNC on it, they all go on past it and the barrier is free for its next BSSY. Lanes that wait at a BSYNC on Bx but are not
// named by the barrier (a callee that shares Bx with its caller) stay.
inline void tier_release_bsync(Warp& w, unsigned x) {
  const Mask members = w.b[x] & w.alive & ~w.exited;
  Mask arrived = tier_waiting_on(w, Wait::BSync, x);   // (the lanes that wait at a BSYNC on Bx)
  if (members) arrived &= members;
  if ((members & ~arrived) == 0 && arrived) {
    tier_wake(w, arrived);
    w.b[x] &= ~arrived;   // they have converged: the barrier is free for its next BSSY
  }
}

// A block barrier's release (exec_control's release_bar): once the barrier has all the threads it waits for -- every live thread of
// the block, or the count it was named with -- the warps' lanes waiting at it go on, every warp learns the reduction's result, and
// the barrier is reset.
inline void tier_release_bar(Block& blk, unsigned id) {
  Block::Barrier& b = blk.bars[id];
  const uint32_t expected = b.expected ? b.expected : blk.live;
  if (b.arrived < expected) return;
  for (Warp& ww : blk.warps) tier_wake(ww, tier_waiting_on(ww, Wait::Bar, id));
  for (Warp& ww : blk.warps) {
    ww.b2r = b.red == 0 ? b.popc : b.red == 1 ? b.and_ : b.or_;
    ww.b2r_pred = b.red == 0 ? b.popc != 0 : b.red == 1 ? b.and_ : b.or_;
  }
  b = Block::Barrier{};
}

// ---- tier 2 (src/sass/jit.hpp) ----------------------------------------------------------------------------------

// What the generated code's escape needs: the handler's context, where the region's first op is, and the op the escape last ran (the one a
// fault names: tier_run's `cur`).
struct JitHost {
  TCtx* c;
  const tier::MOp* ops;      // the region's first op
  const tier::MOp* last;
  bool stats;
};

// One op of a region, as the run loop runs it after counting it: the handler for the lanes that execute it, then the tier's statistics
// (reported even when no lane executes, as the loop reports them). `ex` is the generated code's own count of the guard, the loop's.
void jit_note(jit::Ctx* j, uint32_t i) {
  JitHost& h = *static_cast<JitHost*>(j->host);
  if (h.stats) [[unlikely]] tier_stats().native[static_cast<size_t>(h.ops[i].ins->op)].fetch_add(1, std::memory_order_relaxed);
}

void jit_escape(jit::Ctx* j, uint32_t i, uint32_t ex) {
  JitHost& h = *static_cast<JitHost*>(j->host);
  const tier::MOp& op = h.ops[i];
  h.last = &op;
  if (ex) op.fn(h.c, op, ex);
  if (h.stats) [[unlikely]] tier_stats().native[static_cast<size_t>(op.ins->op)].fetch_add(1, std::memory_order_relaxed);
}

// Runs the region that starts at ops[idx] if it is compiled and the run loop could run all its ops here: what the loop checks for each
// of them is checked once for all -- the 256-instruction run (n + k), the step budget (steps + k), the registers its handlers touch. A region
// holds no store, so the store gate cannot trip inside one. When a check fails, or the region is not compiled yet (counting towards its
// promotion), nothing is run and the ops go the old way, one at a time. The generated code counts every op as the loop does; when it throws,
// `faulting` is the op the escape last ran, which the fault names (the loop's `cur`). Returns the number of ops run, 0 for none.
// (What a tier_run call keeps for its regions: the context the generated code is given, set up once, on the first region it runs.)
struct JitFrame {
  jit::Ctx jc;
  JitHost jh;
  bool init = false;
};

[[gnu::noinline]] uint32_t jit_run_region(JitFrame& f, jit::Unit& ju, uint32_t ri, TCtx& c, Warp& w, exec::LaunchStats& st, const tier::MOp* ops, size_t idx,
                                          unsigned n, Mask group, unsigned glanes, uint64_t max_steps, bool stats, const tier::MOp*& faulting) {
  const jit::Region& rg = ju.regions[ri - 1];
  if (!jit::admissible(n, rg.k, w.steps, max_steps, rg.maxreg, w.nregs)) return 0;
  jit::RegionFn fn = ju.fn(ri - 1);
  if (!fn) {   // not compiled (yet, or not worth it): while the section is being counted, this run counts for it
    if (!ju.counting()) return 0;
    ju.note(ri - 1, rg.k);
    fn = ju.fn(ri - 1);
    if (!fn) return 0;
  }
  if (!f.init) [[unlikely]] {
    f.init = true;
    f.jh = JitHost{&c, ops, nullptr, stats};
    jit::Ctx& jc = f.jc;
    jc = jit::Ctx{};
    jc.abi = jit::kAbiVersion;
    jc.flags = stats ? 1u : 0u;
    jc.up = w.up;
    jc.p = w.p;
    jc.ur = w.ur;
    jc.note = &jit_note;
    jc.steps = &w.steps;
    jc.instructions = &st.instructions;
    jc.thread_instructions = &st.thread_instructions;
    jc.by_class = st.inst_by_class;
    jc.escape = &jit_escape;
    jc.host = &f.jh;
    jc.b0 = c.b0;
    jc.b0n = c.b0n;
    jc.pool = c.pool;
  }
  f.jc.R = c.R;   // (the register file may have moved since the last pass)
  f.jh.ops = &ops[idx];
  f.jh.last = nullptr;
  try {
    fn(&f.jc, group, glanes);
  } catch (...) {
    faulting = f.jh.last;
    if (jit::stats_on()) jit::stats().region_throws.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
  if (jit::stats_on()) [[unlikely]] {
    jit::stats().regions_run.fetch_add(1, std::memory_order_relaxed);
    jit::stats().instructions_run.fetch_add(rg.k, std::memory_order_relaxed);
    jit::stats().generated_run.fetch_add(rg.gen, std::memory_order_relaxed);
  }
  return rg.k;
}

template <bool Jit>
bool Runner::tier_run_t(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp) {
  TCtx c{&w, &blk, this, w.r.data(), bank0_.data(), bank0_.size(), tp.pool.data(), &simd::kernels()};
  exec::LaunchStats& st = blk.st;
  const tier::MOp* ops = tp.ops.data();
  const size_t count = code.count;
  bool regs_ok = true;
  [[maybe_unused]] jit::Unit* const ju = Jit ? tp.jit.get() : nullptr;
  [[maybe_unused]] JitFrame jf;
  [[maybe_unused]] const uint32_t* const region_at = Jit ? ju->region_at.data() : nullptr;
  for (bool again = false;; again = true) {   // one pass is one call of step_warp
  // The warp's register file may have been resized since the last pass (USETMAXREG.TRY_ALLOC grows it, and a vector that grows moves):
  // the row pointer and the check that a program's registers are inside the file are worked out again for every pass.
  c.R = w.r.data();
  regs_ok = tp.max_reg <= w.nregs;
  if (tier_stats_) [[unlikely]] {
    tier_stats().passes.fetch_add(1, std::memory_order_relaxed);
    if (again) tier_stats().chained.fetch_add(1, std::memory_order_relaxed);
  }
  const size_t first = static_cast<size_t>((pick - code.base) / 16);
  size_t idx = first;
  const unsigned glanes = popc(group);
  const tier::MOp* cur = &ops[idx];
  try {
    for (unsigned n = 0;; ++n) {
      // Tier 2: a region that starts here, compiled, is run in place of its ops (jit_run_region says how, and when not).
      if constexpr (Jit) {
        if (const uint32_t ri = region_at[idx]) {
          const tier::MOp* faulting = nullptr;   // (not `cur`: its address would keep it out of a register in the loop below)
          uint32_t k;
          try {
            k = jit_run_region(jf, *ju, ri, c, w, st, ops, idx, n, group, glanes, max_steps_, tier_stats_, faulting);
          } catch (...) {
            if (faulting) cur = faulting;
            throw;
          }
          if (k) {
            cur = &ops[idx + k - 1];
            idx += k;
            n += k - 1;
            continue;
          }
        }
      }
      // step_warp's store gate: a warp's first store to global memory ends its turn, once, so that the block's other warps
      // issue the loads their prologues make before it lands (the comment there says why). The tier must stop where
      // the interpreter does, or a block's warps see each other's stores in another order.
      if (w.store_gate && (ops[idx].ins->op == Op::STG || ops[idx].ins->op == Op::ST)) {
        w.store_gate = false;
        if (blk.warps.size() > 1) {
          if (idx != first) tier_set_pcs(w, group, code.base + 16 * static_cast<uint64_t>(idx));
          w.yield = true;
          return true;
        }
      }
      if (++w.steps > max_steps_)
        throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": a warp ran " + std::to_string(max_steps_) +
                                      " instructions (an infinite loop?)");
      const tier::MOp& op = ops[idx];
      cur = &op;
      if (n < 256 && idx + 1 < count && op.straight) {
        if ((op.fn && (regs_ok || op.maxreg <= w.nregs)) || op.never) {
          const Mask ex = op.unguarded ? group : tier_guard(w, *op.ins, group);
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
    // The instruction that ends the run (cur is it).
    const tier::MOp& last = ops[idx];
    const uint64_t here = code.base + 16 * static_cast<uint64_t>(idx);
    if (last.form == tier::Form::Bra) {
      // exec_control's BRA for a fixed target: the lanes that take it go there, the others on to the next instruction; a condition
      // predicate beside the guard narrows the lanes that take it (with_condition); a group that branches backwards while other
      // lanes wait gives way now and then. Counted as execute() counts it (a branch is divergent when the guard splits the group).
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      Mask take = ex;
      if (last.aux) {
        take &= tier_pmask(c, last.p);
        if (last.aux > 1) take &= tier_pmask(c, last.p2);
      }
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      if (ex && ex != group) ++st.divergent_branches;
      const uint64_t target = code.base + static_cast<uint64_t>(last.imm);
      if (take == group || !take) {
        tier_set_pcs(w, group, take ? target : here + 16);
      } else {
        const uint64_t next = here + 16;   // (a select without a branch: the lanes that take it are unpredictable)
        for (Mask m = group; m; m &= m - 1) {
          const unsigned l = static_cast<unsigned>(std::countr_zero(m));
          w.pc[l] = next ^ ((target ^ next) & (uint64_t{0} - ((take >> l) & 1)));
        }
      }
      if (take && last.imm <= static_cast<int64_t>(last.ins->pc) && (w.runnable() & ~group)) {
        if (++w.spins >= 64) {
          w.spins = 0;
          w.give_way = true;
          w.yield_pc = target;   // the group that branched back, at the top of its loop (exec_control's BRA)
        }
      }
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::BRA)].fetch_add(1, std::memory_order_relaxed);
    } else if (last.fn && !last.straight && (regs_ok || last.maxreg <= w.nregs)) {
      // PLOP3, S2R, SEL, SHFL, VOTE: a handler, and the next address (execute() advances the group when nothing set the counters).
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      if (last.ins->tensor) ++st.tensor_instructions;
      if (ex) last.fn(&c, last, ex);
      tier_set_pcs(w, group, here + 16);
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(last.ins->op)].fetch_add(1, std::memory_order_relaxed);
    } else if (last.form == tier::Form::Bssy || last.form == tier::Form::Nop) {
      // These leave the program counters to execute()'s advance: every lane of the group goes on to the next address.
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      if (last.form == tier::Form::Bssy) {
        // The lanes join the barrier, in nesting order (see exec_control's BSSY).
        w.b[last.dst] |= ex;
        w.bssy_stamp[last.dst & 15] = ++blk.bssy_clock;
      }
      tier_set_pcs(w, group, here + 16);
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(last.ins->op)].fetch_add(1, std::memory_order_relaxed);
    } else if (last.form == tier::Form::Yield) {
      // exec_control's YIELD: the group that yielded gives way (the lowest address's group is the one that ran), whatever its guard says.
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      w.slice = 0;
      w.give_way = true;
      w.yield = true;
      ++w.idle_yields;
      w.yield_pc = here + 16;   // where the lanes that yielded stand once the YIELD is done (exec_control's YIELD)
      tier_set_pcs(w, group, here + 16);
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::YIELD)].fetch_add(1, std::memory_order_relaxed);
    } else if (last.form == tier::Form::Exit && w.waiting == 0 && [&] {
                 for (const Block::Barrier& b : blk.bars)
                   if (b.arrived) return false;
                 return true;
               }()) {
      // exec_control's EXIT when nothing waits for it: no lane waits at a BSYNC or a WARPSYNC and no block barrier has an arrival, so
      // the releases it runs for the lanes that leave (every convergence barrier, the warp-sync sets, every block barrier) have
      // nothing to release. (With a waiting lane, or a barrier part way, the interpreter runs it.)
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      Mask go = ex;
      if (last.aux) {
        go &= tier_pmask(c, last.p);
        if (last.aux > 1) go &= tier_pmask(c, last.p2);
      }
      blk.live -= popc(go & w.alive & ~w.exited);
      w.exited |= go;
      for (Mask m = group & ~go; m; m &= m - 1) w.pc[std::countr_zero(m)] = here + 16;
      cluster_barrier_check(blk);
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::EXIT)].fetch_add(1, std::memory_order_relaxed);
    } else if (last.form == tier::Form::Bar) {
      // exec_control's BAR: the threads that run it arrive (counted: per thread, or 32 for a warp when the barrier has a thread count);
      // BAR.SYNC and BAR.RED make them wait here, BAR.ARV lets them go on; the lanes that do not run it go on. Then the release.
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      if (ex) ++st.barriers;
      if (idx != first) tier_set_pcs(w, group, here);
      const unsigned id = (last.aux & 1) ? last.c : (tier_sc(c, last.s[0], 0) & 15);
      Block::Barrier& b = blk.bars[id];
      b.mode = last.a;
      b.red = last.b;
      const uint32_t count = static_cast<uint32_t>(last.imm);
      if (count) {
        b.expected = count;
        if (ex) b.arrived += 32;
      } else {
        b.arrived += popc(ex);
      }
      if (last.a == 2 && ex) {
        const Mask pm = tier_pmask(c, last.p) & ex;
        b.popc += popc(pm);
        b.and_ = b.and_ && pm == ex;
        b.or_ = b.or_ || pm != 0;
      }
      if (last.a == 1) {   // ARV: arrive without waiting
        tier_set_pcs(w, group, here + 16);
      } else {
        tier_wait(w, ex, Wait::Bar, id);
        for (Mask m = group & ~ex; m; m &= m - 1) w.pc[std::countr_zero(m)] = here + 16;
      }
      tier_release_bar(blk, id);
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::BAR)].fetch_add(1, std::memory_order_relaxed);
    } else if (last.form == tier::Form::Bsync) {
      // exec_control's BSYNC: the lanes that run it wait at the barrier (at this address, until released), the others of the group
      // go on; then the barrier is released if all its lanes have arrived.
      const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
      const unsigned lanes = ex == group ? glanes : popc(ex);
      ++st.instructions;
      st.thread_instructions += lanes;
      st.inst_by_class[last.cls] += lanes;
      if (idx != first) tier_set_pcs(w, group, here);
      const unsigned x = last.c;
      tier_wait(w, ex, Wait::BSync, x);
      for (Mask m = group & ~ex; m; m &= m - 1) w.pc[std::countr_zero(m)] += 16;
      tier_release_bsync(w, x);
      if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::BSYNC)].fetch_add(1, std::memory_order_relaxed);
    } else {
      if (tier_stats_) [[unlikely]] tier_note_fallback(*last.ins, true);
      if (idx != first) tier_set_pcs(w, group, here);
      execute(blk, w, *ops[idx].ins, group);
    }
  } catch (const Fault& f) {
    fault(w, *cur->ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what());
  } catch (const std::out_of_range& f) {   // a register past the warp's file
    fault(w, *cur->ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), f.what(), Err::Internal);
  } catch (const Error& e) {
    if (e.message().rfind("SASS kernel ", 0) == 0) throw;
    fault(w, *cur->ins, static_cast<unsigned>(std::countr_zero(group ? group : 1u)), e.message(), e.code());
  }
  // The chain (see above).
  if (!tier_chain_ || w.budget == 0 || w.yield || w.slice || w.give_way || w.ws_held || w.mactive_force) return true;
  const Mask run = w.runnable();
  if (!run) return true;
  uint64_t lo = w.pc[std::countr_zero(run)];
  Mask at = tier_lanes_at(w, lo);
  if ((run & ~at) != 0) {   // diverged: the group at the lowest address, which is what step_warp picks when nothing else is asked of it
    lo = tier_lowest_pc(w, run, lo);
    at = tier_lanes_at(w, lo);
  }
  if (lo - code.base >= 16 * static_cast<uint64_t>(count) || ((lo - code.base) & 15)) return true;
  --w.budget;
  ++w.extra;
  group = run & at;
  pick = lo;
  }
}

// The run loop without tier 2's hook is the loop of every run that does not ask for the JIT, so it has no trace of it (tier_run_t is always_inline, so
// it is inlined where the loop was before tier 2); step_warp calls tier_run_jit instead for a section that has a unit.
[[gnu::noinline]] bool Runner::tier_run_jit(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp) {
  if (tp.jit->state.load(std::memory_order_relaxed) == static_cast<int>(jit::State::Failed))   // (failed, or nothing worth compiling: the plain loop)
    return tier_run_t<false>(blk, w, group, pick, code, tp);
  return tier_run_t<true>(blk, w, group, pick, code, tp);
}
bool Runner::tier_run(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp) {
  return tier_run_t<false>(blk, w, group, pick, code, tp);
}
