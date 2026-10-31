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
  if (r == d) return;   // (tier_out: the kernel wrote the destination itself)
  if (ex == kAll) tier_copy_row(d, r);
  else c.sk->blend_m(d, r, ex);   // the executing lanes only, without a branch per lane
}

// Where a kernel that works lane by lane (each lane of its result from the same lane of its sources, the sources of a block of lanes all read before
// the block is written: every kernel of vgpu/simd/simd.hpp that the handlers below name) writes its result: the destination register itself when the whole
// warp runs the instruction (the row copy of tier_store is then 16 loads and stores less per instruction, and a destination that is also a source
// is no problem to such a kernel), else the temporary `tmp`, from which tier_store blends the lanes that run.
[[gnu::always_inline]] inline uint32_t* tier_out(const TCtx& c, uint16_t dst, Mask ex, uint32_t* tmp) {
  return ex == kAll && dst != tier::kNoReg ? c.R + dst * 32u : tmp;
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
  alignas(32) uint32_t t[3][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  c.sk->iadd3(r, a, b, d, op.a);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_imad) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
  const uint32_t *a = tier_src<false>(c, op.s[0], t[0]), *b = tier_src<false>(c, op.s[1], t[1]),
                 *d = tier_src<false>(c, op.s[2], t[2]);
  c.sk->imad(r, a, b, d);
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_lop3) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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
  alignas(32) uint32_t t[3][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
  const bool hi = op.aux & 1, x = op.aux & 2, three = op.aux & 4, bneg = op.aux & 8;
  const Mask cin = x ? tier_pmask(c, op.p) : 0;
  const uint32_t* a = tier_src<false>(c, op.s[0], t[0]);
  const uint32_t* b = tier_src<false>(c, op.s[1], t[1]);
  const uint32_t* top = three ? tier_src<false>(c, op.s[2], t[2]) : nullptr;   // (none: the high word is a's sign)
  const Mask carry = exec::sasslea::lea_lanes({hi, bneg, op.a}, r, a, b, top, cin);   // (vgpu/exec/sass_lea.hpp: four lanes to a vector)
  tier_store(c, op.dst, ex, r);
  if (op.dst2 != tier::kNoReg) tier_setp(c, op.dst2, ex, carry);
}

TIER_HANDLER(h_shf) {
  TIER_PRELUDE
  alignas(32) uint32_t t[3][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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
  alignas(32) uint32_t t[3][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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
  alignas(32) uint32_t t[2][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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
  alignas(32) uint32_t t[2][32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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
  alignas(32) uint32_t t[32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
  c.sk->lane_unary(0, r, tier_src<false>(c, op.s[0], t));   // the number of set bits
  tier_store(c, op.dst, ex, r);
}

TIER_HANDLER(h_i2fp) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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

// F2F between f32 and f64 and I2F from a 32-bit integer to f64: the SIMD kernels Runner::fast_float calls (f2f_widen, f2f_narrow, i2d). op.a: 0 f32 to
// f64, 1 f64 to f32 (op.c the rounding mode), 2 and 3 an unsigned or signed integer to f64; op.b the source's |x| (bit 0) and -x (bit 1), which the kernels apply.
TIER_HANDLER(h_cvt64) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;   // nothing is written to RZ
  if (op.a == 1) {
    alignas(32) uint32_t tlo[32], thi[32], rt[32];
    const uint32_t *lo, *hi;
    tier_src64(c, op.s[0], tlo, thi, lo, hi);
    uint32_t* const r = tier_out(c, op.dst, ex, rt);
    if (op.c) {   // a directed rounding mode: the host's, around the kernel
      RoundingScope scope(op.c);
      c.sk->f2f_narrow(lo, hi, op.b, r);
    } else {
      c.sk->f2f_narrow(lo, hi, op.b, r);
    }
    tier_store(c, op.dst, ex, r);
    return;
  }
  alignas(32) uint32_t t[32], rlo[32], rhi[32], sink[32];
  const uint32_t* const a = tier_src<false>(c, op.s[0], t);
  const bool full = ex == kAll;
  uint32_t* const dlo = full ? c.R + op.dst * 32u : rlo;
  uint32_t* const dhi = op.dst + 1u == kRZ ? sink : full ? c.R + (op.dst + 1u) * 32u : rhi;   // (R254's partner is RZ: its half goes nowhere)
  if (op.a == 0) c.sk->f2f_widen(a, op.b, dlo, dhi);
  else c.sk->i2d(op.a == 3, a, dlo, dhi);
  if (!full) {
    c.sk->blend_m(c.R + op.dst * 32u, rlo, ex);
    if (op.dst + 1u != kRZ) c.sk->blend_m(c.R + (op.dst + 1u) * 32u, rhi, ex);
  }
}

// What cvt64 and the f32 forms leave of F2F, I2F and MUFU: Runner::fast_float's code, then exec_float's.
TIER_HANDLER(h_fltinterp) {
  TIER_PRELUDE
  c.r->tier_float_op(*c.blk, *c.w, *op.ins, ex);
}

TIER_HANDLER(h_unary) {
  TIER_PRELUDE
  alignas(32) uint32_t t[32], rt[32];
  uint32_t* const r = tier_out(c, op.dst, ex, rt);
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

// The lanes' global addresses: a register (a pair if `bhi`) and a uniform register plus an offset. A full warp is a counted loop.
inline void tier_global_addresses(const uint32_t* blo, const uint32_t* bhi, uint64_t add, Mask ex, uint64_t* addrs) {
  if (ex == kAll) {
    if (blo && bhi) {
      for (unsigned l = 0; l < 32; ++l) addrs[l] = (blo[l] | (static_cast<uint64_t>(bhi[l]) << 32)) + add;
    } else if (blo) {
      for (unsigned l = 0; l < 32; ++l) addrs[l] = uint64_t{blo[l]} + add;
    } else {
      for (unsigned l = 0; l < 32; ++l) addrs[l] = add;
    }
    return;
  }
  for (unsigned l = 0; l < 32; ++l) addrs[l] = 0;   // (the lanes that do not run read as zero, as the counting functions find them)
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint64_t v = 0;
    if (blo) {
      v = blo[l];
      if (bhi) v |= static_cast<uint64_t>(bhi[l]) << 32;
    }
    addrs[l] = v + add;
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
  constexpr uint64_t kWinEnd = kConstWindow + (uint64_t{1} << 32);
  MemoryManager& mem = c.r->memory();
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  if (ex && ((ex + (ex & -ex)) & ex) == 0) {
    // Consecutive lanes on consecutive 16-byte pieces (a float4 or an int4 array indexed by the thread): one run of memory, made as
    // 8-byte pieces like the general access, with the lanes' four words shuffled between it and the four register rows. Found on the
    // registers' rows, before the lanes' addresses are made (they are made below for anything that is not this).
    const unsigned first = static_cast<unsigned>(std::countr_zero(ex)), count = popc(ex);
    uint64_t g0;
    if (exec::sassmem::global_run(blo, bhi, urv + imm, first, count, 16, g0) &&
        (g0 >= kWinEnd || g0 + uint64_t{count} * 16 - 1 < kConstWindow)) {
      uint32_t buf[32 * 4];
      bool done;
      if (Store) {
        for (unsigned i = 0; i < count; ++i)
          for (unsigned k = 0; k < 4; ++k) buf[4 * i + k] = d ? d[32 * k + first + i] : 0u;
        done = mem.store_run(g0, buf, uint64_t{count} * 16, 8);
      } else {
        done = mem.load_run(g0, buf, uint64_t{count} * 16, 8);
        if (done && d)
          for (unsigned i = 0; i < count; ++i)
            for (unsigned k = 0; k < 4; ++k) d[32 * k + first + i] = buf[4 * i + k];
      }
      if (done) {
        count_run(blk.st, Store, g0, first, count, ex, 16);
        return;
      }
    }
  }
  uint64_t addrs[32];
  tier_global_addresses(blo, bhi, urv + imm, ex, addrs);
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    if (addrs[l] >= kConstWindow && addrs[l] < kWinEnd) {   // nothing is done yet: the interpreter runs the whole instruction
      c.r->tier_exec_mem(blk, w, *op.ins, ex);
      return;
    }
  }
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
    // Whether the lanes are on consecutive elements is asked of the registers' rows, before the lanes' addresses are made (they are made
    // below, for an access that is not a run or that the memory manager declines).
    uint64_t g0;
    const bool run = exec::sassmem::global_run(blo, bhi, urv + imm, first, count, bytes, g0) && !(g0 >= kConstWindow && g0 < kWinEnd);
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
  uint64_t addrs[32];
  if (ex == kAll) {
    // A full warp: the addresses are the batch as they stand (no lane to skip, none to find again when the values go back to the
    // registers).
    uint64_t fv[32];
    const uint64_t nearest = exec::sassmem::full_warp_addresses(blo, bhi, urv + imm, kConstWindow, addrs);
    if (nearest >= kWinEnd - kConstWindow) {   // (no lane in the constant window)
      bool done;
      if (Store) {
        for (unsigned l = 0; l < 32; ++l) {
          uint32_t v[2];
          data(l, v);
          fv[l] = bytes == 8 ? (static_cast<uint64_t>(v[1]) << 32 | v[0]) : bytes == 4 ? v[0] : (v[0] & ((1u << (8 * bytes)) - 1));
        }
        done = mem.store_batch(addrs, 32, bytes, fv);
      } else {
        done = mem.load_batch(addrs, 32, bytes, fv);
        if (done && dreg) {
          uint32_t* d = c.R + op.dst * 32u;
          if (bytes == 4) {
            for (unsigned l = 0; l < 32; ++l) d[l] = static_cast<uint32_t>(fv[l]);
          } else if (bytes < 4) {
            for (unsigned l = 0; l < 32; ++l) d[l] = tier_extend(op.b, static_cast<uint32_t>(fv[l]));
          } else {
            for (unsigned l = 0; l < 32; ++l) {
              d[l] = static_cast<uint32_t>(fv[l]);
              d[32 + l] = static_cast<uint32_t>(fv[l] >> 32);
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
    return;
  }
  tier_global_addresses(blo, bhi, urv + imm, ex, addrs);
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

// A whole warp's 4-byte shared accesses that are not one run of 32 words: each group of eight lanes is moved as 32 bytes when its
// offsets are eight consecutive words (a tile row of 8, 16 or 32 words, or two rows in one warp), else lane by lane. The groups run in
// order, so where two lanes name one word the later lane's value is the one stored, as in the lane loop (the words of one group are all
// different).
template <bool Store>
[[gnu::noinline]] inline void tier_shared_words(uint8_t* sh, uint32_t* d, const uint32_t* offs) {
  for (unsigned g = 0; g < 4; ++g) {
    const uint32_t* o = offs + 8 * g;
    uint32_t* dg = d ? d + 8 * g : nullptr;
#if defined(__SSE2__)
    const __m128i low24 = _mm_set1_epi32(0xffffff);
    const __m128i first = _mm_add_epi32(_mm_set1_epi32(static_cast<int>(o[0] & 0xffffff)), _mm_set_epi32(12, 8, 4, 0));
    const __m128i same0 = _mm_cmpeq_epi32(_mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(o)), low24), first);
    const __m128i same1 = _mm_cmpeq_epi32(_mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(o) + 1), low24),
                                          _mm_add_epi32(first, _mm_set1_epi32(16)));
    if (_mm_movemask_epi8(_mm_and_si128(same0, same1)) == 0xffff) {
      uint8_t* at = sh + (o[0] & 0xffffff);
      if (Store) {
        if (dg) {
          const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dg)), b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dg) + 1);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(at), a);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(at) + 1, b);
        } else {
          std::memset(at, 0, 32);
        }
      } else if (dg) {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(at)), b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(at) + 1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dg), a);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dg) + 1, b);
      }
      continue;
    }
#endif
    for (unsigned i = 0; i < 8; ++i) {
      uint8_t* at = sh + (o[i] & 0xffffff);
      if (Store) {
        const uint32_t v = dg ? dg[i] : 0u;
        std::memcpy(at, &v, 4);
      } else if (dg) {
        uint32_t v;
        std::memcpy(&v, at, 4);
        dg[i] = v;
      }
    }
  }
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
  const uint64_t ssize = blk.shared.size();
  // The offsets of all 32 lanes as 32-bit words, whether they run or not (an offset is the low 32 bits of the sum, so of the parts'
  // low halves; a lane that does not run reads whatever its register holds, and is left out of the window check and of the counting).
  alignas(32) uint32_t offs[32];
  {
    const uint32_t add32 = static_cast<uint32_t>(urv + imm);
    if (!base) {
      for (unsigned l = 0; l < 32; ++l) offs[l] = add32;
    } else if (!stride) {
      for (unsigned l = 0; l < 32; ++l) offs[l] = base[l] + add32;
    } else if (std::has_single_bit(stride)) {
      const unsigned sh = static_cast<unsigned>(std::countr_zero(stride));
      for (unsigned l = 0; l < 32; ++l) offs[l] = (base[l] << sh) + add32;
    } else {
      for (unsigned l = 0; l < 32; ++l) offs[l] = base[l] * stride + add32;
    }
  }
  if (!exec::shared_window_inside_warp(offs, ex, static_cast<uint32_t>(blk.rank), ssize, bytes)) {   // (as exec_mem: see window_check.hpp)
    c.r->tier_exec_mem(blk, w, *op.ins, ex);
    return;
  }
  uint8_t* sh = blk.shared.data();
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  if (bytes == 4 && ex == kAll) {
    // A whole warp's words: the 32 consecutive words of an array indexed by the thread (the commonest shared access), or one word
    // every lane names. The same bytes move as lane by lane (a word every lane stores is the last lane's); the lanes are on one pass
    // through the banks, as the general count finds (a word that is not aligned spreads over one more bank).
    const uint32_t o0 = offs[0] & 0xffffff;
    uint32_t seq = 0, same = 0;
    for (unsigned l = 0; l < 32; ++l) {
      const uint32_t o = offs[l] & 0xffffff;
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
  const bool by_groups = bytes == 4 && ex == kAll;
  if (by_groups) tier_shared_words<Store>(sh, d, offs);
  // (The widths the instruction can have move with copies of a size known at compile time: see store_lane_bytes.)
  const bool moved = by_groups || with_lane_width(bytes, [&](auto width) {
    constexpr unsigned B = decltype(width)::value;
    const auto lane = [&](unsigned l) {
      uint8_t* at = sh + (offs[l] & 0xffffff);
      if (Store) {
        store_lane_bytes<B>(at, [&](unsigned i) -> uint32_t { return d ? d[32 * i + l] : 0u; });
      } else if (d) {
        if constexpr (B == 4) d[l] = load_lane_word<B>(at);
        else if constexpr (B < 4) d[l] = tier_extend(op.b, load_lane_word<B>(at));
        else
          for (unsigned i = 0; i < B / 4; ++i) d[32 * i + l] = load_lane_word<B>(at, i);
      }
    };
    if (ex == kAll) {
      for (unsigned l = 0; l < 32; ++l) lane(l);
    } else {
      for (Mask m = ex; m; m &= m - 1) lane(static_cast<unsigned>(std::countr_zero(m)));
    }
  });
  if (!moved) {
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* at = sh + (offs[l] & 0xffffff);
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
  count_shared_offsets(blk.st, Store, offs, ex, bytes);
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
  uint64_t addrs[32];
  exec::LocalWindowCheck window;
  const uint32_t add32 = static_cast<uint32_t>(urv + imm);   // (an offset is the low 32 bits of the sum, so of the parts' low halves)
  if (ex == kAll) {   // a full warp: a counted loop on 32-bit offsets
    uint32_t highest = 0;
    if (base) {
      for (unsigned l = 0; l < 32; ++l) {
        const uint32_t off = base[l] + add32;
        addrs[l] = off;
        highest = std::max(highest, off);
      }
    } else {
      for (unsigned l = 0; l < 32; ++l) addrs[l] = add32;
      highest = add32;
    }
    window.highest = highest;
  } else {
    for (unsigned l = 0; l < 32; ++l) addrs[l] = 0;   // (the lanes that do not run read as zero, as the counting functions find them)
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      const uint32_t off = (base ? base[l] : 0u) + add32;
      addrs[l] = off;
      window.add(off);
    }
  }
  const bool inside = window.inside(w.local_size, bytes, ex != 0);
  if (!inside) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);   // the lane that faults, and the words of the fault, are its
    return;
  }
  if (w.local.empty()) w.local.assign(size_t{w.local_size} * 32, 0);
  uint32_t* d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  const unsigned words = std::max(1u, bytes / 4);
  uint8_t* const frames = w.local.data();   // (read once: the stores to the registers could change what the compiler thinks they are)
  const size_t frame = w.local_size;
  const bool moved = with_lane_width(bytes, [&](auto width) {
    constexpr unsigned B = decltype(width)::value;
    const auto lane = [&](unsigned l) {
      uint8_t* at = frames + size_t{l} * frame + addrs[l];
      if (Store) {
        store_lane_bytes<B>(at, [&](unsigned i) -> uint32_t { return d ? d[32 * i + l] : 0u; });
      } else if (d) {
        if constexpr (B == 4) d[l] = load_lane_word<B>(at);
        else if constexpr (B < 4) d[l] = tier_extend(op.b, load_lane_word<B>(at));
        else
          for (unsigned i = 0; i < B / 4; ++i) d[32 * i + l] = load_lane_word<B>(at, i);
      }
    };
    if (ex == kAll) {
      for (unsigned l = 0; l < 32; ++l) lane(l);
    } else {
      for (Mask m = ex; m; m &= m - 1) lane(static_cast<unsigned>(std::countr_zero(m)));
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

// FCHK: whether a / b needs the division's slow path. Always: the slow path is exact for every input (exec_float's rule).
TIER_HANDLER(h_fchk) {
  TIER_PRELUDE
  tier_setp(c, op.dst, ex, kAll);
}

TIER_HANDLER(h_s2r) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;
  uint32_t* d = c.R + op.dst * 32u;
  if (c.r->tier_sreg_fill(*c.blk, *c.w, op.a, d, ex)) return;
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
      if (exec::sasslanes::row_uniform(bp, b) && exec::sasslanes::row_uniform(cp, cc)) tier::shfl_shape(op.a, b, cc, sh);
    }
    if (sh.kind != tier::ShflShape::None) {
      tier::shfl_move(sh, vp, out);
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
    if (ex == kAll) {
      for (unsigned l = 0; l < 32; ++l) d[l] = ballot;   // (a full warp: eight vector stores, not a loop over the lanes' bits)
    } else {
      for (Mask m = ex; m; m &= m - 1) d[std::countr_zero(m)] = ballot;
    }
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

// IMMA, the dense integer forms (exec_mma): the 8- or 4-bit elements of A (M x K, row-major) and of B (K x 8) read out of the fragment registers
// as exec_mma lays them out (vgpu/exec/tile_mma.hpp), each product summed with the accumulator in 32 bits (a wrapping sum is the 64-bit sum cut to
// 32) or, under .SAT, clamped to the s32 range after every 128 bits of K, by the tile kernel (vgpu/simd/simd.hpp: tile_i32). The destination is
// written for the executing lanes.
TIER_HANDLER(h_imma) {
  TIER_PRELUDE
  if (op.dst == tier::kNoReg) return;   // (an RZ destination: the interpreter computes and throws it away)
  const unsigned M = op.a, K = op.b;
  const bool sa = op.aux & 1, sb = op.aux & 2, sat = op.aux & 4, four = op.aux & 8;
  const exec::FragType ta = four ? (sa ? exec::FragType::S4 : exec::FragType::U4) : (sa ? exec::FragType::S8 : exec::FragType::U8),
                       tb = four ? (sb ? exec::FragType::S4 : exec::FragType::U4) : (sb ? exec::FragType::S8 : exec::FragType::U8);
  const unsigned per_reg = four ? 8 : 4, a_regs = M * K / (32 * per_reg), b_regs = K * 8 / (32 * per_reg), c_regs = M * 8 / 32;
  static const uint32_t kZeros[32] = {};
  alignas(32) uint32_t xa[8 * 32], xb[4 * 32], xc[4 * 32], xd[4 * 32];
  const auto load = [&](unsigned base, unsigned n, uint32_t* dst) {
    for (unsigned i = 0; i < n; ++i) std::memcpy(dst + 32 * i, base == kRZ ? kZeros : c.R + (base + i) * 32u, 32 * sizeof(uint32_t));
  };
  load(op.s[0].off, a_regs, xa);
  load(op.s[1].off, b_regs, xb);
  load(op.s[2].off, c_regs, xc);
  alignas(32) int32_t ia[16 * 128], ib[128 * 8], ic[16 * 8], id[16 * 8];
  exec::mma_frag_a(ta, M, K, false, xa, nullptr, 0, nullptr, ia);
  exec::mma_frag_b(tb, K, xb, nullptr, ib);
  if (M == 16) exec::wgmma_acc_to_tile_i32(xc, 8, ic);
  else exec::mma_acc8_to_tile_i32(xc, ic);
  c.sk->tile_i32(M, 8, K, ia, ib, ic, 8, sat ? 128 / (four ? 4 : 8) : K, sat, id, 8);   // (.SAT clamps after every 128 bits of K)
  if (M == 16) exec::wgmma_tile_to_acc_i32(id, 8, xd);
  else exec::mma_tile_to_acc8_i32(id, xd);
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    for (unsigned s = 0; s < c_regs; ++s) c.R[(op.dst + s) * 32u + l] = xd[s * 32 + l];
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

// LDGSTS (cp.async without .ZFILL): each executing lane reads `bytes` (4, 8 or 16) of global memory and writes them at its shared address; a lane
// whose predicate operand is false writes zeros. exec_mem's lane loop does a mem_read and a mem_write (each a switch and a bounds check, out of line)
// per lane; here the global reads are one batch (load_batch, in the 8-byte pieces a 16-byte read is) and the shared writes are lane copies, in lane order
// (the last of lanes on one address wins, as there), once every shared access is known to lie inside the block's own array. A batch the memory
// manager declines (a fault, a misaligned or host-mapped access) and a shared access outside the array are exec_mem's, which has not been started.
TIER_HANDLER(h_ldgsts) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Warp& w = *c.w;
  Block& blk = *c.blk;
  const unsigned bytes = op.a;
  const unsigned rs = op.s[0].off, rg = op.s[1].off;
  const uint32_t* sreg = rs != kRZ ? c.R + rs * 32u : nullptr;
  const uint32_t* glo = rg != kRZ ? c.R + rg * 32u : nullptr;
  const uint32_t* ghi = rg != kRZ ? c.R + (rg + 1u) * 32u : nullptr;
  const uint32_t sadd = op.b;
  uint64_t saddr[32], gaddr[32];
  exec::SharedWindowCheck window;
  for (unsigned l = 0; l < 32; ++l) saddr[l] = 0;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t off = static_cast<uint32_t>((sreg ? sreg[l] : 0u) + sadd);
    saddr[l] = off;
    window.add(off, blk.rank);
  }
  if (!window.inside(blk.shared.size(), bytes, ex != 0)) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);
    return;
  }
  tier_global_addresses(glo, ghi, static_cast<uint64_t>(op.imm), ex, gaddr);
  const Mask reads = (op.aux & 1) ? (ex & tier_pmask(c, op.p)) : ex;
  uint64_t fa[32], lo[32], hi[32];
  unsigned n = 0;
  for (Mask m = reads; m; m &= m - 1) fa[n++] = gaddr[std::countr_zero(m)];
  MemoryManager& mem = c.r->memory();
  if (n) {
    bool ok;
    if (bytes == 16) {
      uint64_t fb[32];
      for (unsigned i = 0; i < n; ++i) fb[i] = fa[i] + 8;
      ok = mem.load_batch(fa, n, 8, lo) && mem.load_batch(fb, n, 8, hi);
    } else {
      ok = mem.load_batch(fa, n, bytes, lo);
    }
    if (!ok) {
      c.r->tier_exec_mem(blk, w, *op.ins, ex);
      return;
    }
  }
  w.idle_yields = 0;   // a store is progress (Runner::mem_write)
  uint8_t* const sh = blk.shared.data();
  unsigned i = 0;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    uint8_t* const at = sh + (saddr[l] & 0xffffff);
    const bool read = (reads >> l) & 1;
    const uint64_t a = read ? lo[i] : 0, b = read ? (bytes == 16 ? hi[i] : 0) : 0;
    i += read;
    if (bytes == 4) {
      const uint32_t v = static_cast<uint32_t>(a);
      std::memcpy(at, &v, 4);
    } else if (bytes == 8) {
      std::memcpy(at, &a, 8);
    } else {
      std::memcpy(at, &a, 8);
      std::memcpy(at + 8, &b, 8);
    }
  }
  count_access(blk.st, 0, false, gaddr, ex, bytes);
  count_shared_access(blk.st, true, saddr, ex, bytes);
}

// LD and ST, whose 64-bit addresses name a window of the executor's address space as well as global memory (Runner::classify). The interpreter classifies
// each lane, moves it by its window's rule and counts the warp as the LAST lane's window says; the cases the tier takes are the ones where that is one
// rule: every active lane in the shared window (the lanes move as LDS/STS do, and are counted as shared accesses of the offsets), or no lane in
// any window (plain global memory, which h_global makes as LDG/STG: the interpreter's per-lane mem_read/mem_write and the batch give the same bytes
// and the same counters). A mix, a local, parameter or constant address, an offset outside the block's array and a 32-byte access are exec_mem's.
constexpr uint64_t kWindowsEnd = 7ull << 32;   // the windows are the 2^32-byte ranges at kConstWindow + 0 .. 6 << 32 (const, 3 unused, param, shared, local)
template <bool Store>
static void h_generic(void* vc, const tier::MOp& op, Mask ex) {
  const TCtx& c = *static_cast<const TCtx*>(vc);
  Warp& w = *c.w;
  Block& blk = *c.blk;
  const unsigned bytes = op.a;
  const unsigned ra = op.s[0].off;
  const uint32_t* blo = ra != kRZ ? c.R + ra * 32u : nullptr;
  const uint32_t* bhi = (ra != kRZ && (op.aux & 1)) ? c.R + (ra + 1u) * 32u : nullptr;
  const uint64_t urv = op.s[1].off != ~0u ? w.ur[op.s[1].off] : 0;
  uint64_t addrs[32];
  tier_global_addresses(blo, bhi, urv + static_cast<uint64_t>(op.imm), ex, addrs);
  unsigned in_shared = 0, in_window = 0;
  for (Mask m = ex; m; m &= m - 1) {
    const uint64_t g = addrs[std::countr_zero(m)];
    in_shared += (g - kSharedWindow) < (uint64_t{1} << 32);
    in_window += (g - kConstWindow) < kWindowsEnd;
  }
  if (ex && in_window == 0) {
    h_global<Store>(vc, op, ex);
    return;
  }
  if (!ex || in_shared != static_cast<unsigned>(popc(ex)) || !(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16)) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);
    return;
  }
  exec::SharedWindowCheck window;
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    addrs[l] -= kSharedWindow;   // the offset in the window (< 2^32), as classify gives it
    window.add(addrs[l], blk.rank);
  }
  if (!window.inside(blk.shared.size(), bytes, true)) {
    c.r->tier_exec_mem(blk, w, *op.ins, ex);
    return;
  }
  if constexpr (Store) w.idle_yields = 0;   // a store is progress (Runner::mem_write)
  uint8_t* const sh = blk.shared.data();
  uint32_t* const d = op.dst == tier::kNoReg ? nullptr : c.R + op.dst * 32u;
  const unsigned words = std::max(1u, bytes / 4);
  const bool moved = with_lane_width(bytes, [&](auto width) {
    constexpr unsigned B = decltype(width)::value;
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* const at = sh + (addrs[l] & 0xffffff);
      if (Store) {
        store_lane_bytes<B>(at, [&](unsigned i) -> uint32_t { return d ? d[32 * i + l] : 0u; });
      } else if (d) {
        if constexpr (B == 4) d[l] = load_lane_word<B>(at);
        else if constexpr (B < 4) d[l] = tier_extend(op.b, load_lane_word<B>(at));
        else
          for (unsigned i = 0; i < B / 4; ++i) d[32 * i + l] = load_lane_word<B>(at, i);
      }
    }
  });
  if (!moved) {   // (unreachable: the widths are those of with_lane_width)
    for (Mask m = ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint8_t* const at = sh + (addrs[l] & 0xffffff);
      uint32_t v[4] = {};
      if (Store) {
        for (unsigned i = 0; i < words; ++i) v[i] = d ? d[32 * i + l] : 0;
        std::memcpy(at, v, bytes);
      } else if (d) {
        std::memcpy(v, at, bytes);
        for (unsigned i = 0; i < words; ++i) d[32 * i + l] = v[i];
      }
    }
  }
  count_shared_access(blk.st, Store, addrs, ex, bytes);
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
    case F::Cvt64: return &h_cvt64;
    case F::FltInterp: return &h_fltinterp;
    case F::Dsetp: return &h_dsetp;
    case F::Hop2: return &h_hop2;
    case F::Unary: return &h_unary;
    case F::Iabs: return &h_iabs;
    case F::Ldc: return &h_ldc;
    case F::Shfl: return &h_shfl;
    case F::Vote: return &h_vote;
    case F::Hmma: return &h_hmma;
    case F::Ldgsts: return &h_ldgsts;
    case F::Ld: return &h_generic<false>;
    case F::St: return &h_generic<true>;
    case F::Match: return &h_match;
    case F::Imma: return &h_imma;
    case F::Atom: return &h_atom;
    case F::Fchk: return &h_fchk;
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
[[gnu::aligned(64)]] inline void tier_set_pcs(Warp& w, Mask group, uint64_t v) {
  if (group == kAll) {
    for (unsigned l = 0; l < 32; ++l) w.pc[l] = v;
  } else {
    exec::sasslanes::set_pcs_partial(w.pc, group, v);
  }
}

inline void tier_wait(Warp& w, Mask ex, Wait kind, unsigned arg) { exec::sasslanes::wait(w, ex, kind, arg); }
inline Mask tier_waiting_on(const Warp& w, Wait kind, unsigned arg) { return exec::sasslanes::waiting_on(w, kind, arg); }
inline void tier_wake(Warp& w, Mask m) { exec::sasslanes::wake(w, m); }

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

// The ops of a stretch (Program::run) one after the other for `group`, with their class counts. A function of its own so that the loop keeps its few
// values in registers (inside the run loop of step_warp they were on the stack).
[[gnu::noinline]] void tier_issue(TCtx& c, const tier::MOp* o, unsigned k, Mask group, unsigned glanes, uint64_t* by_class) {
  for (const tier::MOp* const end = o + k; o != end; ++o) {
    by_class[o->cls] += glanes;
    o->fn(&c, *o, group);
  }
}

// The share rule's sample (jit.hpp): the warp instructions a call of the hook loop ran, whatever ran them, are added to the unit's sample on the way out
// (also when a fault leaves). `u` is null once the rule has decided, or for the loop without the hook.
struct JitSampler {
  jit::Unit* u;
  const Warp& w;
  uint64_t steps0;
  ~JitSampler() {
    if (u) u->sample(w.steps - steps0);
  }
};

// CALL, RET and WARPSYNC as they end a run (tier_run_t calls this; res 0: not taken here, the interpreter runs it -- a call of a function VirtualGPU provides).
// `moved`: the run was longer than one instruction, so the lanes' program counters are still those of its start.
[[gnu::noinline]] Runner::CtlOut Runner::tier_ctl(Block& blk, Warp& w, const tier::MOp& last, Mask group, unsigned glanes, uint64_t here, bool moved,
                                                 const Code& code) {
  uint64_t at = 0;
  bool all_at = false;
  exec::LaunchStats& st = blk.st;
  if (last.form == tier::Form::Call) {
    // exec_control's CALL.REL for a fixed target: a function VirtualGPU provides (a builtin at its magic address) is the interpreter's.
    const uint64_t target = code.base + static_cast<uint64_t>(last.imm);
    if (!(m_.builtins.empty() || target < m_.builtins.begin()->first || m_.builtins.find(target) == m_.builtins.end())) return {0, 0};
    // The lanes that run it go to the target, the others on to the next instruction.
    const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
    const unsigned lanes = ex == group ? glanes : popc(ex);
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[last.cls] += lanes;
    if (ex == group || !ex) {
      at = ex ? target : here + 16;
      tier_set_pcs(w, group, at);
      all_at = true;
    } else {
      exec::sasslanes::select_pcs(w.pc, group, ex, target, here + 16);
    }
    if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::CALL)].fetch_add(1, std::memory_order_relaxed);
    return {at, all_at ? 2u : 1u};
  }
  if (last.form == tier::Form::Ret) {
    // exec_control's RET: the lanes that run it go to the address in the register pair (a section-relative offset, or an absolute code
    // address a relocation wrote), the others on to the next instruction.
    const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
    const unsigned lanes = ex == group ? glanes : popc(ex);
    ++st.instructions;
    st.thread_instructions += lanes;
    st.inst_by_class[last.cls] += lanes;
    const unsigned r = last.dst;
    uint64_t first_at = 0;
    bool same = ex == group;
    for (Mask m = group; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      uint64_t t = here + 16;
      if ((ex >> l) & 1) {
        const uint32_t lo = w.reg(r, l), hi = r + 1 < w.nregs ? w.reg(r + 1, l) : 0;
        t = hi == static_cast<uint32_t>(kCodeBase >> 32) ? (uint64_t{hi} << 32 | lo) : code.base + lo;
      }
      if (m == group) first_at = t;
      same = same && t == first_at;
      w.pc[l] = t;
    }
    if (same) {
      at = first_at;
      all_at = true;
    }
    if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::RET)].fetch_add(1, std::memory_order_relaxed);
    return {at, all_at ? 2u : 1u};
  }
  // exec_control's WARPSYNC: the lanes that run it wait there for the lanes their mask names (a thread its own mask leaves out may not
  // run it), the others go on; then the release (release_warpsync).
  const Mask ex = last.unguarded ? group : tier_guard(w, *last.ins, group);
  const unsigned lanes = ex == group ? glanes : popc(ex);
  ++st.instructions;
  st.thread_instructions += lanes;
  st.inst_by_class[last.cls] += lanes;
  if (moved) tier_set_pcs(w, group, here);
  const Instr& wi = *last.ins;
  // The mask is the same for every lane unless it is a register (a constant, a constant-bank word or a uniform register is read once).
  const bool per_lane = !last.aux && wi.src[last.a].kind == Kind::Reg;
  uint32_t mask_all = ~Mask{0};
  if (ex && !last.aux && !per_lane) mask_all = read32(w, wi.src[last.a], static_cast<unsigned>(std::countr_zero(ex)));
  for (Mask m = group; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    if ((ex >> l) & 1) {
      w.waiting |= Mask{1} << l;
      w.wait_kind[l] = Wait::WarpSync;
      w.wait_arg[l] = per_lane ? read32(w, wi.src[last.a], l) : mask_all;
      if (!((w.wait_arg[l] >> l) & 1u))
        throw Error(Err::IllegalInstruction, "SASS kernel " + k_.name + ": WARPSYNC run by lane " + std::to_string(l) +
                                                 ", which its mask " + std::to_string(w.wait_arg[l]) + " leaves out");
    } else {
      w.pc[l] += 16;
    }
  }
  release_warpsync(w);
  if (tier_stats_) [[unlikely]] tier_stats().native[static_cast<size_t>(Op::WARPSYNC)].fetch_add(1, std::memory_order_relaxed);
  return {at, all_at ? 2u : 1u};
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
  [[maybe_unused]] const JitSampler sampler{Jit && ju->sampling.load(std::memory_order_relaxed) ? ju : nullptr, w, w.steps};
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
  bool all_at = false;   // the run ended by setting every lane of the group to one address, `at` (the chain below then knows where they stand)
  uint64_t at = 0;
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
      // A stretch of register-only instructions, each for the whole group, issued back to back: what the loop below does for each of them
      // -- the store gate (none of them is a store), the step count, the guard (none has one) and the counters -- is done for the stretch.
      // Not in tier 2's loop (its regions would be skipped over), nor while the statistics count each instruction.
      if constexpr (!Jit) {
        unsigned k = ops[idx].run;
        if (k > 256 - n) k = 256 - n;
        if (k >= 2 && regs_ok && group && !tier_stats_ && w.steps + k <= max_steps_) {
          w.steps += k;
          st.instructions += k;
          st.thread_instructions += uint64_t{k} * glanes;
          const tier::MOp* const o = ops + idx;
          tier_issue(c, o, k, group, glanes, st.inst_by_class);
          cur = o + k - 1;
          idx += k;
          n += k - 1;
          continue;
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
        at = take ? target : here + 16;
        tier_set_pcs(w, group, at);
        all_at = true;
      } else {
        exec::sasslanes::select_pcs(w.pc, group, take, target, here + 16);   // (a select without a branch: the lanes that take it are unpredictable)
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
      at = here + 16;
      tier_set_pcs(w, group, at);
      all_at = true;
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
      at = here + 16;
      tier_set_pcs(w, group, at);
      all_at = true;
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
    } else if (last.form == tier::Form::Call || last.form == tier::Form::Ret || last.form == tier::Form::Warpsync) {
      // CALL, RET and WARPSYNC (a function of their own: they are rare beside the straight-line work, and the loop is placement-sensitive;
      // what it hands back is two registers, so `at` and `all_at` stay where the loop keeps them)
      const CtlOut ctl = tier_ctl(blk, w, last, group, glanes, here, idx != first, code);
      if (ctl.res == 2) {
        at = ctl.at;
        all_at = true;
      } else if (!ctl.res) {
        if (tier_stats_) [[unlikely]] tier_note_fallback(*last.ins, true);
        if (idx != first) tier_set_pcs(w, group, here);
        execute(blk, w, *ops[idx].ins, group);
      }
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
  uint64_t lo;
  Mask next;
  // The group at the lowest address, which is what step_warp picks when nothing else is asked of it. When the run set the group to one address
  // and no runnable lane stands outside it, that is the runnable lanes at that address, and the 32 addresses need not be read back.
  if (all_at && (run & ~group) == 0) {
    next = run;
    lo = at;
  } else {
    next = exec::sasslanes::pick_group(w.pc, run, lo);
  }
  if (lo - code.base >= 16 * static_cast<uint64_t>(count) || ((lo - code.base) & 15)) return true;
  --w.budget;
  ++w.extra;
  group = next;
  pick = lo;
  }
}

// The run loop without tier 2's hook is the loop of every run that does not ask for the JIT, so it has no trace of it (tier_run_t is always_inline, so
// it is inlined where the loop was before tier 2); step_warp calls tier_run_jit instead for a section that has a unit.
[[gnu::noinline]] bool Runner::tier_run_jit(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp) {
  return tier_run_t<true>(blk, w, group, pick, code, tp);
}
bool Runner::tier_run(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp) {
  return tier_run_t<false>(blk, w, group, pick, code, tp);
}
