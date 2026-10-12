// Throwaway prototype (r7-design, NOT for merging). See engine.hpp.
//
// Semantics are copied from src/sass/exec_ops.inc (fast_alu and the general
// paths it mirrors) for the forms the prototype accepts; anything else throws
// Unsupported at compile time, so a program either runs entirely here or not
// at all. Compile with the same flags as the library for a fair comparison.
#include "engine.hpp"

#include <time.h>

#include <array>
#include <set>

using namespace vgpu;
using namespace vgpu::sass;

namespace r7 {

namespace {

constexpr uint32_t kNoRow = 0xffffffffu;
constexpr unsigned kClsCount = static_cast<unsigned>(exec::InstClass::Count);

inline float as_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t f32_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

#define ROW(i) (c.R + static_cast<size_t>(i) * 32)

inline void put(Ctx& c, uint32_t drow, const uint32_t* r) {
  uint32_t* d = ROW(drow);
  if (c.ex == kAll) {
    std::memcpy(d, r, 128);
  } else {
    for (unsigned l = 0; l < 32; ++l)
      if ((c.ex >> l) & 1) d[l] = r[l];
  }
}
inline void setp(Ctx& c, uint32_t pd, Mask v) {
  if (pd != kPT) c.P[pd] = (c.P[pd] & ~c.ex) | (v & c.ex);
}

// ---- handlers ------------------------------------------------------------------

const MOp* h_header(Ctx& c, const MOp& p) {
  const BlockInfo& bi = c.infos[p.imm];
  const uint64_t pop = static_cast<uint64_t>(std::popcount(c.ex));
  exec::LaunchStats& st = *c.st;
  st.instructions += bi.n_total;
  st.thread_instructions += bi.n_free * pop;
  for (unsigned k = 0; k < kClsCount; ++k) st.inst_by_class[k] += bi.cls_free[k] * pop;
  c.steps += bi.n_total;
  if (c.steps > c.max_steps) [[unlikely]]
    throw Error(Err::ExecLimit, "prototype: a warp ran " + std::to_string(c.max_steps) + " instructions");
  return &p + 1;
}

const MOp* h_guard(Ctx& c, const MOp& p) {
  Mask g = c.P[p.g];
  if (p.gneg) g = ~g;
  const Mask m = c.ex & g;
  const uint64_t pop = static_cast<uint64_t>(std::popcount(m));
  c.st->thread_instructions += pop;
  c.st->inst_by_class[p.cls] += pop;
  if (m) {
    const Mask save = c.ex;
    c.ex = m;
    p.alt(c, p);
    c.ex = save;
  }
  return &p + 1;
}

const MOp* h_mov(Ctx& c, const MOp& p) {
  alignas(64) uint32_t r[32];
  std::memcpy(r, ROW(p.a), 128);
  put(c, p.d, r);
  return &p + 1;
}

// IMAD with a negated or complemented addend (ival: -x, then ~x).
const MOp* h_imadm(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  const bool neg = p.k0, bnot = p.k1;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) {
    uint32_t cv = cc[l];
    if (neg) cv = 0u - cv;
    if (bnot) cv = ~cv;
    r[l] = a[l] * b[l] + cv;
  }
  put(c, p.d, r);
  return &p + 1;
}

const MOp* h_iadd3(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  const uint32_t ma = p.imm, mb = p.imm2, mc = p.e, add = p.k3;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = (a[l] ^ ma) + (b[l] ^ mb) + (cc[l] ^ mc) + add;
  put(c, p.d, r);
  return &p + 1;
}

const MOp* h_iadd3_plain(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = a[l] + b[l] + cc[l];
  put(c, p.d, r);
  return &p + 1;
}

const MOp* h_imad(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = a[l] * b[l] + cc[l];
  put(c, p.d, r);
  return &p + 1;
}

// IMAD.WIDE: a 64-bit product plus a 64-bit addend (rows c, e = low, high); dst pair d, d+1.
const MOp* h_imad_wide(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *clo = ROW(p.c), *chi = ROW(p.e);
  alignas(64) uint32_t lo[32], hi[32];
  const bool sgn = p.k0;
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t prod = sgn ? static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(a[l])) * static_cast<int32_t>(b[l]))
                              : static_cast<uint64_t>(a[l]) * b[l];
    const uint64_t sum = prod + (clo[l] | (static_cast<uint64_t>(chi[l]) << 32));
    lo[l] = static_cast<uint32_t>(sum);
    hi[l] = static_cast<uint32_t>(sum >> 32);
  }
  put(c, p.d, lo);
  put(c, p.d + 1, hi);   // the compiler maps an RZ destination pair to two sink rows
  return &p + 1;
}

const MOp* h_lop3(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  const unsigned lut = p.imm;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = 0;
  for (unsigned t = 0; t < 8; ++t) {
    if (!((lut >> t) & 1)) continue;
    const uint32_t ma = (t & 4) ? 0u : ~0u, mb = (t & 2) ? 0u : ~0u, mc = (t & 1) ? 0u : ~0u;
    for (unsigned l = 0; l < 32; ++l) r[l] |= (a[l] ^ ma) & (b[l] ^ mb) & (cc[l] ^ mc);
  }
  put(c, p.d, r);
  return &p + 1;
}

// The same function, specialised at compile time on the LUT (what pre-decoding allows).
const MOp* h_lop3_xor3(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = a[l] ^ b[l] ^ cc[l];
  put(c, p.d, r);
  return &p + 1;
}
const MOp* h_lop3_xor2(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b);
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = a[l] ^ b[l];
  put(c, p.d, r);
  return &p + 1;
}

const MOp* h_shf(Ctx& c, const MOp& p) {
  const uint32_t *lo = ROW(p.a), *n = ROW(p.b), *top = ROW(p.c);
  const unsigned type = p.k0;
  const bool wrap = p.k1, right = p.k2, hi = p.k3;
  const bool is64 = type < 2, sgn = type == 0 || type == 2;
  const uint32_t width = is64 ? 64 : 32;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) {
    const uint64_t v = lo[l] | (static_cast<uint64_t>(top[l]) << 32);
    const uint32_t k = wrap ? n[l] % width : std::min(n[l], width);
    uint64_t x;
    if (!right) x = k >= 64 ? 0 : v << k;
    else if (sgn) x = static_cast<uint64_t>(static_cast<int64_t>(v) >> std::min(k, 63u));
    else x = k >= 64 ? 0 : v >> k;
    r[l] = static_cast<uint32_t>(hi ? x >> 32 : x);
  }
  put(c, p.d, r);
  return &p + 1;
}

// LEA (fast_alu's rule): (a << s) + b with .HI/.X and a carry out.
const MOp* h_lea(Ctx& c, const MOp& p) {
  const bool hi = p.k0, x = p.k1, three = p.k2;
  const unsigned sh = p.imm;
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *top = ROW(p.c);
  const Mask cin = x ? c.P[p.e & 0xff] ^ ((p.e >> 8) ? ~Mask{0} : 0) : 0;
  const bool bneg = p.k3 & 1, bnot = p.k3 & 2;
  alignas(64) uint32_t r[32];
  Mask carry = 0;
  for (unsigned l = 0; l < 32; ++l) {
    uint32_t bb = b[l];
    if (bneg) bb = 0u - bb;
    if (bnot) bb = ~bb;
    uint32_t shifted;
    if (!hi) {
      shifted = sh >= 32 ? 0 : a[l] << sh;
    } else {
      const uint32_t t = three ? top[l] : static_cast<uint32_t>(static_cast<int32_t>(a[l]) >> 31);
      shifted = static_cast<uint32_t>(((a[l] | (static_cast<uint64_t>(t) << 32)) << sh) >> 32);
    }
    uint64_t sum = static_cast<uint64_t>(shifted) + bb + ((cin >> l) & 1);
    r[l] = static_cast<uint32_t>(sum);
    carry |= static_cast<Mask>(sum >> 32 != 0) << l;
  }
  put(c, p.d, r);
  if (p.imm2 != kNoRow) setp(c, p.imm2, carry);
  return &p + 1;
}

inline bool int_cmp_s(unsigned cmp, int32_t a, int32_t b) {
  switch (cmp) {
    case 0: return false; case 1: return a < b; case 2: return a == b; case 3: return a <= b;
    case 4: return a > b; case 5: return a != b; case 6: return a >= b; default: return true;
  }
}
inline bool int_cmp_u(unsigned cmp, uint32_t a, uint32_t b) {
  switch (cmp) {
    case 0: return false; case 1: return a < b; case 2: return a == b; case 3: return a <= b;
    case 4: return a > b; case 5: return a != b; case 6: return a >= b; default: return true;
  }
}
inline Mask combine(unsigned bop, Mask r, Mask acc) { return bop == 0 ? (r & acc) : bop == 1 ? (r | acc) : (r ^ acc); }

// ISETP: k0 = comparison, k1 = signed, k2 = bool op; e = accumulator predicate (low 8 bits: index, bit 8: negate).
const MOp* h_isetp(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b);
  const unsigned cmp = p.k0, bop = p.k2;
  Mask r = 0;
  if (p.k1) {
    for (unsigned l = 0; l < 32; ++l) r |= static_cast<Mask>(int_cmp_s(cmp, static_cast<int32_t>(a[l]), static_cast<int32_t>(b[l]))) << l;
  } else {
    for (unsigned l = 0; l < 32; ++l) r |= static_cast<Mask>(int_cmp_u(cmp, a[l], b[l])) << l;
  }
  const Mask acc = c.P[p.e & 0xff] ^ ((p.e >> 8) ? ~Mask{0} : 0);
  setp(c, p.d, combine(bop, r, acc));
  if (p.imm != kNoRow) setp(c, p.imm, combine(bop, ~r, acc));
  return &p + 1;
}

inline uint32_t canon(float v) { return v != v ? 0x7fffffffu : f32_bits(v); }

const MOp* h_ffma(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b), *cc = ROW(p.c);
  // abs/neg modifiers as the fast path applies them: and-mask then xor-mask per source
  const uint32_t aa = p.imm & 1 ? 0x7fffffffu : ~0u, ab = p.imm & 2 ? 0x7fffffffu : ~0u, ac = p.imm & 4 ? 0x7fffffffu : ~0u;
  const uint32_t xa = p.imm2 & 1 ? 0x80000000u : 0, xb = p.imm2 & 2 ? 0x80000000u : 0, xc = p.imm2 & 4 ? 0x80000000u : 0;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l)
#ifdef R7_BREAK_FFMA
    r[l] = canon(as_f32((a[l] & aa) ^ xa) * as_f32((b[l] & ab) ^ xb) + as_f32((cc[l] & ac) ^ xc));   // deliberately wrong: two roundings
#else
    r[l] = canon(std::fma(as_f32((a[l] & aa) ^ xa), as_f32((b[l] & ab) ^ xb), as_f32((cc[l] & ac) ^ xc)));
#endif
  put(c, p.d, r);
  return &p + 1;
}
const MOp* h_fadd(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b);
  const uint32_t aa = p.imm & 1 ? 0x7fffffffu : ~0u, ab = p.imm & 2 ? 0x7fffffffu : ~0u;
  const uint32_t xa = p.imm2 & 1 ? 0x80000000u : 0, xb = p.imm2 & 2 ? 0x80000000u : 0;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = canon(as_f32((a[l] & aa) ^ xa) + as_f32((b[l] & ab) ^ xb));
  put(c, p.d, r);
  return &p + 1;
}
const MOp* h_fmul(Ctx& c, const MOp& p) {
  const uint32_t *a = ROW(p.a), *b = ROW(p.b);
  const uint32_t aa = p.imm & 1 ? 0x7fffffffu : ~0u, ab = p.imm & 2 ? 0x7fffffffu : ~0u;
  const uint32_t xa = p.imm2 & 1 ? 0x80000000u : 0, xb = p.imm2 & 2 ? 0x80000000u : 0;
  alignas(64) uint32_t r[32];
  for (unsigned l = 0; l < 32; ++l) r[l] = canon(as_f32((a[l] & aa) ^ xa) * as_f32((b[l] & ab) ^ xb));
  put(c, p.d, r);
  return &p + 1;
}

// I2FP.F32.{S32,U32}, round to nearest: a (u)int32 to float.
const MOp* h_i2fp(Ctx& c, const MOp& p) {
  const uint32_t* a = ROW(p.a);
  alignas(64) uint32_t r[32];
  if (p.k0) {
    for (unsigned l = 0; l < 32; ++l) r[l] = f32_bits(static_cast<float>(static_cast<int32_t>(a[l])));
  } else {
    for (unsigned l = 0; l < 32; ++l) r[l] = f32_bits(static_cast<float>(static_cast<int64_t>(a[l])));
  }
  put(c, p.d, r);
  return &p + 1;
}

// Counts one warp-level global access exactly as count_access does (src/sass/exec_ops.inc).
void count_global(exec::LaunchStats& st, bool store, const uint64_t* addr, Mask lanes, uint32_t bytes, bool fast) {
  const uint64_t n = static_cast<uint64_t>(std::popcount(lanes));
  if (!n) return;
  uint64_t distinct;
  if (fast && lanes == kAll && bytes == 4 && (addr[0] & 3) == 0) {
    // a fully contiguous warp access: consecutive addresses touch consecutive sectors
    bool contiguous = true;
    for (unsigned l = 1; l < 32; ++l) contiguous = contiguous && addr[l] == addr[0] + 4u * l;
    if (contiguous) {
      distinct = (addr[31] / 32) - (addr[0] / 32) + 1;
      goto counted;
    }
  }
  {
    std::vector<uint64_t> sectors;
    for (Mask m = lanes; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      for (uint64_t s = addr[l] / 32; s <= (addr[l] + bytes - 1) / 32; ++s) sectors.push_back(s);
    }
    std::sort(sectors.begin(), sectors.end());
    distinct = static_cast<uint64_t>(std::unique(sectors.begin(), sectors.end()) - sectors.begin());
  }
counted:
  (store ? st.global_stores : st.global_loads) += n;
  (store ? st.global_bytes_written : st.global_bytes_read) += n * bytes;
  st.global_sectors += distinct;
  ++st.global_requests;
  (store ? st.global_sectors_st : st.global_sectors_ld) += distinct;
  ++(store ? st.global_requests_st : st.global_requests_ld);
  (store ? st.global_bytes_st : st.global_bytes_ld) += n * bytes;
}

bool g_fast_counters = true;
bool g_bulk_mem = false;

// LDG.E / STG.E, 32-bit: address = 64-bit register pair a (lo), a+1 (hi) plus imm. d = data row.
const MOp* h_ldg32(Ctx& c, const MOp& p) {
  const uint32_t *lo = ROW(p.a), *hi = ROW(p.a + 1);
  uint64_t addrs[32] = {};
  alignas(64) uint32_t r[32] = {};
  const uint64_t off = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(p.imm)));
  if (g_bulk_mem && c.ex == kAll) {
    for (unsigned l = 0; l < 32; ++l) addrs[l] = (lo[l] | (static_cast<uint64_t>(hi[l]) << 32)) + off;
  }
  bool bulk = false;
  if (g_bulk_mem && c.ex == kAll) {
    bool contiguous = (addrs[0] & 3) == 0 && addrs[0] < 0x6ff9'0000'0000ull;
    for (unsigned l = 1; l < 32 && contiguous; ++l) contiguous = addrs[l] == addrs[0] + 4u * l;
    if (contiguous) bulk = c.mem->load_run(addrs[0], r, 128, 4);
  }
  if (!bulk) {
    for (Mask m = c.ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      const uint64_t g = (lo[l] | (static_cast<uint64_t>(hi[l]) << 32)) + off;
      if ((g & 3) || g >= 0x6ff9'0000'0000ull) throw Unsupported("prototype: unaligned or windowed address");
      addrs[l] = g;
      r[l] = static_cast<uint32_t>(c.mem->load_scalar(g, 4));
    }
  }
  put(c, p.d, r);
  count_global(*c.st, false, addrs, c.ex, 4, g_fast_counters);
  return &p + 1;
}
const MOp* h_stg32(Ctx& c, const MOp& p) {
  const uint32_t *lo = ROW(p.a), *hi = ROW(p.a + 1), *d = ROW(p.d);
  uint64_t addrs[32] = {};
  const uint64_t off = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(p.imm)));
  bool bulk = false;
  if (g_bulk_mem && c.ex == kAll) {
    for (unsigned l = 0; l < 32; ++l) addrs[l] = (lo[l] | (static_cast<uint64_t>(hi[l]) << 32)) + off;
    bool contiguous = (addrs[0] & 3) == 0 && addrs[0] < 0x6ff9'0000'0000ull;
    for (unsigned l = 1; l < 32 && contiguous; ++l) contiguous = addrs[l] == addrs[0] + 4u * l;
    if (contiguous) bulk = c.mem->store_run(addrs[0], d, 128, 4);
  }
  if (!bulk) {
    for (Mask m = c.ex; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      const uint64_t g = (lo[l] | (static_cast<uint64_t>(hi[l]) << 32)) + off;
      if ((g & 3) || g >= 0x6ff9'0000'0000ull) throw Unsupported("prototype: unaligned or windowed address");
      addrs[l] = g;
      c.mem->store_scalar(g, 4, d[l]);
    }
  }
  count_global(*c.st, true, addrs, c.ex, 4, g_fast_counters);
  return &p + 1;
}

// Branch: p.g/p.gneg guard (7/0: none). Only the convergent case.
const MOp* h_bra(Ctx& c, const MOp& p) {
  Mask m = c.ex;
  if (p.g != 7 || p.gneg) {
    Mask g = c.P[p.g];
    if (p.gneg) g = ~g;
    m = c.ex & g;
    const uint64_t pop = static_cast<uint64_t>(std::popcount(m));
    c.st->thread_instructions += pop;
    c.st->inst_by_class[p.cls] += pop;
  }
  if (m == c.ex) return c.ops + p.target;
  if (m == 0) return &p + 1;
  throw Unsupported("prototype: divergent branch at pc " + std::to_string(p.pc));
}

const MOp* h_exit(Ctx& c, const MOp& p) {
  Mask m = c.ex;
  if (p.g != 7 || p.gneg) {
    Mask g = c.P[p.g];
    if (p.gneg) g = ~g;
    m = c.ex & g;
    const uint64_t pop = static_cast<uint64_t>(std::popcount(m));
    c.st->thread_instructions += pop;
    c.st->inst_by_class[p.cls] += pop;
  }
  c.ex &= ~m;
  if (!c.ex) {
    c.done = true;
    return nullptr;
  }
  return &p + 1;
}

// ---- the compiler --------------------------------------------------------------

exec::InstClass cls_of(Op op) {
  using C = exec::InstClass;
  switch (op) {
    case Op::FADD: case Op::FMUL: case Op::FFMA: return C::Fp32;
    case Op::I2FP: return C::BitConvert;
    case Op::BRA: case Op::EXIT: return C::Control;
    case Op::LDG: case Op::STG: return C::Memory;
    case Op::IMAD: case Op::IADD3: case Op::LOP3: case Op::SHF: case Op::LEA: case Op::ISETP: return C::Integer;
    default: return C::Misc;
  }
}

bool is_term(Op op) { return op == Op::BRA || op == Op::EXIT; }

// Row layout, fixed before any instruction is compiled:
//   [0, nregs)           registers R0..   (RZ reads the zero row, writes the sink row)
//   zero, sink           two rows
//   8 special rows       TID.X/Y/Z, CTAID.X/Y/Z, LANEID, spare
//   max_ur rows          uniform registers (broadcast rows)
//   constants            one broadcast row per distinct constant, appended as met
struct Compiler {
  const Module& m;
  const CubinKernel& k;
  const LaunchInfo& li;
  Program& pr;
  std::map<uint64_t, uint32_t> const_idx;
  uint32_t max_ur = 0;
  const unsigned urz;

  Compiler(const Module& mm, const CubinKernel& kk, const LaunchInfo& l, Program& p)
      : m(mm), k(kk), li(l), pr(p), urz(mm.sm >= 100 ? 255u : 63u) {}

  uint32_t bank0_word(uint64_t off) const {
    if (off + 4 > li.bank0.size()) return 0;
    uint32_t v;
    std::memcpy(&v, &li.bank0[off], 4);
    return v;
  }
  uint32_t const_row(uint32_t v, bool launch = false) {
    const uint64_t key = (uint64_t{launch} << 32) | v;
    auto it = const_idx.find(key);
    if (it != const_idx.end()) return pr.first_const + it->second;
    const uint32_t idx = static_cast<uint32_t>(pr.const_rows.size());
    pr.const_rows.push_back(v);
    if (launch) {
      pr.const_k.push_back(static_cast<int32_t>(pr.launch_consts.size()));
      pr.launch_consts.push_back(v);
    } else {
      pr.const_k.push_back(-1);
    }
    const_idx[key] = idx;
    return pr.first_const + idx;
  }
  uint32_t ur_row(uint32_t n) const { return pr.first_ur + n; }
  uint32_t row(const Operand& o, bool dst = false) {
    switch (o.kind) {
      case Kind::Reg:
        if (o.reg == kRZ) return dst ? pr.sink_row : pr.zero_row;
        if (o.reg >= pr.nregs) throw Unsupported("register past the file");
        return o.reg;
      case Kind::UReg:
        if (o.reg >= urz) return dst ? pr.sink_row : pr.zero_row;
        return ur_row(o.reg);
      case Kind::Imm: return const_row(static_cast<uint32_t>(o.imm));
      case Kind::FImm: return const_row(o.fbits);
      case Kind::CBank:
        if (o.reg != 0) throw Unsupported("constant bank other than 0");
        return const_row(bank0_word(static_cast<uint64_t>(o.imm)), true);
      default: throw Unsupported("operand kind");
    }
  }
  static bool plain_src(const Operand& o) {
    return o.kind == Kind::Reg || o.kind == Kind::Imm || o.kind == Kind::FImm || o.kind == Kind::CBank || o.kind == Kind::UReg;
  }
  std::pair<uint32_t, uint32_t> row64(const Operand& o) {
    switch (o.kind) {
      case Kind::Reg:
        if (o.reg == kRZ) return {pr.zero_row, pr.zero_row};
        if (o.reg + 1 >= pr.nregs) throw Unsupported("register pair past the file");
        return {o.reg, o.reg + 1};
      case Kind::CBank:
        if (o.reg != 0) throw Unsupported("constant bank other than 0");
        return {const_row(bank0_word(static_cast<uint64_t>(o.imm)), true), const_row(bank0_word(static_cast<uint64_t>(o.imm) + 4), true)};
      case Kind::Imm: {
        const uint64_t v = static_cast<uint64_t>(o.imm);
        return {const_row(static_cast<uint32_t>(v)), const_row(static_cast<uint32_t>(v >> 32))};
      }
      default: throw Unsupported("64-bit operand kind");
    }
  }
  uint32_t dst_row(const Operand& o) {
    if (o.kind == Kind::Reg) {
      if (o.reg == kRZ) return pr.sink_row;
      if (o.reg >= pr.nregs) throw Unsupported("destination past the file");
      return o.reg;
    }
    if (o.kind == Kind::UReg) throw Unsupported("uniform destination");
    throw Unsupported("destination kind");
  }
  uint32_t pred_src(const Operand& o) const {   // index | negate << 8
    if (o.kind != Kind::Pred) throw Unsupported("predicate source kind");
    return o.reg | (o.neg ? 0x100u : 0u);
  }

  void emit(const Instr& ins, std::vector<MOp>& out) {
    if (ins.guard_uniform && (ins.guard != kPT || ins.guard_not)) throw Unsupported("uniform guard at " + to_text(ins));
    MOp op;
    op.pc = static_cast<uint32_t>(ins.pc);
    op.cls = static_cast<uint8_t>(cls_of(ins.op));
    const bool guarded = ins.guard != kPT || ins.guard_not;
    switch (ins.op) {
      case Op::IMAD: {
        const unsigned kind = ins.f[0];
        if (kind == 0x27 || ins.f[2] || ins.dst.size() > 1 || ins.src.size() < 3) throw Unsupported("IMAD form");
        for (int i = 0; i < 3; ++i)
          if (!plain_src(ins.src[i]) || (i < 2 || kind == 0x25 ? (ins.src[i].neg || ins.src[i].bnot) : false)) throw Unsupported("IMAD modifier");
        if (kind == 0x25) {
          const auto cc = row64(ins.src[2]);
          if (ins.dst[0].kind != Kind::Reg || ins.dst[0].reg == kRZ || ins.dst[0].reg + 1 >= pr.nregs) throw Unsupported("IMAD.WIDE destination");
          op.fn = h_imad_wide;
          op.d = ins.dst[0].reg;
          op.a = row(ins.src[0]);
          op.b = row(ins.src[1]);
          op.c = cc.first;
          op.e = cc.second;
          op.k0 = ins.f[1];
        } else {
          const auto zero = [](const Operand& o) {
            return (o.kind == Kind::Reg && o.reg == kRZ) || (o.kind == Kind::Imm && o.imm == 0);
          };
          op.d = dst_row(ins.dst[0]);
          if (zero(ins.src[0]) || zero(ins.src[1])) {
            op.fn = h_mov;
            op.a = row(ins.src[2]);
          } else if (ins.src[2].neg || ins.src[2].bnot) {
            op.fn = h_imadm;
            op.a = row(ins.src[0]);
            op.b = row(ins.src[1]);
            op.c = row(ins.src[2]);
            op.k0 = ins.src[2].neg;
            op.k1 = ins.src[2].bnot;
          } else {
            op.fn = h_imad;
            op.a = row(ins.src[0]);
            op.b = row(ins.src[1]);
            op.c = row(ins.src[2]);
          }
        }
        break;
      }
      case Op::IADD3: {
        if (ins.f[0] || ins.f[1] || ins.src.size() < 3) throw Unsupported("IADD3 form");
        for (size_t i = 1; i < ins.dst.size(); ++i)
          if (!(ins.dst[i].kind == Kind::Pred && ins.dst[i].reg == kPT)) throw Unsupported("IADD3 carry out");
        for (int i = 0; i < 3; ++i)
          if (!plain_src(ins.src[i])) throw Unsupported("IADD3 source");
        op.d = dst_row(ins.dst[0]);
        op.a = row(ins.src[0]);
        op.b = row(ins.src[1]);
        op.c = row(ins.src[2]);
        const bool any = ins.src[0].neg || ins.src[0].bnot || ins.src[1].neg || ins.src[1].bnot || ins.src[2].neg || ins.src[2].bnot;
        if (!any) {
          op.fn = h_iadd3_plain;
        } else {
          op.fn = h_iadd3;
          op.imm = (ins.src[0].neg || ins.src[0].bnot) ? ~0u : 0;
          op.imm2 = (ins.src[1].neg || ins.src[1].bnot) ? ~0u : 0;
          op.e = (ins.src[2].neg || ins.src[2].bnot) ? ~0u : 0;
          op.k3 = (ins.src[0].neg ? 1 : 0) + (ins.src[1].neg ? 1 : 0) + (ins.src[2].neg ? 1 : 0);
        }
        break;
      }
      case Op::LOP3: {
        if (ins.dst.size() != 1 || ins.src.size() < 3) throw Unsupported("LOP3 form");
        for (int i = 0; i < 3; ++i)
          if (!plain_src(ins.src[i])) throw Unsupported("LOP3 source");
        op.d = dst_row(ins.dst[0]);
        op.a = row(ins.src[0]);
        op.b = row(ins.src[1]);
        op.c = row(ins.src[2]);
        op.imm = ins.f[0];
        op.fn = ins.f[0] == 0x96 ? h_lop3_xor3 : h_lop3;
        break;
      }
      case Op::SHF: {
        if (ins.dst.size() != 1 || ins.src.size() < 3) throw Unsupported("SHF form");
        for (int i = 0; i < 3; ++i)
          if (!plain_src(ins.src[i]) || ins.src[i].neg || ins.src[i].bnot) throw Unsupported("SHF source");
        op.fn = h_shf;
        op.d = dst_row(ins.dst[0]);
        op.a = row(ins.src[0]);
        op.b = row(ins.src[1]);
        op.c = row(ins.src[2]);
        op.k0 = ins.f[0]; op.k1 = ins.f[1]; op.k2 = ins.f[2]; op.k3 = ins.f[3];
        break;
      }
      case Op::LEA: {
        const bool hi = ins.f[0], x = ins.f[1], sx32 = ins.f[2];
        const bool three = hi && !sx32;
        const size_t pidx = three ? 4 : 3;
        if (ins.src.size() < (x ? pidx + 1 : pidx) || ins.dst.size() > 2) throw Unsupported("LEA form");
        if (ins.dst.size() == 2 && ins.dst[1].kind != Kind::Pred) throw Unsupported("LEA carry kind");
        if (!plain_src(ins.src[0]) || !plain_src(ins.src[1]) || ins.src[0].neg || ins.src[0].bnot) throw Unsupported("LEA source");
        if (three && (!plain_src(ins.src[2]) || ins.src[2].neg || ins.src[2].bnot)) throw Unsupported("LEA top");
        if (ins.src[1].neg) throw Unsupported("LEA negated addend");
        op.fn = h_lea;
        op.d = dst_row(ins.dst[0]);
        op.a = row(ins.src[0]);
        op.b = row(ins.src[1]);
        op.c = three ? row(ins.src[2]) : pr.zero_row;
        op.e = x ? pred_src(ins.src[pidx]) : 7;
        op.imm = ins.f[3];
        op.imm2 = ins.dst.size() == 2 ? ins.dst[1].reg : kNoRow;
        op.k0 = hi; op.k1 = x; op.k2 = three;
        op.k3 = ins.src[1].bnot ? 2 : 0;
        break;
      }
      case Op::ISETP: {
        if (ins.f[3] || ins.f[4] || ins.dst.size() < 2 || ins.dst[0].kind != Kind::Pred || ins.dst[1].kind != Kind::Pred ||
            ins.src.size() < 3)
          throw Unsupported("ISETP form");
        for (int i = 0; i < 2; ++i)
          if (!(ins.src[i].kind == Kind::Reg || ins.src[i].kind == Kind::Imm || ins.src[i].kind == Kind::CBank || ins.src[i].kind == Kind::UReg) ||
              ins.src[i].neg || ins.src[i].bnot)
            throw Unsupported("ISETP source");
        op.fn = h_isetp;
        op.a = row(ins.src[0]);
        op.b = row(ins.src[1]);
        op.d = ins.dst[0].reg;
        op.imm = ins.dst[1].reg == kPT ? kNoRow : ins.dst[1].reg;
        op.e = pred_src(ins.src[2]);
        op.k0 = ins.f[0]; op.k1 = ins.f[1]; op.k2 = ins.f[2];
        break;
      }
      case Op::FFMA: case Op::FADD: case Op::FMUL: {
        const unsigned nsrc = ins.op == Op::FFMA ? 3 : 2;
        if (ins.f[0] || ins.f[1] || ins.f[2] || (ins.op == Op::FMUL && ins.f[4] != 4) || ins.src.size() < nsrc)
          throw Unsupported("FP form");
        op.fn = ins.op == Op::FFMA ? h_ffma : ins.op == Op::FADD ? h_fadd : h_fmul;
        op.d = dst_row(ins.dst[0]);
        uint32_t* rows[3] = {&op.a, &op.b, &op.c};
        for (unsigned i = 0; i < nsrc; ++i) {
          const Operand& o = ins.src[i];
          if (!plain_src(o) || o.bnot) throw Unsupported("FP source");
          *rows[i] = row(o);
          if (o.abs) op.imm |= 1u << i;
          if (o.neg) op.imm2 |= 1u << i;
        }
        break;
      }
      case Op::I2FP: {
        // dsize/ssize are both 32-bit for I2FP (see the general path); f[1] is the rounding mode (0: nearest)
        if (ins.f[1] != 0 || ins.src.empty() || !plain_src(ins.src[0]) || ins.src[0].neg || ins.src[0].abs || ins.src[0].bnot) throw Unsupported("I2FP form");
        op.fn = h_i2fp;
        op.d = dst_row(ins.dst[0]);
        op.a = row(ins.src[0]);
        op.k0 = ins.f[0] != 0;
        break;
      }
      case Op::MOV: {
        if (ins.mnemonic != "MOV" || ins.f[1] || ins.f[0] != 0xf || ins.src.empty() || !plain_src(ins.src[0])) throw Unsupported("MOV form");
        op.fn = h_mov;
        op.d = dst_row(ins.dst[0]);
        op.a = row(ins.src[0]);
        break;
      }
      case Op::S2R: {
        const unsigned idx = ins.f[0];
        op.fn = h_mov;
        op.d = dst_row(ins.dst[0]);
        switch (idx) {
          case 0x21: op.a = pr.first_spec + 0; break;   // TID.X
          case 0x22: op.a = pr.first_spec + 1; break;
          case 0x23: op.a = pr.first_spec + 2; break;
          case 0x25: op.a = pr.first_spec + 3; break;   // CTAID.X
          case 0x26: op.a = pr.first_spec + 4; break;
          case 0x27: op.a = pr.first_spec + 5; break;
          case 0x00: op.a = pr.first_spec + 6; break;   // LANEID
          default: throw Unsupported("special register");
        }
        break;
      }
      case Op::S2UR: {
        const unsigned idx = ins.f[0];
        if (ins.dst[0].kind != Kind::UReg || ins.dst[0].reg >= urz || (idx != 0x25 && idx != 0x26 && idx != 0x27)) throw Unsupported("S2UR form");
        op.fn = h_mov;
        op.d = ur_row(ins.dst[0].reg);
        op.a = pr.first_spec + 3 + (idx - 0x25);
        break;
      }
      case Op::ULDC: {
        const unsigned size = ins.f[0];
        if (ins.f[1] || ins.dst[0].kind != Kind::UReg || ins.src[0].kind != Kind::CBank || ins.src[0].reg != 0 || (size != 4 && size != 5) ||
            ins.dst[0].reg + 1 >= urz)
          throw Unsupported("ULDC form");
        const uint32_t words = size == 5 ? 2 : 1;
        for (uint32_t i = 0; i < words; ++i) {
          MOp w = op;
          w.fn = h_mov;
          w.d = ur_row(ins.dst[0].reg + i);
          w.a = const_row(bank0_word(static_cast<uint64_t>(ins.src[0].imm) + 4 * i), true);
          if (i + 1 < words) out.push_back(w);
          else op = w;
        }
        break;
      }
      case Op::LDG: case Op::STG: {
        const bool store = ins.op == Op::STG;
        const bool e = ins.w.bit(72);
        if (ins.f[0] != 4 || !e || m.sm < 80) throw Unsupported("memory form");
        const Operand& a = ins.src[0];
        if (a.kind != Kind::Mem) throw Unsupported("memory operand");
        const unsigned ra = static_cast<unsigned>(ins.w.field(24, 8));
        if (ra == kRZ || ra + 1 >= pr.nregs) throw Unsupported("address register");
        op.fn = store ? h_stg32 : h_ldg32;
        op.a = ra;
        op.imm = static_cast<uint32_t>(a.imm);
        if (store) {
          const Operand& d = ins.src[1];
          if (d.kind != Kind::Reg) throw Unsupported("store data");
          op.d = d.reg == kRZ ? pr.zero_row : d.reg;
        } else {
          op.d = dst_row(ins.dst[0]);
        }
        break;
      }
      case Op::BRA: {
        if (ins.src.empty() || ins.src[0].kind != Kind::Label) throw Unsupported("BRA form");
        op.fn = h_bra;
        op.imm = static_cast<uint32_t>(ins.src[0].imm);   // absolute address; resolved after layout
        break;
      }
      case Op::EXIT:
        op.fn = h_exit;
        break;
      case Op::NOP:
        op.fn = h_mov;
        op.d = pr.sink_row;
        op.a = pr.zero_row;
        break;
      default:
        throw Unsupported(std::string("op ") + op_name(ins.op));
    }
    if (guarded) {
      op.g = ins.guard;
      op.gneg = ins.guard_not;
      if (ins.op != Op::BRA && ins.op != Op::EXIT) {
        op.alt = op.fn;
        op.fn = h_guard;
      }
    }
    out.push_back(op);
  }
};

}  // namespace

Program compile(const Module& m, const CubinKernel& k, const LaunchInfo& li) {
  const auto now = [] { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; };
  const double t0 = now();
  Program pr;
  pr.nregs = std::min<uint32_t>(std::max<uint32_t>(k.regs + 8, 16), 256);
  const Code& code = m.code[m.code_index.at(k.text_section)];
  pr.n_sass = code.count;
  const unsigned urz = m.sm >= 100 ? 255u : 63u;

  // Reachable leaders.
  std::set<uint64_t> leaders;
  std::vector<uint64_t> work{0};
  std::set<uint64_t> seen;
  const auto index = [&](uint64_t pc) { return pc / 16; };
  uint32_t max_ur = 0;
  const auto scan_urs = [&](const Instr& ins) {
    for (const Operand& o : ins.dst) if (o.kind == Kind::UReg && o.reg < urz) max_ur = std::max(max_ur, o.reg + 2);
    for (const Operand& o : ins.src) if (o.kind == Kind::UReg && o.reg < urz) max_ur = std::max(max_ur, o.reg + 2);
  };
  while (!work.empty()) {
    const uint64_t start = work.back();
    work.pop_back();
    if (!seen.insert(start).second) continue;
    leaders.insert(start);
    uint64_t pc = start;
    for (;;) {
      if (index(pc) >= code.count) throw Unsupported("runs off the end of the code");
      const Instr& ins = code.instr(index(pc));
      scan_urs(ins);
      if (is_term(ins.op)) {
        const bool always = ins.guard == kPT && !ins.guard_not;
        if (ins.op == Op::BRA) {
          if (ins.src.empty() || ins.src[0].kind != Kind::Label) throw Unsupported("BRA form");
          work.push_back(static_cast<uint64_t>(ins.src[0].imm));
        }
        if (!always) work.push_back(pc + 16);
        break;
      }
      pc += 16;
      if (seen.count(pc)) break;   // falls into a known leader
    }
  }
  // Fixed row layout.
  pr.zero_row = pr.nregs;
  pr.sink_row = pr.nregs + 1;
  pr.first_spec = pr.nregs + 2;
  pr.first_ur = pr.first_spec + 8;
  pr.first_const = pr.first_ur + max_ur;
  pr.row_tid_x = pr.first_spec;
  pr.row_tid_y = pr.first_spec + 1;
  pr.row_tid_z = pr.first_spec + 2;
  pr.ctaid_rows = {pr.first_spec + 3, pr.first_spec + 4, pr.first_spec + 5};
  pr.row_laneid = pr.first_spec + 6;
  Compiler cc(m, k, li, pr);
  cc.max_ur = max_ur;

  std::map<uint64_t, uint32_t> block_start;   // leader pc -> op index of its header
  const std::vector<uint64_t> order(leaders.begin(), leaders.end());
  for (size_t bi = 0; bi < order.size(); ++bi) {
    const uint64_t start = order[bi];
    const uint64_t limit = bi + 1 < order.size() ? order[bi + 1] : 16 * code.count;
    BlockInfo info;
    std::vector<MOp> body;
    for (uint64_t pc = start; pc < limit; pc += 16) {
      const Instr& ins = code.instr(index(pc));
      const bool free = ins.guard == kPT && !ins.guard_not;
      ++info.n_total;
      if (free) {
        ++info.n_free;
        ++info.cls_free[static_cast<size_t>(cls_of(ins.op))];
      }
      cc.emit(ins, body);
      if (is_term(ins.op)) break;
    }
    MOp hdr;
    hdr.fn = h_header;
    hdr.imm = static_cast<uint32_t>(pr.infos.size());
    pr.infos.push_back(info);
    block_start[start] = static_cast<uint32_t>(pr.ops.size());
    pr.ops.push_back(hdr);
    for (MOp& o : body) pr.ops.push_back(o);
  }
  for (MOp& o : pr.ops)
    if (o.fn == h_bra) {
      const auto it = block_start.find(o.imm);
      if (it == block_start.end()) throw Unsupported("branch target is not a block start");
      o.target = static_cast<int32_t>(it->second);
    }
  pr.nrows = pr.first_const + static_cast<uint32_t>(pr.const_rows.size());
  pr.compile_seconds = now() - t0;
  return pr;
}

// Whether one instruction, on its own, is inside the prototype's subset (for coverage statistics).
bool supports(const Module& m, const CubinKernel& k, const Instr& ins, std::string* why) {
  Program pr;
  LaunchInfo li;
  li.bank0.assign(0x400, 0);
  pr.nregs = std::min<uint32_t>(std::max<uint32_t>(k.regs + 8, 16), 256);
  pr.zero_row = pr.nregs;
  pr.sink_row = pr.nregs + 1;
  pr.first_spec = pr.nregs + 2;
  pr.first_ur = pr.first_spec + 8;
  pr.first_const = pr.first_ur + 70;
  Compiler cc(m, k, li, pr);
  std::vector<MOp> out;
  try {
    cc.emit(ins, out);
    return true;
  } catch (const Unsupported& u) {
    if (why) *why = u.what();
    return false;
  }
}

std::vector<uint8_t> make_bank0(const Module& m, const CubinKernel& k, const LaunchInfo& li,
                                const std::vector<std::vector<uint8_t>>& args) {
  if (m.sm >= 90) throw Unsupported("bank-0 layout for sm_90+ not written in the prototype");
  const CubinSection* tmpl = m.cubin.section(".nv.constant0." + k.name);
  std::vector<uint8_t> b(std::max<size_t>(tmpl ? tmpl->size : 0, k.param_base + k.param_size + 16), 0);
  if (tmpl && !tmpl->bytes.empty()) std::memcpy(b.data(), tmpl->bytes.data(), tmpl->bytes.size());
  if (b.size() < 0x220) b.resize(0x220, 0);
  const auto put32 = [&](size_t off, uint32_t v) { std::memcpy(&b[off], &v, 4); };
  const auto put64 = [&](size_t off, uint64_t v) { std::memcpy(&b[off], &v, 8); };
  for (int i = 0; i < 3; ++i) {
    put32(0x0 + 4 * i, li.block[i]);
    put32(0xc + 4 * i, li.grid[i]);
  }
  put64(0x18, kSharedWindow);
  put64(0x20, kLocalWindow);
  put32(0x28, 1024);   // the stack: the runner's local_size_ for a 1 KiB stack and no frame
  for (size_t i = 0; i < args.size(); ++i) std::memcpy(&b[k.param_base + k.params[i].offset], args[i].data(), k.params[i].size);
  return b;
}

exec::LaunchStats run(const Program& pr, const LaunchInfo& li, MemoryManager& mem) {
  const uint64_t blocks = uint64_t{li.grid[0]} * li.grid[1] * li.grid[2];
  const uint32_t threads = li.block[0] * li.block[1] * li.block[2];
  const uint32_t nwarps = (threads + 31) / 32;
  exec::LaunchStats stats;
  std::vector<uint32_t> file_storage(static_cast<size_t>(pr.nrows) * 32 + 16);
  uint32_t* file = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(file_storage.data()) + 63) & ~uintptr_t{63});
  Ctx c;
  c.R = file;
  c.st = &stats;
  c.max_steps = li.max_steps;
  c.mem = &mem;
  c.infos = pr.infos.data();
  c.ops = pr.ops.data();
  c.K = pr.launch_consts.data();
  for (uint64_t lin = 0; lin < blocks; ++lin) {
    const uint32_t cta[3] = {static_cast<uint32_t>(lin % li.grid[0]), static_cast<uint32_t>((lin / li.grid[0]) % li.grid[1]),
                             static_cast<uint32_t>(lin / (uint64_t{li.grid[0]} * li.grid[1]))};
    stats.warps += nwarps;
    for (uint32_t wi = 0; wi < nwarps; ++wi) {
      // Registers and the zero/sink rows start at zero (the runner zero-fills a warp's file too);
      // then the special rows and the uniform registers; the constant rows are written once per warp.
      std::memset(file, 0, static_cast<size_t>(pr.nregs + 2) * 128);
      uint32_t *tx = file + static_cast<size_t>(pr.row_tid_x) * 32, *ty = file + static_cast<size_t>(pr.row_tid_y) * 32,
               *tz = file + static_cast<size_t>(pr.row_tid_z) * 32, *ln = file + static_cast<size_t>(pr.row_laneid) * 32;
      for (unsigned l = 0; l < 32; ++l) {
        const uint32_t tid = wi * 32 + l;
        tx[l] = tid % li.block[0];
        ty[l] = (tid / li.block[0]) % li.block[1];
        tz[l] = tid / (li.block[0] * li.block[1]);
        ln[l] = l;
      }
      for (int d = 0; d < 3; ++d) {
        uint32_t* r = file + static_cast<size_t>(pr.ctaid_rows[d]) * 32;
        for (unsigned l = 0; l < 32; ++l) r[l] = cta[d];
      }
      if (pr.first_const > pr.first_ur) std::memset(file + static_cast<size_t>(pr.first_ur) * 32, 0, static_cast<size_t>(pr.first_const - pr.first_ur) * 128);
      for (size_t i = 0; i < pr.const_rows.size(); ++i) {
        uint32_t* r = file + static_cast<size_t>(pr.first_const + i) * 32;
        for (unsigned l = 0; l < 32; ++l) r[l] = pr.const_rows[i];
      }
      const uint32_t n = std::min<uint32_t>(32, threads - wi * 32);
      c.ex = n == 32 ? kAll : ((Mask{1} << n) - 1);
      for (int i = 0; i < 7; ++i) c.P[i] = 0;
      c.P[7] = kAll;
      c.done = false;
      c.steps = 0;
      for (const MOp* p = pr.ops.data(); p; p = p->fn(c, *p)) {
      }
    }
  }
  stats.blocks = blocks;
  return stats;
}

void set_fast_counters(bool on) { g_fast_counters = on; }

namespace {
using RegionFn = void (*)(uint32_t* __restrict, uint32_t* __restrict, uint32_t, const uint32_t*);
const MOp* h_region(Ctx& c, const MOp& p) {
  reinterpret_cast<RegionFn>(p.user)(c.R, c.P, c.ex, c.K);
  return &p + 1;
}
}  // namespace

Fn region_handler() { return h_region; }

OpKind kind_of(const MOp& o) {
  const Fn f = o.fn;
  if (f == h_header) return OpKind::Header;
  if (f == h_guard) return OpKind::Guard;
  if (f == h_mov) return OpKind::Mov;
  if (f == h_iadd3) return OpKind::Iadd3;
  if (f == h_iadd3_plain) return OpKind::Iadd3Plain;
  if (f == h_imad) return OpKind::Imad;
  if (f == h_imadm) return OpKind::ImadM;
  if (f == h_imad_wide) return OpKind::ImadWide;
  if (f == h_lop3) return OpKind::Lop3;
  if (f == h_lop3_xor3) return OpKind::Lop3Xor3;
  if (f == h_shf) return OpKind::Shf;
  if (f == h_lea) return OpKind::Lea;
  if (f == h_isetp) return OpKind::Isetp;
  if (f == h_ffma) return OpKind::Ffma;
  if (f == h_fadd) return OpKind::Fadd;
  if (f == h_fmul) return OpKind::Fmul;
  if (f == h_i2fp) return OpKind::I2fp;
  if (f == h_ldg32) return OpKind::Ldg;
  if (f == h_stg32) return OpKind::Stg;
  if (f == h_bra) return OpKind::Bra;
  if (f == h_exit) return OpKind::Exit;
  if (f == h_region) return OpKind::Region;
  return OpKind::Other;
}
void set_bulk_mem(bool on) { g_bulk_mem = on; }

}  // namespace r7
