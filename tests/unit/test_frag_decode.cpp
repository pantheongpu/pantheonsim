// The mma fragment decode (exec::mma_frag_a, mma_frag_b: the registers of A and B as the row-major tiles of the tile kernels) against the
// element-at-a-time form it replaced, written out here as the frozen reference: every element's register, lane and bit position from the PTX ISA's
// layout, its value from the scalar decoders, a sparse A placed by mma.sp's metadata rule. All nine element types, every shape the forms use (dense and
// sparse, m16 and m8), metadata selectors 0 to 3, registers of random bits (every f16, bf16, tf32 and fp8 class: NaNs, infinities, subnormals, zeros of
// both signs, the extremes) and of special words. Bit for bit, NaN payloads included. Run again with VGPU_SIMD=base (the F16C conversion on and off).
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/exec/tile_mma.hpp"
#include "vgpu/simd/fp16.hpp"
#include "vtest.hpp"

using namespace vgpu;
using exec::FragType;

namespace {

// ---- the reference: the first form of the decode, one element at a time ----------------------------------------------------------

float ref_float(FragType t, uint32_t reg, unsigned e) {
  const unsigned bits = exec::frag_bits(t);
  const uint32_t raw = bits == 32 ? reg : reg >> (bits * e);
  switch (t) {
    case FragType::F16: return simd::f16_to_f32(static_cast<uint16_t>(raw));
    case FragType::BF16: return std::bit_cast<float>((raw & 0xffffu) << 16);
    case FragType::TF32: return std::bit_cast<float>(raw & 0xffffe000u);
    case FragType::E4M3: return static_cast<float>(exec::fp8_value(raw & 0xff, false));
    default: return static_cast<float>(exec::fp8_value(raw & 0xff, true));
  }
}
int32_t ref_int(FragType t, uint32_t reg, unsigned e) {
  switch (t) {
    case FragType::S8: return static_cast<int8_t>(reg >> (8 * e));
    case FragType::U8: return static_cast<uint8_t>(reg >> (8 * e));
    case FragType::S4: return static_cast<int32_t>(((reg >> (4 * e)) & 0xfu) << 28) >> 28;
    default: return static_cast<int32_t>((reg >> (4 * e)) & 0xfu);
  }
}

// mma.sp's placement of stored element k of chunk `chunk` in row `row`, from the 32 lanes' metadata words and the selector.
unsigned ref_sparse_column(unsigned bits, unsigned K, unsigned sel, const uint32_t* meta, unsigned row, unsigned chunk, unsigned k) {
  const bool tf32 = bits == 32, int4 = bits == 4, narrow = bits <= 8;
  const unsigned chunk_elems = tf32 ? 2 : int4 ? 8 : 4;
  const unsigned row_bits = tf32 ? 2 * K : int4 ? K / 2 : K;
  const unsigned holders = 2 * row_bits / 32;
  const unsigned group = row % 8, half = row / 8;
  const unsigned src = narrow ? 4 * group + 2 * sel + half + 2 * (chunk / 8) : 4 * group + holders * sel + chunk / 4;
  const unsigned shift = narrow ? (chunk % 8) * 4 : half * 16 + (chunk % 4) * 4;
  const uint32_t nib = (src < 32 ? meta[src] : 0u) >> shift;
  const unsigned idx0 = nib & 3, idx1 = (nib >> 2) & 3;
  unsigned pos;
  if (tf32) pos = idx0 / 2;
  else if (int4) pos = 2 * (k < 2 ? idx0 : idx1) + k % 2;
  else pos = k == 0 ? idx0 : idx1;
  return chunk * chunk_elems + pos;
}

void ref_frag_a(FragType t, unsigned M, unsigned K, bool sparse, const uint32_t* regs, const uint32_t* meta, unsigned sel, float* f, int32_t* iout) {
  const bool is_int = exec::frag_is_int(t);
  const unsigned bits = exec::frag_bits(t), p = 32 / bits;
  const unsigned nregs = M * K / (sparse ? 2 : 1) / (32 * p);
  if (sparse) {
    if (is_int) std::fill(iout, iout + size_t{M} * K, 0);
    else std::fill(f, f + size_t{M} * K, 0.0f);
  }
  const bool tf32 = t == FragType::TF32, int4 = bits == 4;
  const unsigned chunk_elems = tf32 ? 2 : int4 ? 8 : 4, stored = chunk_elems / 2, cpr = p / stored;
  for (unsigned r = 0; r < nregs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, tt = l % 4, row = g + 8 * (r % 2), col0 = tt * p + (r / 2) * 4 * p;
      const uint32_t v = regs[size_t{r} * 32 + l];
      for (unsigned e = 0; e < p; ++e) {
        const unsigned col = sparse ? ref_sparse_column(bits, K, sel, meta, row, tt * cpr + (r / 2) * 4 * cpr + e / stored, e % stored) : col0 + e;
        if (col >= K) continue;
        if (is_int) iout[size_t{row} * K + col] = ref_int(t, v, e);
        else f[size_t{row} * K + col] = ref_float(t, v, e);
      }
    }
}

void ref_frag_b(FragType t, unsigned K, const uint32_t* regs, float* f, int32_t* iout) {
  const bool is_int = exec::frag_is_int(t);
  const unsigned p = 32 / exec::frag_bits(t), nregs = K * 8 / (32 * p);
  for (unsigned r = 0; r < nregs; ++r)
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned g = l / 4, tt = l % 4, row0 = tt * p + r * 4 * p;
      const uint32_t v = regs[size_t{r} * 32 + l];
      for (unsigned e = 0; e < p; ++e) {
        if (is_int) iout[size_t{row0 + e} * 8 + g] = ref_int(t, v, e);
        else f[size_t{row0 + e} * 8 + g] = ref_float(t, v, e);
      }
    }
}

// ---- the shapes ----------------------------------------------------------------------------------------------------------------------

struct Form {
  FragType t;
  const char* name;
  unsigned M, K;
  bool sparse;
};
// (K is the logical K of the instruction; a sparse A stores half of it.)
const Form kForms[] = {
    {FragType::F16, "f16 m16k8", 16, 8, false},      {FragType::F16, "f16 m16k16", 16, 16, false},    {FragType::F16, "f16 sp k16", 16, 16, true},
    {FragType::F16, "f16 sp k32", 16, 32, true},     {FragType::BF16, "bf16 m16k16", 16, 16, false},  {FragType::BF16, "bf16 sp k32", 16, 32, true},
    {FragType::TF32, "tf32 m16k4", 16, 4, false},    {FragType::TF32, "tf32 m16k8", 16, 8, false},    {FragType::TF32, "tf32 sp k8", 16, 8, true},
    {FragType::TF32, "tf32 sp k16", 16, 16, true},   {FragType::E4M3, "e4m3 k16", 16, 16, false},     {FragType::E4M3, "e4m3 k32", 16, 32, false},
    {FragType::E4M3, "e4m3 sp k64", 16, 64, true},   {FragType::E5M2, "e5m2 k32", 16, 32, false},     {FragType::E5M2, "e5m2 sp k64", 16, 64, true},
    {FragType::S8, "s8 m8k16", 8, 16, false},        {FragType::S8, "s8 m16k16", 16, 16, false},      {FragType::S8, "s8 m16k32", 16, 32, false},
    {FragType::S8, "s8 sp k32", 16, 32, true},       {FragType::S8, "s8 sp k64", 16, 64, true},       {FragType::U8, "u8 m16k32", 16, 32, false},
    {FragType::U8, "u8 sp k64", 16, 64, true},       {FragType::S4, "s4 m8k32", 8, 32, false},        {FragType::S4, "s4 m16k32", 16, 32, false},
    {FragType::S4, "s4 m16k64", 16, 64, false},      {FragType::S4, "s4 sp k64", 16, 64, true},       {FragType::S4, "s4 sp k128", 16, 128, true},
    {FragType::U4, "u4 m16k64", 16, 64, false},      {FragType::U4, "u4 sp k128", 16, 128, true},
};
struct BForm {
  FragType t;
  unsigned K;
};
const BForm kBForms[] = {{FragType::F16, 8},   {FragType::F16, 16}, {FragType::F16, 32}, {FragType::BF16, 16}, {FragType::BF16, 32}, {FragType::TF32, 4},
                         {FragType::TF32, 8},  {FragType::TF32, 16}, {FragType::E4M3, 16}, {FragType::E4M3, 32}, {FragType::E4M3, 64}, {FragType::E5M2, 32},
                         {FragType::S8, 16},   {FragType::S8, 32},  {FragType::S8, 64},   {FragType::U8, 32},   {FragType::S4, 32},   {FragType::S4, 64},
                         {FragType::S4, 128},  {FragType::U4, 64},  {FragType::U4, 128}};

uint32_t special_word(std::mt19937& g, FragType t) {
  static const uint32_t half[] = {0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe01, 0x7c01, 0x3c00, 0xbc00, 0x03ff};
  switch (g() % 6) {
    case 0: return 0;
    case 1: return 0xffffffffu;
    case 2: return 0x80808080u;
    case 3:
      if (exec::frag_bits(t) == 16) return half[g() % 15] | half[g() % 15] << 16;
      return 0x7f7f7f7fu & g();
    case 4: return 0x7f800000u | (g() & 0x807fffffu);
    default: return g();
  }
}

bool same(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }

}  // namespace

VTEST(the_a_fragment_decode_matches_the_element_form) {
  std::mt19937 g(0x51a7);
  long compared = 0;
  for (const Form& f : kForms) {
    const unsigned bits = exec::frag_bits(f.t), p = 32 / bits;
    const unsigned nregs = f.M * f.K / (f.sparse ? 2 : 1) / (32 * p);
    for (int trial = 0; trial < 120; ++trial) {
      std::vector<uint32_t> regs(size_t{nregs} * 32), meta(32);
      for (uint32_t& w : regs) w = trial < 20 ? special_word(g, f.t) : g();
      for (uint32_t& w : meta) w = g();
      const unsigned sel = trial % 4;
      const bool is_int = exec::frag_is_int(f.t);
      std::vector<float> gf(size_t{f.M} * f.K, -1.0f), wf(size_t{f.M} * f.K, -1.0f);
      std::vector<int32_t> gi(size_t{f.M} * f.K, -7), wi(size_t{f.M} * f.K, -7);
      exec::mma_frag_a(f.t, f.M, f.K, f.sparse, regs.data(), meta.data(), sel, gf.data(), gi.data());
      ref_frag_a(f.t, f.M, f.K, f.sparse, regs.data(), meta.data(), sel, wf.data(), wi.data());
      for (size_t i = 0; i < gf.size(); ++i) {
        ++compared;
        if (is_int ? gi[i] != wi[i] : !same(gf[i], wf[i])) {
          char buf[200];
          std::snprintf(buf, sizeof buf, "%s trial %d sel %u: element %zu (row %zu, col %zu): %s", f.name, trial, sel, i, i / f.K, i % f.K,
                        is_int ? (std::to_string(gi[i]) + " against " + std::to_string(wi[i])).c_str() : "float bits differ");
          throw vtest::Failure(buf);
        }
      }
    }
  }
  VCHECK(compared > 1000000);
}

VTEST(the_b_fragment_decode_matches_the_element_form) {
  std::mt19937 g(0xb00f);
  long compared = 0;
  for (const BForm& f : kBForms) {
    const unsigned p = 32 / exec::frag_bits(f.t), nregs = f.K * 8 / (32 * p);
    for (int trial = 0; trial < 150; ++trial) {
      std::vector<uint32_t> regs(size_t{nregs} * 32);
      for (uint32_t& w : regs) w = trial < 20 ? special_word(g, f.t) : g();
      const bool is_int = exec::frag_is_int(f.t);
      std::vector<float> gf(size_t{f.K} * 8, -1.0f), wf(size_t{f.K} * 8, -1.0f);
      std::vector<int32_t> gi(size_t{f.K} * 8, -7), wi(size_t{f.K} * 8, -7);
      exec::mma_frag_b(f.t, f.K, regs.data(), gf.data(), gi.data());
      ref_frag_b(f.t, f.K, regs.data(), wf.data(), wi.data());
      for (size_t i = 0; i < gf.size(); ++i) {
        ++compared;
        if (is_int ? gi[i] != wi[i] : !same(gf[i], wf[i]))
          throw vtest::Failure("B type " + std::to_string(static_cast<int>(f.t)) + " K " + std::to_string(f.K) + " trial " + std::to_string(trial) + ": element " + std::to_string(i));
      }
    }
  }
  VCHECK(compared > 300000);
}

VTEST_MAIN
