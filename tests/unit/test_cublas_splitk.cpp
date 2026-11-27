// The cuBLAS split-K model (nvidia/src/cublas_splitk.hpp, gemm_tensor_sum.hpp; issue #734), without a card or the shim:
//   * the tables hold the products that were measured (rows recorded from an A10G and a Tesla T4 on 2026-11-25/26) and say nothing about the rest;
//   * product() with a slice length is the slices' chains added in order in f32, and with no slice (or one as long as K) the plain chain;
//   * product_epilogue() makes the card's epilogue (fused where the card fuses): the unsplit form fma(alpha, acc, beta * c) differs from the unfused
//     one on inputs chosen to show it, the three forms are what they say;
//   * the profiles name their table with a class: measured for the cards, inferred for the A10G's table on an RTX 3080 Ti, none for a card of
//     another multiprocessor count.
// The bits against the cards themselves are nvidia/tests/e2e/cublas_tc_gemm.cu's.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cublas_splitk.hpp"
#include "gemm_tensor_sum.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

// What the shim's runtime answers from the profile, for an Ampere card of 80 multiprocessors.
extern "C" int vgpu_tensor_sum_v2(int, unsigned* chunk, unsigned* chunk_tf32, int* cc, int* splitk) {
  *chunk = 8; *chunk_tf32 = 4; *cc = 86; *splitk = 2;
  return 1;
}
extern "C" int vgpu_tensor_sum_v1(int d, unsigned* a, unsigned* b, int* c) { int s; return vgpu_tensor_sum_v2(d, a, b, c, &s); }

using namespace vgpu::gemmtc;

namespace {
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }

std::vector<float> random_matrix(size_t n, uint64_t seed) {
  std::vector<float> v(n);
  uint64_t s = seed;
  for (auto& x : v) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    // a value exactly representable in f16: 11 significant bits
    x = static_cast<float>(static_cast<int>(s % 4001) - 2000) / 512.0f;
  }
  return v;
}
}  // namespace

VTEST(tables_hold_the_measured_products) {
  // 256x256x4096 f16 -> f32 NN: 8 slices of 512 on the A10G (80 multiprocessors), 448 on the Tesla T4 (40).
  if (plan_of(SplitTable::A10g, Format::F16, false, false, false, 256, 256, 4096).len != 512) throw vtest::Failure("A10G f16 256x256x4096 is not 512");
  if (plan_of(SplitTable::T4, Format::F16, false, false, false, 256, 256, 4096).len != 448) throw vtest::Failure("T4 f16 256x256x4096 is not 448");
  // TF32 128x128x256: 96, 96, 64 (three slices); the A10G's two-slice rows carry the serial epilogue.
  if (plan_of(SplitTable::A10g, Format::Tf32, false, false, false, 128, 128, 256).len != 96) throw vtest::Failure("A10G tf32 128x128x256 is not 96");
  // Not in a table: the chain (0), as for an unmeasured shape and the None table.
  if (plan_of(SplitTable::A10g, Format::F16, false, false, false, 100, 100, 1000).len != 0) throw vtest::Failure("an off-grid shape has a plan");
  if (plan_of(SplitTable::None, Format::F16, false, false, false, 256, 256, 4096).len != 0) throw vtest::Failure("the None table has a plan");
  // A product measured unsplit.
  if (plan_of(SplitTable::A10g, Format::F16, false, false, false, 128, 128, 128).len != 0) throw vtest::Failure("128x128x128 f16 is split");
  bool serial = false;
  for (int k : {128, 256, 512, 768, 1024}) serial |= plan_of(SplitTable::A10g, Format::F16, false, false, false, 64, 64, k).serial;
  if (!serial) throw vtest::Failure("no two-slice f16 row of the A10G carries the serial epilogue");
}

VTEST(product_with_slices_is_the_slices_chains_added_in_order) {
  const Model md = model_of(0);
  const int m = 40, n = 24, k = 300, slice = 96;
  const auto A = random_matrix(static_cast<size_t>(m) * k, 1), B = random_matrix(static_cast<size_t>(k) * n, 2);
  std::vector<float> whole, got, p0, p1, p2, plain;
  product(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, got, slice);
  // By hand: the chain of each slice from zero (A's columns k0.., B's rows k0..), then the f32 sum in order.
  product(md, Format::F16, false, false, m, n, 96, A.data(), m, B.data(), k, p0, 0);
  product(md, Format::F16, false, false, m, n, 96, A.data() + static_cast<size_t>(96) * m, m, B.data() + 96, k, p1, 0);
  product(md, Format::F16, false, false, m, n, 108, A.data() + static_cast<size_t>(192) * m, m, B.data() + 192, k, p2, 0);
  for (size_t i = 0; i < got.size(); ++i) {
    const float want = (p0[i] + p1[i]) + p2[i];
    if (bits(got[i]) != bits(want)) throw vtest::Failure("element " + std::to_string(i) + " is not the slices' sum");
  }
  // Transposed operands cut the same slices.
  std::vector<float> At(A.size()), Bt(B.size()), got_t;
  for (int i = 0; i < m; ++i) for (int p = 0; p < k; ++p) At[static_cast<size_t>(p) + static_cast<size_t>(i) * k] = A[static_cast<size_t>(i) + static_cast<size_t>(p) * m];
  for (int p = 0; p < k; ++p) for (int j = 0; j < n; ++j) Bt[static_cast<size_t>(j) + static_cast<size_t>(p) * n] = B[static_cast<size_t>(p) + static_cast<size_t>(j) * k];
  product(md, Format::F16, true, true, m, n, k, At.data(), k, Bt.data(), n, got_t, slice);
  if (std::memcmp(got.data(), got_t.data(), got.size() * 4)) throw vtest::Failure("TT with slices differs from NN with slices");
  // No slice, or a slice as long as K: the plain chain, and a real cut is not.
  product(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, plain, 0);
  product(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, whole, k);
  if (std::memcmp(plain.data(), whole.data(), plain.size() * 4)) throw vtest::Failure("slice = k is not the plain chain");
  if (!std::memcmp(plain.data(), got.data(), plain.size() * 4)) throw vtest::Failure("the cut product equals the plain chain: the inputs do not show the cut");
}

VTEST(epilogue_forms) {
  const Model md = model_of(0);
  const int m = 16, n = 8, k = 192;
  const auto A = random_matrix(static_cast<size_t>(m) * k, 3), B = random_matrix(static_cast<size_t>(k) * n, 4), C = random_matrix(static_cast<size_t>(m) * n, 5);
  const float alpha = 1.3f, beta = 0.7f;
  std::vector<float> R, P, p0, p1;
  // One chain: fma(alpha, acc, beta * c).
  product(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, P, 0);
  product_epilogue(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, 0, false, alpha, beta, C.data(), R);
  size_t differ_unfused = 0;
  for (size_t i = 0; i < R.size(); ++i) {
    if (bits(R[i]) != bits(std::fma(alpha, P[i], rmul(beta, C[i])))) throw vtest::Failure("the chain's epilogue is not fma(alpha, acc, beta * c)");
    differ_unfused += bits(R[i]) != bits(rmul(alpha, P[i]) + rmul(beta, C[i]));
  }
  if (differ_unfused == 0) throw vtest::Failure("the inputs do not tell the fused epilogue from the unfused one");
  // beta = 0: alpha * acc (C is not read).
  product_epilogue(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, 0, false, alpha, 0.0f, nullptr, R);
  for (size_t i = 0; i < R.size(); ++i)
    if (bits(R[i]) != bits(rmul(alpha, P[i]))) throw vtest::Failure("beta = 0 is not alpha * acc");
  // Three slices (k = 192, slices of 64): fma(beta, c, alpha * sum of the slices).
  product(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, P, 64);
  product_epilogue(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, 64, false, alpha, beta, C.data(), R);
  for (size_t i = 0; i < R.size(); ++i)
    if (bits(R[i]) != bits(std::fma(beta, C[i], rmul(alpha, P[i])))) throw vtest::Failure("three slices: not fma(beta, c, alpha * sum)");
  // Two slices (96 + 96), the serial form: fma(beta, c, alpha * P0), then + alpha * P1 (f16); fma(alpha, P0, beta * c), then fma(alpha, P1, .) (tf32);
  // and without `serial` the sum form.
  product(md, Format::F16, false, false, m, n, 96, A.data(), m, B.data(), k, p0, 0);
  product(md, Format::F16, false, false, m, n, 96, A.data() + static_cast<size_t>(96) * m, m, B.data() + 96, k, p1, 0);
  product_epilogue(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, 96, true, alpha, beta, C.data(), R);
  for (size_t i = 0; i < R.size(); ++i) {
    const float want = std::fma(beta, C[i], rmul(alpha, p0[i])) + rmul(alpha, p1[i]);
    if (bits(R[i]) != bits(want)) throw vtest::Failure("two slices, serial f16: not the serial form");
  }
  product_epilogue(md, Format::Tf32, false, false, m, n, k, A.data(), m, B.data(), k, 96, true, alpha, beta, C.data(), R);
  std::vector<float> t0, t1;
  product(md, Format::Tf32, false, false, m, n, 96, A.data(), m, B.data(), k, t0, 0);
  product(md, Format::Tf32, false, false, m, n, 96, A.data() + static_cast<size_t>(96) * m, m, B.data() + 96, k, t1, 0);
  for (size_t i = 0; i < R.size(); ++i) {
    const float want = std::fma(alpha, t1[i], std::fma(alpha, t0[i], rmul(beta, C[i])));
    if (bits(R[i]) != bits(want)) throw vtest::Failure("two slices, serial tf32: not the serial form");
  }
  product(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, P, 96);
  product_epilogue(md, Format::F16, false, false, m, n, k, A.data(), m, B.data(), k, 96, false, alpha, beta, C.data(), R);
  for (size_t i = 0; i < R.size(); ++i)
    if (bits(R[i]) != bits(std::fma(beta, C[i], rmul(alpha, P[i])))) throw vtest::Failure("two slices, not serial: not the sum form");
}

VTEST(profiles_name_their_split_table) {
  using vgpu::CublasSplitK;
  struct Want { const char* id; CublasSplitK table; vgpu::Provenance cls; };
  const Want wants[] = {{"nvidia/a10g", CublasSplitK::A10g, vgpu::Provenance::Measured},
                        {"nvidia/t4", CublasSplitK::T4, vgpu::Provenance::Measured},
                        {"nvidia/rtx3080ti", CublasSplitK::A10g, vgpu::Provenance::Inferred}};
  for (const Want& w : wants) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(w.id);
    if (p.numerics.cublas_splitk != w.table) throw vtest::Failure(std::string(w.id) + ": numerics.cublas_splitk is not the table that was measured for it");
    const auto cls = p.provenance_of("numerics.cublas_splitk");
    if (!cls || *cls != w.cls) throw vtest::Failure(std::string(w.id) + ": numerics.cublas_splitk has the wrong class");
  }
  // A table belongs to a (compute capability, multiprocessor count) pair: no other profile carries one, and the pairs are the tables'.
  for (const std::string& id : vgpu::available_gpus()) {
    const vgpu::DeviceProfile p = vgpu::load_gpu(id);
    if (p.numerics.cublas_splitk == CublasSplitK::None) continue;
    const bool t4 = p.numerics.cublas_splitk == CublasSplitK::T4;
    if (p.cc_major != (t4 ? 7 : 8) || p.cc_minor != (t4 ? 5 : 6) || p.multiprocessors != (t4 ? 40u : 80u))
      throw vtest::Failure(id + ": carries a split-K table of a different compute capability or multiprocessor count");
  }
}

VTEST_MAIN
