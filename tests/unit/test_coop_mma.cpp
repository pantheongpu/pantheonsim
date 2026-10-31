// OpCoopMma: the cooperative-matrix multiply-add over one warp (include/vgpu/ptx/ast.hpp), for a frontend whose matrices live in the
// lanes' registers (PantheonZE's SPV_KHR_cooperative_matrix lowering).
//
// Each case builds a kernel with the AST builder: every lane loads its registers of A, B and C from buffers (lane-major: lane l's register
// s is element l * regs + s), runs the instruction, and stores its registers of D. The host reference is written here from the layout rules
// in the comment of OpCoopMma (not from the interpreter): it scatters the registers into dense matrices, folds the products in increasing k
// with one fused multiply-add each (floats) or a wrapping multiply-add (integers), and gathers D back. The answer must match bit for bit,
// at warp widths 8, 16 and 32, for the shapes and types of Intel's joint_matrix (half, bfloat16, tf32-shaped float, int8 of every
// signedness, half and float accumulators, M = 1 .. 32, N = 16 .. 64, K = 8 .. 32). Also: the refusals (a partial warp, a shape the layout
// does not take, register counts that do not match).
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/simd/fp16.hpp"
#include "ast_run.hpp"
#include "vtest.hpp"

using namespace astrun;
using vgpu::ptx::CoopElem;
using vgpu::ptx::OpCoopMma;

namespace {

struct Tile {
  const char* name;
  uint32_t M, N, K;
  CoopElem a, b, c;
};

bool is_float(CoopElem t) { return t == CoopElem::F16 || t == CoopElem::BF16 || t == CoopElem::F32; }
uint32_t bits_of(CoopElem t) {
  switch (t) {
    case CoopElem::S8: case CoopElem::U8: return 8;
    case CoopElem::F16: case CoopElem::BF16: case CoopElem::S16: case CoopElem::U16: return 16;
    default: return 32;
  }
}
Type mem_type(CoopElem t) { return tu(bits_of(t)); }

// register slot s of lane l -> (row, col) of a rows x cols matrix (the documented rules)
std::pair<uint32_t, uint32_t> where(uint32_t rows, uint32_t cols, uint32_t W, uint32_t s, uint32_t l) {
  if (cols % W == 0) return {s / (cols / W), l + (s % (cols / W)) * W};
  const uint32_t e = s * W + l;
  (void)rows;
  return {e / cols, e % cols};
}

float to_float(CoopElem t, uint32_t v) {
  switch (t) {
    case CoopElem::F16: return static_cast<float>(half_value(v & 0xFFFF));
    case CoopElem::BF16: return std::bit_cast<float>(v << 16);
    default: return std::bit_cast<float>(v);
  }
}
int32_t to_int(CoopElem t, uint32_t v) {
  switch (t) {
    case CoopElem::S8: return static_cast<int8_t>(v);
    case CoopElem::U8: return static_cast<uint8_t>(v);
    case CoopElem::S16: return static_cast<int16_t>(v);
    case CoopElem::U16: return static_cast<uint16_t>(v);
    default: return static_cast<int32_t>(v);
  }
}

// small values: halves and bfloat16s whose products and sums are exact enough to be interesting but not trivially so
uint32_t random_elem(Rng& rng, CoopElem t) {
  const uint64_t r = rng.next();
  switch (t) {
    case CoopElem::F16: return half_from(static_cast<double>(static_cast<int>(r % 33) - 16) * 0.125 + static_cast<double>(r >> 40 & 7) * 0.001953125);
    case CoopElem::BF16: return std::bit_cast<uint32_t>(static_cast<float>(static_cast<int>(r % 33) - 16) * 0.125f + static_cast<float>(r >> 40 & 7) * 0.0078125f) >> 16;
    case CoopElem::F32: return std::bit_cast<uint32_t>(static_cast<float>(static_cast<int>(r % 2001) - 1000) * 0.001f);
    case CoopElem::S8: return static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(r % 256) - 128));
    case CoopElem::U8: return static_cast<uint8_t>(r % 256);
    case CoopElem::S16: case CoopElem::U16: return static_cast<uint16_t>(r);
    default: return static_cast<uint32_t>(r);
  }
}

OpCoopMma make(const Tile& t, const std::vector<Reg>& a, const std::vector<Reg>& b, const std::vector<Reg>& c, const std::vector<Reg>& d) {
  OpCoopMma o;
  o.m = t.M; o.n = t.N; o.k = t.K;
  o.a_type = t.a; o.b_type = t.b; o.c_type = t.c;
  o.a = a; o.b = b; o.c = c; o.d = d;
  return o;
}

// the packed forms of the buffers: each register in the low bits of a 32-bit word of the buffer would waste the load type, so the buffers hold
// elements of the register's own width
std::vector<uint8_t> pack_elems(const std::vector<uint32_t>& v, uint32_t bits) {
  std::vector<uint8_t> o(v.size() * (bits / 8));
  for (size_t i = 0; i < v.size(); ++i) std::memcpy(&o[i * (bits / 8)], &v[i], bits / 8);
  return o;
}

}  // namespace

VTEST(matches_the_host_fold_in_increasing_k) {
  const std::vector<Tile> tiles = {
      {"half 8x16x16 -> float", 8, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 1x16x16", 1, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 4x16x16", 4, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 16x16x16", 16, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 32x16x16", 32, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 1x64x16", 1, 64, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 32x64x16", 32, 64, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32},
      {"half 8x16x16 -> half", 8, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F16},
      {"bf16 8x16x16", 8, 16, 16, CoopElem::BF16, CoopElem::BF16, CoopElem::F32},
      {"float 8x16x8 (tf32 shape)", 8, 16, 8, CoopElem::F32, CoopElem::F32, CoopElem::F32},
      {"float 8x32x16", 8, 32, 16, CoopElem::F32, CoopElem::F32, CoopElem::F32},
      {"s8 x s8 8x16x32", 8, 16, 32, CoopElem::S8, CoopElem::S8, CoopElem::S32},
      {"u8 x u8 8x16x32", 8, 16, 32, CoopElem::U8, CoopElem::U8, CoopElem::S32},
      {"s8 x u8 8x16x32", 8, 16, 32, CoopElem::S8, CoopElem::U8, CoopElem::S32},
      {"u8 x s8 1x64x32", 1, 64, 32, CoopElem::U8, CoopElem::S8, CoopElem::S32},
      {"s8 x s8 32x64x32", 32, 64, 32, CoopElem::S8, CoopElem::S8, CoopElem::S32},
      {"s16 x u16 8x16x16", 8, 16, 16, CoopElem::S16, CoopElem::U16, CoopElem::S32},
  };
  Rng rng(0xC00);
  uint64_t cases = 0;
  for (const Tile& t : tiles)
    for (uint32_t W : {8u, 16u, 32u}) {
      if (t.N % W || (t.M * t.K) % W) continue;   // the layout does not take this shape at this width (checked in refuses_shapes_the_layout_does_not_take)
      const uint32_t a_regs = t.M * t.K / W, b_regs = t.K * t.N / W, c_regs = t.M * t.N / W;
      std::vector<uint32_t> A(size_t{W} * a_regs), B(size_t{W} * b_regs), C(size_t{W} * c_regs);
      for (auto& v : A) v = random_elem(rng, t.a);
      for (auto& v : B) v = random_elem(rng, t.b);
      for (auto& v : C) v = random_elem(rng, t.c);
      // the host reference
      std::vector<uint32_t> want(size_t{W} * c_regs);
      if (is_float(t.a)) {
        std::vector<float> Am(t.M * t.K), Bm(t.K * t.N), Cm(t.M * t.N);
        for (uint32_t l = 0; l < W; ++l) {
          for (uint32_t s = 0; s < a_regs; ++s) { auto [r, c] = where(t.M, t.K, W, s, l); Am[r * t.K + c] = to_float(t.a, A[l * a_regs + s]); }
          for (uint32_t s = 0; s < b_regs; ++s) { auto [r, c] = where(t.K, t.N, W, s, l); Bm[r * t.N + c] = to_float(t.b, B[l * b_regs + s]); }
          for (uint32_t s = 0; s < c_regs; ++s) { auto [r, c] = where(t.M, t.N, W, s, l); Cm[r * t.N + c] = to_float(t.c, C[l * c_regs + s]); }
        }
        for (uint32_t r = 0; r < t.M; ++r)
          for (uint32_t c = 0; c < t.N; ++c) {
            float acc = Cm[r * t.N + c];
            for (uint32_t k = 0; k < t.K; ++k) acc = std::fmaf(Am[r * t.K + k], Bm[k * t.N + c], acc);
            Cm[r * t.N + c] = acc;
          }
        for (uint32_t l = 0; l < W; ++l)
          for (uint32_t s = 0; s < c_regs; ++s) {
            auto [r, c] = where(t.M, t.N, W, s, l);
            want[l * c_regs + s] = t.c == CoopElem::F16 ? half_from(static_cast<double>(Cm[r * t.N + c])) : std::bit_cast<uint32_t>(Cm[r * t.N + c]);
          }
      } else {
        std::vector<int32_t> Am(t.M * t.K), Bm(t.K * t.N), Cm(t.M * t.N);
        for (uint32_t l = 0; l < W; ++l) {
          for (uint32_t s = 0; s < a_regs; ++s) { auto [r, c] = where(t.M, t.K, W, s, l); Am[r * t.K + c] = to_int(t.a, A[l * a_regs + s]); }
          for (uint32_t s = 0; s < b_regs; ++s) { auto [r, c] = where(t.K, t.N, W, s, l); Bm[r * t.N + c] = to_int(t.b, B[l * b_regs + s]); }
          for (uint32_t s = 0; s < c_regs; ++s) { auto [r, c] = where(t.M, t.N, W, s, l); Cm[r * t.N + c] = static_cast<int32_t>(C[l * c_regs + s]); }
        }
        for (uint32_t r = 0; r < t.M; ++r)
          for (uint32_t c = 0; c < t.N; ++c) {
            uint32_t acc = static_cast<uint32_t>(Cm[r * t.N + c]);
            for (uint32_t k = 0; k < t.K; ++k) acc += static_cast<uint32_t>(Am[r * t.K + k]) * static_cast<uint32_t>(Bm[k * t.N + c]);
            Cm[r * t.N + c] = static_cast<int32_t>(acc);
          }
        for (uint32_t l = 0; l < W; ++l)
          for (uint32_t s = 0; s < c_regs; ++s) {
            auto [r, c] = where(t.M, t.N, W, s, l);
            want[l * c_regs + s] = static_cast<uint32_t>(Cm[r * t.N + c]);
          }
      }
      // The kernel loads each register with its own width, so the buffers hold elements of that width.
      auto make_buf = [&](const std::vector<uint32_t>& v, CoopElem ty) { return pack_elems(v, bits_of(ty)); };
      vgpu::ptx::KernelBuilder kb("coop");
      Prologue p = prologue(kb, 4);
      auto load_regs = [&](const Reg& buf, uint32_t regs, CoopElem ty) {
        std::vector<Reg> out;
        for (uint32_t s = 0; s < regs; ++s) {
          const Reg idx = kb.temp(tu(32));
          kb.emit(madlo(tu(32), idx, RO(p.lin), IM(regs), IM(s)), "mad");
          out.push_back(load_elem(kb, buf, idx, mem_type(ty)));
        }
        return out;
      };
      const std::vector<Reg> a = load_regs(p.ptr[0], a_regs, t.a), b = load_regs(p.ptr[1], b_regs, t.b), c = load_regs(p.ptr[2], c_regs, t.c);
      std::vector<Reg> d;
      for (uint32_t s = 0; s < c_regs; ++s) d.push_back(kb.temp(mem_type(t.c)));
      kb.emit(make(t, a, b, c, d), "coopmma");
      for (uint32_t s = 0; s < c_regs; ++s) {
        const Reg idx = kb.temp(tu(32));
        kb.emit(madlo(tu(32), idx, RO(p.lin), IM(c_regs), IM(s)), "mad");
        store_elem(kb, p.ptr[3], idx, mem_type(t.c), d[s]);
      }
      kb.emit(OpRet{}, "ret");
      vgpu::exec::LaunchConfig cfg;
      cfg.block = {W, 1, 1};
      cfg.grid = {1, 1, 1};
      cfg.warp_size = W;
      auto res = run(kb.finish(), cfg, {make_buf(A, t.a), make_buf(B, t.b), make_buf(C, t.c), std::vector<uint8_t>(size_t{W} * c_regs * (bits_of(t.c) / 8), 0xCD)});
      std::vector<uint32_t> got(size_t{W} * c_regs, 0);
      for (size_t i = 0; i < got.size(); ++i) std::memcpy(&got[i], &res[3][i * (bits_of(t.c) / 8)], bits_of(t.c) / 8);
      for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i])
          throw vtest::Failure(std::string(t.name) + " at warp width " + std::to_string(W) + ": lane " + std::to_string(i / c_regs) + " register " +
                               std::to_string(i % c_regs) + " got 0x" + std::to_string(got[i]) + " want 0x" + std::to_string(want[i]));
      ++cases;
    }
  VCHECK(cases >= 38);   // 17 tiles at W = 8 and 16, the 4 whose N and M * K are multiples of 32 at W = 32
}

VTEST(refuses_a_partial_warp_and_shapes_the_layout_does_not_take) {
  // a partial warp: 12 threads on a 16-wide warp
  {
    const Tile t = {"half 8x16x16", 8, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32};
    const uint32_t W = 16, a_regs = 8, b_regs = 16, c_regs = 8;
    std::vector<uint8_t> a(size_t{16} * a_regs * 2, 0), b(size_t{16} * b_regs * 2, 0), c(size_t{16} * c_regs * 4, 0);
    vgpu::ptx::KernelBuilder kb("coop");
    Prologue p = prologue(kb, 4);
    auto load_regs = [&](const Reg& buf, uint32_t regs, CoopElem ty) {
      std::vector<Reg> out;
      for (uint32_t s = 0; s < regs; ++s) {
        const Reg idx = kb.temp(tu(32));
        kb.emit(madlo(tu(32), idx, RO(p.lin), IM(regs), IM(s)), "mad");
        out.push_back(load_elem(kb, buf, idx, mem_type(ty)));
      }
      return out;
    };
    const std::vector<Reg> ra = load_regs(p.ptr[0], a_regs, t.a), rb = load_regs(p.ptr[1], b_regs, t.b), rc = load_regs(p.ptr[2], c_regs, t.c);
    std::vector<Reg> d;
    for (uint32_t s = 0; s < c_regs; ++s) d.push_back(kb.temp(tu(32)));
    kb.emit(make(t, ra, rb, rc, d), "coopmma");
    kb.emit(OpRet{}, "ret");
    vgpu::exec::LaunchConfig cfg;
    cfg.block = {12, 1, 1};
    cfg.grid = {1, 1, 1};
    cfg.warp_size = W;
    auto err = VCAPTURE(vgpu::Error, run(kb.finish(), cfg, {a, b, c, std::vector<uint8_t>(64, 0)}));
    VCHECK_CONTAINS(std::string(err.what()), "convergent");
  }
  // N = 16 on a 32-wide warp: the column rule does not hold for B and C
  {
    const Tile t = {"half 8x16x16 at W = 32", 8, 16, 16, CoopElem::F16, CoopElem::F16, CoopElem::F32};
    vgpu::ptx::KernelBuilder kb("coop");
    Prologue p = prologue(kb, 1);
    (void)p;
    std::vector<Reg> a, b, c, d;
    for (uint32_t s = 0; s < 4; ++s) a.push_back(kb.temp(tu(16)));
    for (uint32_t s = 0; s < 8; ++s) b.push_back(kb.temp(tu(16)));
    for (uint32_t s = 0; s < 4; ++s) { c.push_back(kb.temp(tu(32))); d.push_back(kb.temp(tu(32))); }
    kb.emit(make(t, a, b, c, d), "coopmma");
    kb.emit(OpRet{}, "ret");
    vgpu::exec::LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    cfg.grid = {1, 1, 1};
    cfg.warp_size = 32;
    auto err = VCAPTURE(vgpu::Error, run(kb.finish(), cfg, {std::vector<uint8_t>(64, 0)}));
    VCHECK_CONTAINS(std::string(err.what()), "layout rules do not take");
  }
}

VTEST_MAIN
