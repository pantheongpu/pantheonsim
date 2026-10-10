// How sm_120's kind::f8f6f4 / mxf8f6f4 mma.sync accumulates, asked of the hardware cell by cell.
//
// nvidia/tests/data/lowprec/ptx120.rtx-pro-6000-server.txt holds only hashes. Eighteen of its 27 dense forms
// agree to the bit with "the exact sum of the products and C, rounded toward zero to fp32" (the rule
// exec_mma in src/exec/interpreter.cpp and QMMA in src/sass/exec_ops.inc follow); the nine "wide" ones -- an
// E4M3 or E5M2 operand against an operand that spans many binades, with NaNs or infinities among the
// operands -- do not, and the hashes cannot say in which cells. This program is the instrument that settles
// it. It runs controlled inputs through the instruction and prints, for each probed cell, the result next to
// the exact value rounded toward zero and to nearest:
//
//   gap   2^16 and a second term 2^(16 - gap) (gap 0..34): how many bits below the largest term survive, with
//         the second term positive, negative, twice, in C, across a cancellation, and with C as the big term;
//   pos   2^16 and n products of 2^-8 (half an ulp of 2^16 per two): the smalls next to the big product, at
//         the far end of K, or before a big product at the last K: whether they are added one at a time (and
//         lost), together (and kept), or in groups of K;
//   cin   C = 2^16 against one small product and the reverse;
//   spec  NaN, infinity and zero operands against each other, each cell adding 1.0 (a second product): which
//         NaN bits come out, whether NaN times zero is NaN, infinity minus infinity, signs of zeros; and the
//         same cells with the second product zero (specalone);
//   zero  exact cancellation, 0 x finite and negative zero in C: the sign of a zero result.
//
// Built for sm_120a and run on the RTX PRO 6000 (nvidia/tools/card-session-remote.sh does), the output is the
// card's answer; on the simulator it is the simulator's, and a line that differs is a cell the rule gets
// wrong. Every line is "<form> <experiment> <label> card=<hex> rz=<hex> rn=<hex>" and ends "  =rz" (the
// result is the exact value rounded toward zero), "  =rn" or "  =neither"; NaN and infinity cells compare
// against NaN (0x7fffffff) and the signed infinity.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- the instruction, one kernel per form ----
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#define MMA_PLAIN(INSTR)                                                                                         \
  asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"                                \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                                                          \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#define MMA_MX(INSTR)                                                                                            \
  asm volatile(INSTR " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, {0, 0}, %15, {0, 0};"      \
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)                                                          \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),      \
                 "r"(0x7f7f7f7fu), "r"(0x7f7f7f7fu));
#else
#define MMA_PLAIN(INSTR)
#define MMA_MX(INSTR)
#endif
#define MMA_KERNEL(NAME, BODY)                                                                                   \
  __global__ void k_##NAME(const unsigned* in, unsigned* out) {                                                 \
    const unsigned l = threadIdx.x;                                                                              \
    const unsigned a0 = in[l], a1 = in[l + 32], a2 = in[l + 64], a3 = in[l + 96], b0 = in[l + 128],            \
                   b1 = in[l + 160];                                                                             \
    const float c0 = __uint_as_float(in[l + 192]), c1 = __uint_as_float(in[l + 224]),                           \
                c2 = __uint_as_float(in[l + 256]), c3 = __uint_as_float(in[l + 288]);                            \
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;                                                                        \
    BODY                                                                                                         \
    out[4 * l] = __float_as_uint(d0);                                                                            \
    out[4 * l + 1] = __float_as_uint(d1);                                                                        \
    out[4 * l + 2] = __float_as_uint(d2);                                                                        \
    out[4 * l + 3] = __float_as_uint(d3);                                                                        \
  }

MMA_KERNEL(e4m3_e4m3, MMA_PLAIN("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32"))
MMA_KERNEL(e5m2_e4m3, MMA_PLAIN("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e4m3.f32"))
MMA_KERNEL(e3m2_e4m3, MMA_PLAIN("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e4m3.f32"))
MMA_KERNEL(e5m2_e2m1, MMA_PLAIN("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e5m2.e2m1.f32"))
MMA_KERNEL(mx_e4m3_e4m3, MMA_MX("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0"))

// ---- element formats: the values the simulator's decoder gives them (E2M1 in f8f6f4 is its 6-bit container read as E2M3) ----
enum Fmt { E4M3, E5M2, E3M2, E2M3, E2M1 };
double decode(Fmt f, unsigned c) {
  switch (f) {
    case E4M3: {
      if ((c & 0x7f) == 0x7f) return NAN;
      const int e = (c >> 3) & 15, m = c & 7;
      const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
      return c & 0x80 ? -v : v;
    }
    case E5M2: {
      const int e = (c >> 2) & 31, m = c & 3;
      if (e == 31) return m ? NAN : (c & 0x80 ? -INFINITY : INFINITY);
      const double v = e ? std::ldexp(1.0 + m / 4.0, e - 15) : std::ldexp(m / 4.0, -14);
      return c & 0x80 ? -v : v;
    }
    case E3M2: {
      c &= 0x3f;
      const int e = (c >> 2) & 7, m = c & 3;
      const double v = e ? std::ldexp(1.0 + m / 4.0, e - 3) : std::ldexp(m / 4.0, -2);
      return c & 0x20 ? -v : v;
    }
    case E2M3: case E2M1: {
      c &= 0x3f;
      const int e = (c >> 3) & 3, m = c & 7;
      const double v = e ? std::ldexp(1.0 + m / 8.0, e - 1) : std::ldexp(m / 8.0, 0);
      return c & 0x20 ? -v : v;
    }
  }
  return 0;
}
// The code of an exactly representable value, -1 when the format has none.
int encode(Fmt f, double v) {
  const unsigned n = (f == E4M3 || f == E5M2) ? 256 : 64;
  for (unsigned c = 0; c < n; ++c) {
    const double d = decode(f, c);
    if (d == v && std::signbit(d) == std::signbit(v)) return (int)c;
  }
  return -1;
}

// ---- one 16x8x32 problem ----
struct Prob {
  uint8_t A[16][32] = {}, B[32][8] = {};
  float C[16][8] = {};
  bool ok = true;   // false once a value could not be encoded: the experiment is skipped
};
struct Cell {
  int r, c;
  std::string label;
};
unsigned bits(float f) { unsigned u; std::memcpy(&u, &f, 4); return u; }

struct Form {
  const char* name;
  void (*kernel)(const unsigned*, unsigned*);
  Fmt fa, fb;
};
const Form kForms[] = {
    {"e4m3_e4m3", k_e4m3_e4m3, E4M3, E4M3},       {"e5m2_e4m3", k_e5m2_e4m3, E5M2, E4M3},
    {"e3m2_e4m3", k_e3m2_e4m3, E3M2, E4M3},       {"e5m2_e2m1", k_e5m2_e2m1, E5M2, E2M1},
    {"mx_e4m3_e4m3", k_mx_e4m3_e4m3, E4M3, E4M3},
};

void put_a(Prob& p, const Form& f, int r, int k, double v) {
  const int c = encode(f.fa, v);
  if (c < 0) p.ok = false; else p.A[r][k] = (uint8_t)c;
}
void put_b(Prob& p, const Form& f, int k, int col, double v) {
  const int c = encode(f.fb, v);
  if (c < 0) p.ok = false; else p.B[k][col] = (uint8_t)c;
}

// The sum a cell asks for, exactly (a 64-bit significand holds every sum the experiments make), or a special.
long double exact_cell(const Prob& p, const Form& f, int r, int c, bool* special, double* special_value) {
  long double s = p.C[r][c];
  *special = false;
  bool nan = std::isnan(p.C[r][c]), pinf = false, ninf = false;
  for (int k = 0; k < 32; ++k) {
    const double a = decode(f.fa, p.A[r][k]), b = decode(f.fb, p.B[k][c]);
    if (std::isnan(a) || std::isnan(b)) { nan = true; continue; }
    if (std::isinf(a) || std::isinf(b)) {
      if (a == 0 || b == 0) { nan = true; continue; }
      (std::signbit(a) != std::signbit(b) ? ninf : pinf) = true;
      continue;
    }
    s += (long double)a * b;
  }
  if (nan || (pinf && ninf)) { *special = true; *special_value = NAN; }
  else if (pinf || ninf) { *special = true; *special_value = pinf ? INFINITY : -INFINITY; }
  return s;
}
float rz32(long double x) {
  float f = (float)x;
  if (std::fabs((long double)f) > std::fabs(x)) f = std::nextafterf(f, 0.f);
  return f;
}

unsigned* g_in = nullptr;
unsigned* g_out = nullptr;
int g_cells = 0, g_differ = 0;

// Runs the problem and prints the probed cells.
void run(const Form& f, const char* exp, const Prob& p, const std::vector<Cell>& cells) {
  if (!p.ok) { std::printf("%s %s SKIP a value this form cannot hold\n", f.name, exp); return; }
  unsigned in[10 * 32];
  for (int l = 0; l < 32; ++l) {
    const int g = l / 4, t = l % 4;
    auto pack_a = [&](int row, int k0) {
      unsigned w = 0;
      for (int e = 0; e < 4; ++e) w |= (unsigned)p.A[row][k0 + e] << (8 * e);
      return w;
    };
    auto pack_b = [&](int k0) {
      unsigned w = 0;
      for (int e = 0; e < 4; ++e) w |= (unsigned)p.B[k0 + e][g] << (8 * e);
      return w;
    };
    in[l] = pack_a(g, 4 * t);
    in[l + 32] = pack_a(g + 8, 4 * t);
    in[l + 64] = pack_a(g, 16 + 4 * t);
    in[l + 96] = pack_a(g + 8, 16 + 4 * t);
    in[l + 128] = pack_b(4 * t);
    in[l + 160] = pack_b(16 + 4 * t);
    in[l + 192] = bits(p.C[g][2 * t]);
    in[l + 224] = bits(p.C[g][2 * t + 1]);
    in[l + 256] = bits(p.C[g + 8][2 * t]);
    in[l + 288] = bits(p.C[g + 8][2 * t + 1]);
  }
  cudaMemcpy(g_in, in, sizeof in, cudaMemcpyHostToDevice);
  f.kernel<<<1, 32>>>(g_in, g_out);
  unsigned out[128];
  cudaMemcpy(out, g_out, sizeof out, cudaMemcpyDeviceToHost);
  unsigned D[16][8];
  for (int l = 0; l < 32; ++l) {
    const int g = l / 4, t = l % 4;
    D[g][2 * t] = out[4 * l];
    D[g][2 * t + 1] = out[4 * l + 1];
    D[g + 8][2 * t] = out[4 * l + 2];
    D[g + 8][2 * t + 1] = out[4 * l + 3];
  }
  for (const Cell& c : cells) {
    bool special;
    double sv;
    const long double x = exact_cell(p, f, c.r, c.c, &special, &sv);
    const unsigned card = D[c.r][c.c];
    unsigned rz, rn;
    if (special) rz = rn = std::isnan(sv) ? 0x7fffffffu : sv > 0 ? 0x7f800000u : 0xff800000u;
    else { rz = bits(rz32(x)); rn = bits((float)x); }
    std::printf("%s %s %s card=%08x rz=%08x rn=%08x%s\n", f.name, exp, c.label.c_str(), card, rz, rn,
                card == rz ? (card == rn ? "" : "  =rz") : card == rn ? "  =rn" : "  =neither");
    ++g_cells;
    g_differ += card != rz;
  }
}

std::string num(int v) { return std::to_string(v); }

void experiments(const Form& f) {
  const bool wide = f.fb == E4M3 && (f.fa == E4M3 || f.fa == E5M2);   // 2^8 is in both operands' ranges
  const double big_a = std::ldexp(1.0, 8), big_b = std::ldexp(1.0, 8);   // 2^16
  // gap: row r holds a small A operand 2^a (a from a0 down), the B operand 2^b of the small product fixed per experiment.
  struct Gap { const char* tag; int b, a0; };
  if (wide)
    for (const Gap& g : {Gap{"b8", 8, 8}, Gap{"b0", 0, 0}, Gap{"bm9", -9, -1}}) {
      Prob p;
      std::vector<Cell> cells;
      for (int r = 0; r < 16; ++r) {
        const int a = g.a0 - r;
        if (encode(f.fa, std::ldexp(1.0, a)) < 0) continue;   // below this format's smallest value: no row
        put_a(p, f, r, 0, big_a);
        put_a(p, f, r, 1, std::ldexp(1.0, a));
        put_a(p, f, r, 2, std::ldexp(1.0, a));
        put_a(p, f, r, 3, big_a);
        const int e = a + g.b;
        p.C[r][3] = std::ldexp(1.0f, std::max(-149, std::min(127, e)));
        p.C[r][6] = -65536.0f;
        p.C[r][7] = 65536.0f;
        for (int c = 0; c < 8; ++c) cells.push_back({r, c, std::string("gap") + num(16 - e) + "_c" + num(c)});
      }
      const double sb = std::ldexp(1.0, g.b);
      put_b(p, f, 0, 0, big_b); put_b(p, f, 1, 0, sb);                                              // c0 big + small
      put_b(p, f, 0, 1, big_b); put_b(p, f, 1, 1, -sb);                                             // c1 big - small
      put_b(p, f, 0, 2, big_b); put_b(p, f, 1, 2, sb); put_b(p, f, 2, 2, sb);                       // c2 big + 2 small
      put_b(p, f, 0, 3, big_b);                                                                     // c3 big + C small
      put_b(p, f, 0, 4, -big_b); put_b(p, f, 1, 4, -sb);                                            // c4 -(big + small)
      put_b(p, f, 0, 5, big_b); put_b(p, f, 1, 5, sb); put_b(p, f, 3, 5, -big_b);                   // c5 big + small - big
      put_b(p, f, 0, 6, big_b); put_b(p, f, 1, 6, sb);                                              // c6 big + small + C = -big
      put_b(p, f, 1, 7, sb);                                                                        // c7 small + C = +big
      run(f, (std::string("gap_") + g.tag).c_str(), p, cells);
    }
  // pos: n = r + 1 smalls of 2^-8 (A and B 2^-4) beside one 2^16.
  if (wide) {
    struct Pos { const char* tag; int big_k; int first; int dir; };   // smalls at first, first+dir, ... (n of them)
    for (const Pos& q : {Pos{"near", 0, 1, +1}, Pos{"far", 0, 31, -1}, Pos{"last", 31, 0, +1}}) {
      Prob p;
      std::vector<Cell> cells;
      for (int r = 0; r < 16; ++r) {
        const int n = r + 1;
        put_a(p, f, r, q.big_k, big_a);
        for (int i = 0; i < n; ++i) put_a(p, f, r, q.first + q.dir * i, std::ldexp(1.0, -4));
        cells.push_back({r, 0, std::string("n") + num(n)});
      }
      put_b(p, f, q.big_k, 0, big_b);
      for (int k = 0; k < 32; ++k)
        if (k != q.big_k) put_b(p, f, k, 0, std::ldexp(1.0, -4));
      run(f, (std::string("pos_") + q.tag).c_str(), p, cells);
    }
  }
  // cin: C = 2^16 against a small product 2^a (rows), and a big product against a small C.
  if (wide) {
    Prob p;
    std::vector<Cell> cells;
    for (int r = 0; r < 16; ++r) {
      put_a(p, f, r, 0, std::ldexp(1.0, 8 - r));   // small A: 2^(8 - r)
      p.C[r][0] = 65536.0f;
      p.C[r][1] = -65536.0f;
      p.C[r][2] = std::ldexp(1.0f, -r * 4 - 1);    // tiny C against the 2^16 product of c2
      p.C[r][3] = -std::ldexp(1.0f, -r * 4 - 1);
      for (int c = 0; c < 4; ++c) cells.push_back({r, c, std::string("r") + num(r) + "_c" + num(c)});
    }
    put_b(p, f, 0, 0, 1.0); put_b(p, f, 0, 1, 1.0); put_b(p, f, 0, 2, 0.0); put_b(p, f, 0, 3, 0.0);
    // c2, c3: a 2^16 product at k = 1
    for (int r = 0; r < 16; ++r) put_a(p, f, r, 1, big_a);
    put_b(p, f, 1, 2, big_b); put_b(p, f, 1, 3, big_b);
    run(f, "cin", p, cells);
  }
  // spec and specalone: row r's first operand against column c's, plus (spec) 1.0 x 1.0 as a second product.
  {
    std::vector<int> arows, bcols;
    if (f.fa == E5M2) arows = {0x7c, 0xfc, 0x7d, 0x7e, 0x7f, 0xfd, 0x7b, 0, 0x80, 0x3c, 0x01, 0xbc, 0xff, 0xfe, 0x7a, 0x04};
    else if (f.fa == E4M3) arows = {0x7f, 0xff, 0x7e, 0xfe, 0, 0x80, 0x38, 0x01, 0x40, 0xb8, 0x7d, 0x08, 0x78, 0xf8, 0x30, 0x77};
    else arows = {0x3f, 0x1f, 0x3e, 0, 0x20, 0x08, 0x04, 0x01, 0x28, 0x0c, 0x10, 0x18, 0x24, 0x2c, 0x14, 0x1c};
    if (f.fb == E4M3) bcols = {0x7f, 0xff, 0x7e, 0, 0x80, 0x38, 0x01, 0x40};
    else bcols = {0x3f, 0x1f, 0, 0x20, 0x08, 0x0c, 0x01, 0x04};
    for (const char* tag : {"spec", "specalone"}) {
      Prob p;
      std::vector<Cell> cells;
      for (int r = 0; r < 16; ++r) {
        p.A[r][0] = (uint8_t)arows[(size_t)r];
        put_a(p, f, r, 1, 1.0);
        for (int c = 0; c < 8; ++c) cells.push_back({r, c, std::string("a") + num(r) + "_b" + num(c)});
      }
      for (int c = 0; c < 8; ++c) {
        p.B[0][c] = (uint8_t)bcols[(size_t)c];
        if (tag[4] == 0) put_b(p, f, 1, c, 1.0);
      }
      run(f, tag, p, cells);
    }
  }
  // zero: the sign of a zero result. Column 0: B = (1, 1).
  {
    Prob p;
    std::vector<Cell> cells;
    put_b(p, f, 0, 0, 1.0); put_b(p, f, 1, 0, 1.0);
    put_a(p, f, 0, 0, 1.0); put_a(p, f, 0, 1, -1.0);                        // z0: 1 - 1
    put_a(p, f, 1, 0, 0.0); put_a(p, f, 1, 1, 0.0);                         // z1: +0 + +0, C = +0
    put_a(p, f, 2, 0, -0.0); put_a(p, f, 2, 1, 0.0);                        // z2: -0 + +0
    put_a(p, f, 3, 0, -0.0); put_a(p, f, 3, 1, -0.0); p.C[3][0] = -0.0f;    // z3: -0 - 0 + C = -0
    put_a(p, f, 4, 0, -0.0); put_a(p, f, 4, 1, -0.0);                       // z4: -0, -0 and C = +0
    put_a(p, f, 5, 0, 0.25); put_a(p, f, 5, 1, -0.25);   // z5: 2^-2 - 2^-2
    put_a(p, f, 6, 0, -1.0);                                                // z6: -1, C = +1
    p.C[6][0] = 1.0f;
    put_a(p, f, 7, 0, -1.0); put_a(p, f, 7, 1, 1.0);                        // z7: -1 + 1
    for (int r = 0; r < 8; ++r) cells.push_back({r, 0, std::string("z") + num(r)});
    run(f, "zero", p, cells);
  }
}

}  // namespace

int main() {
  int major = 0, dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (major != 12) {
    std::printf("SKIP: sm_120's block-scaled mma.sync needs a compute capability 12 GPU (or profile)\nPASS\n");
    return 0;
  }
  cudaMalloc(&g_in, 10 * 32 * 4);
  cudaMalloc(&g_out, 128 * 4);
  for (const Form& f : kForms) experiments(f);
  const cudaError_t e = cudaDeviceSynchronize();
  cudaFree(g_in);
  cudaFree(g_out);
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  std::printf("# %d cells probed, %d differ from the exact value rounded toward zero\nPASS\n", g_cells, g_differ);
  return 0;
}
