// The packed integer instructions of PTX ISA 9.2 -- add and sub with .sat, neg, min and max with .relu, set --
// on .u8x4, .s8x4, .u16x2 and .s16x2 registers, each lane on its own, checked against a host loop. They are
// sm_107f's and sm_120f's, so this is built for sm_120a (ptxas of CUDA 13.2 and later) and run on SASS and
// PTX alike. Derived from the ISA's text, not checked against a card: there is no sm_120 GPU here.
// Prints PASS on the last line.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

enum Form { AddU8, AddSatU8, AddSatS8, SubU8, SubSatU8, SubSatS8, NegS8, MinU8, MinS8, MinReluS8, MaxU8, MaxS8, MaxReluS8,
            AddU16, AddSatU16, AddSatS16, MinS16, MaxU16, SetLtS8, SetLoU8, SetGeS8, SetEqU8, SetNeU16, SetLeS16, kForms };

__global__ void run(const unsigned* a, const unsigned* b, unsigned* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const unsigned x = a[i], y = b[i];
  unsigned d[kForms];
#define F(form, text) asm("{ " text " }" : "=r"(d[form]) : "r"(x), "r"(y))
  F(AddU8, "add.u8x4 %0, %1, %2;"); F(AddSatU8, "add.sat.u8x4 %0, %1, %2;"); F(AddSatS8, "add.sat.s8x4 %0, %1, %2;");
  F(SubU8, "sub.u8x4 %0, %1, %2;"); F(SubSatU8, "sub.sat.u8x4 %0, %1, %2;"); F(SubSatS8, "sub.sat.s8x4 %0, %1, %2;");
  F(NegS8, "neg.s8x4 %0, %1;");
  F(MinU8, "min.u8x4 %0, %1, %2;"); F(MinS8, "min.s8x4 %0, %1, %2;"); F(MinReluS8, "min.relu.s8x4 %0, %1, %2;");
  F(MaxU8, "max.u8x4 %0, %1, %2;"); F(MaxS8, "max.s8x4 %0, %1, %2;"); F(MaxReluS8, "max.relu.s8x4 %0, %1, %2;");
  F(AddU16, "add.u16x2 %0, %1, %2;"); F(AddSatU16, "add.sat.u16x2 %0, %1, %2;"); F(AddSatS16, "add.sat.s16x2 %0, %1, %2;");
  F(MinS16, "min.s16x2 %0, %1, %2;"); F(MaxU16, "max.u16x2 %0, %1, %2;");
  F(SetLtS8, "set.lt.s8x4 %0, %1, %2;"); F(SetLoU8, "set.lo.u8x4 %0, %1, %2;"); F(SetGeS8, "set.ge.s8x4 %0, %1, %2;");
  F(SetEqU8, "set.eq.u8x4 %0, %1, %2;"); F(SetNeU16, "set.ne.u16x2 %0, %1, %2;"); F(SetLeS16, "set.le.s16x2 %0, %1, %2;");
#undef F
  for (int f = 0; f < kForms; ++f) out[size_t(i) * kForms + f] = d[f];
}

// The reference, a lane at a time.
static unsigned lanes(unsigned a, unsigned b, int bits, bool sgn, int kind /*0 add 1 sub 2 neg 3 min 4 max 5 lt 6 lo 7 ge 8 eq 9 ne 10 le*/,
                      bool sat, bool relu) {
  const unsigned mask = (1u << bits) - 1;
  unsigned out = 0;
  for (int k = 0; k < 32 / bits; ++k) {
    auto part = [&](unsigned v) -> long long {
      const unsigned u = (v >> (bits * k)) & mask;
      return sgn ? static_cast<long long>(static_cast<int>(u << (32 - bits)) >> (32 - bits)) : u;
    };
    const long long x = part(a), y = part(b), lo = sgn ? -(1ll << (bits - 1)) : 0, hi = sgn ? (1ll << (bits - 1)) - 1 : (1ll << bits) - 1;
    long long v = 0;
    switch (kind) {
      case 0: v = x + y; break;
      case 1: v = x - y; break;
      case 2: v = -x; break;
      case 3: v = x < y ? x : y; break;
      case 4: v = x > y ? x : y; break;
      case 5: case 6: v = (x < y) ? mask : 0; break;
      case 7: v = (x >= y) ? mask : 0; break;
      case 8: v = (x == y) ? mask : 0; break;
      case 9: v = (x != y) ? mask : 0; break;
      case 10: v = (x <= y) ? mask : 0; break;
    }
    if (sat) v = v < lo ? lo : v > hi ? hi : v;
    if (relu && v < 0) v = 0;
    out |= (static_cast<unsigned>(v) & mask) << (bits * k);
  }
  return out;
}

int main() {
  const int n = 4096;
  std::vector<unsigned> a(n), b(n), out(size_t(n) * kForms);
  unsigned s = 20261009;
  auto next = [&] { s = s * 1664525u + 1013904223u; return s; };
  for (int i = 0; i < n; ++i) {
    a[i] = next(); b[i] = next();
    if (i % 5 == 0) b[i] = a[i];                        // equal lanes
    if (i % 7 == 0) { a[i] &= 0x80ff80ffu; b[i] |= 0x7f807f80u; }   // the edges: -128, 127, 0, 255
  }
  unsigned *da, *db, *dout;
  cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dout, out.size() * 4);
  cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);
  run<<<(n + 127) / 128, 128>>>(da, db, dout, n);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
  struct Ref { const char* name; int bits; bool sgn; int kind; bool sat, relu; };
  const Ref ref[kForms] = {
      {"add.u8x4", 8, false, 0, false, false}, {"add.sat.u8x4", 8, false, 0, true, false}, {"add.sat.s8x4", 8, true, 0, true, false},
      {"sub.u8x4", 8, false, 1, false, false}, {"sub.sat.u8x4", 8, false, 1, true, false}, {"sub.sat.s8x4", 8, true, 1, true, false},
      {"neg.s8x4", 8, true, 2, false, false},
      {"min.u8x4", 8, false, 3, false, false}, {"min.s8x4", 8, true, 3, false, false}, {"min.relu.s8x4", 8, true, 3, false, true},
      {"max.u8x4", 8, false, 4, false, false}, {"max.s8x4", 8, true, 4, false, false}, {"max.relu.s8x4", 8, true, 4, false, true},
      {"add.u16x2", 16, false, 0, false, false}, {"add.sat.u16x2", 16, false, 0, true, false}, {"add.sat.s16x2", 16, true, 0, true, false},
      {"min.s16x2", 16, true, 3, false, false}, {"max.u16x2", 16, false, 4, false, false},
      {"set.lt.s8x4", 8, true, 5, false, false}, {"set.lo.u8x4", 8, false, 6, false, false}, {"set.ge.s8x4", 8, true, 7, false, false},
      {"set.eq.u8x4", 8, false, 8, false, false}, {"set.ne.u16x2", 16, false, 9, false, false}, {"set.le.s16x2", 16, true, 10, false, false}};
  int bad = 0;
  for (int f = 0; f < kForms; ++f) {
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
      const unsigned want = lanes(a[i], b[i], ref[f].bits, ref[f].sgn, ref[f].kind, ref[f].sat, ref[f].relu);
      if (out[size_t(i) * kForms + f] != want) {
        if (!wrong) std::printf("%s: %08x, %08x gives %08x, wanted %08x\n", ref[f].name, a[i], b[i], out[size_t(i) * kForms + f], want);
        ++wrong;
      }
    }
    std::printf("%-16s %d of %d wrong\n", ref[f].name, wrong, n);
    bad += wrong != 0;
  }
  std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
