// MFMA A-matrix broadcast (CBSZ and ABID) for the multi-block shapes, checked against AMD's Matrix Instruction Calculator
// (github.com/ROCm/amd_matrix_instruction_calculator), not against the ISA guides' prose and not against the simulator.
//
// With CBSZ non-zero the instruction feeds each group of 2^CBSZ blocks the A of one block of that group, the one ABID names (the
// calculator's README: "setting CBSZ=2 and ABID=1 would cause block 1 to be broadcast ... in place of blocks 0-3. Block 5 would be
// used in place of blocks 4-7"). That is also the encoding table of the CDNA3 guide ("within each contiguous set of 2^CBSZ blocks, this
// chooses which block of A to broadcast"). Section 7.1.5.1 of the guide describes it by lanes instead (p_a(l) = l % S + S * ABID, S =
// 64 / 2^CBSZ), which is the same only when 2^CBSZ is the whole number of blocks; the calculator is followed where they differ.
//
// How it is checked: every register word of A holds a number that names where it is (lane, register, half), B is a one-hot per column
// (B[b][k][j] = 1 where k == j % K), C is zero, so D[b][i][j] is the number in the A element the instruction used for A[b][i][j % K]. The
// tables in mfma_broadcast_golden.h (written by mfma_broadcast_gen.py from the calculator's output) say where each A, B and D element
// sits and, for each CBSZ and ABID, which A element every position of A reads.
//
// Built for gfx90a, gfx942 and gfx950 (mfma_broadcast.<arch>).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mfma_broadcast_golden.h"

typedef float v4f __attribute__((vector_size(16)));
typedef float v16f __attribute__((vector_size(64)));
typedef float v32f __attribute__((vector_size(128)));
typedef __fp16 v4h __attribute__((vector_size(8)));

// Family indices follow the generator's FAMILIES list: 0-2 are f32 (32x32x1 2 blocks, 16x16x1 4, 4x4x1 16), 3-5 are f16 (32x32x4, 16x16x4, 4x4x4).
template <int F, int CBSZ, int ABID>
__global__ void kern(const uint32_t* a, const uint32_t* b, float* d) {
  const int l = threadIdx.x;
  if constexpr (F < 3) {
    float va, vb;
    __builtin_memcpy(&va, a + l, 4);
    __builtin_memcpy(&vb, b + l, 4);
    if constexpr (F == 0) {
      v32f c = {};
      const v32f r = __builtin_amdgcn_mfma_f32_32x32x1f32(va, vb, c, CBSZ, ABID, 0);
      __builtin_memcpy(d + 32 * l, &r, sizeof r);
    } else if constexpr (F == 1) {
      v16f c = {};
      const v16f r = __builtin_amdgcn_mfma_f32_16x16x1f32(va, vb, c, CBSZ, ABID, 0);
      __builtin_memcpy(d + 16 * l, &r, sizeof r);
    } else {
      v4f c = {};
      const v4f r = __builtin_amdgcn_mfma_f32_4x4x1f32(va, vb, c, CBSZ, ABID, 0);
      __builtin_memcpy(d + 4 * l, &r, sizeof r);
    }
  } else {
    v4h va, vb;
    __builtin_memcpy(&va, a + 2 * l, 8);
    __builtin_memcpy(&vb, b + 2 * l, 8);
    if constexpr (F == 3) {
      v32f c = {};
      const v32f r = __builtin_amdgcn_mfma_f32_32x32x4f16(va, vb, c, CBSZ, ABID, 0);
      __builtin_memcpy(d + 32 * l, &r, sizeof r);
    } else if constexpr (F == 4) {
      v16f c = {};
      const v16f r = __builtin_amdgcn_mfma_f32_16x16x4f16(va, vb, c, CBSZ, ABID, 0);
      __builtin_memcpy(d + 16 * l, &r, sizeof r);
    } else {
      v4f c = {};
      const v4f r = __builtin_amdgcn_mfma_f32_4x4x4f16(va, vb, c, CBSZ, ABID, 0);
      __builtin_memcpy(d + 4 * l, &r, sizeof r);
    }
  }
}

struct Launch {
  int family, cbsz, abid;
  void (*kernel)(const uint32_t*, const uint32_t*, float*);
  const uint16_t* a;
};
#define X(F, C, A, TABLE) {F, C, A, &kern<F, C, A>, TABLE},
static const Launch kLaunch[] = {MFMA_CONFIGS(X)};
#undef X

static uint16_t half_bits(float f) {
  const __fp16 h = static_cast<__fp16>(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}

// One word-sized slot of A or B: element `sub` of register `reg` of `lane`, f32 (one per word) or f16 (two).
static void put(std::vector<uint32_t>& words, int regs, int kind, int reg, int lane, int sub, float value) {
  uint32_t& w = words[lane * regs + reg];
  if (kind == 0) {
    std::memcpy(&w, &value, 4);
  } else {
    w = (w & ~(0xFFFFu << (16 * sub))) | static_cast<uint32_t>(half_bits(value)) << (16 * sub);
  }
}

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 2;
  std::printf("device %s\n", prop.gcnArchName);
  int failed = 0;
  for (const Launch& L : kLaunch) {
    const Family& f = kFamilies[L.family];
    const int a_words = 64 * f.a_regs, b_words = 64 * f.b_regs, d_words = 64 * f.d_regs;
    // A: the number 1 + (position of the half or word) in every slot, so that D says which slot it came from.
    std::vector<uint32_t> a(a_words, 0), b(b_words, 0);
    const int per_word = f.kind == 0 ? 1 : 2;
    for (int lane = 0; lane < 64; ++lane)
      for (int reg = 0; reg < f.a_regs; ++reg)
        for (int sub = 0; sub < per_word; ++sub) put(a, f.a_regs, f.kind, reg, lane, sub, static_cast<float>(1 + (lane * f.a_regs + reg) * per_word + sub));
    // B: the one-hot per column, at the positions the calculator gives for B.
    for (int bl = 0; bl < f.blocks; ++bl)
      for (int k = 0; k < f.K; ++k)
        for (int j = 0; j < f.N; ++j) {
          const uint16_t loc = f.b[(bl * f.K + k) * f.N + j];
          put(b, f.b_regs, f.kind, (loc >> 6) & 3, loc & 63, (loc >> 8) & 3, k == j % f.K ? 1.0f : 0.0f);
        }
    uint32_t *da, *db;
    float* dd;
    if (hipMalloc(&da, a_words * 4) != hipSuccess || hipMalloc(&db, b_words * 4) != hipSuccess || hipMalloc(&dd, d_words * 4) != hipSuccess) return 2;
    if (hipMemcpy(da, a.data(), a_words * 4, hipMemcpyHostToDevice) != hipSuccess || hipMemcpy(db, b.data(), b_words * 4, hipMemcpyHostToDevice) != hipSuccess) return 2;
    hipLaunchKernelGGL(L.kernel, dim3(1), dim3(64), 0, 0, da, db, dd);
    std::vector<float> d(d_words, -1.0f);
    if (hipDeviceSynchronize() != hipSuccess || hipMemcpy(d.data(), dd, d_words * 4, hipMemcpyDeviceToHost) != hipSuccess) {
      std::printf("%s cbsz:%d abid:%d: the kernel failed\n", f.key, L.cbsz, L.abid);
      ++failed;
      continue;
    }
    (void)hipFree(da), (void)hipFree(db), (void)hipFree(dd);
    int wrong = 0;
    for (int bl = 0; bl < f.blocks; ++bl)
      for (int i = 0; i < f.M; ++i)
        for (int j = 0; j < f.N; ++j) {
          // The slot the calculator says A[bl][i][j % K] reads, and the number in it.
          const uint16_t src = L.a[(bl * f.M + i) * f.K + j % f.K];
          const int lane = src & 63, reg = (src >> 6) & 3, sub = (src >> 8) & 3;
          const float want = static_cast<float>(1 + (lane * f.a_regs + reg) * per_word + sub);
          const uint16_t at = f.d[(bl * f.M + i) * f.N + j];   // where D[bl][i][j] sits: lane | register << 6
          const float got = d[(at & 63) * f.d_regs + (at >> 6)];
          if (got != want) {
            if (wrong < 2) std::printf("  D[%d][%d][%d]: got %g want %g\n", bl, i, j, got, want);
            ++wrong;
          }
        }
    std::printf("%s cbsz:%d abid:%d: %d of %d wrong\n", f.key, L.cbsz, L.abid, wrong, f.blocks * f.M * f.N);
    failed += wrong != 0;
  }
  std::printf("mfma_broadcast %s\n", failed ? "FAILED" : "ok");
  return failed != 0;
}
