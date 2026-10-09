// gfx1250 (CDNA 5, MI455X) cross-lane permutes, each against the host (permlane1250.gfx1250, wave32). The oracle is AMD's
// CDNA5 ISA (section 15.14): permlane16_var and permlanex16_var gather within a row or across a pair of rows by a
// per-lane index; permlane_bcast, _up, _down and _xor work in lane groups of a width a scalar gives; permlane_idx_gen
// makes the address ds_bpermute wants; permlane16_swap trades a row of one register with a row of another.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

// The input of lane l is 100 + l; the lane's index operand is idx[l].
__global__ void k_var(const uint32_t* idx, uint32_t* out) {
  const uint32_t l = threadIdx.x, v = 100 + l;
  out[l] = __builtin_amdgcn_permlane16_var(0u, v, idx[l], false, false);
  out[32 + l] = __builtin_amdgcn_permlanex16_var(0u, v, idx[l], false, false);
}
template <int W, int S>
__global__ void k_group(uint32_t* out) {
  const uint32_t l = threadIdx.x, v = 100 + l;
  out[l] = __builtin_amdgcn_permlane_bcast(v, S, W);
  out[32 + l] = __builtin_amdgcn_permlane_up(v, S, W);
  out[64 + l] = __builtin_amdgcn_permlane_down(v, S, W);
  out[96 + l] = __builtin_amdgcn_permlane_xor(v, S, W);
}
__global__ void k_idx(uint32_t* out) {
  const uint32_t l = threadIdx.x;
  // Group of 8; every lane picks lane (l * 3 + 1) of its group.
  const uint32_t a = __builtin_amdgcn_permlane_idx_gen(l * 3 + 1, 8);
  out[l] = a;
  out[32 + l] = __builtin_amdgcn_ds_bpermute(a, 100 + l);
}
__global__ void k_swap(uint32_t* out) {
  const uint32_t l = threadIdx.x;
    const auto r = __builtin_amdgcn_permlane16_swap(100 + l, 200 + l, false, false);
  out[l] = r[0];
  out[32 + l] = r[1];
}

static void report(const char* what, int wrong, int of) { std::printf("%s: %d of %d wrong\n", what, wrong, of); }

template <void (*K)(uint32_t*)>
static std::vector<uint32_t> run_and_get(size_t n) {
  uint32_t* d = nullptr;
  std::vector<uint32_t> out(n);
  if (hipMalloc(&d, n * 4) != hipSuccess) return out;
  K<<<1, 32>>>(d);
  if (hipDeviceSynchronize() != hipSuccess) return out;
  if (hipMemcpy(out.data(), d, n * 4, hipMemcpyDeviceToHost) != hipSuccess) return out;
  return out;
}

template <int W, int S>
static int check_group() {
  const auto out = run_and_get<k_group<W, S>>(128);
  int wrong = 0;
  for (int l = 0; l < 32; ++l) {
    const int base = l / W * W, j = l - base;
    const int delta = S < W ? S : W;
    const int bcast = base + (S & 63 & (W - 1));
    const int up = j < delta ? l : l - delta;
    const int down = j + delta < W ? l + delta : l;
    int x = l;
    if (S < 32) {
      x = l ^ (S & 63);
      if (x >= base + W) x = l;
    }
    wrong += out[l] != 100u + bcast;
    wrong += out[32 + l] != 100u + up;
    wrong += out[64 + l] != 100u + down;
    wrong += out[96 + l] != 100u + x;
  }
  char name[96];
  std::snprintf(name, sizeof name, "permlane_bcast, up, down and xor, groups of %d, S = %d", W, S);
  report(name, wrong, 128);
  return wrong != 0;
}

int main(int argc, char** argv) {
  // An argument picks one check by its position, for finding which one a change broke.
  const int only = argc > 1 ? std::atoi(argv[1]) : 0;
  int n = 0;
  int failed = 0;
  if (++n && (!only || only == n)) {
    std::vector<uint32_t> idx(32);
    for (int l = 0; l < 32; ++l) idx[l] = (l * 7 + 3) & 15;   // the low four bits are what counts
    for (int l = 0; l < 32; ++l) idx[l] |= (l % 3) << 4;      // the rest must not
    uint32_t *d_idx = nullptr, *d_out = nullptr;
    CHECK(hipMalloc(&d_idx, 32 * 4));
    CHECK(hipMalloc(&d_out, 64 * 4));
    CHECK(hipMemcpy(d_idx, idx.data(), 32 * 4, hipMemcpyHostToDevice));
    k_var<<<1, 32>>>(d_idx, d_out);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(64);
    CHECK(hipMemcpy(out.data(), d_out, 64 * 4, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const int row = l / 16;
      wrong += out[l] != 100u + row * 16 + (idx[l] & 15);
      wrong += out[32 + l] != 100u + (row ^ 1) * 16 + (idx[l] & 15);
    }
    report("permlane16_var and permlanex16_var", wrong, 64);
    failed += wrong != 0;
  }
  if (++n && (!only || only == n)) failed += check_group<32, 5>();
  if (++n && (!only || only == n)) failed += check_group<16, 3>();
  if (++n && (!only || only == n)) failed += check_group<8, 2>();
  if (++n && (!only || only == n)) failed += check_group<4, 1>();
  if (++n && (!only || only == n)) failed += check_group<8, 40>();   // a mask or delta past the wave or the group
  if (++n && (!only || only == n)) {
    const auto out = run_and_get<k_idx>(64);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      const int pick = ((l * 3 + 1) & 7) + (l & ~7);
      wrong += out[l] != static_cast<uint32_t>(pick << 2);
      wrong += out[32 + l] != 100u + pick;
    }
    report("permlane_idx_gen with ds_bpermute", wrong, 64);
    failed += wrong != 0;
  }
  if (++n && (!only || only == n)) {
    const auto out = run_and_get<k_swap>(64);
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      // The destination (first, 100 + l) takes the source's (second, 200 + l) row 0 into its row 1, and the source takes
      // the destination's row 1 into its row 0; the other rows stay.
      const uint32_t first = l >= 16 ? 200u + (l - 16) : 100u + l;
      const uint32_t second = l < 16 ? 100u + (l + 16) : 200u + l;
      wrong += out[l] != first;
      wrong += out[32 + l] != second;
    }
    report("permlane16_swap", wrong, 64);
    failed += wrong != 0;
  }
  std::printf("permlane1250: 8 checks, %d failed\n", failed);
  return failed != 0;
}
