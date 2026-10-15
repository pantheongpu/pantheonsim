// Shared and local memory accessed 1, 2, 4, 8 and 16 bytes at a time, by lane-dependent indices.
//
// The SASS executor moves a lane's bytes between the register file and the block's shared array (or the warp's local
// window) with one copy per register word (src/sass/exec_ops.inc, store_lane_bytes / load_lane_word), and the tier
// has its own handlers for the same accesses. This program stores through a permutation of the lanes and loads through
// another, so that every lane's data lands in another lane's registers, at every width, with and without the
// sign/zero extension of the narrow loads, and checks every result word against a host computation. It also runs
// against NVIDIA's real GPU and library (compile with nvcc and run on a card): nothing in it is simulator-specific.
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int kThreads = 64;
constexpr int kBlocks = 3;

__device__ __host__ inline uint32_t mix(uint32_t a, uint32_t b) { return (a ^ (b * 2654435761u)) + (a >> 3) + 0x9e3779b9u; }

// Shared memory: a permutation of 64 lanes to the slots (5 is odd, so t * 5 + 3 mod 64 is a bijection), and another to read.
__global__ void k_shared(const uint32_t* in, uint32_t* out) {
  __shared__ uint4 s16[kThreads];
  __shared__ uint2 s8[kThreads];
  __shared__ uint32_t s4[kThreads];
  __shared__ int16_t s2[kThreads];
  __shared__ int8_t s1[kThreads];
  const int t = threadIdx.x, g = blockIdx.x * kThreads + t;
  const uint32_t a = in[4 * g], b = in[4 * g + 1], c = in[4 * g + 2], d = in[4 * g + 3];
  const int p = (t * 5 + 3) & (kThreads - 1), q = (t * 9 + 1) & (kThreads - 1);
  s16[p] = make_uint4(a, b, c, d);
  s8[p] = make_uint2(b, c);
  s4[p] = d;
  s2[p] = static_cast<int16_t>(a);
  s1[p] = static_cast<int8_t>(b);
  __syncthreads();
  const uint4 v16 = s16[q];
  const uint2 v8 = s8[q];
  const uint32_t v4 = s4[q];
  const int v2 = s2[q];                              // sign-extended
  const unsigned v2u = static_cast<uint16_t>(s2[q]);   // zero-extended
  const int v1 = s1[q];
  const unsigned v1u = static_cast<uint8_t>(s1[q]);
  uint32_t* o = out + 12 * g;
  o[0] = v16.x; o[1] = v16.y; o[2] = v16.z; o[3] = v16.w;
  o[4] = v8.x; o[5] = v8.y; o[6] = v4;
  o[7] = static_cast<uint32_t>(v2); o[8] = v2u;
  o[9] = static_cast<uint32_t>(v1); o[10] = v1u;
  o[11] = mix(v16.x, v8.y);
}

// Local memory: per-thread arrays indexed by a value read from memory, so that the compiler cannot keep them in registers.
__global__ void k_local(const uint32_t* in, uint32_t* out) {
  const int t = threadIdx.x, g = blockIdx.x * kThreads + t;
  uint4 l16[8];
  uint2 l8[8];
  uint32_t l4[8];
  int16_t l2[8];
  int8_t l1[8];
  for (int i = 0; i < 8; ++i) {
    const uint32_t x = in[4 * g + (i & 3)] + static_cast<uint32_t>(i) * 7u;
    l16[i] = make_uint4(x, x + 1, x + 2, x + 3);
    l8[i] = make_uint2(x ^ 5u, x + 9u);
    l4[i] = x * 3u;
    l2[i] = static_cast<int16_t>(x);
    l1[i] = static_cast<int8_t>(x >> 3);
  }
  // A second pass overwrites some slots through other indices.
  const uint32_t j0 = in[4 * g] & 7u, j1 = in[4 * g + 1] & 7u, j2 = in[4 * g + 2] & 7u;
  l16[j0] = make_uint4(j1, j2, j0, 0xdeadbeefu);
  l8[j1] = make_uint2(j0 * 3u, j2 * 5u);
  l4[j2] = j0 + j1 + j2;
  l2[j0] = static_cast<int16_t>(0x8001 + j1);
  l1[j2] = static_cast<int8_t>(0x80 + j0);
  const uint32_t k0 = in[4 * g + 3] & 7u, k1 = (k0 + j0) & 7u;
  const uint4 a16 = l16[k0], b16 = l16[k1];   // (whole vectors: 16- and 8-byte loads)
  const uint2 a8 = l8[k0], b8 = l8[k1];
  uint32_t* o = out + 12 * g;
  o[0] = a16.x; o[1] = a16.y; o[2] = b16.z; o[3] = b16.w;
  o[4] = a8.x; o[5] = b8.y; o[6] = l4[k0] + l4[k1];
  o[7] = static_cast<uint32_t>(static_cast<int>(l2[k0])); o[8] = static_cast<uint16_t>(l2[k1]);
  o[9] = static_cast<uint32_t>(static_cast<int>(l1[k0])); o[10] = static_cast<uint8_t>(l1[k1]);
  o[11] = mix(a16.x, b8.y);
}

// The host's version of both kernels.
static void expect_shared(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
  for (int blk = 0; blk < kBlocks; ++blk) {
    uint32_t s16[kThreads][4], s8[kThreads][2], s4[kThreads];
    int16_t s2[kThreads];
    int8_t s1[kThreads];
    for (int t = 0; t < kThreads; ++t) {
      const int g = blk * kThreads + t, p = (t * 5 + 3) & (kThreads - 1);
      const uint32_t a = in[4 * g], b = in[4 * g + 1], c = in[4 * g + 2], d = in[4 * g + 3];
      s16[p][0] = a; s16[p][1] = b; s16[p][2] = c; s16[p][3] = d;
      s8[p][0] = b; s8[p][1] = c;
      s4[p] = d;
      s2[p] = static_cast<int16_t>(a);
      s1[p] = static_cast<int8_t>(b);
    }
    for (int t = 0; t < kThreads; ++t) {
      const int g = blk * kThreads + t, q = (t * 9 + 1) & (kThreads - 1);
      uint32_t* o = &out[12 * g];
      o[0] = s16[q][0]; o[1] = s16[q][1]; o[2] = s16[q][2]; o[3] = s16[q][3];
      o[4] = s8[q][0]; o[5] = s8[q][1]; o[6] = s4[q];
      o[7] = static_cast<uint32_t>(static_cast<int>(s2[q])); o[8] = static_cast<uint16_t>(s2[q]);
      o[9] = static_cast<uint32_t>(static_cast<int>(s1[q])); o[10] = static_cast<uint8_t>(s1[q]);
      o[11] = mix(s16[q][0], s8[q][1]);
    }
  }
}

static void expect_local(const std::vector<uint32_t>& in, std::vector<uint32_t>& out) {
  for (int g = 0; g < kBlocks * kThreads; ++g) {
    uint32_t l16[8][4], l8[8][2], l4[8];
    int16_t l2[8];
    int8_t l1[8];
    for (int i = 0; i < 8; ++i) {
      const uint32_t x = in[4 * g + (i & 3)] + static_cast<uint32_t>(i) * 7u;
      l16[i][0] = x; l16[i][1] = x + 1; l16[i][2] = x + 2; l16[i][3] = x + 3;
      l8[i][0] = x ^ 5u; l8[i][1] = x + 9u;
      l4[i] = x * 3u;
      l2[i] = static_cast<int16_t>(x);
      l1[i] = static_cast<int8_t>(x >> 3);
    }
    const uint32_t j0 = in[4 * g] & 7u, j1 = in[4 * g + 1] & 7u, j2 = in[4 * g + 2] & 7u;
    l16[j0][0] = j1; l16[j0][1] = j2; l16[j0][2] = j0; l16[j0][3] = 0xdeadbeefu;
    l8[j1][0] = j0 * 3u; l8[j1][1] = j2 * 5u;
    l4[j2] = j0 + j1 + j2;
    l2[j0] = static_cast<int16_t>(0x8001 + j1);
    l1[j2] = static_cast<int8_t>(0x80 + j0);
    const uint32_t k0 = in[4 * g + 3] & 7u, k1 = (k0 + j0) & 7u;
    uint32_t* o = &out[12 * g];
    o[0] = l16[k0][0]; o[1] = l16[k0][1]; o[2] = l16[k1][2]; o[3] = l16[k1][3];
    o[4] = l8[k0][0]; o[5] = l8[k1][1]; o[6] = l4[k0] + l4[k1];
    o[7] = static_cast<uint32_t>(static_cast<int>(l2[k0])); o[8] = static_cast<uint16_t>(l2[k1]);
    o[9] = static_cast<uint32_t>(static_cast<int>(l1[k0])); o[10] = static_cast<uint8_t>(l1[k1]);
    o[11] = mix(l16[k0][0], l8[k1][1]);
  }
}

#define CHECK_CUDA(x)                                                           \
  do {                                                                          \
    cudaError_t e_ = (x);                                                       \
    if (e_ != cudaSuccess) {                                                    \
      std::printf("FAIL: %s: %s\n", #x, cudaGetErrorString(e_));                \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

int main() {
  const int n = kBlocks * kThreads;
  std::vector<uint32_t> in(4 * n), want(12 * n), got(12 * n);
  uint64_t s = 0x243F6A8885A308D3ull;
  for (auto& x : in) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    x = static_cast<uint32_t>(s >> 11);
  }
  uint32_t *d_in = nullptr, *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in, in.size() * 4));
  CHECK_CUDA(cudaMalloc(&d_out, got.size() * 4));
  CHECK_CUDA(cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
  int bad = 0;
  for (int which = 0; which < 2; ++which) {
    CHECK_CUDA(cudaMemset(d_out, 0xff, got.size() * 4));
    if (which == 0) k_shared<<<kBlocks, kThreads>>>(d_in, d_out);
    else k_local<<<kBlocks, kThreads>>>(d_in, d_out);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost));
    if (which == 0) expect_shared(in, want);
    else expect_local(in, want);
    int mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i)
      if (got[i] != want[i]) {
        if (mismatches++ < 5)
          std::printf("%s: word %zu (thread %zu, field %zu): got %08x, expected %08x\n", which ? "local" : "shared", i,
                      i / 12, i % 12, got[i], want[i]);
      }
    std::printf("%s: %d of %zu words differ\n", which ? "local" : "shared", mismatches, got.size());
    bad += mismatches;
  }
  cudaFree(d_in);
  cudaFree(d_out);
  std::printf(bad ? "FAIL\n" : "PASS\n");
  return bad ? 1 : 0;
}
