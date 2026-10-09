// gfx1250 (CDNA 5, MI455X) block loads and stores (block1250.gfx1250, wave32). The oracle is AMD's CDNA5 ISA: each lane moves
// up to 32 consecutive dwords between memory and 32 consecutive registers, those whose bit M0 has set, skipping the "holes" in
// both. Registers a mask leaves out keep what they held, and memory it leaves out is not written.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#pragma clang diagnostic ignored "-Winline-asm"

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef uint32_t v32 __attribute__((ext_vector_type(32)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
#define GP(T, p) ((__attribute__((address_space(1))) T*)(p))
#define LP(T, p) ((__attribute__((address_space(3))) T*)(p))

// Each lane loads the dwords `load_mask` picks of its own 32, into registers that start as 0xA5A5A5A5, and stores the dwords
// `store_mask` picks of those registers into its own 32 dwords of the output (which start as 0x5A5A5A5A).
__global__ void k_block(const uint32_t* in, uint32_t* out, uint32_t load_mask, uint32_t store_mask) {
  const int l = threadIdx.x;
  v32 r;
  for (int i = 0; i < 32; ++i) r[i] = 0xA5A5A5A5u;
  const uint32_t* src = in + l * 32;
  uint32_t* dst = out + l * 32;
  asm volatile("s_mov_b32 m0, %2\n\tglobal_load_block %0, %1, off\n\ts_wait_loadcnt 0" : "+v"(r) : "v"(src), "s"(load_mask) : "m0", "memory");
  asm volatile("s_mov_b32 m0, %2\n\tglobal_store_block %0, %1, off\n\ts_wait_storecnt 0" : : "v"(dst), "v"(r), "s"(store_mask) : "m0", "memory");
}

// Cluster loads, outside a cluster: global loads. Into registers, and into LDS (and then out of it, to be seen).
__global__ void k_cluster(const uint32_t* in, uint32_t* out) {
  __shared__ uint32_t lds[32 * 4];
  const int l = threadIdx.x;
  const uint32_t a = __builtin_amdgcn_cluster_load_b32(GP(int, in + l), 0, 1);
  const v2i b = __builtin_amdgcn_cluster_load_b64(GP(v2i, in + 64 + 2 * l), 0, 1);
  const v4i c = __builtin_amdgcn_cluster_load_b128(GP(v4i, in + 192 + 4 * l), 0, 1);
  __builtin_amdgcn_cluster_load_async_to_lds_b128(GP(v4i, in + 320 + 4 * l), LP(v4i, lds + 4 * (31 - l)), 0, 0, 1);
  __builtin_amdgcn_s_wait_asynccnt(0);
  __syncthreads();
  out[l] = a;
  out[32 + 2 * l] = b[0];
  out[32 + 2 * l + 1] = b[1];
  for (int k = 0; k < 4; ++k) out[96 + 4 * l + k] = c[k];
  for (int k = 0; k < 4; ++k) out[224 + 4 * l + k] = lds[4 * l + k];
}

int main() {
  std::vector<uint32_t> in(32 * 32);
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<uint32_t>(i * 2654435761u + 1);
  uint32_t *d_in = nullptr, *d_out = nullptr;
  CHECK(hipMalloc(&d_in, in.size() * 4));
  CHECK(hipMalloc(&d_out, in.size() * 4));
  CHECK(hipMemcpy(d_in, in.data(), in.size() * 4, hipMemcpyHostToDevice));
  int failed = 0;
  const uint32_t masks[][2] = {{0xFFFFFFFFu, 0xFFFFFFFFu}, {0x55555555u, 0xFFFFFFFFu}, {0xFFFFFFFFu, 0xAAAAAAAAu},
                               {0x0000FFF0u, 0x0000FFF0u}, {0x80000001u, 0x80000001u}, {0x12345678u, 0xF0F0F0F0u}};
  for (const auto& m : masks) {
    std::vector<uint32_t> init(in.size(), 0x5A5A5A5Au);
    CHECK(hipMemcpy(d_out, init.data(), init.size() * 4, hipMemcpyHostToDevice));
    k_block<<<1, 32>>>(d_in, d_out, m[0], m[1]);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(in.size());
    CHECK(hipMemcpy(out.data(), d_out, out.size() * 4, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < 32; ++l)
      for (int i = 0; i < 32; ++i) {
        const uint32_t reg = (m[0] >> i) & 1 ? in[l * 32 + i] : 0xA5A5A5A5u;   // what the register held after the load
        const uint32_t want = (m[1] >> i) & 1 ? reg : 0x5A5A5A5Au;
        wrong += out[l * 32 + i] != want;
      }
    std::printf("block load mask %08x, store mask %08x: %d of 1024 wrong\n", m[0], m[1], wrong);
    failed += wrong != 0;
  }
  {
    std::vector<uint32_t> big(512);
    for (size_t i = 0; i < big.size(); ++i) big[i] = static_cast<uint32_t>(i * 40503u + 17);
    uint32_t *c_in = nullptr, *c_out = nullptr;
    CHECK(hipMalloc(&c_in, big.size() * 4));
    CHECK(hipMalloc(&c_out, 352 * 4));
    CHECK(hipMemcpy(c_in, big.data(), big.size() * 4, hipMemcpyHostToDevice));
    k_cluster<<<1, 32>>>(c_in, c_out);
    CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(352);
    CHECK(hipMemcpy(out.data(), c_out, out.size() * 4, hipMemcpyDeviceToHost));
    int wrong = 0;
    for (int l = 0; l < 32; ++l) {
      wrong += out[l] != big[l];
      for (int k = 0; k < 2; ++k) wrong += out[32 + 2 * l + k] != big[64 + 2 * l + k];
      for (int k = 0; k < 4; ++k) wrong += out[96 + 4 * l + k] != big[192 + 4 * l + k];
      // the 128-bit load of lane l went to the LDS slot of lane 31 - l, which lane 31 - l then reads
      for (int k = 0; k < 4; ++k) wrong += out[224 + 4 * (31 - l) + k] != big[320 + 4 * l + k];
    }
    std::printf("cluster loads, outside a cluster: %d of %d wrong\n", wrong, 32 + 64 + 128 + 128);
    failed += wrong != 0;
  }
  std::printf("block1250: %d failed\n", failed);
  return failed != 0;
}
