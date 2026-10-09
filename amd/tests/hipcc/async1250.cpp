// gfx1250 (CDNA 5, MI455X) asynchronous copies between global memory and LDS, each against the host
// (async1250.gfx1250, wave32). The oracle is AMD's CDNA5 ISA (section 10.8): every lane moves 1, 4, 8 or 16 bytes between
// its own global address and its own LDS address, and a wave waits on ASYNCcnt before it uses what a load wrote. Each
// kernel loads a block of global memory into LDS with a lane-reversed address, waits, reads the LDS back through
// another lane, and stores it to global memory by an asynchronous store from LDS at yet another place.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));

#define GP(T, p) ((__attribute__((address_space(1))) T*)(p))
#define LP(T, p) ((__attribute__((address_space(3))) T*)(p))

// Lane l's element of the input goes to the LDS slot of lane 31 - l; the store sends the slot of lane l to the output
// element of lane (l + 5) mod 32. So out[(l + 5) % 32] = in[l] after the round trip, which the host can say.
template <class T, int N>
struct Copy;
template <>
struct Copy<uint8_t, 1> {
  static __device__ void load(const uint8_t* g, uint8_t* l) { __builtin_amdgcn_global_load_async_to_lds_b8(GP(char, g), LP(char, l), 0, 0); }
  static __device__ void store(uint8_t* g, uint8_t* l) { __builtin_amdgcn_global_store_async_from_lds_b8(GP(char, g), LP(char, l), 0, 0); }
};
template <>
struct Copy<uint32_t, 1> {
  static __device__ void load(const uint32_t* g, uint32_t* l) { __builtin_amdgcn_global_load_async_to_lds_b32(GP(int, g), LP(int, l), 0, 0); }
  static __device__ void store(uint32_t* g, uint32_t* l) { __builtin_amdgcn_global_store_async_from_lds_b32(GP(int, g), LP(int, l), 0, 0); }
};
template <>
struct Copy<uint32_t, 2> {
  static __device__ void load(const uint32_t* g, uint32_t* l) { __builtin_amdgcn_global_load_async_to_lds_b64(GP(v2i, g), LP(v2i, l), 0, 0); }
  static __device__ void store(uint32_t* g, uint32_t* l) { __builtin_amdgcn_global_store_async_from_lds_b64(GP(v2i, g), LP(v2i, l), 0, 0); }
};
template <>
struct Copy<uint32_t, 4> {
  static __device__ void load(const uint32_t* g, uint32_t* l) { __builtin_amdgcn_global_load_async_to_lds_b128(GP(v4i, g), LP(v4i, l), 0, 0); }
  static __device__ void store(uint32_t* g, uint32_t* l) { __builtin_amdgcn_global_store_async_from_lds_b128(GP(v4i, g), LP(v4i, l), 0, 0); }
};

template <class T, int N>
__global__ void k_copy(const T* in, T* out) {
  __shared__ T lds[32 * N];
  const int l = threadIdx.x;
  Copy<T, N>::load(in + l * N, lds + (31 - l) * N);
  __builtin_amdgcn_s_wait_asynccnt(0);
  __syncthreads();
  Copy<T, N>::store(out + ((l + 5) % 32) * N, lds + (31 - l) * N);
  __builtin_amdgcn_s_wait_asynccnt(0);
}

// The same with an instruction offset of 64, which moves both the global and the LDS address.
__global__ void k_offset(const uint32_t* in, uint32_t* out) {
  __shared__ uint32_t lds[32 + 16];
  const int l = threadIdx.x;
  __builtin_amdgcn_global_load_async_to_lds_b32(GP(int, in + l), LP(int, lds + l), 64, 0);
  __builtin_amdgcn_s_wait_asynccnt(0);
  __syncthreads();
  out[l] = lds[(l + 1) % 32 + 16];
}

template <class T>
static T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  if (hipMalloc(&p, v.size() * sizeof(T)) != hipSuccess) return nullptr;
  if (hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess) return nullptr;
  return p;
}
static void report(const char* what, int wrong, int of) { std::printf("%s: %d of %d wrong\n", what, wrong, of); }

template <class T, int N>
static int check(const char* what) {
  std::vector<T> in(32 * N), out(32 * N, T{0});
  for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<T>(i * 2654435761u + 7);
  T* d_in = upload(in);
  T* d_out = upload(out);
  k_copy<T, N><<<1, 32>>>(d_in, d_out);
  if (hipDeviceSynchronize() != hipSuccess) return 1;
  if (hipMemcpy(out.data(), d_out, out.size() * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) return 1;
  int wrong = 0;
  for (int l = 0; l < 32; ++l)
    for (int e = 0; e < N; ++e) wrong += out[((l + 5) % 32) * N + e] != in[l * N + e];
  report(what, wrong, 32 * N);
  return wrong != 0;
}

int main() {
  int failed = 0;
  failed += check<uint8_t, 1>("global_load_async_to_lds_b8 and global_store_async_from_lds_b8");
  failed += check<uint32_t, 1>("global_load_async_to_lds_b32 and global_store_async_from_lds_b32");
  failed += check<uint32_t, 2>("global_load_async_to_lds_b64 and global_store_async_from_lds_b64");
  failed += check<uint32_t, 4>("global_load_async_to_lds_b128 and global_store_async_from_lds_b128");
  {
    std::vector<uint32_t> in(32 + 16, 0), out(32, 0);
    for (int i = 0; i < 48; ++i) in[i] = 1000 + i;
    uint32_t* d_in = upload(in);
    uint32_t* d_out = upload(out);
    k_offset<<<1, 32>>>(d_in, d_out);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(out.data(), d_out, 32 * sizeof(uint32_t), hipMemcpyDeviceToHost));
    // Lane l loads input element l + 16 (64 bytes in) into LDS element l + 16, and then reads the element lane (l + 1) % 32
    // loaded, so out[l] is input element (l + 1) % 32 + 16.
    int wrong = 0;
    for (int l = 0; l < 32; ++l) wrong += out[l] != in[(l + 1) % 32 + 16];
    report("global_load_async_to_lds_b32 with an instruction offset", wrong, 32);
    failed += wrong != 0;
  }
  std::printf("async1250: 5 checks, %d failed\n", failed);
  return failed != 0;
}
