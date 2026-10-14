// The runtime functions the toolkit declares that this library once lacked, one after another with the argument
// cases they were measured on -- every answer an RTX 3060 gave under NVIDIA's CUDA 13.0 runtime
// (nvidia/tests/data/exports_sweep_runtime.rtx3060.expected). Not here: a call that crashes the card (a garbage
// handle, a callback unregistered twice) and what the simulator refuses on purpose where the card works (host
// atomics over a peer path, no card here to check): the sweep's documentation lists them. Cases of functions newer
// than the toolkit at hand are not compiled and carry [12.8] or [13], so the runner can leave them out of the
// expected file.
//
// Each case runs in a process of its own (the card crashes on some arguments), and reads libk.cubin
// (exports_sweep_module.cu, built by the runner) from the directory the program runs in.
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

__device__ int sym[16];
__global__ void kern(int* p) {
  extern __shared__ int sm[];
  __shared__ int st[64];
  st[threadIdx.x & 63] = 1;
  sm[threadIdx.x] = 2;
  p[threadIdx.x] = st[0] + sm[0] + sym[0];
}
__global__ void other(float* p) { p[threadIdx.x] = 2.f; }
void hostfn() {}

// Kernels for the attribute cases (cudaFuncGetAttributes / cudaFuncSetAttribute): static shared memory of several
// sizes, dynamic shared memory, launch bounds, enough registers to limit the block, and module constants.
template <int N> __global__ void kstat(int* p) {
  __shared__ char s[N];
  s[threadIdx.x % N] = static_cast<char>(threadIdx.x);
  __syncthreads();
  p[0] = s[1 % N];
}
__global__ void kdyn(int* p) {
  extern __shared__ char dsm[];
  dsm[threadIdx.x] = 1;
  __syncthreads();
  p[0] = dsm[0];
}
__global__ void kdyn2(int* p) {
  extern __shared__ char dsm[];
  dsm[threadIdx.x] = 1;
  __syncthreads();
  p[0] = dsm[0];
}
__global__ __launch_bounds__(128) void klb128(int* p) { p[0] = 1; }
__global__ __launch_bounds__(512, 2) void klb512(int* p) { p[0] = 1; }
__global__ __launch_bounds__(96) void klb96(int* p) { p[0] = 1; }
template <int N> __global__ void kheavy(float* o) {
  float a[N];
#pragma unroll
  for (int i = 0; i < N; i++) a[i] = o[i * 32 + threadIdx.x];
  float s = 0;
#pragma unroll
  for (int i = 0; i < N; i++) s += a[i] * a[(i * 7 + 1) % N];
#pragma unroll
  for (int i = 0; i < N; i++) o[i * 32 + threadIdx.x] = a[i] * s + a[(i + 1) % N];
}
__constant__ int ctab[100];
__constant__ char cflag[5];
__constant__ double cwide[3];
__global__ void kconst(int* p) { p[0] = ctab[threadIdx.x] + cflag[1] + static_cast<int>(cwide[1]); }

static std::string g_notes;
static void note(const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  g_notes += g_notes.empty() ? "" : "; ";
  g_notes += buf;
}

static cudaStream_t stream_made() {
  cudaStream_t s = nullptr;
  cudaStreamCreate(&s);
  return s;
}

// What cudaFuncGetAttributes says of a kernel (not the registers and local memory: they are the compiler's).
static void note_attrs(const void* f) {
  cudaFuncAttributes a;
  const cudaError_t e = cudaFuncGetAttributes(&a, f);
  if (e != cudaSuccess) {
    note("get %s", cudaGetErrorName(e));
    return;
  }
  note("static %zu, const %zu, maxThreads %d, ptx %d, binary %d, cacheCA %d, maxDynamic %d, carveout %d, clusterMustBeSet %d, "
       "cluster %d %d %d, policy %d, nonPortable %d",
       a.sharedSizeBytes, a.constSizeBytes, a.maxThreadsPerBlock, a.ptxVersion, a.binaryVersion, a.cacheModeCA,
       a.maxDynamicSharedSizeBytes, a.preferredShmemCarveout, a.clusterDimMustBeSet, a.requiredClusterWidth,
       a.requiredClusterHeight, a.requiredClusterDepth, a.clusterSchedulingPolicyPreference,
       a.nonPortableClusterSizeAllowed);
}
static int max_dynamic(const void* f) {
  cudaFuncAttributes a;
  return cudaFuncGetAttributes(&a, f) == cudaSuccess ? a.maxDynamicSharedSizeBytes : -99;
}
// A launch of one warp with `dyn` bytes of dynamic shared memory: whether it was refused. The code is not printed:
// an RTX 3060 under CUDA 13.0 says cudaErrorInvalidValue and this library, like the CUDA 12 runtime, says
// cudaErrorInvalidConfiguration for every launch configuration it refuses.
static bool launch_refused(const void* f, size_t dyn) {
  int* d = nullptr;
  cudaMalloc(&d, 4096);
  void* args[1] = {&d};
  const cudaError_t e = cudaLaunchKernel(f, dim3(1), dim3(32), args, dyn, 0);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaGetLastError();
  cudaFree(d);
  return e != cudaSuccess;
}
static cudaError_t set_dynamic(const void* f, int v) {
  const cudaError_t e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, v);
  note("now %d", max_dynamic(f));
  return e;
}

struct Case {
  const char* label;
  const char* tag;
  std::function<cudaError_t()> run;
};
#define C(l, ...) cases.push_back({l, "", [&]() -> cudaError_t { __VA_ARGS__ }})
#if CUDART_VERSION >= 12080
#define C128(l, ...) cases.push_back({l, " [12.8]", [&]() -> cudaError_t { __VA_ARGS__ }})
#else
#define C128(l, ...) (void)0
#endif
#if CUDART_VERSION >= 13000
#define C13(l, ...) cases.push_back({l, " [13]", [&]() -> cudaError_t { __VA_ARGS__ }})
#else
#define C13(l, ...) (void)0
#endif

int main() {
  std::vector<Case> cases;
  // ---- the shared memory configuration, deprecated
  C("cudaDeviceGetSharedMemConfig", { cudaSharedMemConfig c = (cudaSharedMemConfig)99; cudaError_t e = cudaDeviceGetSharedMemConfig(&c); note("%d", (int)c); return e; });
  C("cudaDeviceSetSharedMemConfig eight, then get", { cudaError_t e = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte); cudaSharedMemConfig c = (cudaSharedMemConfig)99; cudaDeviceGetSharedMemConfig(&c); note("get %d", (int)c); return e; });
  C("cudaDeviceSetSharedMemConfig 7", { return cudaDeviceSetSharedMemConfig((cudaSharedMemConfig)7); });
  C("cudaDeviceGetSharedMemConfig null", { return cudaDeviceGetSharedMemConfig(nullptr); });
  C("cudaFuncSetSharedMemConfig", { return cudaFuncSetSharedMemConfig((const void*)kern, cudaSharedMemBankSizeEightByte); });
  C("cudaFuncSetSharedMemConfig 7", { return cudaFuncSetSharedMemConfig((const void*)kern, (cudaSharedMemConfig)7); });
  C("cudaFuncSetSharedMemConfig null", { return cudaFuncSetSharedMemConfig(nullptr, cudaSharedMemBankSizeEightByte); });
  C("cudaFuncSetSharedMemConfig stack address", { int x; return cudaFuncSetSharedMemConfig((const void*)&x, cudaSharedMemBankSizeEightByte); });
  // ---- cudaSetValidDevices
  C("cudaSetValidDevices {1,0}", { int a[2] = {1, 0}; return cudaSetValidDevices(a, 2); });
  C("cudaSetValidDevices len 0", { int a[2] = {1, 0}; return cudaSetValidDevices(a, 0); });
  C("cudaSetValidDevices null", { return cudaSetValidDevices(nullptr, 1); });
  C("cudaSetValidDevices with a device that does not exist", { int a[4] = {1, 0, -1, 5}; return cudaSetValidDevices(a, 4); });
  C("cudaSetValidDevices len -1", { int a[2] = {1, 0}; return cudaSetValidDevices(a, -1); });
  // ---- the widest linear 1D texture
  C("cudaDeviceGetTexture1DLinearMaxWidth float", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaError_t e = cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); note("w=%zu", w); return e; });
  C("cudaDeviceGetTexture1DLinearMaxWidth uchar4", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc<uchar4>(); cudaError_t e = cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); note("w=%zu", w); return e; });
  C("cudaDeviceGetTexture1DLinearMaxWidth float4", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc<float4>(); cudaError_t e = cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); note("w=%zu", w); return e; });
  C("cudaDeviceGetTexture1DLinearMaxWidth uchar", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc<unsigned char>(); cudaError_t e = cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); note("w=%zu", w); return e; });
  C("cudaDeviceGetTexture1DLinearMaxWidth 3 channels", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindFloat); return cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); });
  C("cudaDeviceGetTexture1DLinearMaxWidth 2 x 8 bits", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned); cudaError_t e = cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); note("w=%zu", w); return e; });
  C("cudaDeviceGetTexture1DLinearMaxWidth no channels", { size_t w = 77; cudaChannelFormatDesc f = cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindFloat); return cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 0); });
  C("cudaDeviceGetTexture1DLinearMaxWidth null width", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); return cudaDeviceGetTexture1DLinearMaxWidth(nullptr, &f, 0); });
  C("cudaDeviceGetTexture1DLinearMaxWidth null format", { size_t w; return cudaDeviceGetTexture1DLinearMaxWidth(&w, nullptr, 0); });
  C("cudaDeviceGetTexture1DLinearMaxWidth device 9", { size_t w; cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); return cudaDeviceGetTexture1DLinearMaxWidth(&w, &f, 9); });
  // ---- dynamic shared memory per block at an occupancy (a kernel with 256 bytes of static shared memory)
  static const int occ[][2] = {{1, 128}, {2, 128}, {4, 256}, {8, 128}, {12, 128}, {6, 256}, {100, 128}, {0, 128}, {1, 0}, {1, 2048}, {1, 1024}, {3, 512}, {16, 64}, {32, 32}, {-1, 128}, {1, -5}, {2, 1024}, {3, 1024}};
  for (const auto& o : occ) {
    static char labels[32][96];
    static int n = 0;
    std::snprintf(labels[n], sizeof labels[n], "cudaOccupancyAvailableDynamicSMemPerBlock blocks %d size %d", o[0], o[1]);
    const char* label = labels[n++];
    const int blocks = o[0], size = o[1];
    cases.push_back({label, "", [blocks, size]() -> cudaError_t { size_t s = 77; cudaError_t e = cudaOccupancyAvailableDynamicSMemPerBlock(&s, (const void*)kern, blocks, size); note("%zu", s); return e; }});
  }
  C("cudaOccupancyAvailableDynamicSMemPerBlock null", { return cudaOccupancyAvailableDynamicSMemPerBlock(nullptr, (const void*)kern, 1, 128); });
  C("cudaOccupancyAvailableDynamicSMemPerBlock null function", { size_t s; return cudaOccupancyAvailableDynamicSMemPerBlock(&s, nullptr, 1, 128); });
  // ---- kernel attributes: what cudaFuncGetAttributes says before anything is set, and what cudaFuncSetAttribute
  // accepts (nvidia/src/func_attrs.hpp has the rules; every line is an RTX 3060 under CUDA 13.0)
  C("cudaFuncGetAttributes, no static shared memory", { note_attrs((const void*)kdyn); return cudaSuccess; });
  C("cudaFuncGetAttributes, 256 bytes static", { note_attrs((const void*)kstat<256>); return cudaSuccess; });
  C("cudaFuncGetAttributes, 40000 bytes static", { note_attrs((const void*)kstat<40000>); return cudaSuccess; });
  C("cudaFuncGetAttributes, 48 KiB static", { note_attrs((const void*)kstat<49152>); return cudaSuccess; });
  C("cudaFuncGetAttributes, __launch_bounds__(128)", { note_attrs((const void*)klb128); return cudaSuccess; });
  C("cudaFuncGetAttributes, __launch_bounds__(512, 2)", { note_attrs((const void*)klb512); return cudaSuccess; });
  C("cudaFuncGetAttributes, __launch_bounds__(96)", { note_attrs((const void*)klb96); return cudaSuccess; });
  C("cudaFuncGetAttributes, registers limit the block", { note_attrs((const void*)kheavy<400>); return cudaSuccess; });
  C("cudaFuncGetAttributes, module constants", { note_attrs((const void*)kconst); return cudaSuccess; });
  C("cudaFuncGetAttributes null attributes", { return cudaFuncGetAttributes(nullptr, (const void*)kdyn); });
  C("cudaFuncGetAttributes null function", { cudaFuncAttributes a; return cudaFuncGetAttributes(&a, nullptr); });
  C("cudaFuncGetAttributes stack address", { cudaFuncAttributes a; int x; return cudaFuncGetAttributes(&a, (const void*)&x); });
  C("cudaFuncGetAttributes null both", { return cudaFuncGetAttributes(nullptr, nullptr); });
  static const int dyn_values[] = {-1, 0, 1, 1024, 49152, 49153, 65536, 101376, 101377, 102400};
  for (const int v : dyn_values) {
    static char labels[16][96];
    static int n = 0;
    std::snprintf(labels[n], sizeof labels[n], "cudaFuncSetAttribute MaxDynamicSharedMemorySize %d, no static", v);
    const char* label = labels[n++];
    cases.push_back({label, "", [v]() -> cudaError_t { return set_dynamic((const void*)kdyn, v); }});
  }
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 101120, 256 static", { return set_dynamic((const void*)kstat<256>, 101120); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 101121, 256 static", { return set_dynamic((const void*)kstat<256>, 101121); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 48896, 256 static", { return set_dynamic((const void*)kstat<256>, 48896); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 61376, 40000 static", { return set_dynamic((const void*)kstat<40000>, 61376); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 61377, 40000 static", { return set_dynamic((const void*)kstat<40000>, 61377); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 52224, 48 KiB static", { return set_dynamic((const void*)kstat<49152>, 52224); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 52225, 48 KiB static", { return set_dynamic((const void*)kstat<49152>, 52225); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize 100000, 48 KiB static", { return set_dynamic((const void*)kstat<49152>, 100000); });
  C("cudaFuncSetAttribute MaxDynamicSharedMemorySize twice", { cudaError_t e = set_dynamic((const void*)kdyn, 70000); set_dynamic((const void*)kdyn, 3000); return e; });
  C("cudaFuncSetAttribute attribute 99", { return cudaFuncSetAttribute((const void*)kdyn, (cudaFuncAttribute)99, 1); });
  C("cudaFuncSetAttribute attribute -1", { return cudaFuncSetAttribute((const void*)kdyn, (cudaFuncAttribute)-1, 1); });
  C("cudaFuncSetAttribute attribute 16", { return cudaFuncSetAttribute((const void*)kdyn, (cudaFuncAttribute)16, 1); });
  C("cudaFuncSetAttribute attribute 0", { return cudaFuncSetAttribute((const void*)kdyn, (cudaFuncAttribute)0, 1); });
  C("cudaFuncSetAttribute attribute 1", { return cudaFuncSetAttribute((const void*)kdyn, (cudaFuncAttribute)1, 1); });
  C("cudaFuncSetAttribute attribute 7", { return cudaFuncSetAttribute((const void*)kdyn, (cudaFuncAttribute)7, 1); });
  C("cudaFuncSetAttribute ClusterDimMustBeSet", { return cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeClusterDimMustBeSet, 1); });
  C("cudaFuncSetAttribute null function", { return cudaFuncSetAttribute(nullptr, cudaFuncAttributeMaxDynamicSharedMemorySize, 100); });
  C("cudaFuncSetAttribute stack address", { int x; return cudaFuncSetAttribute((const void*)&x, cudaFuncAttributeMaxDynamicSharedMemorySize, 100); });
  C("cudaFuncSetAttribute stack address, carveout", { int x; return cudaFuncSetAttribute((const void*)&x, cudaFuncAttributePreferredSharedMemoryCarveout, 1); });
  C("cudaFuncSetAttribute stack address, attribute 99", { int x; return cudaFuncSetAttribute((const void*)&x, (cudaFuncAttribute)99, 1); });
  C("cudaFuncSetAttribute stack address, size -1", { int x; return cudaFuncSetAttribute((const void*)&x, cudaFuncAttributeMaxDynamicSharedMemorySize, -1); });
  static const int carve_values[] = {-2, -1, 0, 1, 50, 100, 101};
  for (const int v : carve_values) {
    static char labels[16][96];
    static int n = 0;
    std::snprintf(labels[n], sizeof labels[n], "cudaFuncSetAttribute PreferredSharedMemoryCarveout %d", v);
    const char* label = labels[n++];
    cases.push_back({label, "", [v]() -> cudaError_t { cudaError_t e = cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributePreferredSharedMemoryCarveout, v); cudaFuncAttributes a; cudaFuncGetAttributes(&a, (const void*)klb128); note("now %d", a.preferredShmemCarveout); return e; }});
  }
  C("cudaFuncSetAttribute required cluster dimensions", { cudaError_t e = cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeRequiredClusterWidth, 3); cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeRequiredClusterHeight, 2); cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeRequiredClusterDepth, 5); note_attrs((const void*)klb128); return e; });
  C("cudaFuncSetAttribute required cluster width -1", { return cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeRequiredClusterWidth, -1); });
  C("cudaFuncSetAttribute required cluster height -1", { return cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeRequiredClusterHeight, -1); });
  C("cudaFuncSetAttribute required cluster depth 0", { cudaError_t e = cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeRequiredClusterDepth, 0); note_attrs((const void*)klb128); return e; });
  C("cudaFuncSetAttribute ClusterSchedulingPolicyPreference 2", { cudaError_t e = cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeClusterSchedulingPolicyPreference, 2); note_attrs((const void*)klb128); return e; });
  C("cudaFuncSetAttribute ClusterSchedulingPolicyPreference 3", { return cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeClusterSchedulingPolicyPreference, 3); });
  C("cudaFuncSetAttribute ClusterSchedulingPolicyPreference -1", { return cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeClusterSchedulingPolicyPreference, -1); });
  C("cudaFuncSetAttribute NonPortableClusterSizeAllowed 7", { cudaError_t e = cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeNonPortableClusterSizeAllowed, 7); note_attrs((const void*)klb128); return e; });
  C("cudaFuncSetAttribute NonPortableClusterSizeAllowed -1", { cudaError_t e = cudaFuncSetAttribute((const void*)klb128, cudaFuncAttributeNonPortableClusterSizeAllowed, -1); note_attrs((const void*)klb128); return e; });
  C("cudaFuncSetAttribute everything on one kernel, then the other", { cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeMaxDynamicSharedMemorySize, 70000); cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributePreferredSharedMemoryCarveout, 40); note("other kernel %d", max_dynamic((const void*)kdyn2)); note_attrs((const void*)kdyn); return cudaSuccess; });
  C("cudaFuncSetAttribute on device 0, read on device 1", { cudaSetDevice(0); cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeMaxDynamicSharedMemorySize, 70000); cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributePreferredSharedMemoryCarveout, 40); int on0 = max_dynamic((const void*)kdyn); cudaSetDevice(1); int on1 = max_dynamic((const void*)kdyn); cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeMaxDynamicSharedMemorySize, 3000); int after = max_dynamic((const void*)kdyn); cudaSetDevice(0); note("device 0 %d, device 1 %d, after setting 3000 on 1: %d, device 0 again %d", on0, on1, after, max_dynamic((const void*)kdyn)); return cudaSuccess; });
  // A launch may ask for what the kernel's limit allows, no more.
  C("launch dynamic shared memory, not opted in", { note("49152 refused %d, 49153 refused %d, 65536 refused %d", launch_refused((const void*)kdyn, 49152), launch_refused((const void*)kdyn, 49153), launch_refused((const void*)kdyn, 65536)); return cudaSuccess; });
  C("launch dynamic shared memory, 40000 bytes static", { note("9152 refused %d, 9153 refused %d", launch_refused((const void*)kstat<40000>, 9152), launch_refused((const void*)kstat<40000>, 9153)); return cudaSuccess; });
  C("launch dynamic shared memory, 48 KiB static", { note("0 refused %d, 1 refused %d", launch_refused((const void*)kstat<49152>, 0), launch_refused((const void*)kstat<49152>, 1)); return cudaSuccess; });
  C("launch dynamic shared memory, opted in to 60000", { cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeMaxDynamicSharedMemorySize, 60000); note("49153 refused %d, 60000 refused %d, 60001 refused %d", launch_refused((const void*)kdyn, 49153), launch_refused((const void*)kdyn, 60000), launch_refused((const void*)kdyn, 60001)); return cudaSuccess; });
  C("launch dynamic shared memory, opted in to 100", { cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeMaxDynamicSharedMemorySize, 100); note("100 refused %d, 101 refused %d, 49152 refused %d", launch_refused((const void*)kdyn, 100), launch_refused((const void*)kdyn, 101), launch_refused((const void*)kdyn, 49152)); return cudaSuccess; });
  C("launch dynamic shared memory, opted in to the most", { cudaFuncSetAttribute((const void*)kdyn, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376); note("101376 refused %d, 101377 refused %d", launch_refused((const void*)kdyn, 101376), launch_refused((const void*)kdyn, 101377)); return cudaSuccess; });
  C("launch with more threads than __launch_bounds__", { int* d; cudaMalloc(&d, 4096); void* args[1] = {&d}; cudaError_t e1 = cudaLaunchKernel((const void*)klb128, dim3(1), dim3(128), args, 0, 0); cudaError_t e2 = cudaLaunchKernel((const void*)klb128, dim3(1), dim3(129), args, 0, 0); cudaError_t e3 = cudaLaunchKernel((const void*)klb128, dim3(1), dim3(64, 2), args, 0, 0); cudaError_t e4 = cudaLaunchKernel((const void*)klb128, dim3(1), dim3(64, 3), args, 0, 0); cudaGetLastError(); note("128 refused %d, 129 refused %d, 64x2 refused %d, 64x3 refused %d", e1 != cudaSuccess, e2 != cudaSuccess, e3 != cudaSuccess, e4 != cudaSuccess); return cudaSuccess; });
  // The occupancy calls follow the limit.
  C("cudaOccupancyMaxActiveBlocksPerMultiprocessor, dynamic shared memory past the limit", { int n[5] = {-9, -9, -9, -9, -9}; const size_t dyn[5] = {0, 49152, 49153, 65536, 101376}; cudaError_t e = cudaSuccess; for (int i = 0; i < 5; ++i) { cudaError_t x = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n[i], (const void*)kdyn2, 128, dyn[i]); if (x != cudaSuccess) e = x; } note("blocks at 0, 49152, 49153, 65536, 101376: %d %d %d %d %d", n[0], n[1], n[2], n[3], n[4]); return e; });
  C("cudaOccupancyMaxActiveBlocksPerMultiprocessor, opted in to 60000", { cudaFuncSetAttribute((const void*)kdyn2, cudaFuncAttributeMaxDynamicSharedMemorySize, 60000); int n[5] = {-9, -9, -9, -9, -9}; const size_t dyn[5] = {0, 49152, 49153, 65536, 101376}; cudaError_t e = cudaSuccess; for (int i = 0; i < 5; ++i) { cudaError_t x = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n[i], (const void*)kdyn2, 128, dyn[i]); if (x != cudaSuccess) e = x; } note("blocks at 0, 49152, 49153, 65536, 101376: %d %d %d %d %d", n[0], n[1], n[2], n[3], n[4]); return e; });
  C("cudaOccupancyAvailableDynamicSMemPerBlock, opted in to 60000", { cudaFuncSetAttribute((const void*)kdyn2, cudaFuncAttributeMaxDynamicSharedMemorySize, 60000); size_t a[3] = {77, 77, 77}; cudaError_t e = cudaSuccess; const int blocks[3] = {1, 2, 4}; for (int i = 0; i < 3; ++i) { cudaError_t x = cudaOccupancyAvailableDynamicSMemPerBlock(&a[i], (const void*)kdyn2, blocks[i], 128); if (x != cudaSuccess) e = x; } note("1, 2, 4 blocks: %zu %zu %zu", a[0], a[1], a[2]); return e; });
  C("cudaOccupancyAvailableDynamicSMemPerBlock, not opted in", { size_t a[3] = {77, 77, 77}; cudaError_t e = cudaSuccess; const int blocks[3] = {1, 2, 4}; for (int i = 0; i < 3; ++i) { cudaError_t x = cudaOccupancyAvailableDynamicSMemPerBlock(&a[i], (const void*)kdyn2, blocks[i], 128); if (x != cudaSuccess) e = x; } note("1, 2, 4 blocks: %zu %zu %zu", a[0], a[1], a[2]); return e; });
  C("cudaOccupancyAvailableDynamicSMemPerBlock, opted in to the most", { cudaFuncSetAttribute((const void*)kdyn2, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376); size_t a[3] = {77, 77, 77}; cudaError_t e = cudaSuccess; const int blocks[3] = {1, 2, 4}; for (int i = 0; i < 3; ++i) { cudaError_t x = cudaOccupancyAvailableDynamicSMemPerBlock(&a[i], (const void*)kdyn2, blocks[i], 128); if (x != cudaSuccess) e = x; } note("1, 2, 4 blocks: %zu %zu %zu", a[0], a[1], a[2]); return e; });
  // ---- NvSci, the export table
  C("cudaDeviceGetNvSciSyncAttributes", { char b[64] = {}; return cudaDeviceGetNvSciSyncAttributes(b, 0, 0); });
  C("cudaDeviceGetNvSciSyncAttributes null", { return cudaDeviceGetNvSciSyncAttributes(nullptr, 0, 0); });
  C("cudaDeviceGetNvSciSyncAttributes device 9", { char b[64]; return cudaDeviceGetNvSciSyncAttributes(b, 9, 0); });
  C("cudaGetExportTable", { const void* t = nullptr; cudaUUID_t u; std::memset(&u, 0, sizeof u); return cudaGetExportTable(&t, &u); });
  C("cudaGetExportTable null table", { cudaUUID_t u; std::memset(&u, 0, sizeof u); return cudaGetExportTable(nullptr, &u); });
  C("cudaGetExportTable null id", { const void* t; return cudaGetExportTable(&t, nullptr); });
  // ---- a kernel as a CUfunction
  C("cudaGetFuncBySymbol", { cudaFunction_t f = nullptr, g = nullptr, h = nullptr; cudaError_t e = cudaGetFuncBySymbol(&f, (const void*)kern); cudaGetFuncBySymbol(&g, (const void*)kern); cudaGetFuncBySymbol(&h, (const void*)other); note("set %d, same again %d, other kernel differs %d", f != nullptr, f == g, f != h); return e; });
  C("cudaGetFuncBySymbol null result", { return cudaGetFuncBySymbol(nullptr, (const void*)kern); });
  C("cudaGetFuncBySymbol null symbol", { cudaFunction_t f = (cudaFunction_t)0x1234; cudaError_t e = cudaGetFuncBySymbol(&f, nullptr); note("untouched %d", f == (cudaFunction_t)0x1234); return e; });
  C("cudaGetFuncBySymbol a host function", { cudaFunction_t f = (cudaFunction_t)0x1234; cudaError_t e = cudaGetFuncBySymbol(&f, (const void*)hostfn); note("untouched %d", f == (cudaFunction_t)0x1234); return e; });
  C("cudaGetFuncBySymbol a variable", { int junk; cudaFunction_t f = (cudaFunction_t)0x1234; return cudaGetFuncBySymbol(&f, (const void*)&junk); });
  C("cudaGetFuncBySymbol, then cudaFuncGetAttributes of the kernel", { cudaFunction_t f = nullptr; cudaGetFuncBySymbol(&f, (const void*)kern); cudaFuncAttributes fa; cudaError_t e = cudaFuncGetAttributes(&fa, (const void*)kern); note("static shared %zu", fa.sharedSizeBytes); return e; });
  // ---- arrays
  C("cudaArrayGetMemoryRequirements", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMallocArray(&a, &f, 64, 16, 0); cudaArrayMemoryRequirements r; return cudaArrayGetMemoryRequirements(&r, a, 0); });
  C("cudaArrayGetMemoryRequirements null result", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMallocArray(&a, &f, 64, 16, 0); return cudaArrayGetMemoryRequirements(nullptr, a, 0); });
  C("cudaArrayGetMemoryRequirements null array", { cudaArrayMemoryRequirements r; return cudaArrayGetMemoryRequirements(&r, nullptr, 0); });
  C("cudaArrayGetMemoryRequirements device 5", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMallocArray(&a, &f, 64, 16, 0); cudaArrayMemoryRequirements r; return cudaArrayGetMemoryRequirements(&r, a, 5); });
  C("cudaMipmappedArrayGetMemoryRequirements", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaMipmappedArray_t m; cudaMallocMipmappedArray(&m, &f, make_cudaExtent(64, 64, 0), 4, 0); cudaArrayMemoryRequirements r; return cudaMipmappedArrayGetMemoryRequirements(&r, m, 0); });
  C("cudaMipmappedArrayGetMemoryRequirements null", { cudaArrayMemoryRequirements r; return cudaMipmappedArrayGetMemoryRequirements(&r, nullptr, 0); });
  C("cudaArrayGetSparseProperties", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMallocArray(&a, &f, 64, 16, 0); cudaArraySparseProperties p; return cudaArrayGetSparseProperties(&p, a); });
  C("cudaArrayGetSparseProperties null result", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMallocArray(&a, &f, 64, 16, 0); return cudaArrayGetSparseProperties(nullptr, a); });
  C("cudaArrayGetSparseProperties null array", { cudaArraySparseProperties p; return cudaArrayGetSparseProperties(&p, nullptr); });
  C("cudaMipmappedArrayGetSparseProperties", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaMipmappedArray_t m; cudaMallocMipmappedArray(&m, &f, make_cudaExtent(64, 64, 0), 4, 0); cudaArraySparseProperties p; return cudaMipmappedArrayGetSparseProperties(&p, m); });
  C("cudaMipmappedArrayGetSparseProperties null", { cudaArraySparseProperties p; return cudaMipmappedArrayGetSparseProperties(&p, nullptr); });
  C("cudaArrayGetPlane", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a, pl = nullptr; cudaMallocArray(&a, &f, 64, 16, 0); return cudaArrayGetPlane(&pl, a, 0); });
  C("cudaArrayGetPlane plane 1", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a, pl = nullptr; cudaMallocArray(&a, &f, 64, 16, 0); return cudaArrayGetPlane(&pl, a, 1); });
  C("cudaArrayGetPlane null result", { cudaChannelFormatDesc f = cudaCreateChannelDesc<float>(); cudaArray_t a; cudaMallocArray(&a, &f, 64, 16, 0); return cudaArrayGetPlane(nullptr, a, 0); });
  C("cudaArrayGetPlane null array", { cudaArray_t pl; return cudaArrayGetPlane(&pl, nullptr, 0); });
  // ---- graph copies by symbol
  C("cudaGraphAddMemcpyNodeToSymbol, then FromSymbol, launched", {
    cudaGraph_t g; cudaGraphNode_t n1, n2; int host[16] = {1, 2, 3, 4}, out[16] = {};
    cudaGraphCreate(&g, 0);
    cudaError_t e = cudaGraphAddMemcpyNodeToSymbol(&n1, g, nullptr, 0, sym, host, 64, 0, cudaMemcpyHostToDevice);
    cudaError_t e2 = cudaGraphAddMemcpyNodeFromSymbol(&n2, g, &n1, 1, out, sym, 64, 0, cudaMemcpyDeviceToHost);
    cudaGraphExec_t ge; cudaGraphInstantiate(&ge, g, 0); cudaGraphLaunch(ge, 0); cudaDeviceSynchronize();
    int host2[16] = {9, 8, 7, 6};
    cudaError_t e3 = cudaGraphMemcpyNodeSetParamsToSymbol(n1, sym, host2, 64, 0, cudaMemcpyHostToDevice);
    cudaError_t e4 = cudaGraphExecMemcpyNodeSetParamsToSymbol(ge, n1, sym, host2, 64, 0, cudaMemcpyHostToDevice);
    cudaGraphLaunch(ge, 0); cudaDeviceSynchronize();
    int out2[4] = {out[0], out[1], out[2], out[3]};
    note("add %s, %s; set %s, exec set %s; first launch %d %d, after %d %d", cudaGetErrorName(e), cudaGetErrorName(e2), cudaGetErrorName(e3), cudaGetErrorName(e4), 1 * (out2[0] == 9) , 1 * (out2[3] == 6), out2[0], out2[3]);
    return cudaSuccess; });
  C("cudaGraphAddMemcpyNodeToSymbol null symbol", { cudaGraph_t g; cudaGraphNode_t n; int host[16]; cudaGraphCreate(&g, 0); return cudaGraphAddMemcpyNodeToSymbol(&n, g, nullptr, 0, nullptr, host, 64, 0, cudaMemcpyHostToDevice); });
  C("cudaGraphAddMemcpyNodeToSymbol offset 8", { cudaGraph_t g; cudaGraphNode_t n; int host[16]; cudaGraphCreate(&g, 0); return cudaGraphAddMemcpyNodeToSymbol(&n, g, nullptr, 0, sym, host, 64, 8, cudaMemcpyHostToDevice); });
  C("cudaGraphAddMemcpyNodeToSymbol offset 64", { cudaGraph_t g; cudaGraphNode_t n; int host[16]; cudaGraphCreate(&g, 0); return cudaGraphAddMemcpyNodeToSymbol(&n, g, nullptr, 0, sym, host, 64, 64, cudaMemcpyHostToDevice); });
  C("cudaGraphAddMemcpyNodeToSymbol device to host", { cudaGraph_t g; cudaGraphNode_t n; int host[16]; cudaGraphCreate(&g, 0); return cudaGraphAddMemcpyNodeToSymbol(&n, g, nullptr, 0, sym, host, 64, 0, cudaMemcpyDeviceToHost); });
  C("cudaGraphAddMemcpyNodeToSymbol default kind", { cudaGraph_t g; cudaGraphNode_t n; int host[16]; cudaGraphCreate(&g, 0); return cudaGraphAddMemcpyNodeToSymbol(&n, g, nullptr, 0, sym, host, 64, 0, cudaMemcpyDefault); });
  C("cudaGraphAddMemcpyNodeToSymbol null node", { cudaGraph_t g; int host[16]; cudaGraphCreate(&g, 0); return cudaGraphAddMemcpyNodeToSymbol(nullptr, g, nullptr, 0, sym, host, 64, 0, cudaMemcpyHostToDevice); });
  C("cudaGraphMemcpyNodeSetParamsToSymbol null node", { int host[16]; return cudaGraphMemcpyNodeSetParamsToSymbol(nullptr, sym, host, 64, 0, cudaMemcpyHostToDevice); });
  C("cudaGraphExecMemcpyNodeSetParamsToSymbol null exec", { int host[16]; cudaGraph_t g; cudaGraphNode_t n; cudaGraphCreate(&g, 0); cudaGraphAddMemcpyNodeToSymbol(&n, g, nullptr, 0, sym, host, 64, 0, cudaMemcpyHostToDevice); return cudaGraphExecMemcpyNodeSetParamsToSymbol(nullptr, n, sym, host, 64, 0, cudaMemcpyHostToDevice); });
  // ---- memory pools by location
  C13("cudaMemGetDefaultMemPool", { cudaMemPool_t p = nullptr, d = nullptr; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; cudaDeviceGetDefaultMemPool(&d, 0); cudaError_t e = cudaMemGetDefaultMemPool(&p, &l, cudaMemAllocationTypePinned); note("is the device's default %d", p == d); return e; });
  C13("cudaMemGetMemPool", { cudaMemPool_t p = nullptr, d = nullptr; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; cudaDeviceGetMemPool(&d, 0); cudaError_t e = cudaMemGetMemPool(&p, &l, cudaMemAllocationTypePinned); note("is the device's pool %d", p == d); return e; });
  C13("cudaMemGetDefaultMemPool invalid location", { cudaMemPool_t p; cudaMemLocation l{cudaMemLocationTypeInvalid, 0}; return cudaMemGetDefaultMemPool(&p, &l, cudaMemAllocationTypePinned); });
  C13("cudaMemGetDefaultMemPool device 9", { cudaMemPool_t p; cudaMemLocation l{cudaMemLocationTypeDevice, 9}; return cudaMemGetDefaultMemPool(&p, &l, cudaMemAllocationTypePinned); });
  C13("cudaMemGetDefaultMemPool invalid type", { cudaMemPool_t p; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; return cudaMemGetDefaultMemPool(&p, &l, cudaMemAllocationTypeInvalid); });
  C13("cudaMemGetDefaultMemPool managed", { cudaMemPool_t p; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; return cudaMemGetDefaultMemPool(&p, &l, cudaMemAllocationTypeManaged); });
  C13("cudaMemGetDefaultMemPool null pool", { cudaMemLocation l{cudaMemLocationTypeDevice, 0}; return cudaMemGetDefaultMemPool(nullptr, &l, cudaMemAllocationTypePinned); });
  C13("cudaMemGetDefaultMemPool null location", { cudaMemPool_t p; return cudaMemGetDefaultMemPool(&p, nullptr, cudaMemAllocationTypePinned); });
  C13("cudaMemSetMemPool, then cudaDeviceGetMemPool", { cudaMemPoolProps pp; std::memset(&pp, 0, sizeof pp); pp.allocType = cudaMemAllocationTypePinned; pp.location.type = cudaMemLocationTypeDevice; pp.location.id = 0; cudaMemPool_t mine, cur; cudaMemPoolCreate(&mine, &pp); cudaMemLocation l{cudaMemLocationTypeDevice, 0}; cudaError_t e = cudaMemSetMemPool(&l, cudaMemAllocationTypePinned, mine); cudaDeviceGetMemPool(&cur, 0); note("the device's pool is now mine %d", cur == mine); return e; });
  C13("cudaMemSetMemPool null pool", { cudaMemLocation l{cudaMemLocationTypeDevice, 0}; return cudaMemSetMemPool(&l, cudaMemAllocationTypePinned, nullptr); });
  C13("cudaMemSetMemPool null location", { cudaMemPool_t d; cudaDeviceGetDefaultMemPool(&d, 0); return cudaMemSetMemPool(nullptr, cudaMemAllocationTypePinned, d); });
  C13("cudaMemSetMemPool device 9", { cudaMemPool_t d; cudaMemLocation l{cudaMemLocationTypeDevice, 9}; cudaDeviceGetDefaultMemPool(&d, 0); return cudaMemSetMemPool(&l, cudaMemAllocationTypePinned, d); });
  // ---- one call to change any node
  C128("cudaGraphNodeSetParams null node", { alignas(cudaGraphNodeParams) char b[sizeof(cudaGraphNodeParams)]; std::memset(b, 0, sizeof b); reinterpret_cast<cudaGraphNodeParams*>(b)->type = cudaGraphNodeTypeMemcpy; return cudaGraphNodeSetParams(nullptr, reinterpret_cast<cudaGraphNodeParams*>(b)); });
  C128("cudaGraphNodeSetParams null parameters", { cudaGraph_t g; cudaGraphNode_t n; cudaGraphCreate(&g, 0); cudaGraphAddEmptyNode(&n, g, nullptr, 0); return cudaGraphNodeSetParams(n, nullptr); });
  C128("cudaGraphNodeSetParams unknown type", { cudaGraph_t g; cudaGraphNode_t n; cudaGraphCreate(&g, 0); cudaGraphAddEmptyNode(&n, g, nullptr, 0); alignas(cudaGraphNodeParams) char b[sizeof(cudaGraphNodeParams)]; std::memset(b, 0, sizeof b); reinterpret_cast<cudaGraphNodeParams*>(b)->type = (cudaGraphNodeType)99; return cudaGraphNodeSetParams(n, reinterpret_cast<cudaGraphNodeParams*>(b)); });
  C128("cudaGraphExecNodeSetParams null exec", { cudaGraph_t g; cudaGraphNode_t n; cudaGraphCreate(&g, 0); cudaGraphAddEmptyNode(&n, g, nullptr, 0); alignas(cudaGraphNodeParams) char b[sizeof(cudaGraphNodeParams)]; std::memset(b, 0, sizeof b); reinterpret_cast<cudaGraphNodeParams*>(b)->type = cudaGraphNodeTypeMemcpy; return cudaGraphExecNodeSetParams(nullptr, n, reinterpret_cast<cudaGraphNodeParams*>(b)); });
  // ---- the log
  C13("cudaLogsCurrent", { cudaLogIterator it = 777; cudaError_t e = cudaLogsCurrent(&it, 0); note("it=%u", it); return e; });
  C13("cudaLogsCurrent null", { return cudaLogsCurrent(nullptr, 0); });
  C13("cudaLogsDumpToMemory", { cudaLogIterator it = 0; char b[64]; size_t sz = 64; cudaError_t e = cudaLogsDumpToMemory(&it, b, &sz, 0); note("size %zu", sz); return e; });
  C13("cudaLogsDumpToMemory null buffer", { cudaLogIterator it = 0; size_t sz = 64; return cudaLogsDumpToMemory(&it, nullptr, &sz, 0); });
  C13("cudaLogsDumpToMemory null size", { cudaLogIterator it = 0; char b[64]; return cudaLogsDumpToMemory(&it, b, nullptr, 0); });
  C13("cudaLogsDumpToFile", { cudaLogIterator it = 0; std::remove("cuda_logs_sweep.txt"); cudaError_t e = cudaLogsDumpToFile(&it, "cuda_logs_sweep.txt", 0); std::ifstream f("cuda_logs_sweep.txt"); std::string s((std::istreambuf_iterator<char>(f)), {}); note("file made %d, %zu bytes", (int)f.good(), s.size()); return e; });
  C13("cudaLogsDumpToFile null path", { cudaLogIterator it = 0; return cudaLogsDumpToFile(&it, nullptr, 0); });
  C13("cudaLogsRegisterCallback, then cudaLogsUnregisterCallback", { cudaLogsCallbackHandle h = nullptr; cudaError_t e = cudaLogsRegisterCallback([](void*, cudaLogLevel, char*, size_t) {}, nullptr, &h); note("handle set %d, unregister %s", h != nullptr, cudaGetErrorName(cudaLogsUnregisterCallback(h))); return e; });
  C13("cudaLogsRegisterCallback null callback", { cudaLogsCallbackHandle h; return cudaLogsRegisterCallback(nullptr, nullptr, &h); });
  C13("cudaLogsUnregisterCallback null", { return cudaLogsUnregisterCallback(nullptr); });
  // ---- notifications about the device
  C128("cudaDeviceRegisterAsyncNotification, then Unregister", { cudaAsyncCallbackHandle_t h = nullptr; cudaError_t e = cudaDeviceRegisterAsyncNotification(0, [](cudaAsyncNotificationInfo_t*, void*, cudaAsyncCallbackHandle_t) {}, nullptr, &h); note("handle set %d, unregister %s", h != nullptr, cudaGetErrorName(cudaDeviceUnregisterAsyncNotification(0, h))); return e; });
  C128("cudaDeviceRegisterAsyncNotification null callback", { cudaAsyncCallbackHandle_t h; return cudaDeviceRegisterAsyncNotification(0, nullptr, nullptr, &h); });
  C128("cudaDeviceRegisterAsyncNotification device 9", { cudaAsyncCallbackHandle_t h; return cudaDeviceRegisterAsyncNotification(9, [](cudaAsyncNotificationInfo_t*, void*, cudaAsyncCallbackHandle_t) {}, nullptr, &h); });
  C128("cudaDeviceUnregisterAsyncNotification null", { return cudaDeviceUnregisterAsyncNotification(0, nullptr); });
  // ---- atomics over a link
  C13("cudaDeviceGetHostAtomicCapabilities", { unsigned caps[3] = {9, 9, 9}; cudaAtomicOperation ops[3] = {cudaAtomicOperationIntegerAdd, cudaAtomicOperationFloatAdd, cudaAtomicOperationCAS}; cudaError_t e = cudaDeviceGetHostAtomicCapabilities(caps, ops, 3, 0); note("%u %u %u", caps[0], caps[1], caps[2]); return e; });
  C13("cudaDeviceGetHostAtomicCapabilities bad operation", { unsigned caps[1]; cudaAtomicOperation ops[1] = {(cudaAtomicOperation)77}; return cudaDeviceGetHostAtomicCapabilities(caps, ops, 1, 0); });
  C13("cudaDeviceGetHostAtomicCapabilities null", { cudaAtomicOperation ops[1] = {cudaAtomicOperationIntegerAdd}; return cudaDeviceGetHostAtomicCapabilities(nullptr, ops, 1, 0); });
  C13("cudaDeviceGetHostAtomicCapabilities count 0", { unsigned caps[1]; cudaAtomicOperation ops[1] = {cudaAtomicOperationIntegerAdd}; return cudaDeviceGetHostAtomicCapabilities(caps, ops, 0, 0); });
  C13("cudaDeviceGetHostAtomicCapabilities device 7", { unsigned caps[1]; cudaAtomicOperation ops[1] = {cudaAtomicOperationIntegerAdd}; return cudaDeviceGetHostAtomicCapabilities(caps, ops, 1, 7); });
  C13("cudaDeviceGetP2PAtomicCapabilities 1->0", { unsigned caps[3] = {9, 9, 9}; cudaAtomicOperation ops[3] = {cudaAtomicOperationIntegerAdd, cudaAtomicOperationFloatAdd, cudaAtomicOperationCAS}; cudaError_t e = cudaDeviceGetP2PAtomicCapabilities(caps, ops, 3, 1, 0); note("%u %u %u", caps[0], caps[1], caps[2]); return e; });
  C13("cudaDeviceGetP2PAtomicCapabilities 0->0", { unsigned caps[1]; cudaAtomicOperation ops[1] = {cudaAtomicOperationIntegerAdd}; return cudaDeviceGetP2PAtomicCapabilities(caps, ops, 1, 0, 0); });
  C13("cudaDeviceGetP2PAtomicCapabilities 0->7", { unsigned caps[1]; cudaAtomicOperation ops[1] = {cudaAtomicOperationIntegerAdd}; return cudaDeviceGetP2PAtomicCapabilities(caps, ops, 1, 0, 7); });
  C13("cudaDeviceGetP2PAtomicCapabilities null", { cudaAtomicOperation ops[1] = {cudaAtomicOperationIntegerAdd}; return cudaDeviceGetP2PAtomicCapabilities(nullptr, ops, 1, 0, 1); });
  // ---- libraries (the cubin is built with the program)
  C128("cudaLibraryLoadFromFile", { cudaLibrary_t l = nullptr; cudaError_t e = cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); note("loaded %d", l != nullptr); return e; });
  C128("cudaLibraryLoadFromFile missing", { cudaLibrary_t l; return cudaLibraryLoadFromFile(&l, "/nonexistent.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C128("cudaLibraryLoadFromFile null library", { return cudaLibraryLoadFromFile(nullptr, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C128("cudaLibraryLoadFromFile null path", { cudaLibrary_t l; return cudaLibraryLoadFromFile(&l, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0); });
  C128("cudaLibraryGetKernelCount", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); unsigned n = 77; cudaError_t e = cudaLibraryGetKernelCount(&n, l); note("n=%u", n); return e; });
  C128("cudaLibraryGetKernelCount null", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); return cudaLibraryGetKernelCount(nullptr, l); });
  C128("cudaLibraryGetKernelCount null library", { unsigned n; return cudaLibraryGetKernelCount(&n, nullptr); });
  C128("cudaLibraryEnumerateKernels", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); cudaKernel_t ks[8] = {}; cudaError_t e = cudaLibraryEnumerateKernels(ks, 3, l); note("%d %d %d", ks[0] != nullptr, ks[1] != nullptr, ks[2] != nullptr); return e; });
  C128("cudaLibraryEnumerateKernels null", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); return cudaLibraryEnumerateKernels(nullptr, 3, l); });
  C128("cudaLibraryEnumerateKernels null library", { cudaKernel_t ks[8]; return cudaLibraryEnumerateKernels(ks, 3, nullptr); });
  C128("cudaLibraryGetGlobal", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p = nullptr; size_t n = 77; cudaError_t e = cudaLibraryGetGlobal(&p, &n, l, "gvar"); note("bytes %zu, address set %d", n, p != nullptr); return e; });
  C128("cudaLibraryGetGlobal missing", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p; size_t n; return cudaLibraryGetGlobal(&p, &n, l, "nosuch"); });
  C128("cudaLibraryGetGlobal a managed variable", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p; size_t n; return cudaLibraryGetGlobal(&p, &n, l, "mvar"); });
  C128("cudaLibraryGetGlobal a kernel", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p; size_t n; return cudaLibraryGetGlobal(&p, &n, l, "k1"); });
  C128("cudaLibraryGetGlobal null name", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p; size_t n; return cudaLibraryGetGlobal(&p, &n, l, nullptr); });
  C128("cudaLibraryGetGlobal null library", { void* p; size_t n; return cudaLibraryGetGlobal(&p, &n, nullptr, "gvar"); });
  C128("cudaLibraryGetManaged", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p = nullptr; size_t n = 77; cudaError_t e = cudaLibraryGetManaged(&p, &n, l, "mvar"); note("bytes %zu, address set %d", n, p != nullptr); return e; });
  C128("cudaLibraryGetManaged a plain variable", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* p; size_t n; return cudaLibraryGetManaged(&p, &n, l, "gvar"); });
  C128("cudaLibraryGetUnifiedFunction", { cudaLibrary_t l; cudaLibraryLoadFromFile(&l, "libk.cubin", nullptr, nullptr, 0, nullptr, nullptr, 0); void* f; return cudaLibraryGetUnifiedFunction(&f, l, "k1"); });
  C128("cudaLibraryGetUnifiedFunction null library", { void* f; return cudaLibraryGetUnifiedFunction(&f, nullptr, "k1"); });
  // ---- copies in batches
  C128("cudaGetKernel", { cudaKernel_t k = nullptr; cudaError_t e = cudaGetKernel(&k, (const void*)kern); note("set %d", k != nullptr); return e; });
  C128("cudaGetKernel null", { return cudaGetKernel(nullptr, (const void*)kern); });
#if CUDART_VERSION >= 12080
  C128("cudaMemcpyBatchAsync", { char *d1, *d2, *h1; cudaMalloc(&d1, 4096); cudaMalloc(&d2, 4096); cudaMallocHost(&h1, 4096); for (int i = 0; i < 4096; ++i) h1[i] = (char)i; cudaStream_t s; cudaStreamCreate(&s);
    void* dsts[2] = {d1, d2}; const void* srcs[2] = {h1, h1 + 1024}; size_t sizes[2] = {1024, 512}; size_t idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; size_t fail = 77;
#if CUDART_VERSION >= 13000
    cudaError_t e = cudaMemcpyBatchAsync(dsts, srcs, sizes, 2, &at, idx, 1, s);
#else
    cudaError_t e = cudaMemcpyBatchAsync(dsts, const_cast<void**>(srcs), sizes, 2, &at, idx, 1, &fail, s);
#endif
    cudaStreamSynchronize(s); char b1[1024], b2[512]; cudaMemcpy(b1, d1, 1024, cudaMemcpyDeviceToHost); cudaMemcpy(b2, d2, 512, cudaMemcpyDeviceToHost); (void)fail; note("copies right %d", b1[5] == 5 && b2[5] == (char)(1024 + 5)); return e; });
#if CUDART_VERSION >= 13000
#define BATCH(dsts, srcs, sizes, count, at, idx, num, stream) cudaMemcpyBatchAsync(dsts, srcs, sizes, count, at, idx, num, stream)
#else
#define BATCH(dsts, srcs, sizes, count, at, idx, num, stream) cudaMemcpyBatchAsync(dsts, const_cast<void**>(srcs), sizes, count, at, idx, num, &fail_, stream)
#endif
  C128("cudaMemcpyBatchAsync count 0", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; return BATCH(d, s, z, 0, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync null destinations", { size_t fail_; char b[16]; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; return BATCH(nullptr, s, z, 1, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync null attributes", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; return BATCH(d, s, z, 1, nullptr, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync attribute count 0", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; return BATCH(d, s, z, 1, &at, idx, 0, stream_made()); });
  C128("cudaMemcpyBatchAsync attribute index 1", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {1}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; return BATCH(d, s, z, 1, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync invalid access order", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); return BATCH(d, s, z, 1, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync access order 9", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = (cudaMemcpySrcAccessOrder)9; return BATCH(d, s, z, 1, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync flags 0x80", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {1}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; at.flags = 0x80; return BATCH(d, s, z, 1, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync a size of 0", { size_t fail_; char b[16]; void* d[1] = {b}; const void* s[1] = {b}; size_t z[1] = {0}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; return BATCH(d, s, z, 1, &at, idx, 1, stream_made()); });
  C128("cudaMemcpyBatchAsync pageable to device, then back", { size_t fail_; static char pageable[256], back[256]; for (int i = 0; i < 256; ++i) pageable[i] = (char)(i * 3); char* d; cudaMalloc(&d, 256); size_t z[1] = {256}, idx[1] = {0}; cudaMemcpyAttributes at; std::memset(&at, 0, sizeof at); at.srcAccessOrder = cudaMemcpySrcAccessOrderAny; void* dd[1] = {d}; const void* ss[1] = {pageable};
    cudaStream_t st = stream_made(); cudaError_t e = BATCH(dd, ss, z, 1, &at, idx, 1, st); cudaStreamSynchronize(st); at.srcAccessOrder = cudaMemcpySrcAccessOrderStream; void* bd[1] = {back}; const void* bs[1] = {d}; cudaError_t e2 = BATCH(bd, bs, z, 1, &at, idx, 1, st); cudaStreamSynchronize(st); note("back %s, round trip right %d", cudaGetErrorName(e2), back[5] == (char)15); return e; });
#undef BATCH
#endif
#if CUDART_VERSION >= 13000
  C13("cudaMemPrefetchBatchAsync", { void* m; cudaMallocManaged(&m, 1 << 20); void* dp[1] = {m}; size_t sz[1] = {1 << 20}; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; size_t li[1] = {0}; return cudaMemPrefetchBatchAsync(dp, sz, 1, &l, li, 1, 0, stream_made()); });
  C13("cudaMemPrefetchBatchAsync count 0", { void* m; cudaMallocManaged(&m, 1 << 20); void* dp[1] = {m}; size_t sz[1] = {1 << 20}; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; size_t li[1] = {0}; return cudaMemPrefetchBatchAsync(dp, sz, 0, &l, li, 1, 0, stream_made()); });
  C13("cudaMemPrefetchBatchAsync flags 1", { void* m; cudaMallocManaged(&m, 1 << 20); void* dp[1] = {m}; size_t sz[1] = {1 << 20}; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; size_t li[1] = {0}; return cudaMemPrefetchBatchAsync(dp, sz, 1, &l, li, 1, 1, stream_made()); });
  C13("cudaMemDiscardBatchAsync", { void* m; cudaMallocManaged(&m, 1 << 20); void* dp[1] = {m}; size_t sz[1] = {1 << 20}; return cudaMemDiscardBatchAsync(dp, sz, 1, 0, stream_made()); });
  C13("cudaMemDiscardBatchAsync null", { size_t sz[1] = {1 << 20}; return cudaMemDiscardBatchAsync(nullptr, sz, 1, 0, stream_made()); });
  C13("cudaMemDiscardAndPrefetchBatchAsync", { void* m; cudaMallocManaged(&m, 1 << 20); void* dp[1] = {m}; size_t sz[1] = {1 << 20}; cudaMemLocation l{cudaMemLocationTypeDevice, 0}; size_t li[1] = {0}; return cudaMemDiscardAndPrefetchBatchAsync(dp, sz, 1, &l, li, 1, 0, stream_made()); });
#endif

  int bad = 0;
  for (const Case& c : cases) {
    std::fflush(stdout);
    const pid_t k = fork();
    if (k == 0) {
      g_notes.clear();
      const cudaError_t e = c.run();
      std::printf("%s%s: %s%s%s\n", c.label, c.tag, cudaGetErrorName(e), g_notes.empty() ? "" : " -- ", g_notes.c_str());
      std::fflush(stdout);
      _exit(0);
    }
    int status = 0;
    waitpid(k, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      std::printf("%s%s: CRASHED (signal %d)\n", c.label, c.tag, WIFSIGNALED(status) ? WTERMSIG(status) : 0);
      ++bad;
    }
  }
  std::printf("%s: %zu cases\n", bad ? "FAIL" : "PASS", cases.size());
  return bad != 0;
}
