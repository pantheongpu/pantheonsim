// The program run_host_gpu_stub.sh runs with a stand-in driver (host_gpu_stub_driver.c): `big` launches a kernel of the exact set (integer arithmetic) on a grid
// worth a card, `small` the same kernel on a few warps, `mufu` a kernel outside the set (expf); `div`, `rcp`, `divexp` an IEEE division, a bare rcp.approx and a division with an expf (the division idiom: in the set
// only with VGPU_HOST_GPU_DIV=1, and only when it is the sole thing outside it). It prints the results on stdout, which must not depend on the
// host-GPU mode, and on stderr how long the first launch took in milliseconds ("first_launch_ms=").
// For turbo mode (VGPU_HOST_GPU=turbo, docs/perf-design/12-host-gpu-turbo.md): `iatom` an integer atomicAdd whose old value is used (admitted),
// `fatom` a float atomicAdd whose old value is used and `cas` a compare-and-swap (both only with VGPU_HOST_GPU_ATOMICS=1). Each thread's result
// does not depend on the order the threads run in, so these print the same lines wherever they run. `hist` is an integer reduction onto 64 words (the
// exact set), for the verify run of step 16: its sums are the same in any order, unless an update is lost. `none` is `acct` with no launch at all: what a
// card that computes nothing leaves (step 18: a sampled launch sent to such a card must print exactly this).
// `few` is `big` on 8 blocks with a long loop in every thread (Round sp61: a grid too small to sample by whole blocks, sampled inside its blocks),
// `fewshort` the same with a short loop, and `fewnone` `few` with no launch (steps 21 to 23).
// `streak` (Round sp65, steps 24 to 26) alternates `scramble` over 4 MiB with `bump`, one warp adding 1 to 32 counters, eight times each, and reads the
// memory back only at the end: once `scramble` runs on the card, the card holds the whole live memory when each `bump` comes, and `bump` (far too
// short to repay a card trip by itself) goes there too. `streakread` reads both buffers back after every `scramble`, so the card holds nothing the
// simulator does not when `bump` comes. Both print the counters and a hash of the 4 MiB.
// `zerogrid` (step 27) launches `scramble` four times over 4 MiB and then on a grid of 0 blocks, while the card holds the memory: the last launch must
// get the simulator's own answer (cudaErrorInvalidConfiguration), as with the mode off, and never reach the card.
// `sqrt`, `rsqrt`, `sqrtdiv` (Round sp68, step 29): an IEEE sqrtf on inputs that take both of its paths (tiny floats the slow path, negatives a NaN), a bare
// rsqrtf, and sqrtf followed by a division (the square root idiom: in the set only with VGPU_HOST_GPU_SQRT=1; the last also needs VGPU_HOST_GPU_DIV=1).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

__global__ void scramble(uint32_t* d, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = d[i] ^ (uint32_t(i) * 2654435761u);
  for (int r = 0; r < rounds; ++r) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x += uint32_t(r) * 0x9e3779b9u;
  }
  d[i] = x;
}
// `scramble` with its loop kept whole (`few`, Round sp61): nvcc unrolls `scramble`'s loop by 4 and counts the unrolled part through a register it adds
// to the counter, which the in-block sample does not follow (it follows a counter compared to a bound, plus a constant at most).
__global__ void scramble1(uint32_t* d, int n, int rounds) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t x = d[i] ^ (uint32_t(i) * 2654435761u);
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x += uint32_t(r) * 0x9e3779b9u;
  }
  d[i] = x;
}
__global__ void bump(uint32_t* c) { c[threadIdx.x] += 1; }
__global__ void use_expf(float* d, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = expf(d[i] * 0.001f) + float(i);
}

__global__ void divk(float* d, int n) {   // IEEE division: MUFU.RCP, FFMAs, FCHK, a call to the slow path (the division idiom)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = (d[i] * 1.0001f + 1.5f) / (float(i % 977) + 1.25f);
}
__global__ void rcpk(float* d, int n) {   // a bare rcp.approx: MUFU.RCP with nothing that hides its last bit
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(float(i % 977) + 1.25f));
    d[i] = r + d[i];
  }
}
__global__ void divexp(float* d, int n) {   // the division idiom and an EX2 in one kernel
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = expf(d[i] * 0.001f) / (float(i % 977) + 1.25f);
}

__global__ void sqrtk(float* d, int n) {   // IEEE sqrtf: MUFU.RSQ, the range check, two FMULs and two FFMAs, a call to the slow path (the square root idiom)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = (i & 1) ? d[i] : d[i] * 1.0001f + float(i);   // memset 1: the odd elements are 2.4e-38 at first, below 2^-101: the slow path
  if (i % 97 == 0) v = -v;                                // a NaN
  d[i] = sqrtf(v);
}
__global__ void rsqrtk(float* d, int n) {   // a bare rsqrtf: MUFU.RSQ with nothing that hides its last bits
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = rsqrtf(fabsf(d[i]) + float(i) + 1.0f);
}
__global__ void sqrtdivk(float* d, int n) {   // the square root idiom and the division idiom in one kernel
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = sqrtf(fabsf(d[i]) + float(i)) / (float(i % 977) + 1.25f);
}

__global__ void iatomk(unsigned* d, int n) {   // an integer atomic with a returned value: ATOMG.ADD
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned acc = 0;
  for (unsigned r = 0; r < 16; ++r) acc += atomicAdd(&d[i], r + 1);   // its own element: the old values are the same in any order
  d[i] += acc;
}
__global__ void fatomk(float* d, int n) {   // a float atomic with a returned value (ATOMG.ADD.F32 or a compare-and-swap loop, by toolkit and architecture)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = 0;
  for (int r = 0; r < 16; ++r) acc += atomicAdd(&d[i], 1.0f);   // small integers: exact in any order
  d[i] += acc;
}
__global__ void cask(unsigned* d, int n) {   // compare-and-swap: ATOMG.CAS
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned acc = 0;
  for (unsigned r = 0; r < 16; ++r) acc += atomicCAS(&d[i], d[i], d[i] + r);
  d[i] ^= acc;
}

__global__ void histk(unsigned* d, int n) {   // an integer reduction onto 64 words that every block hits: the simulator's host threads contend for them
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (unsigned r = 0; r < 8; ++r) atomicAdd(&d[(i + r) & 63], r + 1);
}

void leak_heap();   // host_gpu_lazy_heap.cu: a kernel of a module of its own (a kernel that calls malloc keeps every kernel of its module in the simulator)

static int streak_program(bool read_between) {
  const int n = 1 << 20;
  void *d = nullptr, *c = nullptr;
  cudaMalloc(&d, n * 4);
  cudaMemset(d, 1, n * 4);
  cudaMalloc(&c, 32 * 4);
  cudaMemset(c, 0, 32 * 4);
  std::vector<unsigned char> back(n * 4);
  std::vector<uint32_t> cnt(32);
  for (int launch = 0; launch < 8; ++launch) {
    scramble<<<n / 256, 256>>>(static_cast<uint32_t*>(d), n, 64);
    if (read_between) {
      cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(cnt.data(), c, 32 * 4, cudaMemcpyDeviceToHost);
    }
    bump<<<1, 32>>>(static_cast<uint32_t*>(c));
  }
  cudaDeviceSynchronize();
  cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(cnt.data(), c, 32 * 4, cudaMemcpyDeviceToHost);
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char b : back) h = (h ^ b) * 0x100000001b3ull;
  unsigned lo = cnt[0], hi = cnt[0];
  for (uint32_t v : cnt) lo = v < lo ? v : lo, hi = v > hi ? v : hi;
  std::printf("bumps %u to %u, scrambled %016llx\n", lo, hi, static_cast<unsigned long long>(h));
  cudaFree(c);
  cudaFree(d);
  return 0;
}

static int zerogrid_program() {
  const int n = 1 << 20;
  void* d = nullptr;
  cudaMalloc(&d, n * 4);
  cudaMemset(d, 1, n * 4);
  for (int launch = 0; launch < 4; ++launch) scramble<<<n / 256, 256>>>(static_cast<uint32_t*>(d), n, 64);
  cudaDeviceSynchronize();
  scramble<<<0, 256>>>(static_cast<uint32_t*>(d), n, 64);
  const cudaError_t e = cudaGetLastError();
  cudaDeviceSynchronize();
  std::printf("zero grid: %s\n", cudaGetErrorName(e));
  cudaFree(d);
  return 0;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "big";
  if (!std::strcmp(mode, "zerogrid")) return zerogrid_program();
  if (!std::strcmp(mode, "streak") || !std::strcmp(mode, "streakread")) return streak_program(!std::strcmp(mode, "streakread"));
  const bool small = !std::strcmp(mode, "small"), mufu = !std::strcmp(mode, "mufu");
  const bool div = !std::strcmp(mode, "div"), rcp = !std::strcmp(mode, "rcp"), divexp_mode = !std::strcmp(mode, "divexp");
  const bool hist = !std::strcmp(mode, "hist");
  const bool sqrt_mode = !std::strcmp(mode, "sqrt"), rsqrt_mode = !std::strcmp(mode, "rsqrt"), sqrtdiv_mode = !std::strcmp(mode, "sqrtdiv");
  const bool iatom = !std::strcmp(mode, "iatom"), fatom = !std::strcmp(mode, "fatom"), cas = !std::strcmp(mode, "cas");
  const bool fewnone = !std::strcmp(mode, "fewnone");
  const bool none = !std::strcmp(mode, "none") || fewnone;
  const bool few = !std::strcmp(mode, "few") || !std::strcmp(mode, "fewshort") || fewnone;
  const int rounds = !std::strcmp(mode, "acct") ? 2048 : !std::strcmp(mode, "few") ? 1 << 15 : 64;   // `acct`: `big` with 32 times the work per thread, a launch of a fraction of a second (run_host_gpu_stub.sh step 8)
  const bool heap = !std::strcmp(mode, "heap");   // `big`, after a kernel left a block of the device heap in use: no launch may go to the card
  // `keep`: `big` with an 8 MiB allocation beside it that no kernel touches, and a host memset of its first 64 KiB after the second launch (run_host_gpu_stub.sh
  // step 28: what a verify run pushes again)
  const bool keep = !std::strcmp(mode, "keep");
  const int n = small ? 64 : few ? 8 * 256 : 1 << 18;
  void* d = nullptr;
  if (heap) leak_heap();
  cudaMalloc(&d, n * 4);
  cudaMemset(d, 1, n * 4);
  void* side = nullptr;
  if (keep) {
    cudaMalloc(&side, 8 << 20);
    cudaMemset(side, 2, 8 << 20);
  }
  uint64_t h = 0xcbf29ce484222325ull;
  for (int launch = 0; launch < 4; ++launch) {
    const auto t0 = std::chrono::steady_clock::now();
    if (none) {}
    else if (mufu) use_expf<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (div) divk<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (rcp) rcpk<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (divexp_mode) divexp<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (sqrt_mode) sqrtk<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (rsqrt_mode) rsqrtk<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (sqrtdiv_mode) sqrtdivk<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (hist) histk<<<(n + 255) / 256, 256>>>(static_cast<unsigned*>(d), n);
    else if (iatom) iatomk<<<(n + 255) / 256, 256>>>(static_cast<unsigned*>(d), n);
    else if (fatom) fatomk<<<(n + 255) / 256, 256>>>(static_cast<float*>(d), n);
    else if (cas) cask<<<(n + 255) / 256, 256>>>(static_cast<unsigned*>(d), n);
    else if (few) scramble1<<<(n + 255) / 256, 256>>>(static_cast<uint32_t*>(d), n, rounds);
    else scramble<<<(n + 255) / 256, 256>>>(static_cast<uint32_t*>(d), n, rounds);
    cudaDeviceSynchronize();
    if (launch == 0) std::fprintf(stderr, "first_launch_ms=%.0f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    std::vector<unsigned char> back(n * 4);
    cudaMemcpy(back.data(), d, n * 4, cudaMemcpyDeviceToHost);
    for (unsigned char c : back) h = (h ^ c) * 0x100000001b3ull;
    std::printf("launch %d: %016llx\n", launch, static_cast<unsigned long long>(h));
    if (keep && launch == 1) cudaMemset(side, 3, 64 << 10);
  }
  if (keep) {
    std::vector<unsigned char> s(8 << 20);
    cudaMemcpy(s.data(), side, s.size(), cudaMemcpyDeviceToHost);
    for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
    std::printf("side: %016llx\n", static_cast<unsigned long long>(h));
    cudaFree(side);
  }
  cudaFree(d);
  return 0;
}
