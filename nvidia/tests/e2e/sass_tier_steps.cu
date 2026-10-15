// What Runner::step_warp's straight-line loop does besides executing an instruction (src/sass/exec.cpp), which the tier's run
// loop (Runner::tier_run, src/sass/tier_ops.inc) must do the same way, because the order in which a block's warps see one
// another's memory -- and so the answer of a program with a race in it, and the instruction count of one that waits for
// another warp -- depends on it. run_sass_tier.sh runs this program with the interpreter (VGPU_TIER=0) and the tier
// (VGPU_TIER=1), with the performance counters on, and requires the same output byte for byte.
//
// The items, and the test of each (the numbers are those in the list at the top of tier_ops.inc):
//   1. The store gate: a warp's first STG or ST ends its turn, once, if the block has other warps.
//        gate_plain (STG), gate_long (the store comes more than 256 straight instructions into a run), gate_generic (ST),
//        gate_diverged (the two halves of a warp reach their stores at different times).
//   2. The step count and its limit (VGPU_MAX_STEPS): every instruction, the one that ends a run included, is a step.
//        `steps`: the smallest limit a kernel finishes under, and the text of the failure one step below it.
//   3. At most 256 straight instructions, then the next one, a call; a section's last instruction ends a run.
//        spin_long: a loop body of 700 straight instructions waits for another warp.
//   4. A call of step_warp is the unit of a warp's turn (4096 of them, then the next warp goes): the instruction count of a
//      warp that spins on another's store says how many calls an iteration took.
//        spin_plain, spin_shfl, spin_atomic, spin_vote, spin_membar, spin_diverged, spin_long.
//   5. Which lanes execute: the guard, the counters (instructions, thread instructions, per class, barriers, divergent
//      branches), BSSY/BSYNC reconvergence, a block that is not whole warps.   reconverge, barriers.
//   6. A fault names the instruction, the lane and the address, however far into a run it is.   `fault <name>`.
//   7. The counter reads of S2R (CTAID, TID, laneid, clock-free ones) and uniform-datapath operands.   specials.
//
// The interpreter is the reference; the other tests hold it to the card.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define CK(x) do { cudaError_t e_ = (x); if (e_) { std::printf("error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

// n dependent additions the compiler cannot fold (each asm is volatile), SASS: n IADD3s in a straight run.
#define ADD1(v) asm volatile("add.u32 %0, %0, 1;" : "+r"(v));
#define ADD8(v) ADD1(v) ADD1(v) ADD1(v) ADD1(v) ADD1(v) ADD1(v) ADD1(v) ADD1(v)
#define ADD64(v) ADD8(v) ADD8(v) ADD8(v) ADD8(v) ADD8(v) ADD8(v) ADD8(v) ADD8(v)
#define ADD256(v) ADD64(v) ADD64(v) ADD64(v) ADD64(v)

// ---- 1. the store gate -------------------------------------------------------------------------------------------
// Each warp's first lane loads the word, adds one and stores it. With the gate every warp loads before any stores: all end
// with `word` 1 and every `seen` 0.
__global__ void gate_plain(u32* word, u32* seen) {
  if ((threadIdx.x & 31) == 0) {
    const u32 v = *word;
    seen[threadIdx.x >> 5] = v;
    *word = v + 1;
  }
}
// The first store comes after a run of straight instructions longer than a chunk: the gate must stop at the store itself.
__global__ void gate_long(u32* word, u32* seen) {
  if ((threadIdx.x & 31) == 0) {
    const u32 v = *word;
    u32 pad = threadIdx.x;
    ADD256(pad) ADD64(pad) ADD8(pad)
    seen[threadIdx.x >> 5] = v + (pad & 0x80000000u);
    *word = v + 1 + (pad >> 31);
  }
}
// A generic store (ST): the pointer is global at run time but the compiler cannot know.
__global__ void gate_generic(u32* word, u32* seen, int pick) {
  __shared__ u32 sh[4];
  if ((threadIdx.x & 31) == 0) {
    u32* p = pick ? word : &sh[threadIdx.x >> 5];
    const u32 v = *p;
    seen[threadIdx.x >> 5] = v;
    *p = v + 1;
  }
}
// The lanes of a warp take different paths to their own stores (two groups, two store instructions).
__global__ void gate_diverged(u32* word, u32* seen) {
  const unsigned lane = threadIdx.x & 31;
  if (lane == 0 || lane == 1) {
    const u32 v = *word;
    seen[(threadIdx.x >> 5) * 2 + lane] = v;
    if (lane == 0) {
      u32 pad = v;
      ADD64(pad)
      *word = v + 1 + (pad >> 31);
    } else {
      *word = v + 2;
    }
  }
}

// ---- 3 and 4. a warp that spins on another's store ---------------------------------------------------------------
// Warp 0 waits for a flag warp 1 sets after some work. The warps take turns of 4096 calls of step_warp, so what warp 0 counts
// before its turn ends is a measure of the calls an iteration of its loop takes: it is the output, and the instruction count.
template <int Kind>
__device__ __forceinline__ u32 spin_body(volatile u32* flag, u32 acc) {
  if (Kind == 1) acc += __shfl_xor_sync(0xffffffffu, acc, 1);                       // SHFL ends a run
  if (Kind == 2) acc += atomicAdd(const_cast<u32*>(flag) + 1, 0u);                  // ATOM ends a run
  if (Kind == 3) acc += __popc(__ballot_sync(0xffffffffu, acc & 1));                // VOTE
  if (Kind == 4) __threadfence();                                                   // MEMBAR
  if (Kind == 5) { if (acc & 2) acc += 3; else acc ^= 5; }                          // a branch lanes may split on
  if (Kind == 6) { ADD256(acc) ADD256(acc) ADD64(acc) ADD8(acc) }                   // a body longer than two chunks
  return acc + 1;
}
template <int Kind>
__global__ void spin(u32* flag, u32* out) {
  volatile u32* vf = flag;
  const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp == 0) {
    u32 acc = lane;
    u32 n = 0;
    while (*vf == 0 && n < 100000) {
      acc = spin_body<Kind>(vf, acc);
      ++n;
    }
    out[lane] = acc;
    out[32 + lane] = n;
  } else if (warp == 1 && lane == 0) {
    u32 pad = 0;
    ADD64(pad) ADD64(pad)
    *vf = 1 + (pad >> 31);
  }
}

// Every thread of two warps takes a spin lock (the lanes of a warp compete for it: a YIELD loop, convergence barriers, the lane
// group that won and the groups that gave way), adds to a counter inside and releases it.
__global__ void spinlock(u32* lock, u32* counter, u32* out) {
  bool done = false;
  while (!done) {
    if (atomicCAS(lock, 0u, 1u) == 0u) {
      const u32 v = *reinterpret_cast<volatile u32*>(counter);
      *reinterpret_cast<volatile u32*>(counter) = v + 1;
      __threadfence();
      atomicExch(lock, 0u);
      done = true;
    }
  }
  out[threadIdx.x] = threadIdx.x + 1;
}

// ---- 2. the step limit -------------------------------------------------------------------------------------------
// One warp: loads and stores, a loop, a shuffle, a vote, a barrier, a branch that splits the lanes, an exit by every lane.
__global__ void steps_kernel(u32* data, u32* out) {
  const unsigned t = threadIdx.x;
  __shared__ u32 sh[64];
  u32 acc = data[t];
  for (int i = 0; i < 7; ++i) {
    acc = acc * 3u + i;
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    sh[(t + i) & 63] = acc;
    __syncwarp();
    acc ^= sh[(t * 5 + i) & 63];
    if (acc & 4) acc += 11; else acc -= 3;
  }
  acc += __popc(__ballot_sync(0xffffffffu, acc & 8));
  out[t] = acc;
}

// ---- 5. reconvergence, barriers, blocks that are not whole warps ----------------------------------------------------
__global__ void reconverge(const u32* in, u32* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ u32 sh[256];
  u32 v = i < n ? in[i] : 7u;
  sh[threadIdx.x & 255] = v;
  __syncthreads();
  if (v & 1) {
    if (v & 2) v = v * 5u + sh[(threadIdx.x + 1) & 255];
    else v ^= sh[(threadIdx.x + 3) & 255];
  } else if (v & 4) {
    for (int k = 0; k < (int)(v & 7); ++k) v += sh[(threadIdx.x + k) & 255];
  }
  __syncthreads();
  sh[threadIdx.x & 255] = v;
  __syncthreads();
  if (i < n) out[i] = v + sh[(threadIdx.x + 17) & 255] + (u32)__popc(__activemask());
}

// ---- 7. special registers and kernel arguments ------------------------------------------------------------------
__global__ void specials(u32* out, u32 a, u32 b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i * 8 + 0] = threadIdx.x + blockIdx.x * 1000u;
  out[i * 8 + 1] = (u32)(threadIdx.x & 31) + (threadIdx.x >> 5) * 100u;
  out[i * 8 + 2] = a * (u32)i + b;
  out[i * 8 + 3] = blockDim.x * 7u + gridDim.x;
  out[i * 8 + 4] = (u32)warpSize;
  u32 lane;
  asm("mov.u32 %0, %%laneid;" : "=r"(lane));
  out[i * 8 + 5] = lane;
  u32 m;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
  out[i * 8 + 6] = m;
  out[i * 8 + 7] = (u32)__activemask();
}

// ---- 6. faults ---------------------------------------------------------------------------------------------------
// One lane of the warp reaches outside what it may, after a long run of straight instructions, so that the fault is not in
// the first instruction of a call. Each runs in a process of its own.
__global__ void f_gather(const u32* in, u32* out, long far) {
  const int t = threadIdx.x;
  u32 pad = t;
  ADD256(pad) ADD64(pad)
  out[t] = in[(t == 5) ? far : t] + (pad >> 31);
}
__global__ void f_scatter(u32* out, long far) {
  const int t = threadIdx.x;
  u32 pad = t;
  ADD256(pad) ADD8(pad)
  out[(t == 9) ? far : t] = t + (pad >> 31);
}
__global__ void f_local(u32* out, int far) {
  u32 arr[256];
  const int t = threadIdx.x;
  for (int i = 0; i < 256; ++i) arr[i] = i * t;
  volatile int idx = (t == 7) ? far : t;
  u32 pad = t;
  ADD64(pad)
  out[t] = arr[idx] + (pad >> 31);
}
__global__ void f_shared(u32* out, int far) {
  __shared__ u32 s[64];
  const int t = threadIdx.x;
  s[t] = t;
  __syncthreads();
  volatile int idx = (t == 11) ? far : t;
  u32 pad = t;
  ADD64(pad) ADD8(pad)
  out[t] = s[idx] + (pad >> 31);
}
__global__ void f_atomic(u32* out, long far) {
  const int t = threadIdx.x;
  atomicAdd(&out[(t == 3) ? far : t % 4], 1u);
}
__global__ void f_misaligned(const u32* in, u32* out) {
  const int t = threadIdx.x;
  const char* p = reinterpret_cast<const char*>(in) + (t == 2 ? 3 : 0) + 4 * t;
  out[t] = *reinterpret_cast<const u32*>(p);
}
__global__ void f_trap(u32* out) {
  u32 pad = threadIdx.x;
  ADD64(pad)
  if (threadIdx.x == 4) asm volatile("trap;");
  out[threadIdx.x] = pad;
}
__global__ void f_forever(u32* out) {   // a loop with no end: the step limit
  u32 pad = threadIdx.x;
  while (true) {
    ADD8(pad)
    if (*reinterpret_cast<volatile u32*>(out) == 0xdeadbeefu) break;
  }
  out[threadIdx.x] = pad;
}

static int run_fault(const std::string& n) {
  u32 *a, *b;
  CK(cudaMalloc(&a, 4096));
  CK(cudaMalloc(&b, 4096));
  CK(cudaMemset(a, 0, 4096));
  CK(cudaMemset(b, 0, 4096));
  if (n == "gather") f_gather<<<1, 32>>>(a, b, 1l << 34);
  else if (n == "scatter") f_scatter<<<1, 32>>>(a, 1l << 34);
  else if (n == "local") f_local<<<1, 32>>>(a, 1 << 22);
  else if (n == "shared") f_shared<<<1, 32>>>(a, 1 << 20);
  else if (n == "atomic") f_atomic<<<1, 32>>>(a, 1l << 34);
  else if (n == "misaligned") f_misaligned<<<1, 32>>>(a, b);
  else if (n == "trap") f_trap<<<1, 32>>>(a);
  else if (n == "forever") f_forever<<<1, 32>>>(a);
  else return 2;
  const cudaError_t e = cudaDeviceSynchronize();
  std::printf("fault %s: %s\n", n.c_str(), cudaGetErrorString(e));
  return e == cudaSuccess ? 1 : 0;   // it must fault
}

// `steps`: one launch of steps_kernel under whatever VGPU_MAX_STEPS the parent set.
static int run_steps_child() {
  u32 *d, *o;
  CK(cudaMalloc(&d, 4096));
  CK(cudaMalloc(&o, 4096));
  std::vector<u32> h(1024);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (u32)(i * 2654435761u + 12345u);
  CK(cudaMemcpy(d, h.data(), 4096, cudaMemcpyHostToDevice));
  steps_kernel<<<1, 32>>>(d, o);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e) {
    std::printf("fail %s\n", cudaGetErrorString(e));
    return 3;
  }
  std::vector<u32> r(32);
  CK(cudaMemcpy(r.data(), o, 128, cudaMemcpyDeviceToHost));
  u64 hsh = 1469598103934665603ull;
  for (u32 v : r) hsh = (hsh ^ v) * 1099511628211ull;
  std::printf("ok %016llx\n", (unsigned long long)hsh);
  return 0;
}

static std::string child(const char* self, const std::string& args, const std::string& env_prefix, int* rc) {
  const std::string cmd = env_prefix + " " + self + " " + args + " 2>&1";
  FILE* p = popen(cmd.c_str(), "r");
  std::string out;
  if (!p) { *rc = -1; return out; }
  char buf[512];
  while (fgets(buf, sizeof buf, p)) out += buf;
  *rc = pclose(p);
  return out;
}

static u64 hash_bytes(const void* p, size_t n) {
  u64 h = 1469598103934665603ull;
  const unsigned char* c = (const unsigned char*)p;
  for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
  return h;
}

template <class T>
static int report(const char* name, T* d, size_t count) {
  std::vector<T> h(count);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost));
  std::printf("%-26s %016llx\n", name, (unsigned long long)hash_bytes(h.data(), count * sizeof(T)));
  return 0;
}

static int gate_result(const char* name, u32* word, u32* seen, int warps) {
  u32 w = 0, s[8] = {};
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&w, word, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s, seen, 4 * warps, cudaMemcpyDeviceToHost));
  std::printf("%-26s word=%u seen=", name, w);
  for (int i = 0; i < warps; ++i) std::printf("%u%s", s[i], i + 1 < warps ? "," : "");
  std::printf("\n");
  return 0;
}

template <int Kind>
static int run_spin(const char* name, int threads) {
  u32 *flag, *out;
  CK(cudaMalloc(&flag, 64));
  CK(cudaMalloc(&out, 64 * 4));
  CK(cudaMemset(flag, 0, 64));
  CK(cudaMemset(out, 0, 64 * 4));
  spin<Kind><<<1, threads>>>(flag, out);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e) {
    std::printf("%-26s error %s\n", name, cudaGetErrorString(e));
    return 0;
  }
  std::vector<u32> h(64);
  CK(cudaMemcpy(h.data(), out, 256, cudaMemcpyDeviceToHost));
  std::printf("%-26s n(lane0)=%u hash=%016llx\n", name, h[32], (unsigned long long)hash_bytes(h.data(), 256));
  CK(cudaFree(flag));
  CK(cudaFree(out));
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "fault") return run_fault(argv[2]);
  if (argc > 1 && std::string(argv[1]) == "steps") return run_steps_child();

  u32 *word, *seen;
  CK(cudaMalloc(&word, 64));
  CK(cudaMalloc(&seen, 64));
  for (int warps : {2, 4}) {
    CK(cudaMemset(word, 0, 4));
    CK(cudaMemset(seen, 0xff, 64));
    gate_plain<<<1, 32 * warps>>>(word, seen);
    if (gate_result(warps == 2 ? "gate_plain 2 warps" : "gate_plain 4 warps", word, seen, warps)) return 1;
    CK(cudaMemset(word, 0, 4));
    CK(cudaMemset(seen, 0xff, 64));
    gate_long<<<1, 32 * warps>>>(word, seen);
    if (gate_result(warps == 2 ? "gate_long 2 warps" : "gate_long 4 warps", word, seen, warps)) return 1;
    CK(cudaMemset(word, 0, 4));
    CK(cudaMemset(seen, 0xff, 64));
    gate_generic<<<1, 32 * warps>>>(word, seen, 1);
    if (gate_result(warps == 2 ? "gate_generic 2 warps" : "gate_generic 4 warps", word, seen, warps)) return 1;
    CK(cudaMemset(word, 0, 4));
    CK(cudaMemset(seen, 0xff, 64));
    gate_diverged<<<1, 32 * warps>>>(word, seen);
    if (gate_result(warps == 2 ? "gate_diverged 2 warps" : "gate_diverged 4 warps", word, seen, warps * 2)) return 1;
  }
  // One warp has no warp-mate to wait for: its store is not delayed (the result is the same either way, the counters say it).
  CK(cudaMemset(word, 0, 4));
  CK(cudaMemset(seen, 0xff, 64));
  gate_plain<<<1, 32>>>(word, seen);
  if (gate_result("gate_plain 1 warp", word, seen, 1)) return 1;

  // The spin loops, with two warps and with four (the extra ones do nothing).
  for (int threads : {64, 128}) {
    char nm[64];
    std::snprintf(nm, sizeof nm, "spin_plain  %d", threads);
    if (run_spin<0>(nm, threads)) return 1;
    std::snprintf(nm, sizeof nm, "spin_shfl   %d", threads);
    if (run_spin<1>(nm, threads)) return 1;
    std::snprintf(nm, sizeof nm, "spin_atomic %d", threads);
    if (run_spin<2>(nm, threads)) return 1;
    std::snprintf(nm, sizeof nm, "spin_vote   %d", threads);
    if (run_spin<3>(nm, threads)) return 1;
    std::snprintf(nm, sizeof nm, "spin_membar %d", threads);
    if (run_spin<4>(nm, threads)) return 1;
    std::snprintf(nm, sizeof nm, "spin_branch %d", threads);
    if (run_spin<5>(nm, threads)) return 1;
    std::snprintf(nm, sizeof nm, "spin_long   %d", threads);
    if (run_spin<6>(nm, threads)) return 1;
  }

  {
    u32 *lock, *counter, *out;
    CK(cudaMalloc(&lock, 4));
    CK(cudaMalloc(&counter, 4));
    CK(cudaMalloc(&out, 256 * 4));
    for (int threads : {32, 64, 96}) {
      CK(cudaMemset(lock, 0, 4));
      CK(cudaMemset(counter, 0, 4));
      CK(cudaMemset(out, 0, 256 * 4));
      spinlock<<<1, threads>>>(lock, counter, out);
      u32 n = 0;
      const cudaError_t e = cudaDeviceSynchronize();
      if (e) {
        std::printf("spinlock %d: error %s\n", threads, cudaGetErrorString(e));
        return 0;
      }
      CK(cudaMemcpy(&n, counter, 4, cudaMemcpyDeviceToHost));
      std::printf("spinlock %-17d counter=%u (of %d)\n", threads, n, threads);
    }
    CK(cudaFree(lock)); CK(cudaFree(counter)); CK(cudaFree(out));
  }

  // Reconvergence and barriers, with block sizes that are not whole warps.
  {
    const int n = 3000;
    std::vector<u32> h(n);
    for (int i = 0; i < n; ++i) h[i] = (u32)(i * 2246822519u + 374761393u) ^ (u32)(i >> 3);
    u32 *din, *dout;
    CK(cudaMalloc(&din, n * 4));
    CK(cudaMalloc(&dout, n * 4));
    CK(cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice));
    for (int threads : {96, 100, 33, 256, 61}) {
      reconverge<<<(n + threads - 1) / threads, threads>>>(din, dout, n);
      char nm[64];
      std::snprintf(nm, sizeof nm, "reconverge %d", threads);
      if (report(nm, dout, n)) return 1;
    }
    CK(cudaFree(din)); CK(cudaFree(dout));
  }
  {
    u32* out;
    CK(cudaMalloc(&out, 400 * 8 * 4));
    for (int threads : {50, 128}) {
      specials<<<3, threads>>>(out, 0x9e3779b9u, 12345u);
      char nm[64];
      std::snprintf(nm, sizeof nm, "specials %d", threads);
      if (report(nm, out, (size_t)3 * threads * 8)) return 1;
    }
    CK(cudaFree(out));
  }
  CK(cudaFree(word)); CK(cudaFree(seen));
  CK(cudaDeviceSynchronize());

  // The step limit: the smallest VGPU_MAX_STEPS steps_kernel finishes under (a bisection by processes of its own, since a
  // launch that fails ends the context), and the failure one step below it.
  {
    int rc = 0;
    long lo = 1, hi = 100000;   // lo fails, hi passes
    std::string ok = child(argv[0], "steps", "env VGPU_MAX_STEPS=100000", &rc);
    if (rc != 0) {
      std::printf("steps: even a limit of %ld fails: %s", hi, ok.c_str());
      return 1;
    }
    while (hi - lo > 1) {
      const long mid = (lo + hi) / 2;
      std::string r = child(argv[0], "steps", "env VGPU_MAX_STEPS=" + std::to_string(mid), &rc);
      if (rc == 0) hi = mid; else lo = mid;
    }
    std::printf("steps: the kernel finishes under VGPU_MAX_STEPS=%ld\n", hi);
    std::string below = child(argv[0], "steps", "env VGPU_MAX_STEPS=" + std::to_string(hi - 1), &rc);
    std::printf("steps: one below: rc=%d %s", rc != 0, below.c_str());
    std::printf("steps: at the limit: %s", child(argv[0], "steps", "env VGPU_MAX_STEPS=" + std::to_string(hi), &rc).c_str());
  }
  std::printf("done\n");
  return 0;
}
