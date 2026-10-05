// Device atomics on mapped host memory, racing the host's own atomics.
//
// A kernel's atomics on pinned (or managed, or CUDA-IPC-shared) memory are
// updates the host can make at the same moment: host code with the CPU's
// atomics, or a kernel in another process -- NVSHMEM's PEs add into one
// another's heaps this way. The simulator once serialized its atomics only
// with a lock of its own, which neither of those takes, so a concurrent host
// update could be lost. Here the kernel holds itself open until a host thread
// has finished adding into the same four words, so the two always overlap,
// and each word must come out as the sum of every update from both sides.
//
// This is the simulator's guarantee, not the card's: an RTX 3060 reports
// cudaDevAttrHostNativeAtomicSupported 0 (so does the simulator), and run on
// it this program lost 1-38% of the updates (CUDA 13.0, under WSL). What it
// checks is the mechanism that keeps two processes' kernels atomic against
// each other on CUDA-IPC-shared memory (ipc.cu races that case too), here
// for both engines. On a real GPU it skips.
#include <dlfcn.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>

constexpr int kBlocks = 32, kThreads = 128, kIters = 16;
constexpr unsigned kHostAdds = 200000;

struct Words {
  unsigned u32;            // atomicAdd
  unsigned cas;            // an atomicCAS loop
  unsigned long long u64;  // atomicAdd, 64-bit
  float f32;               // atomicAdd on whole numbers, exact below 2^24
  int started, host_done;
};

__global__ void race(Words* w) {
  if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch_system(&w->started, 1);
  for (int i = 0; i < kIters; ++i) {
    atomicAdd_system(&w->u32, 1u);
    atomicAdd_system(&w->u64, 1ull);
    atomicAdd_system(&w->f32, 1.0f);
    unsigned old = *(volatile unsigned*)&w->cas;
    for (;;) {
      const unsigned got = atomicCAS_system(&w->cas, old, old + 1);
      if (got == old) break;
      old = got;
    }
  }
  // Keep the kernel running until the host has made all its updates.
  if (blockIdx.x == 0 && threadIdx.x == 0)
    while (atomicAdd_system(&w->host_done, 0) == 0) {
    }
}

int main() {
  if (!dlsym(RTLD_DEFAULT, "_Z22vgpu_device_allocationPKvPPvPm")) {
    std::printf("SKIP: not the simulator; device atomics on host memory are not atomic against the host here\n");
    return 0;
  }
  Words* h = nullptr;
  if (cudaHostAlloc(&h, sizeof(Words), cudaHostAllocMapped) != cudaSuccess) {
    std::printf("FAIL: cudaHostAlloc\n");
    return 1;
  }
  std::memset(h, 0, sizeof(Words));
  Words* d = nullptr;
  cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0);

  std::thread host([h] {
    while (__atomic_load_n(&h->started, __ATOMIC_SEQ_CST) == 0) std::this_thread::yield();
    for (unsigned i = 0; i < kHostAdds; ++i) {
      __atomic_fetch_add(&h->u32, 1u, __ATOMIC_SEQ_CST);
      __atomic_fetch_add(&h->cas, 1u, __ATOMIC_SEQ_CST);
      __atomic_fetch_add(&h->u64, 1ull, __ATOMIC_SEQ_CST);
      auto* fb = reinterpret_cast<uint32_t*>(&h->f32);
      uint32_t o = __atomic_load_n(fb, __ATOMIC_SEQ_CST), n;
      do {
        float f;
        std::memcpy(&f, &o, 4);
        f += 1.0f;
        std::memcpy(&n, &f, 4);
      } while (!__atomic_compare_exchange_n(fb, &o, n, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
    __atomic_store_n(&h->host_done, 1, __ATOMIC_SEQ_CST);
  });
  race<<<kBlocks, kThreads>>>(d);
  const cudaError_t e = cudaDeviceSynchronize();
  host.join();
  if (e != cudaSuccess) {
    std::printf("FAIL: kernel: %s\n", cudaGetErrorString(e));
    return 1;
  }
  const unsigned long long want = 1ull * kBlocks * kThreads * kIters + kHostAdds;
  int bad = 0;
  auto check = [&](const char* what, unsigned long long got) {
    const bool ok = got == want;
    std::printf("%s %s: %llu (want %llu)\n", ok ? "ok  " : "FAIL", what, got, want);
    bad += !ok;
  };
  check("u32 atomicAdd", h->u32);
  check("u32 atomicCAS loop", h->cas);
  check("u64 atomicAdd", h->u64);
  check("f32 atomicAdd", static_cast<unsigned long long>(h->f32));
  cudaFreeHost(h);
  std::printf(bad ? "FAIL: %d lost-update check(s)\n" : "PASS\n", bad);
  return bad ? 1 : 0;
}
