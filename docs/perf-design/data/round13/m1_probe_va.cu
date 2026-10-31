// M1 (host-gpu spike): can the real card hold the simulator's device windows at the same addresses?
// Every risky driver call runs in a forked child with a watchdog (SIGKILL after a timeout), because on WSL2 an out-of-range
// cuMemAddressReserve hint HANGS instead of failing. Build: nvcc -arch=sm_86 m1_probe_va.cu -o m1_probe_va -lcuda
// Usage: m1_probe_va sweep | map <hint> <MiB> | conc <n>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
__global__ void fill(unsigned* p, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) p[i] = 0xC0DE0000u + (unsigned)i * 3u; }
static void init() { cuInit(0); CUdevice d; cuDeviceGet(&d, 0); CUcontext c; cuDevicePrimaryCtxRetain(&c, d); cuCtxSetCurrent(c); }
static CUmemAllocationProp prop() { CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = 0; return p; }
// result codes: exit status of the child: 0 exact, 1 other address, 2 error (see stderr), 3 map/kernel/readback wrong
static int do_reserve(unsigned long long hint, size_t size, bool map, double* t_reserve) {
  init();
  CUdeviceptr p = 0; double t0 = now();
  CUresult r = cuMemAddressReserve(&p, size, 0, hint, 0);
  *t_reserve = now() - t0;
  if (r != CUDA_SUCCESS) { const char* s; cuGetErrorName(r, &s); printf("err=%s", s); return 2; }
  int rc = p == hint ? 0 : 1; if (rc) printf("got=0x%llx ", (unsigned long long)p);
  if (map && rc == 0) {
    CUmemAllocationProp pr = prop(); CUmemGenericAllocationHandle h;
    CUresult c = cuMemCreate(&h, size, &pr, 0); if (c) { printf("cuMemCreate=%d ", c); return 2; }
    c = cuMemMap(p, size, 0, h, 0); if (c) { printf("cuMemMap=%d ", c); return 2; }
    CUmemAccessDesc ad = {}; ad.location = pr.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    c = cuMemSetAccess(p, size, &ad, 1); if (c) { printf("setaccess=%d ", c); return 2; }
    size_t n = size / 4 < (64u << 20) ? size / 4 : (64u << 20);
    fill<<<(unsigned)((n + 255) / 256), 256>>>((unsigned*)p, n); if (cuCtxSynchronize()) { printf("kernel failed "); return 3; }
    std::vector<unsigned> o(n); cuMemcpyDtoH(o.data(), p, n * 4);
    for (size_t i = 0; i < n; ++i) if (o[i] != 0xC0DE0000u + (unsigned)i * 3u) { printf("readback wrong at %zu ", i); return 3; }
    // second kernel reading what the first wrote through the same pointer, and writing the tail of the range
    cuMemUnmap(p, size); cuMemRelease(h);
  }
  cuMemAddressFree(p, size); return rc;
}
// run fn in a child; returns exit code, or -1 on timeout (child killed)
template <class F> static int guarded(F fn, double timeout_s, double* wall) {
  fflush(stdout); double t0 = now(); pid_t pid = fork();
  if (pid == 0) { int rc = fn(); fflush(stdout); _exit(rc); }
  int st = 0; for (;;) { pid_t w = waitpid(pid, &st, WNOHANG); if (w == pid) break; if (now() - t0 > timeout_s) { kill(pid, SIGKILL); waitpid(pid, &st, 0); *wall = now() - t0; return -1; } usleep(2000); }
  *wall = now() - t0; return WIFEXITED(st) ? WEXITSTATUS(st) : -2;
}
static const char* name(int rc) { return rc == 0 ? "EXACT" : rc == 1 ? "other-address" : rc == 2 ? "error" : rc == 3 ? "WRONG" : rc == -1 ? "HANG(killed)" : "crash"; }
int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "sweep";
  if (!strcmp(mode, "sweep")) {
    double w0; guarded([&] { init(); size_t gran = 0; CUmemAllocationProp pr = prop();
      cuMemGetAllocationGranularity(&gran, &pr, CU_MEM_ALLOC_GRANULARITY_MINIMUM); printf("granularity min %zu", gran);
      cuMemGetAllocationGranularity(&gran, &pr, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED); printf(" recommended %zu\n", gran);
      size_t fr, tot; cuMemGetInfo(&fr, &tot); printf("memory free %zu total %zu MiB\n", fr >> 20, tot >> 20); return 0; }, 20.0, &w0);
    // power-of-two hints: 2^32 .. 2^48 and the simulator's own windows
    std::vector<unsigned long long> hints;
    for (int k = 32; k <= 48; ++k) hints.push_back(1ull << k);
    unsigned long long extra[] = {0x200000000000ull, 0x210000000000ull, 0x2f0000000000ull, 0x2fff00000000ull, 0x6ffb00000000ull, 0x7ffe00000000ull, 0x7fff00000000ull, 0x800000000000ull, 0xffff00000000ull};
    for (auto e : extra) hints.push_back(e);
    for (auto h : hints) {
      double tr = 0, wall = 0; int rc = guarded([&] { return do_reserve(h, 64ull << 20, true, &tr); }, 10.0, &wall);
      printf("hint 0x%012llx 64MiB reserve+map+kernel+readback: %-13s wall %.3f s\n", h, name(rc), wall);
    }
  } else if (!strcmp(mode, "map")) {
    unsigned long long h = strtoull(argv[2], 0, 0); size_t sz = strtoull(argv[3], 0, 0) << 20; double tr, wall;
    int rc = guarded([&] { return do_reserve(h, sz, true, &tr); }, 10.0, &wall); printf("map 0x%llx %zu MiB: %s wall %.3f\n", h, sz >> 20, name(rc), wall);
  } else if (!strcmp(mode, "windows")) {  // reserve (no map) each simulator device window of 1 TiB in full, then in one child hold all 16 at once
    for (int d = 0; d < 16; ++d) { unsigned long long h = 0x200000000000ull + (unsigned long long)d * 0x10000000000ull; double tr, wall;
      int rc = guarded([&] { return do_reserve(h, 1ull << 40, false, &tr); }, 10.0, &wall); printf("window %2d 0x%012llx 1 TiB reserve: %s wall %.3f\n", d, h, name(rc), wall); }
    double wall; int rc = guarded([&] { init(); int bad = 0; for (int d = 0; d < 16; ++d) { CUdeviceptr p; unsigned long long h = 0x200000000000ull + (unsigned long long)d * 0x10000000000ull;
      if (cuMemAddressReserve(&p, 1ull << 40, 0, h, 0) || p != h) bad = 1; } printf("all 16 at once: %s ", bad ? "FAILED" : "ok"); return bad; }, 20.0, &wall); printf("-> %s\n", name(rc));
  } else if (!strcmp(mode, "gran")) {  // granularity rules of the VMM API
    double wall; int rc = guarded([&] { init(); CUmemAllocationProp pr = prop(); CUmemGenericAllocationHandle h; CUdeviceptr p;
      printf("cuMemAddressReserve size 64 KiB (not a granule): %d\n", (int)cuMemAddressReserve(&p, 64 << 10, 0, 0x200000000000ull, 0));
      printf("cuMemAddressReserve hint not 2MiB-aligned (0x200000010000, 2 MiB): %d", (int)cuMemAddressReserve(&p, 2 << 20, 0, 0x200000010000ull, 0)); printf(" got 0x%llx\n", (unsigned long long)p);
      printf("cuMemCreate 64 KiB: %d\n", (int)cuMemCreate(&h, 64 << 10, &pr, 0));
      printf("cuMemCreate 3 MiB: %d\n", (int)cuMemCreate(&h, 3 << 20, &pr, 0));
      CUresult r = cuMemAddressReserve(&p, 8 << 20, 0, 0x200000000000ull, 0); printf("reserve 8 MiB at window base: %d\n", (int)r);
      r = cuMemCreate(&h, 2 << 20, &pr, 0); printf("cuMemCreate 2 MiB: %d\n", (int)r);
      printf("cuMemMap 2 MiB at +0: %d\n", (int)cuMemMap(p, 2 << 20, 0, h, 0));
      printf("cuMemMap same handle at +4 MiB (two views of one physical granule): %d\n", (int)cuMemMap(p + (4 << 20), 2 << 20, 0, h, 0));
      printf("cuMemMap again at +0 (overlap): %d\n", (int)cuMemMap(p, 2 << 20, 0, h, 0));
      CUmemAccessDesc ad = {}; ad.location = pr.location; ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      printf("SetAccess on the unmapped +2 MiB hole: %d\n", (int)cuMemSetAccess(p + (2 << 20), 2 << 20, &ad, 1));
      printf("cuMemUnmap of half a granule: %d\n", (int)cuMemUnmap(p, 1 << 20));
      return 0; }, 20.0, &wall); printf("-> %s\n", name(rc));
  } else if (!strcmp(mode, "conc")) {  // N processes at the same fixed addresses at once, each with its own physical memory, each running kernels
    int n = atoi(argv[2]); std::vector<pid_t> pids; double t0 = now();
    for (int i = 0; i < n; ++i) { pid_t p = fork(); if (p == 0) { double tr; int bad = 0; for (int it = 0; it < 20; ++it) bad |= do_reserve(0x200000000000ull, 256ull << 20, true, &tr); _exit(bad); } pids.push_back(p); }
    int bad = 0; for (auto p : pids) { int st; waitpid(p, &st, 0); bad |= WEXITSTATUS(st); }
    printf("conc %d processes x 20 reserve+map+kernel at 0x200000000000: %s, wall %.2f s\n", n, bad ? "FAILED" : "all EXACT and verified", now() - t0);
  }
  return 0;
}
