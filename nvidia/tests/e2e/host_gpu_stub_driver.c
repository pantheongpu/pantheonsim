/* A stand-in for the real driver (libcuda.so.1) for run_host_gpu_stub.sh: the host-GPU mode loads it through VGPU_HOST_GPU_LIBCUDA and starts it exactly as it
 * starts the real one, so the mode's start-up logic is tested on machines with no card. It reports the card the script says (STUB_SM, STUB_MP), backs the
 * reserved window without memory, sleeps STUB_INIT_MS in cuInit (the real driver's context creation takes about 150 ms), and refuses to load a module, so a
 * launch that reaches it falls back to the simulator and the program's output cannot change. Every call is appended to $STUB_LOG, one name per line. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned long long u64;
static void note(const char* name) {
  const char* f = getenv("STUB_LOG");
  if (!f) return;
  FILE* o = fopen(f, "a");
  if (!o) return;
  fprintf(o, "%s\n", name);
  fclose(o);
}
static int env_int(const char* n, int d) { const char* v = getenv(n); return v && *v ? atoi(v) : d; }

int cuInit(unsigned flags) {
  (void)flags;
  note("cuInit");
  const int ms = env_int("STUB_INIT_MS", 0);
  if (ms > 0) { struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000L}; nanosleep(&t, 0); }
  return env_int("STUB_FAIL_INIT", 0) ? 100 : 0;
}
int cuDeviceGet(int* d, int i) { note("cuDeviceGet"); *d = i; return 0; }
int cuDeviceGetAttribute(int* v, int attr, int d) {
  (void)d;
  *v = attr == 16 ? env_int("STUB_MP", 28) : attr == 75 ? env_int("STUB_SM", 86) / 10 : attr == 76 ? env_int("STUB_SM", 86) % 10 : 0;
  return 0;
}
int cuDevicePrimaryCtxRetain(void** c, int d) { (void)d; note("cuDevicePrimaryCtxRetain"); *c = (void*)0x1000; return 0; }
int cuCtxSetCurrent(void* c) { (void)c; return 0; }
int cuCtxSynchronize(void) { return 0; }
int cuMemAddressReserve(u64* p, size_t size, size_t align, u64 addr, u64 flags) { (void)size; (void)align; (void)flags; note("cuMemAddressReserve"); *p = addr; return 0; }
int cuMemCreate(u64* h, size_t size, const void* prop, u64 flags) { (void)size; (void)prop; (void)flags; *h = 1; return 0; }
int cuMemMap(u64 p, size_t size, size_t off, u64 h, u64 flags) { (void)p; (void)size; (void)off; (void)h; (void)flags; return 0; }
int cuMemSetAccess(u64 p, size_t size, const void* d, size_t n) { (void)p; (void)size; (void)d; (void)n; return 0; }
int cuMemsetD8_v2(u64 p, unsigned char v, size_t n) { (void)p; (void)v; (void)n; return 0; }
int cuMemcpyHtoD_v2(u64 d, const void* s, size_t n) { (void)d; (void)s; (void)n; note("cuMemcpyHtoD"); return 0; }
int cuMemcpyDtoH_v2(void* d, u64 s, size_t n) { (void)d; (void)s; (void)n; return 0; }
int cuMemGetInfo_v2(size_t* f, size_t* t) { *f = 1ull << 40; *t = 1ull << 40; return 0; }
int cuModuleLoadData(void** m, const void* image) { (void)m; (void)image; note("cuModuleLoadData"); return 1; }
int cuModuleGetFunction(void** f, void* m, const char* name) { (void)f; (void)m; (void)name; return 1; }
int cuFuncSetAttribute(void* f, int a, int v) { (void)f; (void)a; (void)v; return 0; }
int cuLaunchKernel(void* f, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e, unsigned g, unsigned sh, void* s, void** p, void** x) {
  (void)f; (void)a; (void)b; (void)c; (void)d; (void)e; (void)g; (void)sh; (void)s; (void)p; (void)x; note("cuLaunchKernel"); return 1;
}
