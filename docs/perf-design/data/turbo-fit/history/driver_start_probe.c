/* Times the driver start the way Impl::begin_start does it (host_gpu.cpp), one call at a time. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef int (*f_i_u)(unsigned);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double T0;
static void lap(const char* what) { double t = now(); printf("%-28s %7.1f ms (at %7.1f)\n", what, 0.0, (t - T0) * 1e3); }
static double last;
static void step(const char* what) { double t = now(); printf("%-28s %7.1f ms\n", what, (t - last) * 1e3); last = now(); }
int main(int argc, char** argv) {
  int deepbind = argc > 1 && argv[1][0] == 'd';
  T0 = last = now();
  void* h = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_NOW | RTLD_LOCAL | (deepbind ? RTLD_DEEPBIND : 0));
  step("dlopen libcuda");
  if (!h) { puts("no libcuda"); return 1; }
  int (*cuInit)(unsigned) = dlsym(h, "cuInit");
  int (*cuDeviceGet)(int*, int) = dlsym(h, "cuDeviceGet");
  int (*cuDeviceGetAttribute)(int*, int, int) = dlsym(h, "cuDeviceGetAttribute");
  int (*cuDevicePrimaryCtxRetain)(void**, int) = dlsym(h, "cuDevicePrimaryCtxRetain");
  int (*cuCtxSetCurrent)(void*) = dlsym(h, "cuCtxSetCurrent");
  int (*cuMemGetInfo)(size_t*, size_t*) = dlsym(h, "cuMemGetInfo_v2");
  int (*cuMemAddressReserve)(unsigned long long*, size_t, size_t, unsigned long long, unsigned long long) = dlsym(h, "cuMemAddressReserve");
  step("dlsym x7");
  int r = cuInit(0); step("cuInit"); if (r) printf("cuInit=%d\n", r);
  int d; cuDeviceGet(&d, 0); step("cuDeviceGet");
  int v; for (int i = 0; i < 12; ++i) cuDeviceGetAttribute(&v, 1 + i, d); step("cuDeviceGetAttribute x12");
  void* ctx; r = cuDevicePrimaryCtxRetain(&ctx, d); step("cuDevicePrimaryCtxRetain"); if (r) printf("retain=%d\n", r);
  r = cuCtxSetCurrent(ctx); step("cuCtxSetCurrent");
  size_t fr, tot; cuMemGetInfo(&fr, &tot); step("cuMemGetInfo");
  unsigned long long p = 0; r = cuMemAddressReserve(&p, 1ull << 40, 0, 0x200000000000ull, 0); step("cuMemAddressReserve 1TiB"); if (r) printf("reserve=%d\n", r);
  printf("total %.1f ms\n", (now() - T0) * 1e3);
  return 0;
}
