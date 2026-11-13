/* A stand-in for the real driver (libcuda.so.1) for run_host_gpu_stub.sh: the host-GPU mode loads it through VGPU_HOST_GPU_LIBCUDA and starts it exactly as it
 * starts the real one, so the mode's start-up logic is tested on machines with no card. It reports the card the script says (STUB_SM, STUB_MP), backs the
 * reserved window without memory, sleeps STUB_INIT_MS in cuInit (the real driver's context creation takes about 150 ms), and refuses to load a module, so a
 * launch that reaches it falls back to the simulator and the program's output cannot change. Every call is appended to $STUB_LOG, one name per line. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
/* STUB_SYNC_MS: the wait for the kernel takes this long (a kernel that does not come back, for the mode's VGPU_HOST_GPU_TIMEOUT). */
int cuCtxSynchronize(void) {
  const int ms = env_int("STUB_SYNC_MS", 0);
  if (ms > 0) { struct timespec t = {ms / 1000, (long)(ms % 1000) * 1000000L}; nanosleep(&t, 0); }
  return 0;
}
int cuMemAddressReserve(u64* p, size_t size, size_t align, u64 addr, u64 flags) { (void)size; (void)align; (void)flags; note("cuMemAddressReserve"); *p = addr; return 0; }
int cuMemCreate(u64* h, size_t size, const void* prop, u64 flags) { (void)size; (void)prop; (void)flags; *h = 1; return 0; }
/* STUB_KEEP_MEMORY=1: the card has memory. Every mapped range is backed (zeros, as the mode fills a new mapping), copies to the card are kept and copies
 * from it give them back. With STUB_CARD_OK=1 this is a card whose kernels compute nothing: after a launch "on the card" the program reads exactly what
 * was pushed for it, which is how run_host_gpu_stub.sh sees WHICH bytes the mode pushed for a launch it moved from the simulator. */
static struct { u64 lo, size; unsigned char* bytes; } g_maps[64];
static int g_nmaps;
static unsigned char* backing(u64 p, size_t n) {
  for (int i = 0; i < g_nmaps; ++i)
    if (p >= g_maps[i].lo && p + n <= g_maps[i].lo + g_maps[i].size) return g_maps[i].bytes + (p - g_maps[i].lo);
  return 0;
}
int cuMemMap(u64 p, size_t size, size_t off, u64 h, u64 flags) {
  (void)off; (void)h; (void)flags;
  if (env_int("STUB_KEEP_MEMORY", 0)) {
    if (g_nmaps == 64) return 2;
    /* (the mode grows its mapping upwards, one range after another: a copy never spans two unless they are joined, so adjacent ranges are) */
    if (g_nmaps && g_maps[g_nmaps - 1].lo + g_maps[g_nmaps - 1].size == p) {
      unsigned char* b = realloc(g_maps[g_nmaps - 1].bytes, g_maps[g_nmaps - 1].size + size);
      if (!b) return 2;
      memset(b + g_maps[g_nmaps - 1].size, 0, size);
      g_maps[g_nmaps - 1].bytes = b;
      g_maps[g_nmaps - 1].size += size;
    } else {
      g_maps[g_nmaps].lo = p;
      g_maps[g_nmaps].size = size;
      g_maps[g_nmaps].bytes = calloc(1, size);
      if (!g_maps[g_nmaps].bytes) return 2;
      ++g_nmaps;
    }
  }
  return 0;
}
int cuMemSetAccess(u64 p, size_t size, const void* d, size_t n) { (void)p; (void)size; (void)d; (void)n; return 0; }
int cuMemsetD8_v2(u64 p, unsigned char v, size_t n) { unsigned char* b = backing(p, n); if (b) memset(b, v, n); return 0; }
int cuMemcpyHtoD_v2(u64 d, const void* s, size_t n) { unsigned char* b = backing(d, n); note("cuMemcpyHtoD"); if (b) memcpy(b, s, n); return 0; }
int cuMemcpyDtoH_v2(void* d, u64 s, size_t n) { unsigned char* b = backing(s, n); if (b) memcpy(d, b, n); return 0; }
int cuMemGetInfo_v2(size_t* f, size_t* t) { *f = 1ull << 40; *t = 1ull << 40; return 0; }
/* STUB_CARD_OK=1: the module loads and a launch succeeds without doing anything (the card's memory is whatever the stub's copies left: nothing). For a run that only
 * needs the mode to believe a launch ran on the card, such as VGPU_HOST_GPU_VERIFY=1, whose result is the simulator's whatever the card did. Never for a run whose
 * output matters in the plain mode: the offloaded launches would not have run at all. */
/* The image the mode loads (Round sp66): the stand-in reads its symbols and relocations, logs "relocs_to_variables <n>" -- the relocations whose symbol lies
 * in a section of variables (.nv.global, .nv.global.init, .nv.constant3), which the mode applies itself, with the simulator's addresses, and takes out -- and
 * answers cuModuleGetGlobal from the symbol table: a variable of section s at offset v is at 0x7f0000000000 + (s << 24) + v. */
static const unsigned char* g_img;
static u64 rd(const unsigned char* p, int n) { u64 v = 0; memcpy(&v, p, n); return v; }
static const char* sec_name(const unsigned char* img, u64 i) {
  const u64 shoff = rd(img + 0x28, 8), es = rd(img + 0x3a, 2), str = rd(img + 0x3e, 2);
  return (const char*)img + rd(img + shoff + str * es + 24, 8) + rd(img + shoff + i * es, 4);
}
static int var_section(const char* n) { return !strcmp(n, ".nv.global") || !strcmp(n, ".nv.global.init") || !strcmp(n, ".nv.constant3"); }
static void count_relocs(const unsigned char* img) {
  const u64 shoff = rd(img + 0x28, 8), es = rd(img + 0x3a, 2), num = rd(img + 0x3c, 2);
  unsigned long n = 0;
  for (u64 i = 0; i < num; ++i) {
    const unsigned char* h = img + shoff + i * es;
    const u64 type = rd(h + 4, 4), off = rd(h + 24, 8), size = rd(h + 32, 8), link = rd(h + 40, 4);
    if (type != 9 && type != 4) continue;
    const u64 ee = type == 4 ? 24 : 16;
    const unsigned char* sh = img + shoff + link * es;
    const u64 symoff = rd(sh + 24, 8), syment = rd(sh + 56, 8);
    for (u64 o = off; o + ee <= off + size; o += ee) {
      const u64 sym = rd(img + o + 8, 8) >> 32, shndx = rd(img + symoff + sym * syment + 6, 2);
      if (shndx && shndx < num && var_section(sec_name(img, shndx))) ++n;
    }
  }
  char line[64];
  snprintf(line, sizeof line, "relocs_to_variables %lu", n);
  note(line);
}
int cuModuleLoadData(void** m, const void* image) {
  note("cuModuleLoadData");
  g_img = (const unsigned char*)image;
  count_relocs(g_img);
  if (env_int("STUB_CARD_OK", 0)) { *m = (void*)0x2000; return 0; }
  return 1;
}
int cuModuleGetGlobal_v2(u64* dptr, size_t* bytes, void* m, const char* name) {
  (void)m;
  note("cuModuleGetGlobal");
  if (!g_img) return 400;
  const u64 shoff = rd(g_img + 0x28, 8), es = rd(g_img + 0x3a, 2), num = rd(g_img + 0x3c, 2);
  for (u64 i = 0; i < num; ++i) {
    const unsigned char* h = g_img + shoff + i * es;
    if (rd(h + 4, 4) != 2) continue;   /* SHT_SYMTAB */
    const u64 off = rd(h + 24, 8), size = rd(h + 32, 8), ent = rd(h + 56, 8), str = rd(g_img + shoff + rd(h + 40, 4) * es + 24, 8);
    for (u64 o = off; ent && o + ent <= off + size; o += ent) {
      const u64 shndx = rd(g_img + o + 6, 2);
      if (!shndx || shndx >= num || strcmp((const char*)g_img + str + rd(g_img + o, 4), name)) continue;
      *dptr = 0x7f0000000000ull + (shndx << 24) + rd(g_img + o + 8, 8);
      if (bytes) *bytes = rd(g_img + o + 16, 8);
      return 0;
    }
  }
  return 500;   /* CUDA_ERROR_NOT_FOUND */
}
int cuMemcpyDtoD_v2(u64 d, u64 s, size_t n) {
  unsigned char* b = backing(d, n);
  unsigned char* a = backing(s, n);
  note("cuMemcpyDtoD");
  if (a && b) memmove(b, a, n);
  return 0;
}
int cuModuleGetFunction(void** f, void* m, const char* name) { (void)m; (void)name; if (env_int("STUB_CARD_OK", 0)) { *f = (void*)0x3000; return 0; } return 1; }
int cuFuncSetAttribute(void* f, int a, int v) { (void)f; (void)a; (void)v; return 0; }
int cuLaunchKernel(void* f, unsigned a, unsigned b, unsigned c, unsigned d, unsigned e, unsigned g, unsigned sh, void* s, void** p, void** x) {
  (void)f; (void)a; (void)b; (void)c; (void)d; (void)e; (void)g; (void)sh; (void)s; (void)p; (void)x; note("cuLaunchKernel"); return env_int("STUB_CARD_OK", 0) ? 0 : 1;
}
