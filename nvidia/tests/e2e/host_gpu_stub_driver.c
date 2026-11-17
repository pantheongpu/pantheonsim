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
/* The card's launch limits (turbo mode's fit check): an sm_86 card's by default (RTX 3060: 1024 threads a block, dimensions 1024 x 1024 x 64, a grid of
 * 2^31-1 x 65535 x 65535, 99 KiB of shared memory a block with the opt-in, 65536 registers a block), each overridable to put a launch outside the card. */
int cuDeviceGetAttribute(int* v, int attr, int d) {
  (void)d;
  switch (attr) {
    case 1: *v = env_int("STUB_MAX_THREADS", 1024); break;
    case 2: *v = env_int("STUB_MAX_BLOCK_X", 1024); break;
    case 3: *v = 1024; break;
    case 4: *v = 64; break;
    case 5: *v = env_int("STUB_MAX_GRID_X", 2147483647); break;
    case 6: *v = 65535; break;
    case 7: *v = 65535; break;
    case 12: *v = env_int("STUB_REGS_PER_BLOCK", 65536); break;
    case 16: *v = env_int("STUB_MP", 28); break;
    case 75: *v = env_int("STUB_SM", 86) / 10; break;
    case 76: *v = env_int("STUB_SM", 86) % 10; break;
    case 97: *v = env_int("STUB_SHARED_OPTIN", 101376); break;
    default: *v = 0;
  }
  return 0;
}
/* A function's attributes: its registers, static shared memory and the most threads it can run (cuFuncGetAttribute 4, 1, 0). */
int cuFuncGetAttribute(int* v, int attr, void* f) {
  (void)f;
  note("cuFuncGetAttribute");
  *v = attr == 0 ? env_int("STUB_FUNC_MAX_THREADS", 1024) : attr == 1 ? env_int("STUB_FUNC_SHARED", 0) : attr == 4 ? env_int("STUB_FUNC_REGS", 32) : 0;
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
/* STUB_TOTAL_MIB: the card has this much memory (all of it free), for a window of the simulator's memory the card cannot hold. */
int cuMemGetInfo_v2(size_t* f, size_t* t) {
  const int mib = env_int("STUB_TOTAL_MIB", 0);
  *f = *t = mib ? (size_t)mib << 20 : 1ull << 40;
  return 0;
}
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
  if (memcmp(g_img, "\x7f" "ELF", 4) != 0) {   /* PTX text: the driver compiles it (turbo mode gives the card PTX when no cubin of its own runs) */
    note("image ptx");
    {   /* the target line the driver is given: a retargeted module says sm_86 on an sm_86 card */
      const char* t = strstr((const char*)image, ".target");
      char line[48] = "ptx_target ?";
      if (t) { int i = 0; const char* e = t + 7; while (*e == ' ') ++e; while (e[i] && e[i] != '\n' && e[i] != ',' && i < 20) ++i; snprintf(line, sizeof line, "ptx_target %.*s", i, e); }
      note(line);
    }
    g_img = 0;
    if (env_int("STUB_CARD_OK", 0)) { *m = (void*)0x2000; return 0; }
    return 1;
  }
  {   /* the architecture of the cubin: e_flags' second byte from ELF ABI version 8 (CUDA 13), its low byte before */
    const unsigned eflags = (unsigned)rd(g_img + 0x30, 4);
    char line[48];
    snprintf(line, sizeof line, "image elf sm_%u", g_img[7] == 0x33 ? eflags & 0xff : (eflags >> 8) & 0xff);
    note(line);
  }
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
  (void)f; (void)a; (void)b; (void)c; (void)d; (void)e; (void)g; (void)sh; (void)s; (void)p; (void)x; note("cuLaunchKernel");
  /* STUB_LAUNCH_FAIL=<error>: the card takes the module and the function and then refuses the launch with this error (701, too many resources, say) */
  if (env_int("STUB_LAUNCH_FAIL", 0)) return env_int("STUB_LAUNCH_FAIL", 0);
  return env_int("STUB_CARD_OK", 0) ? 0 : 1;
}
