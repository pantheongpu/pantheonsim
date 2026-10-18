/* A sampling profiler for hosts where perf is not allowed (perf_event_paranoid = 4, no root).
 *
 *   gcc -O2 -shared -fPIC -o libsigprof.so sigprof.c -ldl
 *   LD_PRELOAD=./libsigprof.so SIGPROF_OUT=samples.txt SIGPROF_HZ=1000 program ...
 *
 * An ITIMER_PROF timer (process CPU time, user + system) raises SIGPROF; the handler records the interrupted
 * instruction pointer, and at exit the samples are written as "<count> <object>+<offset>" lines, which
 * tools/bench-amd/profile/report.py turns into function and source-line tables with addr2line. With SIGPROF_STACK=1
 * the caller chain (up to 10 frames, backtrace()) is kept too, for inclusive times.
 * Single-thread safe by construction (one lock-free ring per process); with several host threads a sample is
 * taken on whichever thread the kernel delivers the signal to, so VGPU_THREADS=1 gives the cleanest profile.
 */
#define _GNU_SOURCE
#include <link.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#include <execinfo.h>

#define MAXS (1 << 20)
#define DEPTH 10
static uint64_t samples[MAXS][DEPTH];
static int use_stack;
static volatile uint64_t nsamples;
static uint64_t lost;

static void on_prof(int sig, siginfo_t* si, void* uc_) {
  (void)sig;
  (void)si;
  ucontext_t* uc = (ucontext_t*)uc_;
  uint64_t n = nsamples;
  if (n >= MAXS) {
    ++lost;
    return;
  }
  samples[n][0] = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
  if (use_stack) {
    void* fr[DEPTH + 2];
    int k = backtrace(fr, DEPTH + 2);
    /* frames: the handler, the signal trampoline, then the interrupted function's callers' chain: the
       interrupted pc is frame 2 (== REG_RIP); keep what follows it */
    for (int i = 3; i < k && i - 2 < DEPTH; ++i) samples[n][i - 2] = (uint64_t)fr[i];
  }
  nsamples = n + 1;
}

struct Obj { uint64_t lo, hi, base; char name[512]; };
static struct Obj objs[512];
static int nobjs;
static int cb(struct dl_phdr_info* i, size_t sz, void* d) {
  (void)sz; (void)d;
  if (nobjs >= 512) return 0;
  struct Obj* o = &objs[nobjs];
  o->base = i->dlpi_addr;
  o->lo = ~0ull; o->hi = 0;
  for (int k = 0; k < i->dlpi_phnum; ++k)
    if (i->dlpi_phdr[k].p_type == PT_LOAD && (i->dlpi_phdr[k].p_flags & PF_X)) {
      uint64_t a = i->dlpi_addr + i->dlpi_phdr[k].p_vaddr;
      if (a < o->lo) o->lo = a;
      if (a + i->dlpi_phdr[k].p_memsz > o->hi) o->hi = a + i->dlpi_phdr[k].p_memsz;
    }
  if (o->hi) {
    snprintf(o->name, sizeof o->name, "%s", i->dlpi_name[0] ? i->dlpi_name : "/proc/self/exe");
    ++nobjs;
  }
  return 0;
}

static int cmp(const void* a, const void* b) { return memcmp(a, b, sizeof(uint64_t) * DEPTH); }
static const char* where(uint64_t pc, uint64_t* off) {
  for (int k = 0; k < nobjs; ++k)
    if (pc >= objs[k].lo && pc < objs[k].hi) { *off = pc - objs[k].base; return objs[k].name; }
  *off = pc;
  return "?";
}

__attribute__((destructor)) static void finish(void) {
  struct itimerval off = {{0, 0}, {0, 0}};
  setitimer(ITIMER_PROF, &off, 0);
  const char* path = getenv("SIGPROF_OUT");
  if (!path) return;
  dl_iterate_phdr(cb, 0);
  FILE* f = fopen(path, "w");
  if (!f) return;
  uint64_t n = nsamples;
  qsort(samples, n, sizeof samples[0], cmp);
  fprintf(f, "# samples %llu lost %llu\n", (unsigned long long)n, (unsigned long long)lost);
  for (uint64_t i = 0; i < n;) {
    uint64_t j = i;
    while (j < n && memcmp(samples[j], samples[i], sizeof samples[0]) == 0) ++j;
    fprintf(f, "%llu", (unsigned long long)(j - i));
    for (int d = 0; d < DEPTH && samples[i][d]; ++d) {
      uint64_t off;
      const char* name = where(samples[i][d], &off);
      fprintf(f, " %s+0x%llx", name, (unsigned long long)off);
    }
    fprintf(f, "\n");
    i = j;
  }
  fclose(f);
}

__attribute__((constructor)) static void start(void) {
  use_stack = getenv("SIGPROF_STACK") != NULL;
  if (use_stack) { void* w[4]; backtrace(w, 4); }   /* load libgcc's unwinder before the first signal */
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = on_prof;
  sa.sa_flags = SA_SIGINFO | SA_RESTART;
  sigaction(SIGPROF, &sa, 0);
  const char* hz = getenv("SIGPROF_HZ");
  int h = hz ? atoi(hz) : 1000;
  if (h < 1) h = 1;
  struct itimerval t = {{0, 1000000 / h}, {0, 1000000 / h}};
  setitimer(ITIMER_PROF, &t, 0);
}
