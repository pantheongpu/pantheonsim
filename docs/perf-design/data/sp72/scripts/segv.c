// LD_PRELOAD: prints where a SIGSEGV happened (the instruction, the address, the call chain) and keeps other libraries from replacing the handler.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
static int (*real_sigaction)(int, const struct sigaction*, struct sigaction*);
static void on_segv(int sig, siginfo_t* si, void* uc_) {
  ucontext_t* uc = (ucontext_t*)uc_;
  char b[256];
  int n = snprintf(b, sizeof b, "\nSEGV-TRACE sig=%d code=%d addr=%p rip=%p rsp=%p rbp=%p\n", sig, si->si_code, si->si_addr,
                   (void*)uc->uc_mcontext.gregs[REG_RIP], (void*)uc->uc_mcontext.gregs[REG_RSP], (void*)uc->uc_mcontext.gregs[REG_RBP]);
  if (write(2, b, n) < 0) _exit(140);
  unsigned long* lp = dlsym(RTLD_DEFAULT, "sprof_last_pc");
  unsigned long* ls = dlsym(RTLD_DEFAULT, "sprof_last_sp");
  unsigned long* lb = dlsym(RTLD_DEFAULT, "sprof_last_bp");
  if (lp) {
    n = snprintf(b, sizeof b, "SAMPLED pc=%p sp=%p bp=%p\n", (void*)*lp, (void*)*ls, (void*)*lb);
    if (write(2, b, n) < 0) _exit(140);
    void* p = (void*)*lp;
    backtrace_symbols_fd(&p, 1, 2);
    // the words at the sampled stack pointer (return addresses among them)
    unsigned long* sp = (unsigned long*)*ls;
    for (int i = 0; i < 48; ++i) {
      void* q = (void*)sp[i];
      Dl_info di;
      if (dladdr(q, &di) && di.dli_fname) {
        n = snprintf(b, sizeof b, "  sp[%d]=%p %s+%#lx\n", i, q, di.dli_fname, (unsigned long)((char*)q - (char*)di.dli_fbase));
        if (write(2, b, n) < 0) _exit(140);
      }
    }
  }
  {   // the SIGPROF frame under the unwinder's: its restorer's address, then the interrupted context
    struct sigaction pa;
    memset(&pa, 0, sizeof pa);
    real_sigaction(SIGPROF, 0, &pa);
    unsigned long rest = (unsigned long)pa.sa_restorer;
    unsigned long* p = (unsigned long*)uc->uc_mcontext.gregs[REG_RSP];
    for (int i = 0; rest && i < 6000; ++i) {
      if (p[i] != rest) continue;
      ucontext_t* u = (ucontext_t*)(p + i + 1);
      unsigned long ipc = u->uc_mcontext.gregs[REG_RIP], isp = u->uc_mcontext.gregs[REG_RSP];
      { Dl_info d0; if (!ipc || !dladdr((void*)ipc, &d0) || !d0.dli_fname || isp < (unsigned long)p || isp > (unsigned long)p + (64ul << 20)) continue; }
      n = snprintf(b, sizeof b, "INTERRUPTED pc=%p sp=%p bp=%p rbx=%p r12=%p r13=%p r14=%p r15=%p\n", (void*)ipc, (void*)isp, (void*)u->uc_mcontext.gregs[REG_RBP],
                   (void*)u->uc_mcontext.gregs[REG_RBX], (void*)u->uc_mcontext.gregs[REG_R12], (void*)u->uc_mcontext.gregs[REG_R13], (void*)u->uc_mcontext.gregs[REG_R14], (void*)u->uc_mcontext.gregs[REG_R15]);
      if (write(2, b, n) < 0) _exit(140);
      void* q0 = (void*)ipc;
      backtrace_symbols_fd(&q0, 1, 2);
      unsigned long* sp = (unsigned long*)isp;
      for (int k = 0; k < 64; ++k) {
        void* q = (void*)sp[k];
        Dl_info di;
        if (dladdr(q, &di) && di.dli_fname) {
          n = snprintf(b, sizeof b, "  sp[%d]=%p %s+%#lx\n", k, q, di.dli_fname, (unsigned long)((char*)q - (char*)di.dli_fbase));
          if (write(2, b, n) < 0) _exit(140);
        }
      }
      break;
    }
    n = snprintf(b, sizeof b, "restorer=%p\n", (void*)rest);
    if (write(2, b, n) < 0) _exit(140);
  }
  void* ip = (void*)uc->uc_mcontext.gregs[REG_RIP];
  backtrace_symbols_fd(&ip, 1, 2);
  _exit(139);
}
int sigaction(int sig, const struct sigaction* act, struct sigaction* old) {
  if (!real_sigaction) real_sigaction = dlsym(RTLD_NEXT, "sigaction");
  if ((sig == SIGSEGV || sig == SIGBUS) && act) {   // (reported as done; ours stays)
    if (old) memset(old, 0, sizeof *old);
    return 0;
  }
  return real_sigaction(sig, act, old);
}
__attribute__((constructor)) static void init(void) {
  if (!real_sigaction) real_sigaction = dlsym(RTLD_NEXT, "sigaction");
  static char stack[1 << 16];
  stack_t ss = {stack, 0, sizeof stack};
  sigaltstack(&ss, 0);
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_sigaction = on_segv;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
  real_sigaction(SIGSEGV, &sa, 0);
  real_sigaction(SIGBUS, &sa, 0);
  void* warm[4];
  backtrace(warm, 4);
}
