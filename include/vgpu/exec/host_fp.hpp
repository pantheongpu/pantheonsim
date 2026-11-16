// The host's floating-point environment around a kernel's execution.
//
// Device arithmetic never raises a host signal: an inf or a NaN a kernel makes is a value, on any GPU. But the engines do
// that arithmetic with the host's own instructions, on the launching thread and on worker threads that take the launching
// thread's environment (worker_pool.cpp, amd/src/gcn_pool.hpp), so a program that enabled floating-point traps
// (feenableexcept, gfortran -ffpe-trap=invalid,zero,overflow) was killed with SIGFPE by the first kernel that divided by
// zero (pantheonsim #605). A launch runs inside a DeviceFpScope: the traps are masked and the sticky flags cleared for
// the kernel (feholdexcept), and the caller's whole environment -- traps, flags, rounding mode -- is put back when it
// ends, by return or by exception, so the host sees no flag a kernel raised, as on hardware. The rounding mode is left as
// it was for the kernel.
//
// The flush-to-zero controls (x86 MXCSR FTZ and DAZ, ARM FPCR FZ) are cleared for the kernel. A device does double
// arithmetic with subnormals as numbers; the engines do it with the host's instructions, and a host program that turned
// the flush on for itself (nvc -fast does at startup, as do gcc and clang with -ffast-math) made every subnormal operand
// or result of a kernel zero. NVHPC 26.9's fmod of a double reads a subnormal's exponent through a multiply by 2^54, and
// returned garbage on the simulated H100 (pantheonsim #672). Where a kernel flushes (.FTZ, ptx .ftz) the engines do it
// themselves, explicitly.
#pragma once

#include <cfenv>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

namespace vgpu::exec {

class DeviceFpScope {
 public:
  DeviceFpScope() {
    std::feholdexcept(&saved_);
    clear_flush_controls();
  }
  ~DeviceFpScope() { std::fesetenv(&saved_); }
  DeviceFpScope(const DeviceFpScope&) = delete;
  DeviceFpScope& operator=(const DeviceFpScope&) = delete;

 private:
  static void clear_flush_controls() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(_mm_getcsr() & ~(0x8000u /* FTZ */ | 0x0040u /* DAZ */));
#elif defined(__aarch64__)
    unsigned long fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr &= ~(1ul << 24);   // FZ
    asm volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
  }

  std::fenv_t saved_;
};

}  // namespace vgpu::exec
