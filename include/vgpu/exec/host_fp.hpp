// The host's floating-point environment around a kernel's execution.
//
// Device arithmetic never raises a host signal: an inf or a NaN a kernel makes is a value, on any GPU. But the engines do
// that arithmetic with the host's own instructions, on the launching thread and on worker threads that take the launching
// thread's environment (worker_pool.cpp, amd/src/gcn_pool.hpp), so a program that enabled floating-point traps
// (feenableexcept, gfortran -ffpe-trap=invalid,zero,overflow) was killed with SIGFPE by the first kernel that divided by
// zero (pantheonsim #605). A launch runs inside a DeviceFpScope: the traps are masked and the sticky flags cleared for
// the kernel (feholdexcept), and the caller's whole environment -- traps, flags, rounding mode -- is put back when it
// ends, by return or by exception, so the host sees no flag a kernel raised, as on hardware. The rounding mode and the
// x86 flush/denormal controls are left as they were for the kernel: what the engines compute does not change.
#pragma once

#include <cfenv>

namespace vgpu::exec {

class DeviceFpScope {
 public:
  DeviceFpScope() { std::feholdexcept(&saved_); }
  ~DeviceFpScope() { std::fesetenv(&saved_); }
  DeviceFpScope(const DeviceFpScope&) = delete;
  DeviceFpScope& operator=(const DeviceFpScope&) = delete;

 private:
  std::fenv_t saved_;
};

}  // namespace vgpu::exec
