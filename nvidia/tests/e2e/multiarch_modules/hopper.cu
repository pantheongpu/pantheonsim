// Built for sm_90 and compute_90 only: no code for an A100.
#include "kernel.cuh"
extern "C" const void* hopper_stub_shared() { return reinterpret_cast<const void*>(&tier_kernel<7>); }
extern "C" const void* hopper_stub_only() { return reinterpret_cast<const void*>(&tier_kernel<9>); }
