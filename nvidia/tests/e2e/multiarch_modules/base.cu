// Built for sm_80, sm_90 and compute_90: runs on Ampere through Hopper.
#include "kernel.cuh"
extern "C" const void* base_stub_shared() { return reinterpret_cast<const void*>(&tier_kernel<7>); }
