// The load callbacks of fft_lto_callbacks.cpp, compiled to LTO-IR by run_fft_lto_callbacks.sh
// (nvcc -dc -dlto -arch=lto_86 -fatbin): the form NVIDIA documents for cufftXtSetJITCallback.
// Their names are looked up by symbol, mangled as C++. The loads and the stores are separate
// images because cuFFT links the image once for each callback it is given: both callbacks from
// one image define every symbol twice, and NVIDIA's plan fails to link (CUFFT_INTERNAL_ERROR).
#include <cufftXt.h>

// Load: the input element scaled by the float callerInfo points at.
__device__ cufftComplex ld_scale(void* in, unsigned long long off, void* info, void*) {
  cufftComplex v = static_cast<cufftComplex*>(in)[off];
  const float s = *static_cast<float*>(info);
  return make_cuFloatComplex(v.x * s, v.y * s);
}

// Load (real input): the element plus its offset.
__device__ cufftReal ld_real(void* in, unsigned long long off, void*, void*) {
  return static_cast<cufftReal*>(in)[off] + static_cast<cufftReal>(off);
}
