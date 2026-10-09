// The store callbacks of fft_lto_callbacks.cpp, as LTO-IR (see fft_lto_callback_loads.cu).
#include <cufftXt.h>

// Store: the conjugate of the result.
__device__ void st_conj(void* out, unsigned long long off, cufftComplex e, void*, void*) {
  static_cast<cufftComplex*>(out)[off] = make_cuFloatComplex(e.x, -e.y);
}

// Store (double complex): half the result, through a helper that LTO inlines.
__device__ static double halve(double x) { return 0.5 * x; }
__device__ void st_half(void* out, unsigned long long off, cufftDoubleComplex e, void*, void*) {
  static_cast<cufftDoubleComplex*>(out)[off] = make_cuDoubleComplex(halve(e.x), halve(e.y));
}
