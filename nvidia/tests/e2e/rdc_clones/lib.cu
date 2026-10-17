// Device functions whose addresses are taken into __device__ pointers, as GooFit's PDFs are; see
// run_rdc_clones.sh.
#include "clones.h"
#include <cmath>
// Each uses libdevice's pow, exp and log, which the device link clones for the kernels that reach them.
__device__ double pdf_argus(const double* e, double s) {
    double x = e[0], t = x / 1.0;
    if (t >= 1) return 0;
    t = 1 - t * t;
    return x * pow(t, 0.5) * exp(s * t);
}
__device__ double pdf_logexp(const double* e, double s) { return log(1.0 + e[0]) * exp(-s * e[0]); }
__device__ pdf_fn ptr_argus = pdf_argus;
__device__ pdf_fn ptr_logexp = pdf_logexp;
