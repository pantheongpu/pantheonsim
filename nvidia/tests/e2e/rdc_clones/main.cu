// The kernels of a separately compiled build that reach libdevice's pow, exp, log and trigonometric helpers, and
// a call through a table of device functions that use them; see run_rdc_clones.sh.
#include "clones.h"
#include <cmath>
#include <cstdio>
#include <vector>
extern __device__ pdf_fn ptr_argus;
extern __device__ pdf_fn ptr_logexp;
__device__ pdf_fn d_table[2];

// Kernels that reach the transcendental functions statically, in different mixes: each gets its own clones of
// the libdevice helpers, with the constants it needs at offsets of its own.
__global__ void k_direct_a(const double* x, double* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = pow(x[i], 1.5) + exp(x[i]);
}
__global__ void k_direct_b(const double* x, double* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = log(x[i] + 2.0) * pow(x[i] + 1.0, 0.25) + sin(x[i]);
}
__global__ void k_direct_c(const double* x, double* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = exp(-x[i]) * log10(x[i] + 1.0) + cos(x[i]) + pow(x[i], 3.0);
}
// The call through the table, as a thrust transform calls a PDF.
__global__ void k_table(const double* x, double* o, int n, double s, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = d_table[which](&x[i], s);
}

static int check(const char* what, const std::vector<double>& got, const std::vector<double>& want) {
    int bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        if (!(std::fabs(got[i] - want[i]) <= 1e-9 * std::fabs(want[i]) + 1e-12)) {
            if (bad < 3) printf("  %s[%zu] = %.17g, want %.17g\n", what, i, got[i], want[i]);
            ++bad;
        }
    printf("%s: %s\n", what, bad ? "MISMATCH" : "ok");
    return bad;
}

int main() {
    const int n = 100;
    std::vector<double> x(n), o(n), w(n);
    for (int i = 0; i < n; ++i) x[i] = (i + 0.5) / n;
    double *dx, *dout;
    cudaMalloc(&dx, n * sizeof(double));
    cudaMalloc(&dout, n * sizeof(double));
    cudaMemcpy(dx, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    int bad = 0;
    auto run = [&](const char* what, auto launch, auto ref) {
        launch();
        cudaDeviceSynchronize();
        cudaMemcpy(o.data(), dout, n * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) w[i] = ref(x[i]);
        bad += check(what, o, w);
    };
    run("direct a", [&] { k_direct_a<<<1, 128>>>(dx, dout, n); }, [](double v) { return std::pow(v, 1.5) + std::exp(v); });
    run("direct b", [&] { k_direct_b<<<1, 128>>>(dx, dout, n); },
        [](double v) { return std::log(v + 2.0) * std::pow(v + 1.0, 0.25) + std::sin(v); });
    run("direct c", [&] { k_direct_c<<<1, 128>>>(dx, dout, n); },
        [](double v) { return std::exp(-v) * std::log10(v + 1.0) + std::cos(v) + std::pow(v, 3.0); });
    pdf_fn p[2];
    cudaMemcpyFromSymbol(&p[0], ptr_argus, sizeof(pdf_fn));
    cudaMemcpyFromSymbol(&p[1], ptr_logexp, sizeof(pdf_fn));
    cudaMemcpyToSymbol(d_table, p, sizeof(p));
    for (double s : {1.0, 2.0}) {
        run("table argus", [&] { k_table<<<1, 128>>>(dx, dout, n, s, 0); },
            [&](double v) { double t = 1 - v * v; return v * std::sqrt(t) * std::exp(s * t); });
        run("table logexp", [&] { k_table<<<1, 128>>>(dx, dout, n, s, 1); },
            [&](double v) { return std::log(1.0 + v) * std::exp(-s * v); });
    }
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
