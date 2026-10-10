// cuFFT's Exec calls inside a captured CUDA graph (see graph_capture_common.h): the plan is made
// beforehand and kept (NVIDIA's kernels use the plan's tables), the transform is recorded, and the
// graph launched with new inputs. run_graph_capture.sh fft cufft --card runs the same program on
// NVIDIA's cuFFT.
#include <cufft.h>
#include <cufftXt.h>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)r_); ok = false; } } while (0)

struct Plan {
  cufftHandle h = 0;
  Plan(cudaStream_t st, cufftType type, int nx, int ny = 0, int batch = 1) {
    if (ny) {
      int n[2] = {nx, ny};
      cufftPlanMany(&h, 2, n, nullptr, 1, 0, nullptr, 1, 0, type, batch);
    } else {
      cufftPlan1d(&h, nx, type, batch);
    }
    cufftSetStream(h, st);
  }
  ~Plan() { cufftDestroy(h); }
};

int main() {
  Runner r;
  const int n = 64, batch = 3;
  {
    Plan p(r.st, CUFFT_C2C, n, 0, batch);
    cufftComplex *in = r.alloc<cufftComplex>(n * batch), *out = r.alloc<cufftComplex>(n * batch);
    r.run("cufftExecC2C forward", [&] {
      bool ok = true;
      fill(r.st, r.counter, reinterpret_cast<float*>(in), 2 * n * batch);
      OK(cufftExecC2C(p.h, in, out, CUFFT_FORWARD));
      return ok;
    }, {{out, (size_t)2 * n * batch, Dt::F32}}, 1e-4);
    r.run("cufftExecC2C inverse, in place", [&] {
      bool ok = true;
      fill(r.st, r.counter, reinterpret_cast<float*>(in), 2 * n * batch);
      OK(cufftExecC2C(p.h, in, in, CUFFT_INVERSE));
      return ok;
    }, {{in, (size_t)2 * n * batch, Dt::F32}}, 1e-4);
  }
  {
    Plan p(r.st, CUFFT_R2C, n, 0, batch), q(r.st, CUFFT_C2R, n, 0, batch);
    cufftReal *x = r.alloc<cufftReal>(n * batch), *back = r.alloc<cufftReal>(n * batch);
    cufftComplex* spec = r.alloc<cufftComplex>((n / 2 + 1) * batch);
    r.run("cufftExecR2C + cufftExecC2R", [&] {
      bool ok = true;
      fill(r.st, r.counter, x, n * batch);
      OK(cufftExecR2C(p.h, x, spec));
      OK(cufftExecC2R(q.h, spec, back));
      return ok;
    }, {{spec, (size_t)2 * (n / 2 + 1) * batch, Dt::F32}, {back, (size_t)n * batch, Dt::F32}}, 1e-4);
  }
  {
    Plan p(r.st, CUFFT_Z2Z, 16, 16, 2);
    cufftDoubleComplex *in = r.alloc<cufftDoubleComplex>(16 * 16 * 2), *out = r.alloc<cufftDoubleComplex>(16 * 16 * 2);
    r.run("cufftExecZ2Z (2D, batched)", [&] {
      bool ok = true;
      fill(r.st, r.counter, reinterpret_cast<double*>(in), 2 * 16 * 16 * 2);
      OK(cufftExecZ2Z(p.h, in, out, CUFFT_FORWARD));
      return ok;
    }, {{out, (size_t)2 * 16 * 16 * 2, Dt::F64}}, 1e-9);
  }
  {
    Plan p(r.st, CUFFT_D2Z, 32, 0, 2), q(r.st, CUFFT_Z2D, 32, 0, 2);
    cufftDoubleReal *x = r.alloc<cufftDoubleReal>(32 * 2), *back = r.alloc<cufftDoubleReal>(32 * 2);
    cufftDoubleComplex* spec = r.alloc<cufftDoubleComplex>(17 * 2);
    r.run("cufftExecD2Z + cufftExecZ2D", [&] {
      bool ok = true;
      fill(r.st, r.counter, x, 32 * 2);
      OK(cufftExecD2Z(p.h, x, spec));
      OK(cufftExecZ2D(q.h, spec, back));
      return ok;
    }, {{spec, (size_t)2 * 17 * 2, Dt::F64}, {back, (size_t)32 * 2, Dt::F64}}, 1e-9);
  }
  {
    Plan p(r.st, CUFFT_C2C, n, 0, 1);
    cufftComplex *in = r.alloc<cufftComplex>(n), *out = r.alloc<cufftComplex>(n);
    r.run("cufftXtExec", [&] {
      bool ok = true;
      fill(r.st, r.counter, reinterpret_cast<float*>(in), 2 * n);
      OK(cufftXtExec(p.h, in, out, CUFFT_FORWARD));
      return ok;
    }, {{out, (size_t)2 * n, Dt::F32}}, 1e-4);
  }
  return finish();
}
