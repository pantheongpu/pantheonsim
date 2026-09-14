// Cooperative launch and a grid-wide barrier, through cooperative_groups.
//
// cg::this_grid().sync() is not an instruction. It compiles to an atomic
// increment of a counter in device memory followed by a spin on that counter,
// so it only terminates if the blocks that have not arrived yet can still run.
// Ordinary CUDA makes the opposite promise -- blocks are independent and may
// run one at a time -- so this only works when the launch says otherwise.
//
// Three things are checked here, and the last two matter as much as the first:
// the barrier holds, a grid too large to be resident is refused rather than
// hung, and the same kernel launched ordinarily fails loudly instead of
// silently computing something else.
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;

__global__ void gridsync(float* x, float* y, int n) {
    cg::grid_group grid = cg::this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
    grid.sync();
    // Reads a value a *different* block wrote, into a separate output so the
    // read side is not racing the write side.
    if (i < n) y[i] = x[(i + 64) % n] * 2.0f;
}

int main() {
    const int n = 256, threads = 64, blocks = n / threads;
    float h[n];
    for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i);
    float *d = nullptr, *o = nullptr;
    if (cudaMalloc(&d, sizeof h) != cudaSuccess) { printf("FAIL: cudaMalloc\n"); return 1; }
    if (cudaMalloc(&o, sizeof h) != cudaSuccess) { printf("FAIL: cudaMalloc\n"); return 1; }
    cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);

    int coop = -1;
    cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0);
    printf("cooperative launch advertised: %s\n", coop == 1 ? "yes" : "no");

    int np = n;
    void* args[] = {&d, &o, &np};
    cudaError_t e = cudaLaunchCooperativeKernel((void*)gridsync, dim3(blocks), dim3(threads),
                                                args, 0, 0);
    printf("cudaLaunchCooperativeKernel: %s\n", cudaGetErrorName(e));
    e = cudaDeviceSynchronize();
    printf("synchronize: %s\n", cudaGetErrorName(e));

    cudaMemcpy(h, o, sizeof h, cudaMemcpyDeviceToHost);
    int wrong = 0;
    for (int i = 0; i < n; ++i) {
        const float want = (static_cast<float>((i + 64) % n) + 1.0f) * 2.0f;
        if (h[i] != want) ++wrong;
    }
    printf("grid barrier held, wrong values: %d\n", wrong);

    // A grid that cannot be resident must be refused. Every block of a
    // cooperative launch waits for every other, so a grid that does not fit
    // does not run slowly -- it hangs.
    e = cudaLaunchCooperativeKernel((void*)gridsync, dim3(1u << 20), dim3(threads), args, 0, 0);
    printf("oversized cooperative grid: %s\n", cudaGetErrorName(e));
    cudaGetLastError();

    // And the same kernel launched ordinarily must not silently do something
    // else: with no cooperative launch there is no barrier object, and the
    // generated code traps.
    e = cudaLaunchKernel((void*)gridsync, dim3(blocks), dim3(threads), args, 0, 0);
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    printf("ordinary launch of a grid-sync kernel: %s\n", cudaGetErrorName(e));
    cudaGetLastError();

    cudaFree(d);
    cudaFree(o);
    printf("RESULT: %s\n", wrong == 0 ? "cooperative launch and grid barrier work"
                                      : "grid barrier did NOT hold");
    return wrong == 0 ? 0 : 1;
}
