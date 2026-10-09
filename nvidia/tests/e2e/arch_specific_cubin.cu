// The kernel arch_specific_cubins runs: p[t] = 5t + 2.
extern "C" __global__ void fill(int* p) { p[threadIdx.x] = static_cast<int>(threadIdx.x) * 5 + 2; }
