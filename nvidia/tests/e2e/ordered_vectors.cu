// A .relaxed (or .volatile, .acquire, .release) vector access is one access.
//
// CUB's decoupled look-back publishes each tile's {status, value} with one
// st.relaxed.gpu.v2 and the next tile polls it with one ld.relaxed.gpu.v2; it
// relies on never seeing a new status beside an old value. The simulator runs
// a grid's blocks on several host threads and used to perform a vector access
// one element at a time, so a reader on another thread could see half of a
// store. NanoVDB's PointsToGrid then built a grid from a torn prefix sum and
// wrote through garbage offsets, but only sometimes.
//
// Block 0 stores {i, ~i} over and over; the other blocks read the pair and
// count every one whose halves disagree. There must be none, on an RTX 3060 or
// here.
#include <cstdio>

__device__ __forceinline__ void st2(unsigned long long* p, unsigned long long a, unsigned long long b) {
    asm volatile("st.relaxed.gpu.global.v2.u64 [%0], {%1, %2};" :: "l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void ld2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
    asm volatile("ld.relaxed.gpu.global.v2.u64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ void st2(unsigned* p, unsigned a, unsigned b) {
    asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(a), "r"(b) : "memory");
}
__device__ __forceinline__ void ld2(const unsigned* p, unsigned& a, unsigned& b) {
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "l"(p) : "memory");
}

template <typename T>
__global__ void publish_and_poll(T* slot, unsigned* torn, int iters) {
    if (threadIdx.x != 0) return;
    if (blockIdx.x == 0) {
        for (int i = 1; i <= iters; ++i) st2(slot, T(i), T(~T(i)));
        return;
    }
    for (int i = 0; i < iters; ++i) {
        T a, b;
        ld2(slot, a, b);
        if (b != T(~a)) atomicAdd(torn, 1u);
    }
}

template <typename T>
int run(const char* what) {
    T* slot = nullptr;
    unsigned* torn = nullptr;
    const T init[2] = {T(0), T(~T(0))};
    cudaMalloc(&slot, sizeof init);
    cudaMalloc(&torn, sizeof(unsigned));
    cudaMemcpy(slot, init, sizeof init, cudaMemcpyHostToDevice);
    cudaMemset(torn, 0, sizeof(unsigned));
    publish_and_poll<T><<<8, 32>>>(slot, torn, 20000);
    unsigned h = 0;
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(&h, torn, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL %s: %s\n", what, cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaFree(slot);
    cudaFree(torn);
    if (h) std::printf("FAIL %s: %u torn reads\n", what, h);
    return h ? 1 : 0;
}

int main() {
    int bad = run<unsigned long long>("st/ld.relaxed.gpu.v2.u64") + run<unsigned>("st/ld.volatile.v2.u32");
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
