// 1-bit tensor-core matrix multiply (BMMA, wmma experimental::precision::b1, shape 8x8x128, sm_75+; the AND form needs sm_80)
// on random bit matrices, through three kernels that arrange the tiles differently, for XOR+popc and AND+popc.
// Own program (the CUDA samples have no BMMA sample, #661). Every result is compared with a plain host popcount reference, and
// a 64-bit FNV-1a checksum over all integer results is printed: it is the same on a card and on the simulator because the
// result is an integer sum, so equality is byte-exact.
//   usage: bmma_gemm            (prints per-case lines, "checksum 0x...", then PASS or FAIL)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mma.h>
#include <cuda_runtime.h>
using namespace nvcuda;
namespace bx = nvcuda::wmma::experimental;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

enum { XOR = 0, AND = 1 };
typedef wmma::fragment<wmma::matrix_a, 8, 8, 128, bx::precision::b1, wmma::row_major> FragA;
typedef wmma::fragment<wmma::matrix_b, 8, 8, 128, bx::precision::b1, wmma::col_major> FragB;
typedef wmma::fragment<wmma::accumulator, 8, 8, 128, int> FragC;

template <int OP> __device__ __forceinline__ void mma(FragC& d, const FragA& a, const FragB& b, const FragC& c) {
    if (OP == XOR) wmma::bmma_sync(d, a, b, c, bx::bmmaBitOpXOR, bx::bmmaAccumulateOpPOPC);
    else           wmma::bmma_sync(d, a, b, c, bx::bmmaBitOpAND, bx::bmmaAccumulateOpPOPC);
}

// Kernel 1: tiles pre-arranged by the host ([tile][k chunk][8 rows][128 bits]); loaded straight from global with ldm = 128.
// One warp per 8x8 output tile, zero accumulator.
template <int OP>
__global__ void k_tiled(int* out, const unsigned* at, const unsigned* bt, int ntm, int ntn, int nkc, int ldo) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp >= ntm * ntn) return;
    int tm = warp / ntn, tn = warp % ntn;
    FragC acc; wmma::fill_fragment(acc, 0);
    for (int kc = 0; kc < nkc; ++kc) {
        FragA fa; FragB fb;
        wmma::load_matrix_sync(fa, at + ((size_t)tm * nkc + kc) * 32, 128);
        wmma::load_matrix_sync(fb, bt + ((size_t)tn * nkc + kc) * 32, 128);
        mma<OP>(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(out + (size_t)tm * 8 * ldo + tn * 8, acc, ldo, wmma::mem_row_major);
}

// Kernel 2: row-major A (M x K) and column-major B (K x N) bit matrices as the host stores them; each warp stages its two
// tiles through shared memory, owns two neighbouring output tiles (two accumulators), starts from a loaded C matrix.
template <int OP>
__global__ void k_staged(int* out, const int* cin, const unsigned* a, const unsigned* b, int ntm, int ntn, int kw, int ldo) {
    __shared__ __align__(32) unsigned sa[4][32];
    __shared__ __align__(32) unsigned sb[4][2][32];
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    int tm = blockIdx.y, tn0 = (blockIdx.x * 4 + w) * 2;
    if (tm >= ntm) return;
    bool two = tn0 + 1 < ntn, any = tn0 < ntn;
    FragC c0, c1;
    if (any) wmma::load_matrix_sync(c0, cin + (size_t)tm * 8 * ldo + tn0 * 8, ldo, wmma::mem_row_major);
    if (two) wmma::load_matrix_sync(c1, cin + (size_t)tm * 8 * ldo + (tn0 + 1) * 8, ldo, wmma::mem_row_major);
    int row = lane / 4, wi = lane % 4;
    for (int kc = 0; kc < kw / 4; ++kc) {
        sa[w][lane] = a[(size_t)(tm * 8 + row) * kw + kc * 4 + wi];
        for (int j = 0; j < 2; ++j) sb[w][j][lane] = (tn0 + j < ntn) ? b[(size_t)((tn0 + j) * 8 + row) * kw + kc * 4 + wi] : 0u;
        __syncwarp();
        FragA fa; FragB fb0, fb1;
        wmma::load_matrix_sync(fa, sa[w], 128);
        if (any) { wmma::load_matrix_sync(fb0, sb[w][0], 128); mma<OP>(c0, fa, fb0, c0); }
        if (two) { wmma::load_matrix_sync(fb1, sb[w][1], 128); mma<OP>(c1, fa, fb1, c1); }
        __syncwarp();
    }
    if (any) wmma::store_matrix_sync(out + (size_t)tm * 8 * ldo + tn0 * 8, c0, ldo, wmma::mem_row_major);
    if (two) wmma::store_matrix_sync(out + (size_t)tm * 8 * ldo + (tn0 + 1) * 8, c1, ldo, wmma::mem_row_major);
}

// Kernel 3: split K over blockIdx.z, one warp per tile, partial sums added with atomicAdd (results stored column-major).
template <int OP>
__global__ void k_split(int* out, const unsigned* at, const unsigned* bt, int ntm, int ntn, int nkc, int splits, int ldo_rows) {
    int tile = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (tile >= ntm * ntn) return;
    int tm = tile / ntn, tn = tile % ntn, per = nkc / splits;
    FragC acc; wmma::fill_fragment(acc, 0);
    for (int kc = blockIdx.z * per; kc < (blockIdx.z + 1) * per; ++kc) {
        FragA fa; FragB fb;
        wmma::load_matrix_sync(fa, at + ((size_t)tm * nkc + kc) * 32, 128);
        wmma::load_matrix_sync(fb, bt + ((size_t)tn * nkc + kc) * 32, 128);
        mma<OP>(acc, fa, fb, acc);
    }
    __shared__ __align__(32) int tilebuf[4][64];
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    wmma::store_matrix_sync(tilebuf[w], acc, 8, wmma::mem_col_major);   // tilebuf[col*8+row]
    __syncwarp();
    for (int i = lane; i < 64; i += 32) {
        int col = i / 8, row = i % 8;
        atomicAdd(out + (size_t)(tn * 8 + col) * ldo_rows + tm * 8 + row, tilebuf[w][i]);
    }
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static unsigned rnd32() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return (unsigned)(rng_state >> 16); }
static uint64_t fnv(uint64_t h, const void* p, size_t n) { const unsigned char* c = (const unsigned char*)p; for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 0x100000001b3ull; } return h; }

struct Shape { int m, n, k; };

int main() {
    const Shape shapes[] = {{8, 8, 128}, {16, 24, 256}, {64, 64, 512}, {40, 72, 1024}, {128, 96, 2048}, {32, 128, 4096}};
    uint64_t total = 0xcbf29ce484222325ull;
    int bad = 0, cases = 0;
    for (const Shape& s : shapes) {
        int M = s.m, N = s.n, K = s.k, kw = K / 32, nkc = K / 128, ntm = M / 8, ntn = N / 8;
        std::vector<unsigned> A((size_t)M * kw), B((size_t)N * kw);          // A row-major bits; B column-major (column n = row n of B)
        for (auto& x : A) x = rnd32() ^ (rnd32() << 7);
        for (auto& x : B) x = rnd32() ^ (rnd32() << 9);
        // a sparse/dense mix in the last rows and columns, so the popcount is not always near K/2
        for (int i = 0; i < kw; ++i) { A[(size_t)(M - 1) * kw + i] = 0xffffffffu; B[(size_t)(N - 1) * kw + i] = (i & 1) ? 0u : 0xffffffffu; A[i] = 0; }
        std::vector<int> C((size_t)M * N);
        for (auto& x : C) x = (int)(rnd32() % 2001) - 1000;
        std::vector<unsigned> At((size_t)ntm * nkc * 32), Bt((size_t)ntn * nkc * 32);
        for (int tm = 0; tm < ntm; ++tm) for (int kc = 0; kc < nkc; ++kc) for (int r = 0; r < 8; ++r) for (int w = 0; w < 4; ++w)
            At[(((size_t)tm * nkc + kc) * 8 + r) * 4 + w] = A[(size_t)(tm * 8 + r) * kw + kc * 4 + w];
        for (int tn = 0; tn < ntn; ++tn) for (int kc = 0; kc < nkc; ++kc) for (int r = 0; r < 8; ++r) for (int w = 0; w < 4; ++w)
            Bt[(((size_t)tn * nkc + kc) * 8 + r) * 4 + w] = B[(size_t)(tn * 8 + r) * kw + kc * 4 + w];
        unsigned *dA, *dB, *dAt, *dBt; int *dC, *dO;
        CK(cudaMalloc(&dA, A.size() * 4)); CK(cudaMalloc(&dB, B.size() * 4)); CK(cudaMalloc(&dAt, At.size() * 4)); CK(cudaMalloc(&dBt, Bt.size() * 4));
        CK(cudaMalloc(&dC, C.size() * 4)); CK(cudaMalloc(&dO, C.size() * 4));
        CK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dAt, At.data(), At.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dBt, Bt.data(), Bt.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
        for (int op = 0; op < 2; ++op) {
            std::vector<int> ref((size_t)M * N);
            for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
                int p = 0;
                for (int w = 0; w < kw; ++w) { unsigned x = A[(size_t)m * kw + w], y = B[(size_t)n * kw + w]; p += __builtin_popcount(op == XOR ? (x ^ y) : (x & y)); }
                ref[(size_t)m * N + n] = p;
            }
            for (int kern = 0; kern < 3; ++kern) {
                std::vector<int> out((size_t)M * N), want = ref;
                CK(cudaMemset(dO, 0, out.size() * 4));
                int splits = (nkc % 4 == 0) ? 4 : (nkc % 2 == 0 ? 2 : 1);
                if (kern == 0) {
                    int blocks = (ntm * ntn + 3) / 4;
                    if (op == XOR) k_tiled<XOR><<<blocks, 128>>>(dO, dAt, dBt, ntm, ntn, nkc, N); else k_tiled<AND><<<blocks, 128>>>(dO, dAt, dBt, ntm, ntn, nkc, N);
                } else if (kern == 1) {
                    dim3 g((ntn + 7) / 8, ntm);
                    if (op == XOR) k_staged<XOR><<<g, 128>>>(dO, dC, dA, dB, ntm, ntn, kw, N); else k_staged<AND><<<g, 128>>>(dO, dC, dA, dB, ntm, ntn, kw, N);
                    for (size_t i = 0; i < want.size(); ++i) want[i] += C[i];
                } else {
                    dim3 g((ntm * ntn + 3) / 4, 1, splits);
                    if (op == XOR) k_split<XOR><<<g, 128>>>(dO, dAt, dBt, ntm, ntn, nkc, splits, M); else k_split<AND><<<g, 128>>>(dO, dAt, dBt, ntm, ntn, nkc, splits, M);
                    // column-major result: transpose the reference to compare
                    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) want[(size_t)n * M + m] = ref[(size_t)m * N + n];
                }
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(out.data(), dO, out.size() * 4, cudaMemcpyDeviceToHost));
                int diff = 0; for (size_t i = 0; i < out.size(); ++i) diff += out[i] != want[i];
                uint64_t h = fnv(0xcbf29ce484222325ull, out.data(), out.size() * 4);
                total = fnv(total, &h, 8); ++cases;
                printf("%dx%dx%d %s kernel%d: checksum 0x%016llx mismatches %d\n", M, N, K, op == XOR ? "xor" : "and", kern + 1, (unsigned long long)h, diff);
                bad += diff != 0;
            }
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dAt); cudaFree(dBt); cudaFree(dC); cudaFree(dO);
    }
    printf("cases %d checksum 0x%016llx\n", cases, (unsigned long long)total);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad != 0;
}
