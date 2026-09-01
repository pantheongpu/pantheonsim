// cuBLASLt and cuRAND differential test.
//
// cuBLASLt results must match real cuBLASLt numerically (it is deterministic
// linear algebra). cuRAND cannot match bit-for-bit -- the generator's internal
// state layout is not published in the detail needed to reproduce it -- so the
// generator is checked for the properties that actually matter: correct
// distribution, correct range, and reproducibility for a given seed.
#include <cstdio>
#include <cmath>
#include <vector>
#include <cublasLt.h>
#include <curand.h>
#include <cuda_runtime.h>

static float* up(const std::vector<float>& h) {
    float* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(float)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> down(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
static std::vector<float> fill(size_t n, unsigned seed) {
    std::vector<float> v(n); unsigned s = seed;
    for (size_t i = 0; i < n; ++i) { s = s*1664525u+1013904223u; v[i] = (float)((int)(s>>16)%400-200)/64.0f; }
    return v;
}
static void emit(const char* tag, const std::vector<float>& v) {
    double sum=0, ab=0; for (float x : v) { sum+=x; ab+=std::fabs(x); }
    printf("%-24s n=%-6zu sum=%.4f abssum=%.4f first=%.4f last=%.4f\n",
           tag, v.size(), sum, ab, v.front(), v.back());
}

static int lt_matmul(const char* tag, cublasLtEpilogue_t epi, const float* bias) {
    cublasLtHandle_t lt; if (cublasLtCreate(&lt)) { printf("LTFAIL create\n"); return 1; }
    const int m=48,n=32,k=40;
    auto A=fill((size_t)m*k,21), B=fill((size_t)k*n,22);
    float *dA=up(A),*dB=up(B),*dD=nullptr;
    cudaMalloc(&dD,(size_t)m*n*sizeof(float));
    cublasLtMatmulDesc_t desc; cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    int32_t opn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&opn,sizeof opn);
    cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof opn);
    if (epi != CUBLASLT_EPILOGUE_DEFAULT) {
        int32_t e = epi;
        cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_EPILOGUE,&e,sizeof e);
        if (bias) cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof bias);
    }
    cublasLtMatrixLayout_t la,lb,ld;
    cublasLtMatrixLayoutCreate(&la,CUDA_R_32F,m,k,m);
    cublasLtMatrixLayoutCreate(&lb,CUDA_R_32F,k,n,k);
    cublasLtMatrixLayoutCreate(&ld,CUDA_R_32F,m,n,m);
    float al=1.0f, be=0.0f;
    cublasStatus_t s = cublasLtMatmul(lt,desc,&al,dA,la,dB,lb,&be,dD,ld,dD,ld,nullptr,nullptr,0,0);
    if (s == CUBLAS_STATUS_SUCCESS) emit(tag, down(dD,(size_t)m*n));
    else printf("%-24s NOT_SUPPORTED\n", tag);
    cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(ld);
    cublasLtMatmulDescDestroy(desc); cublasLtDestroy(lt);
    cudaFree(dA);cudaFree(dB);cudaFree(dD);
    return 0;
}

int main() {
    lt_matmul("lt matmul", CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    lt_matmul("lt matmul relu", CUBLASLT_EPILOGUE_RELU, nullptr);
    {
        auto b = fill(48, 23); float* db = up(b);
        lt_matmul("lt matmul bias", CUBLASLT_EPILOGUE_BIAS, db);
        lt_matmul("lt matmul relu+bias", CUBLASLT_EPILOGUE_RELU_BIAS, db);
        cudaFree(db);
    }
    // cuRAND: properties, not exact values.
    {
        const size_t N = 100000;
        float* d = nullptr; cudaMalloc(&d, N*sizeof(float));
        curandGenerator_t g; curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(g, 1234ULL);
        curandGenerateUniform(g, d, N);
        auto u = down(d, N);
        double mean=0, lo=1e9, hi=-1e9; for (float x:u){mean+=x; lo=std::fmin(lo,x); hi=std::fmax(hi,x);} mean/=N;
        double var=0; for (float x:u) var += (x-mean)*(x-mean); var/=N;
        printf("curand uniform         mean=%.3f var=%.4f min>0=%d max<=1=%d\n",
               mean, var, lo>0.0, hi<=1.0);

        curandSetPseudoRandomGeneratorSeed(g, 99ULL);
        curandGenerateNormal(g, d, N, 2.0f, 3.0f);
        auto nn = down(d, N);
        double m2=0; for (float x:nn) m2+=x; m2/=N;
        double v2=0; for (float x:nn) v2 += (x-m2)*(x-m2); v2/=N;
        printf("curand normal          mean=%.2f stddev=%.2f\n", m2, std::sqrt(v2));

        // Same seed must reproduce the same stream.
        curandSetPseudoRandomGeneratorSeed(g, 4242ULL);
        curandGenerateUniform(g, d, 1024); auto r1 = down(d, 1024);
        curandSetPseudoRandomGeneratorSeed(g, 4242ULL);
        curandGenerateUniform(g, d, 1024); auto r2 = down(d, 1024);
        bool same = true; for (size_t i=0;i<1024;i++) same = same && r1[i]==r2[i];
        printf("curand reproducible    %s\n", same ? "yes" : "NO");
        curandDestroyGenerator(g); cudaFree(d);
    }
    return 0;
}
