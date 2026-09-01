// cuBLAS differential test: the same GEMM/BLAS calls run against real cuBLAS
// on a physical GPU and against VirtualGPU's implementation. Results must
// agree to floating-point tolerance -- exercising transposes, non-unit leading
// dimensions, alpha/beta including the beta==0 write-only case, batching, and
// the level-1/level-2 routines.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CK(x) do { cublasStatus_t s_=(x); if(s_){printf("BLASFAIL %d @%d\n",(int)s_,__LINE__);return 1;} } while(0)

static float* upload(const std::vector<float>& h) {
    float* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(float)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static std::vector<float> download(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
// Deterministic, reproducible on both sides.
static std::vector<float> fill(size_t n, unsigned seed) {
    std::vector<float> v(n);
    unsigned s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = (float)((int)(s >> 16) % 2000 - 1000) / 128.0f;
    }
    return v;
}
static void emit(const char* tag, const std::vector<float>& v) {
    // Print a checksum plus a few elements: enough to catch any disagreement
    // without depending on exact bit patterns of every entry.
    double sum = 0, absum = 0;
    for (float x : v) { sum += x; absum += x < 0 ? -x : x; }
    printf("%-26s n=%-6zu sum=%.4f abssum=%.4f first=%.4f mid=%.4f last=%.4f\n",
           tag, v.size(), sum, absum, v.front(), v[v.size()/2], v.back());
}

int main() {
    cublasHandle_t h;
    CK(cublasCreate(&h));
    int ver = 0; cublasGetVersion(h, &ver);

    // 1. Square GEMM, no transpose.
    {
        const int m=64,n=48,k=32; auto A=fill((size_t)m*k,1), B=fill((size_t)k*n,2), C=fill((size_t)m*n,3);
        float *dA=upload(A),*dB=upload(B),*dC=upload(C); float al=1.25f, be=0.5f;
        CK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,m,dB,k,&be,dC,m));
        emit("sgemm NN", download(dC,(size_t)m*n));
        cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // 2. Both operands transposed.
    {
        const int m=40,n=24,k=56; auto A=fill((size_t)k*m,4), B=fill((size_t)n*k,5), C=fill((size_t)m*n,6);
        float *dA=upload(A),*dB=upload(B),*dC=upload(C); float al=-0.75f, be=2.0f;
        CK(cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&al,dA,k,dB,n,&be,dC,m));
        emit("sgemm TT", download(dC,(size_t)m*n));
        cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // 3. beta == 0: C is write-only and must not be read.
    {
        const int m=32,n=32,k=32; auto A=fill((size_t)m*k,7), B=fill((size_t)k*n,8);
        std::vector<float> C((size_t)m*n, 1.0f/0.0f);  // poison: reading it would give NaN
        float *dA=upload(A),*dB=upload(B),*dC=upload(C); float al=1.0f, be=0.0f;
        CK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,m,dB,k,&be,dC,m));
        emit("sgemm beta=0", download(dC,(size_t)m*n));
        cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // 4. Padded leading dimensions.
    {
        const int m=20,n=16,k=12,lda=32,ldb=24,ldc=40;
        auto A=fill((size_t)lda*k,9), B=fill((size_t)ldb*n,10), C=fill((size_t)ldc*n,11);
        float *dA=upload(A),*dB=upload(B),*dC=upload(C); float al=0.5f, be=-1.0f;
        CK(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,lda,dB,ldb,&be,dC,ldc));
        emit("sgemm padded ld", download(dC,(size_t)ldc*n));
        cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // 5. Strided batched.
    {
        const int m=16,n=16,k=16,batch=4;
        auto A=fill((size_t)m*k*batch,12), B=fill((size_t)k*n*batch,13), C=fill((size_t)m*n*batch,14);
        float *dA=upload(A),*dB=upload(B),*dC=upload(C); float al=1.0f, be=0.25f;
        CK(cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,m,(long long)m*k,
                                     dB,k,(long long)k*n,&be,dC,m,(long long)m*n,batch));
        emit("sgemm batched", download(dC,(size_t)m*n*batch));
        cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // 6. Double precision.
    {
        const int m=24,n=24,k=24;
        std::vector<double> A(m*k),B(k*n),C(m*n);
        for(size_t i=0;i<A.size();++i)A[i]=(double)((i*37)%101)/13.0;
        for(size_t i=0;i<B.size();++i)B[i]=(double)((i*53)%97)/11.0;
        for(size_t i=0;i<C.size();++i)C[i]=(double)((i*29)%89)/7.0;
        double *dA,*dB,*dC;
        cudaMalloc(&dA,A.size()*8);cudaMalloc(&dB,B.size()*8);cudaMalloc(&dC,C.size()*8);
        cudaMemcpy(dA,A.data(),A.size()*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dB,B.data(),B.size()*8,cudaMemcpyHostToDevice);
        cudaMemcpy(dC,C.data(),C.size()*8,cudaMemcpyHostToDevice);
        double al=1.5,be=0.5;
        CK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,m,dB,k,&be,dC,m));
        std::vector<double> out(C.size());
        cudaMemcpy(out.data(),dC,out.size()*8,cudaMemcpyDeviceToHost);
        double s=0; for(double x:out) s+=x;
        printf("%-26s n=%-6zu sum=%.4f\n","dgemm NN",out.size(),s);
        cudaFree(dA);cudaFree(dB);cudaFree(dC);
    }
    // 7. Level 2 and level 1.
    {
        const int m=48,n=36; auto A=fill((size_t)m*n,15), x=fill(n,16), y=fill(m,17);
        float *dA=upload(A),*dx=upload(x),*dy=upload(y); float al=1.1f, be=-0.3f;
        CK(cublasSgemv(h,CUBLAS_OP_N,m,n,&al,dA,m,dx,1,&be,dy,1));
        emit("sgemv N", download(dy,m));
        float a2=2.5f; CK(cublasSaxpy(h,m,&a2,dy,1,dy,1)); emit("saxpy", download(dy,m));
        float a3=0.25f; CK(cublasSscal(h,m,&a3,dy,1)); emit("sscal", download(dy,m));
        float dot=0; CK(cublasSdot(h,m,dy,1,dy,1,&dot)); printf("%-26s %.5f\n","sdot",dot);
        float nrm=0; CK(cublasSnrm2(h,m,dy,1,&nrm)); printf("%-26s %.5f\n","snrm2",nrm);
        cudaFree(dA);cudaFree(dx);cudaFree(dy);
    }
    CK(cublasDestroy(h));
    return 0;
}
