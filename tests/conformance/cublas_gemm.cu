// cuBLAS differential test: the same GEMM/BLAS calls run against real cuBLAS
// on a physical GPU and against VirtualGPU's implementation. Results must
// agree to floating-point tolerance -- exercising transposes, non-unit leading
// dimensions, alpha/beta including the beta==0 write-only case, batching, and
// the level-1/level-2 routines.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
    // 8. Mixed precision. Values are chosen to be exactly representable in the
    // narrow formats, and k is small, so the accumulation is exact on both
    // sides -- a disagreement here is a semantics bug, not rounding.
    {
        const int m=16,n=12,k=8;
        auto small = [](size_t cnt, unsigned seed) {
            std::vector<float> v(cnt); unsigned s=seed;
            for (size_t i=0;i<cnt;++i){ s=s*1664525u+1013904223u; v[i]=(float)((int)(s>>20)%17-8)*0.25f; }
            return v;
        };
        auto A=small((size_t)m*k,21), B=small((size_t)k*n,22), C=small((size_t)m*n,23);

        auto run_half = [&](const char* tag, cudaDataType ctype, cublasComputeType_t comp) {
            std::vector<__half> ha(A.size()), hb(B.size()), hc(C.size());
            for (size_t i=0;i<A.size();++i) ha[i]=__float2half(A[i]);
            for (size_t i=0;i<B.size();++i) hb[i]=__float2half(B[i]);
            for (size_t i=0;i<C.size();++i) hc[i]=__float2half(C[i]);
            void *dA,*dB,*dC;
            cudaMalloc(&dA,ha.size()*2); cudaMalloc(&dB,hb.size()*2);
            cudaMalloc(&dC,hc.size()* (ctype==CUDA_R_32F?4:2));
            cudaMemcpy(dA,ha.data(),ha.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hb.data(),hb.size()*2,cudaMemcpyHostToDevice);
            if (ctype==CUDA_R_32F) cudaMemcpy(dC,C.data(),C.size()*4,cudaMemcpyHostToDevice);
            else cudaMemcpy(dC,hc.data(),hc.size()*2,cudaMemcpyHostToDevice);
            float al=1.0f, be=0.5f;
            cublasStatus_t st = cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,CUDA_R_16F,m,
                                             dB,CUDA_R_16F,k,&be,dC,ctype,m,comp,
                                             CUBLAS_GEMM_DEFAULT);
            if (st) { printf("%-26s status=%d\n",tag,(int)st); }
            else {
                std::vector<float> out(C.size());
                if (ctype==CUDA_R_32F) cudaMemcpy(out.data(),dC,out.size()*4,cudaMemcpyDeviceToHost);
                else { std::vector<__half> t(C.size());
                       cudaMemcpy(t.data(),dC,t.size()*2,cudaMemcpyDeviceToHost);
                       for (size_t i=0;i<t.size();++i) out[i]=__half2float(t[i]); }
                emit(tag,out);
            }
            cudaFree(dA);cudaFree(dB);cudaFree(dC);
        };
        run_half("gemmEx f16 in f16 out", CUDA_R_16F, CUBLAS_COMPUTE_32F);
        run_half("gemmEx f16 in f32 out", CUDA_R_32F, CUBLAS_COMPUTE_32F);

        {   // bfloat16 in and out
            std::vector<__nv_bfloat16> ba(A.size()), bb(B.size()), bc(C.size());
            for (size_t i=0;i<A.size();++i) ba[i]=__float2bfloat16(A[i]);
            for (size_t i=0;i<B.size();++i) bb[i]=__float2bfloat16(B[i]);
            for (size_t i=0;i<C.size();++i) bc[i]=__float2bfloat16(C[i]);
            void *dA,*dB,*dC;
            cudaMalloc(&dA,ba.size()*2); cudaMalloc(&dB,bb.size()*2); cudaMalloc(&dC,bc.size()*2);
            cudaMemcpy(dA,ba.data(),ba.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,bb.data(),bb.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dC,bc.data(),bc.size()*2,cudaMemcpyHostToDevice);
            float al=1.0f, be=0.5f;
            cublasStatus_t st = cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,CUDA_R_16BF,m,
                                             dB,CUDA_R_16BF,k,&be,dC,CUDA_R_16BF,m,
                                             CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
            if (st) printf("%-26s status=%d\n","gemmEx bf16",(int)st);
            else {
                std::vector<__nv_bfloat16> t(C.size());
                cudaMemcpy(t.data(),dC,t.size()*2,cudaMemcpyDeviceToHost);
                std::vector<float> out(t.size());
                for (size_t i=0;i<t.size();++i) out[i]=__bfloat162float(t[i]);
                emit("gemmEx bf16",out);
            }
            cudaFree(dA);cudaFree(dB);cudaFree(dC);
        }

        {   // int8 operands, int32 accumulator and output
            std::vector<int8_t> ia(A.size()), ib(B.size());
            std::vector<int32_t> ic(C.size());
            for (size_t i=0;i<A.size();++i) ia[i]=(int8_t)(A[i]*4.0f);
            for (size_t i=0;i<B.size();++i) ib[i]=(int8_t)(B[i]*4.0f);
            for (size_t i=0;i<C.size();++i) ic[i]=(int32_t)(C[i]*4.0f);
            void *dA,*dB,*dC;
            cudaMalloc(&dA,ia.size()); cudaMalloc(&dB,ib.size()); cudaMalloc(&dC,ic.size()*4);
            cudaMemcpy(dA,ia.data(),ia.size(),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,ib.data(),ib.size(),cudaMemcpyHostToDevice);
            cudaMemcpy(dC,ic.data(),ic.size()*4,cudaMemcpyHostToDevice);
            int32_t al=1, be=2;
            // int8 GEMM wants column counts that are multiples of 4 on hardware;
            // m, n and k here already satisfy that.
            cublasStatus_t st = cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,CUDA_R_8I,m,
                                             dB,CUDA_R_8I,k,&be,dC,CUDA_R_32I,m,
                                             CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT);
            if (st) printf("%-26s status=%d\n","gemmEx int8",(int)st);
            else {
                std::vector<int32_t> out(ic.size());
                cudaMemcpy(out.data(),dC,out.size()*4,cudaMemcpyDeviceToHost);
                long long s2=0, a2=0;
                for (int32_t x:out){ s2+=x; a2+= x<0?-x:x; }
                printf("%-26s n=%-6zu sum=%lld abssum=%lld first=%d last=%d\n",
                       "gemmEx int8",out.size(),s2,a2,out.front(),out.back());
            }
            cudaFree(dA);cudaFree(dB);cudaFree(dC);
        }

        {   // Strided-batched mixed precision, the shape attention uses.
            const int batch=3;
            auto Ab=small((size_t)m*k*batch,31), Bb=small((size_t)k*n*batch,32),
                 Cb=small((size_t)m*n*batch,33);
            std::vector<__half> ha(Ab.size()), hb(Bb.size()), hc(Cb.size());
            for (size_t i=0;i<Ab.size();++i) ha[i]=__float2half(Ab[i]);
            for (size_t i=0;i<Bb.size();++i) hb[i]=__float2half(Bb[i]);
            for (size_t i=0;i<Cb.size();++i) hc[i]=__float2half(Cb[i]);
            void *dA,*dB,*dC;
            cudaMalloc(&dA,ha.size()*2); cudaMalloc(&dB,hb.size()*2); cudaMalloc(&dC,hc.size()*2);
            cudaMemcpy(dA,ha.data(),ha.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hb.data(),hb.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dC,hc.data(),hc.size()*2,cudaMemcpyHostToDevice);
            float al=1.0f, be=0.0f;
            cublasStatus_t st = cublasGemmStridedBatchedEx(
                h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,CUDA_R_16F,m,(long long)m*k,
                dB,CUDA_R_16F,k,(long long)k*n,&be,dC,CUDA_R_16F,m,(long long)m*n,batch,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
            if (st) printf("%-26s status=%d\n","gemmStridedBatchedEx",(int)st);
            else {
                std::vector<__half> t(hc.size());
                cudaMemcpy(t.data(),dC,t.size()*2,cudaMemcpyDeviceToHost);
                std::vector<float> out(t.size());
                for (size_t i=0;i<t.size();++i) out[i]=__half2float(t[i]);
                emit("gemmStridedBatchedEx",out);
            }
            cudaFree(dA);cudaFree(dB);cudaFree(dC);
        }

        {   // cublasHgemm: half in, half out, half alpha/beta.
            std::vector<__half> ha(A.size()), hb(B.size()), hc(C.size());
            for (size_t i=0;i<A.size();++i) ha[i]=__float2half(A[i]);
            for (size_t i=0;i<B.size();++i) hb[i]=__float2half(B[i]);
            for (size_t i=0;i<C.size();++i) hc[i]=__float2half(C[i]);
            __half *dA,*dB,*dC;
            cudaMalloc(&dA,ha.size()*2); cudaMalloc(&dB,hb.size()*2); cudaMalloc(&dC,hc.size()*2);
            cudaMemcpy(dA,ha.data(),ha.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hb.data(),hb.size()*2,cudaMemcpyHostToDevice);
            cudaMemcpy(dC,hc.data(),hc.size()*2,cudaMemcpyHostToDevice);
            __half al=__float2half(1.0f), be=__float2half(0.5f);
            cublasStatus_t st = cublasHgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,dA,m,dB,k,&be,dC,m);
            if (st) printf("%-26s status=%d\n","hgemm",(int)st);
            else {
                std::vector<__half> t(hc.size());
                cudaMemcpy(t.data(),dC,t.size()*2,cudaMemcpyDeviceToHost);
                std::vector<float> out(t.size());
                for (size_t i=0;i<t.size();++i) out[i]=__half2float(t[i]);
                emit("hgemm",out);
            }
            cudaFree(dA);cudaFree(dB);cudaFree(dC);
        }
    }
    CK(cublasDestroy(h));
    return 0;
}
