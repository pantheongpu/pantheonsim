// Texture and surface objects, through the ordinary CUDA runtime API.
//
// Three paths, which are the three a real program takes: tex1Dfetch over linear
// device memory (which is how ML code uses textures -- as a cached load),
// tex2D over a cudaArray with point filtering and clamped addressing, and a
// surface read-modify-write over that same array.
//
// The last check samples the array through a linear filter at texel centres,
// where the filter must return each texel exactly. (Its behaviour between
// texels is checked bit for bit against hardware by texture_filtering.cu.)
#include <cstdio>
#include <cmath>
__global__ void k_fetch(cudaTextureObject_t t, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex1Dfetch<float>(t, i);
}
__global__ void k_tex2d(cudaTextureObject_t t, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[y * w + x] = tex2D<float>(t, x + 0.5f, y + 0.5f);
}
__global__ void k_surf(cudaSurfaceObject_t s, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        float v = surf2Dread<float>(s, x * (int)sizeof(float), y);
        surf2Dwrite(v * 2.0f, s, x * (int)sizeof(float), y);
    }
}
#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { printf("FAIL %s -> %s\n", #x, cudaGetErrorName(e_)); return 1; } } while(0)
int main() {
    const int n = 64;
    float h[n]; for (int i = 0; i < n; ++i) h[i] = i * 0.5f;
    float* lin = nullptr; CK(cudaMalloc(&lin, sizeof h));
    CK(cudaMemcpy(lin, h, sizeof h, cudaMemcpyHostToDevice));
    float* out = nullptr; CK(cudaMalloc(&out, sizeof h));

    // --- 1. tex1Dfetch over linear memory ---
    cudaResourceDesc rd{}; rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = lin;
    rd.res.linear.desc = cudaCreateChannelDesc<float>();
    rd.res.linear.sizeInBytes = sizeof h;
    cudaTextureDesc td{}; td.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex = 0;
    CK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));
    k_fetch<<<1, n>>>(tex, out, n);
    CK(cudaDeviceSynchronize());
    float got[n]; CK(cudaMemcpy(got, out, sizeof got, cudaMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < n; ++i) if (got[i] != h[i]) ++bad;
    printf("tex1Dfetch wrong: %d\n", bad);
    CK(cudaDestroyTextureObject(tex));

    // --- 2. tex2D over a cudaArray ---
    const int W = 8, H = 8;
    float img[W*H]; for (int i = 0; i < W*H; ++i) img[i] = 100.0f + i;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();
    cudaArray_t arr = nullptr;
    CK(cudaMallocArray(&arr, &fmt, W, H, cudaArraySurfaceLoadStore));
    CK(cudaMemcpy2DToArray(arr, 0, 0, img, W*sizeof(float), W*sizeof(float), H,
                           cudaMemcpyHostToDevice));
    cudaResourceDesc ard{}; ard.resType = cudaResourceTypeArray; ard.res.array.array = arr;
    cudaTextureDesc atd{};
    atd.addressMode[0] = cudaAddressModeClamp; atd.addressMode[1] = cudaAddressModeClamp;
    atd.filterMode = cudaFilterModePoint; atd.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex2 = 0;
    CK(cudaCreateTextureObject(&tex2, &ard, &atd, nullptr));
    float* out2 = nullptr; CK(cudaMalloc(&out2, W*H*sizeof(float)));
    k_tex2d<<<dim3(1,1), dim3(W,H)>>>(tex2, out2, W, H);
    CK(cudaDeviceSynchronize());
    float got2[W*H]; CK(cudaMemcpy(got2, out2, sizeof got2, cudaMemcpyDeviceToHost));
    bad = 0; for (int i = 0; i < W*H; ++i) if (got2[i] != img[i]) ++bad;
    printf("tex2D wrong: %d\n", bad);
    CK(cudaDestroyTextureObject(tex2));

    // --- 3. surface read-modify-write over the same array ---
    cudaSurfaceObject_t surf = 0;
    CK(cudaCreateSurfaceObject(&surf, &ard));
    k_surf<<<dim3(1,1), dim3(W,H)>>>(surf, W, H);
    CK(cudaDeviceSynchronize());
    float back[W*H];
    CK(cudaMemcpy2DFromArray(back, W*sizeof(float), arr, 0, 0, W*sizeof(float), H,
                             cudaMemcpyDeviceToHost));
    bad = 0; for (int i = 0; i < W*H; ++i) if (back[i] != img[i]*2.0f) ++bad;
    printf("surface wrong: %d\n", bad);

    // Linear filtering at texel centres: every weight falls on one texel.
    cudaTextureDesc ltd{};
    ltd.addressMode[0] = cudaAddressModeClamp;
    ltd.addressMode[1] = cudaAddressModeClamp;
    ltd.filterMode = cudaFilterModeLinear;
    ltd.readMode = cudaReadModeElementType;
    cudaTextureObject_t lin_tex = 0;
    CK(cudaCreateTextureObject(&lin_tex, &ard, &ltd, nullptr));
    float* lout = nullptr;
    CK(cudaMalloc(&lout, W * H * sizeof(float)));
    k_tex2d<<<dim3(1, 1), dim3(W, H)>>>(lin_tex, lout, W, H);
    CK(cudaDeviceSynchronize());
    float lback[W*H];
    CK(cudaMemcpy(lback, lout, sizeof lback, cudaMemcpyDeviceToHost));
    bad = 0; for (int i = 0; i < W*H; ++i) if (lback[i] != img[i]*2.0f) ++bad;
    printf("linear filtering at texel centres wrong: %d\n", bad);
    cudaFree(lout);
    cudaDestroyTextureObject(lin_tex);
    cudaGetLastError();

    CK(cudaDestroySurfaceObject(surf));
    CK(cudaFreeArray(arr));
    cudaFree(lin); cudaFree(out); cudaFree(out2);
    printf("RESULT: textures and surfaces work\n");
    return 0;
}
