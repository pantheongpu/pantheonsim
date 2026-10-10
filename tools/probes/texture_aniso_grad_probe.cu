// Anisotropic tex2DGrad probe. usage: aprobe <W> <H> <aniso> <filt 0/1> <mipfilt 0/1> <addr 0 wrap 1 clamp 2 mirror 3 border> <data.bin> <in.bin> <out.bin> [bias minclamp maxclamp] [normalized=1]
// data.bin: levels 0.. concatenated float (levels computed from W,H). in.bin: records of 6 floats x,y,dxdx,dxdy,dydx,dydy (normalized coords). out: float per record.
// mode "grad" always. env AMODE=lod: use tex2DLod with record (x,y,lod)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
__global__ void kg(cudaTextureObject_t t, const float* g, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float* p = g + 6 * i;
  out[i] = tex2DGrad<float>(t, p[0], p[1], make_float2(p[2], p[3]), make_float2(p[4], p[5]));
}
__global__ void kl(cudaTextureObject_t t, const float* g, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float* p = g + 6 * i;
  out[i] = tex2DLod<float>(t, p[0], p[1], p[2]);
}
int main(int argc, char** argv) {
  if (argc < 10) { fprintf(stderr, "usage\n"); return 2; }
  size_t W = atoi(argv[1]), H = atoi(argv[2]); int aniso = atoi(argv[3]); int filt = atoi(argv[4]), mf = atoi(argv[5]), addr = atoi(argv[6]);
  float bias = argc > 10 ? atof(argv[10]) : 0, minc = argc > 11 ? atof(argv[11]) : 0, maxc = argc > 12 ? atof(argv[12]) : 20;
  int norm = argc > 13 ? atoi(argv[13]) : 1;
  size_t mx = W > H ? W : H; int levels = 1; while (((size_t)1 << levels) <= mx) ++levels;
  if (getenv("LEVELS")) levels = atoi(getenv("LEVELS"));
  cudaFree(0);
  cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
  cudaMipmappedArray_t mip;
  if (cudaMallocMipmappedArray(&mip, &d, make_cudaExtent(W, H, 0), levels) != cudaSuccess) { fprintf(stderr, "alloc fail\n"); return 1; }
  FILE* f = fopen(argv[7], "rb");
  for (int l = 0; l < levels; ++l) {
    cudaArray_t a; cudaGetMipmappedArrayLevel(&a, mip, l);
    size_t w = W >> l ? W >> l : 1, h = H >> l ? H >> l : 1;
    std::vector<float> host(w * h);
    if (fread(host.data(), 4, w * h, f) != w * h) { fprintf(stderr, "short data\n"); return 3; }
    cudaMemcpy2DToArray(a, 0, 0, host.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice);
  }
  fclose(f);
  cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeMipmappedArray; rd.res.mipmap.mipmap = mip;
  cudaTextureDesc td = {};
  cudaTextureAddressMode am[] = {cudaAddressModeWrap, cudaAddressModeClamp, cudaAddressModeMirror, cudaAddressModeBorder};
  td.addressMode[0] = td.addressMode[1] = am[addr];
  td.filterMode = filt ? cudaFilterModeLinear : cudaFilterModePoint;
  td.mipmapFilterMode = mf ? cudaFilterModeLinear : cudaFilterModePoint;
  td.readMode = cudaReadModeElementType; td.normalizedCoords = norm;
  td.maxAnisotropy = aniso; td.mipmapLevelBias = bias; td.minMipmapLevelClamp = minc; td.maxMipmapLevelClamp = maxc;
  cudaTextureObject_t t;
  if (cudaCreateTextureObject(&t, &rd, &td, nullptr) != cudaSuccess) { fprintf(stderr, "tex fail\n"); return 1; }
  f = fopen(argv[8], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  int n = sz / 24; std::vector<float> in(n * 6); if (fread(in.data(), 4, n * 6, f) != (size_t)n * 6) return 3; fclose(f);
  float *dg, *dout; cudaMalloc(&dg, n * 24); cudaMalloc(&dout, n * 4);
  cudaMemcpy(dg, in.data(), n * 24, cudaMemcpyHostToDevice);
  const char* m = getenv("AMODE");
  if (m && m[0] == 'l') kl<<<(n + 255) / 256, 256>>>(t, dg, dout, n); else kg<<<(n + 255) / 256, 256>>>(t, dg, dout, n);
  std::vector<float> out(n); cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "kernel error\n"); return 1; }
  f = fopen(argv[9], "wb"); fwrite(out.data(), 4, n, f); fclose(f);
  return 0;
}
