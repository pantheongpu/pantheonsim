// Round-5 card probe for texCubemapGrad: a mipmapped cube map whose level k holds the constant k (see texture_grad_lod_probe.cu).
// usage: <S> <in.bin> <out.bin>; a record is 9 floats (P[3], dPdx[3], dPdy[3]).
// Card probe for tex.grad on a cube map: level k holds the constant k, trilinear: the fetched value is the LOD.
// usage: tgc <S> <in.bin> <out.bin> ; in: N records of 9 floats (P[3], ddx[3], ddy[3]); out: N floats
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
__global__ void k(cudaTextureObject_t t, const float* g, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float* p = g + 9 * i;
  out[i] = texCubemapGrad<float>(t, p[0], p[1], p[2], make_float4(p[3], p[4], p[5], 0), make_float4(p[6], p[7], p[8], 0));
}
int main(int argc, char** argv) {
  size_t S = atoi(argv[1]);
  int levels = 1; while (((size_t)1 << levels) <= S) ++levels;
  cudaFree(0);
  cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
  cudaMipmappedArray_t mip;
  if (cudaMallocMipmappedArray(&mip, &d, make_cudaExtent(S, S, 6), levels, cudaArrayCubemap) != cudaSuccess) { fprintf(stderr, "alloc fail\n"); return 1; }
  for (int l = 0; l < levels; ++l) {
    cudaArray_t a; cudaGetMipmappedArrayLevel(&a, mip, l);
    size_t w = S >> l ? S >> l : 1;
    std::vector<float> host(w * w * 6, (float)l);
    cudaMemcpy3DParms p = {}; p.srcPtr = make_cudaPitchedPtr(host.data(), w * 4, w, w); p.dstArray = a; p.extent = make_cudaExtent(w, w, 6); p.kind = cudaMemcpyHostToDevice;
    if (cudaMemcpy3D(&p) != cudaSuccess) { fprintf(stderr, "copy fail\n"); return 1; }
  }
  cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeMipmappedArray; rd.res.mipmap.mipmap = mip;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
  td.filterMode = cudaFilterModeLinear; td.mipmapFilterMode = cudaFilterModeLinear;
  td.readMode = cudaReadModeElementType; td.normalizedCoords = 1; td.maxAnisotropy = 1;
  td.minMipmapLevelClamp = 0; td.maxMipmapLevelClamp = 20;
  td.seamlessCubemap = getenv("SEAM") ? 1 : 0;
  cudaTextureObject_t t;
  if (cudaCreateTextureObject(&t, &rd, &td, nullptr) != cudaSuccess) { fprintf(stderr, "tex fail\n"); return 1; }
  FILE* f = fopen(argv[2], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  int n = sz / 36; std::vector<float> in(n * 9); if (fread(in.data(), 4, n * 9, f) != (size_t)n * 9) return 3; fclose(f);
  float *dg, *dout; cudaMalloc(&dg, n * 36); cudaMalloc(&dout, n * 4);
  cudaMemcpy(dg, in.data(), n * 36, cudaMemcpyHostToDevice);
  k<<<(n + 255) / 256, 256>>>(t, dg, dout, n);
  std::vector<float> out(n); cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "kernel error\n"); return 1; }
  f = fopen(argv[3], "wb"); fwrite(out.data(), 4, n, f); fclose(f);
  return 0;
}
