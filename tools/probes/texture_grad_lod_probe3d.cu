// Round-5 card probe for tex3DGrad's LOD: like texture_grad_lod_probe.cu, but a record is 9 floats (P[3], dPdx[3], dPdy[3]) because
// the position matters for 3D (the unit sees the quad's coordinates P, P + d). Set dims to 3. TGP is unused here.
// Card probe for tex.grad's LOD: a mipmapped float texture whose level k is the constant k, trilinear in the
// mip chain, so the fetched value is the (clamped, quantised) LOD itself.
// usage: texture_grad_lod_probe <dims> <W> <H> <D> <aniso> <in.bin> <out.bin> <bias> <minclamp> <maxclamp> <filter>
// in.bin: N records of 6 floats (dpdx[3], dpdy[3]); out.bin: N floats.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__constant__ float PC[3];
template <int D> __global__ void k(cudaTextureObject_t t, const float* g, float* out, int n);
template <> __global__ void k<1>(cudaTextureObject_t t, const float* g, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float* p = g + 6 * i;
  out[i] = tex1DGrad<float>(t, PC[0], p[0], p[3]);
}
template <> __global__ void k<2>(cudaTextureObject_t t, const float* g, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float* p = g + 6 * i;
  out[i] = tex2DGrad<float>(t, PC[0], PC[1], make_float2(p[0], p[1]), make_float2(p[3], p[4]));
}
template <> __global__ void k<3>(cudaTextureObject_t t, const float* g, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  const float* p = g + 9 * i;
  out[i] = tex3DGrad<float>(t, p[0], p[1], p[2], make_float4(p[3], p[4], p[5], 0), make_float4(p[6], p[7], p[8], 0));
}

int main(int argc, char** argv) {
  if (argc < 12) { fprintf(stderr, "usage\n"); return 2; }
  int dims = atoi(argv[1]); size_t W = atoi(argv[2]), H = atoi(argv[3]), D = atoi(argv[4]); int aniso = atoi(argv[5]);
  float bias = atof(argv[8]), minc = atof(argv[9]), maxc = atof(argv[10]); int filt = atoi(argv[11]);
  size_t mx = W; if (dims >= 2 && H > mx) mx = H; if (dims >= 3 && D > mx) mx = D;
  int levels = 1; while (((size_t)1 << levels) <= mx) ++levels;
  cudaFree(0);
  { float pc[3]={0.37f,0.61f,0.43f}; if (const char* e = getenv("TGP")) sscanf(e, "%f,%f,%f", &pc[0], &pc[1], &pc[2]); cudaMemcpyToSymbol(PC, pc, 12); }
  cudaChannelFormatDesc d = cudaCreateChannelDesc<float>();
  cudaMipmappedArray_t mip;
  cudaExtent ext = make_cudaExtent(W, dims >= 2 ? H : 0, dims >= 3 ? D : 0);
  if (cudaMallocMipmappedArray(&mip, &d, ext, levels) != cudaSuccess) { fprintf(stderr, "alloc fail\n"); return 1; }
  for (int l = 0; l < levels; ++l) {
    cudaArray_t a; cudaGetMipmappedArrayLevel(&a, mip, l);
    size_t w = W >> l ? W >> l : 1, h = dims >= 2 ? (H >> l ? H >> l : 1) : 1, dd = dims >= 3 ? (D >> l ? D >> l : 1) : 1;
    std::vector<float> host(w * h * dd, (float)l);
    if (dims == 1) cudaMemcpy2DToArray(a, 0, 0, host.data(), w * 4, w * 4, 1, cudaMemcpyHostToDevice);
    else if (dims == 2) cudaMemcpy2DToArray(a, 0, 0, host.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice);
    else { cudaMemcpy3DParms p = {}; p.srcPtr = make_cudaPitchedPtr(host.data(), w * 4, w, h); p.dstArray = a; p.extent = make_cudaExtent(w, h, dd); p.kind = cudaMemcpyHostToDevice; cudaMemcpy3D(&p); }
  }
  cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeMipmappedArray; rd.res.mipmap.mipmap = mip;
  cudaTextureDesc td = {};
  td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
  td.filterMode = filt ? cudaFilterModeLinear : cudaFilterModePoint;
  td.mipmapFilterMode = cudaFilterModeLinear;
  td.readMode = cudaReadModeElementType; td.normalizedCoords = getenv("NORM") ? atoi(getenv("NORM")) : 1;
  td.maxAnisotropy = aniso; td.mipmapLevelBias = bias; td.minMipmapLevelClamp = minc; td.maxMipmapLevelClamp = maxc;
  cudaTextureObject_t t;
  if (cudaCreateTextureObject(&t, &rd, &td, nullptr) != cudaSuccess) { fprintf(stderr, "tex fail\n"); return 1; }
  FILE* f = fopen(argv[6], "rb"); fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  int n = sz / 36; std::vector<float> in(n * 9); if (fread(in.data(), 4, n * 9, f) != (size_t)n * 9) return 3; fclose(f);
  float *dg, *dout; cudaMalloc(&dg, n * 36); cudaMalloc(&dout, n * 4);
  cudaMemcpy(dg, in.data(), n * 36, cudaMemcpyHostToDevice);
  if (dims == 1) k<1><<<(n + 255) / 256, 256>>>(t, dg, dout, n);
  else if (dims == 2) k<2><<<(n + 255) / 256, 256>>>(t, dg, dout, n);
  else k<3><<<(n + 255) / 256, 256>>>(t, dg, dout, n);
  std::vector<float> out(n); cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "kernel error\n"); return 1; }
  f = fopen(argv[7], "wb"); fwrite(out.data(), 4, n, f); fclose(f);
  return 0;
}
