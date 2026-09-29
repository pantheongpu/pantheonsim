// Texture fetches with integer coordinates, as an RTX 3060 performs them:
// the texel they name, whatever the texture's filter mode, and zero outside
// the extent whatever its address mode and border colour. tex1Dfetch from
// linear memory set up for linear filtering (CUDA Samples' convolutionFFT2D
// binds its buffers that way), and tex.2d.*.s32 on a 4x4 array with point
// and linear filtering, clamp and border addressing, a border colour of 7,
// and coordinates from -1 to 4. The float-coordinate fetches alongside show
// that the same textures do filter, clamp and use the border colour.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fetch(cudaTextureObject_t lin, float* o) { o[threadIdx.x] = tex1Dfetch<float>(lin, threadIdx.x); }

__global__ void sample(cudaTextureObject_t arr, float* o) {
  const int i = threadIdx.x;
  const int x = i % 6 - 1, y = i / 6 - 1;
  float r, g, b, a, fr;
  asm("tex.2d.v4.f32.s32 {%0,%1,%2,%3}, [%4, {%5, %6}];" : "=f"(r), "=f"(g), "=f"(b), "=f"(a) : "l"(arr), "r"(x), "r"(y));
  asm("{ .reg .f32 fx, fy, g2, b2, a2; cvt.rn.f32.s32 fx, %2; cvt.rn.f32.s32 fy, %3; add.f32 fx, fx, 0f3F000000;"
      " add.f32 fy, fy, 0f3F000000; tex.2d.v4.f32.f32 {%0, g2, b2, a2}, [%1, {fx, fy}]; }"
      : "=f"(fr) : "l"(arr), "r"(x), "r"(y));
  o[i] = r;
  o[36 + i] = fr;
}

int main() {
  int bad = 0;
  float h[16];
  for (int i = 0; i < 16; ++i) h[i] = i * 10.0f + 1;
  float* d;
  cudaMalloc(&d, sizeof h);
  cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
  float* o;
  cudaMalloc(&o, 72 * sizeof(float));

  cudaResourceDesc rl = {};
  rl.resType = cudaResourceTypeLinear;
  rl.res.linear.devPtr = d;
  rl.res.linear.desc = cudaCreateChannelDesc<float>();
  rl.res.linear.sizeInBytes = sizeof h;
  cudaTextureDesc tl = {};
  tl.filterMode = cudaFilterModeLinear;
  tl.readMode = cudaReadModeElementType;
  cudaTextureObject_t lin;
  cudaCreateTextureObject(&lin, &rl, &tl, nullptr);
  fetch<<<1, 16>>>(lin, o);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("tex1Dfetch failed\nFAIL\n"); return 1; }
  float got[72];
  cudaMemcpy(got, o, 16 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 16; ++i)
    if (got[i] != h[i] && bad++ < 10) std::printf("tex1Dfetch %d: %g, want %g\n", i, got[i], h[i]);

  cudaArray_t arr;
  auto cd = cudaCreateChannelDesc<float>();
  cudaMallocArray(&arr, &cd, 4, 4);
  cudaMemcpy2DToArray(arr, 0, 0, h, 16, 16, 4, cudaMemcpyHostToDevice);
  cudaResourceDesc ra = {};
  ra.resType = cudaResourceTypeArray;
  ra.res.array.array = arr;
  // What the card gave: integer coordinates, the same for every mode below,
  // and float coordinates under clamp and under a border colour of 7.
  const float ints[36] = {0, 0, 0, 0, 0, 0, 0, 1, 11, 21, 31, 0, 0, 41, 51, 61, 71, 0,
                          0, 81, 91, 101, 111, 0, 0, 121, 131, 141, 151, 0, 0, 0, 0, 0, 0, 0};
  const float clamped[36] = {1, 1, 11, 21, 31, 31, 1, 1, 11, 21, 31, 31, 41, 41, 51, 61, 71, 71,
                             81, 81, 91, 101, 111, 111, 121, 121, 131, 141, 151, 151, 121, 121, 131, 141, 151, 151};
  float bordered[36];
  for (int i = 0; i < 36; ++i) bordered[i] = ints[i] == 0 ? 7 : ints[i];
  for (int f = 0; f < 2; ++f)
    for (int border = 0; border < 2; ++border) {
      cudaTextureDesc td = {};
      td.filterMode = f ? cudaFilterModeLinear : cudaFilterModePoint;
      td.readMode = cudaReadModeElementType;
      td.addressMode[0] = td.addressMode[1] = border ? cudaAddressModeBorder : cudaAddressModeClamp;
      for (int c = 0; c < 4; ++c) td.borderColor[c] = 7;
      cudaTextureObject_t tex;
      cudaCreateTextureObject(&tex, &ra, &td, nullptr);
      sample<<<1, 36>>>(tex, o);
      if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("tex.2d failed\nFAIL\n"); return 1; }
      cudaMemcpy(got, o, sizeof got, cudaMemcpyDeviceToHost);
      for (int i = 0; i < 36; ++i) {
        const float want_f = border ? bordered[i] : clamped[i];
        if (got[i] != ints[i] && bad++ < 20)
          std::printf("%s %s, integer (%d,%d): %g, want %g\n", f ? "linear" : "point", border ? "border" : "clamp",
                      i % 6 - 1, i / 6 - 1, got[i], ints[i]);
        if (got[36 + i] != want_f && bad++ < 20)
          std::printf("%s %s, float (%d,%d): %g, want %g\n", f ? "linear" : "point", border ? "border" : "clamp",
                      i % 6 - 1, i / 6 - 1, got[36 + i], want_f);
      }
      cudaDestroyTextureObject(tex);
    }
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
