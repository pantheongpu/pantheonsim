// suld/sust's out-of-range policies, .clamp and .zero, checked against an RTX
// 3060 (sm_86): loads and stores past each edge of 1D, 2D, 3D and layered 2D
// surfaces, 8-bit loads clamped by the byte, and .v4 accesses that run only
// partly past a row. The results are compared with the card's, recorded in
// kExpected (`surface_oob --print` prints them). Prints PASS on the last line,
// and runs the same on a GPU.
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

static const unsigned kExpected[] = {
#include "surface_oob_expected.inc"
};

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

__global__ void ld2(cudaSurfaceObject_t s, const int* xy, int n, int mode, unsigned* out) {
  const int i = threadIdx.x;
  if (i >= n) return;
  unsigned v;
  if (mode == 0) asm volatile("suld.b.2d.b32.clamp {%0}, [%1, {%2, %3}];" : "=r"(v) : "l"(s), "r"(xy[2 * i]), "r"(xy[2 * i + 1]));
  else asm volatile("suld.b.2d.b32.zero {%0}, [%1, {%2, %3}];" : "=r"(v) : "l"(s), "r"(xy[2 * i]), "r"(xy[2 * i + 1]));
  out[i] = v;
}
__global__ void ld2b8(cudaSurfaceObject_t s, const int* x, int n, unsigned* out) {
  const int i = threadIdx.x;
  if (i >= n) return;
  unsigned short v;
  asm volatile("suld.b.2d.b8.clamp {%0}, [%1, {%2, %3}];" : "=h"(v) : "l"(s), "r"(x[i]), "r"(1));
  out[i] = v & 0xFF;
}
__global__ void st2(cudaSurfaceObject_t s, int mode, int x, int y, unsigned v) {
  if (mode == 0) asm volatile("sust.b.2d.b32.clamp [%0, {%1, %2}], {%3};" :: "l"(s), "r"(x), "r"(y), "r"(v));
  else asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], {%3};" :: "l"(s), "r"(x), "r"(y), "r"(v));
}
__global__ void ld1v4(cudaSurfaceObject_t s, int x, int mode, unsigned* out) {
  unsigned a, b, c, d;
  if (mode == 0) asm volatile("suld.b.1d.v4.b32.clamp {%0,%1,%2,%3}, [%4, {%5}];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(s), "r"(x));
  else asm volatile("suld.b.1d.v4.b32.zero {%0,%1,%2,%3}, [%4, {%5}];" : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(s), "r"(x));
  out[0] = a; out[1] = b; out[2] = c; out[3] = d;
}
__global__ void st1v4(cudaSurfaceObject_t s, int x, int mode) {
  if (mode == 0) asm volatile("sust.b.1d.v4.b32.clamp [%0, {%1}], {%2,%3,%4,%5};" :: "l"(s), "r"(x), "r"(0xA0), "r"(0xA1), "r"(0xA2), "r"(0xA3));
  else asm volatile("sust.b.1d.v4.b32.zero [%0, {%1}], {%2,%3,%4,%5};" :: "l"(s), "r"(x), "r"(0xA0), "r"(0xA1), "r"(0xA2), "r"(0xA3));
}
__global__ void ld3(cudaSurfaceObject_t s, const int* c, int n, int mode, unsigned* out) {
  const int i = threadIdx.x;
  if (i >= n) return;
  unsigned v;
  if (mode == 0) asm volatile("suld.b.3d.b32.clamp {%0}, [%1, {%2, %3, %4, %4}];" : "=r"(v) : "l"(s), "r"(c[3 * i]), "r"(c[3 * i + 1]), "r"(c[3 * i + 2]));
  else asm volatile("suld.b.3d.b32.zero {%0}, [%1, {%2, %3, %4, %4}];" : "=r"(v) : "l"(s), "r"(c[3 * i]), "r"(c[3 * i + 1]), "r"(c[3 * i + 2]));
  out[i] = v;
}
__global__ void lda2(cudaSurfaceObject_t s, const int* c, int n, int mode, unsigned* out) {
  const int i = threadIdx.x;
  if (i >= n) return;
  unsigned v;
  if (mode == 0) asm volatile("suld.b.a2d.b32.clamp {%0}, [%1, {%2, %3, %4, %4}];" : "=r"(v) : "l"(s), "r"(c[3 * i]), "r"(c[3 * i + 1]), "r"(c[3 * i + 2]));
  else asm volatile("suld.b.a2d.b32.zero {%0}, [%1, {%2, %3, %4, %4}];" : "=r"(v) : "l"(s), "r"(c[3 * i]), "r"(c[3 * i + 1]), "r"(c[3 * i + 2]));
  out[i] = v;
}

static std::vector<unsigned> got;

static cudaSurfaceObject_t surface(cudaArray_t a) {
  cudaResourceDesc rd{};
  rd.resType = cudaResourceTypeArray;
  rd.res.array.array = a;
  cudaSurfaceObject_t s = 0;
  cudaCreateSurfaceObject(&s, &rd);
  return s;
}

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  cudaChannelFormatDesc cd = cudaCreateChannelDesc<unsigned>();
  unsigned* d;
  int* dc;
  CK(cudaMalloc(&d, 1024));
  CK(cudaMalloc(&dc, 1024));

  // 2D: 8 x 4 u32 texels holding 0x1000 * y + x (32 bytes a row).
  cudaArray_t a2;
  CK(cudaMallocArray(&a2, &cd, 8, 4, cudaArraySurfaceLoadStore));
  unsigned h2[32];
  auto reset2 = [&]() {
    for (int i = 0; i < 32; ++i) h2[i] = 0x1000u * (i / 8) + i % 8;
    return cudaMemcpy2DToArray(a2, 0, 0, h2, 32, 32, 4, cudaMemcpyHostToDevice);
  };
  CK(reset2());
  const cudaSurfaceObject_t s2 = surface(a2);
  {
    const int xs[] = {-8, -4, 0, 4, 28, 32, 36, 64, 1000}, ys[] = {-2, -1, 0, 3, 4, 9};
    std::vector<int> xy;
    for (int y : ys)
      for (int x : xs) xy.insert(xy.end(), {x, y});
    const int n = static_cast<int>(xy.size() / 2);
    CK(cudaMemcpy(dc, xy.data(), xy.size() * 4, cudaMemcpyHostToDevice));
    for (int mode = 0; mode < 2; ++mode) {
      ld2<<<1, 64>>>(s2, dc, n, mode, d);
      CK(cudaDeviceSynchronize());
      std::vector<unsigned> o(n);
      CK(cudaMemcpy(o.data(), d, n * 4, cudaMemcpyDeviceToHost));
      got.insert(got.end(), o.begin(), o.end());
    }
    const int x8[] = {-3, -1, 0, 1, 29, 30, 31, 32, 33, 35, 100};
    CK(cudaMemcpy(dc, x8, sizeof x8, cudaMemcpyHostToDevice));
    ld2b8<<<1, 32>>>(s2, dc, 11, d);
    CK(cudaDeviceSynchronize());
    unsigned o8[11];
    CK(cudaMemcpy(o8, d, sizeof o8, cudaMemcpyDeviceToHost));
    got.insert(got.end(), o8, o8 + 11);
    const int sx[] = {-4, -8, 32, 28, 36, 100, 8, 8}, sy[] = {1, 1, 1, 1, 1, 1, -1, 7};
    for (int mode = 0; mode < 2; ++mode)
      for (int k = 0; k < 8; ++k) {
        CK(reset2());
        st2<<<1, 1>>>(s2, mode, sx[k], sy[k], 0xABCD0000u + k);
        CK(cudaDeviceSynchronize());
        unsigned back[32];
        CK(cudaMemcpy2DFromArray(back, 32, a2, 0, 0, 32, 4, cudaMemcpyDeviceToHost));
        unsigned where = 0xFFFFFFFFu;   // the texel the store changed, as y * 8 + x
        for (int i = 0; i < 32; ++i)
          if (back[i] != 0x1000u * (i / 8) + i % 8) where = i;
        got.push_back(where);
      }
  }
  // 1D: 6 u32 texels holding i (24 bytes): a 16-byte .v4 at 16 runs past.
  cudaArray_t a1;
  CK(cudaMallocArray(&a1, &cd, 6, 0, cudaArraySurfaceLoadStore));
  unsigned h1[6];
  auto reset1 = [&]() {
    for (int i = 0; i < 6; ++i) h1[i] = i;
    return cudaMemcpy2DToArray(a1, 0, 0, h1, 24, 24, 1, cudaMemcpyHostToDevice);
  };
  CK(reset1());
  const cudaSurfaceObject_t s1 = surface(a1);
  for (int mode = 0; mode < 2; ++mode)
    for (int x : {-16, 0, 16, 32, 48}) {
      ld1v4<<<1, 1>>>(s1, x, mode, d);
      CK(cudaDeviceSynchronize());
      unsigned o[4];
      CK(cudaMemcpy(o, d, 16, cudaMemcpyDeviceToHost));
      got.insert(got.end(), o, o + 4);
    }
  for (int mode = 0; mode < 2; ++mode)
    for (int x : {-16, 16, 32, 48}) {
      CK(reset1());
      st1v4<<<1, 1>>>(s1, x, mode);
      CK(cudaDeviceSynchronize());
      unsigned b[6];
      CK(cudaMemcpy2DFromArray(b, 24, a1, 0, 0, 24, 1, cudaMemcpyDeviceToHost));
      got.insert(got.end(), b, b + 6);
    }
  // 3D (4 x 2 x 3) and layered 2D (4 x 2, 3 layers): 0x100 * z + 0x10 * y + x.
  unsigned h3[24];
  for (int z = 0; z < 3; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 4; ++x) h3[(z * 2 + y) * 4 + x] = 0x100 * z + 0x10 * y + x;
  cudaMemcpy3DParms p{};
  p.srcPtr = make_cudaPitchedPtr(h3, 16, 4, 2);
  p.extent = make_cudaExtent(4, 2, 3);
  p.kind = cudaMemcpyHostToDevice;
  cudaArray_t a3, al;
  CK(cudaMalloc3DArray(&a3, &cd, make_cudaExtent(4, 2, 3), cudaArraySurfaceLoadStore));
  p.dstArray = a3;
  CK(cudaMemcpy3D(&p));
  CK(cudaMalloc3DArray(&al, &cd, make_cudaExtent(4, 2, 3), cudaArraySurfaceLoadStore | cudaArrayLayered));
  p.dstArray = al;
  CK(cudaMemcpy3D(&p));
  const cudaSurfaceObject_t s3 = surface(a3), sl = surface(al);
  {
    const int c3[] = {4, 1, 1, 16, 1, 1, 4, 5, 1, 4, 1, 3, 4, 1, -1, -4, -1, 7};
    CK(cudaMemcpy(dc, c3, sizeof c3, cudaMemcpyHostToDevice));
    for (int mode = 0; mode < 2; ++mode) {
      ld3<<<1, 32>>>(s3, dc, 6, mode, d);
      CK(cudaDeviceSynchronize());
      unsigned o[6];
      CK(cudaMemcpy(o, d, sizeof o, cudaMemcpyDeviceToHost));
      got.insert(got.end(), o, o + 6);
    }
    const int cl[] = {1, 4, 1, 3, 4, 1, 5, 4, 1, -1, 4, 1, 2, 16, 3};   // layer, x, y
    CK(cudaMemcpy(dc, cl, sizeof cl, cudaMemcpyHostToDevice));
    for (int mode = 0; mode < 2; ++mode) {
      lda2<<<1, 32>>>(sl, dc, 5, mode, d);
      CK(cudaDeviceSynchronize());
      unsigned o[5];
      CK(cudaMemcpy(o, d, sizeof o, cudaMemcpyDeviceToHost));
      got.insert(got.end(), o, o + 5);
    }
  }

  if (print) {
    for (size_t i = 0; i < got.size(); ++i) printf("0x%x,%s", got[i], (i % 8 == 7) ? "\n" : " ");
    printf("\n");
    return 0;
  }
  const size_t n = sizeof kExpected / sizeof kExpected[0];
  int bad = 0;
  if (got.size() != n) {
    printf("%zu results, the card gave %zu\n", got.size(), n);
    bad = 1;
  }
  for (size_t i = 0; i < got.size() && i < n; ++i)
    if (got[i] != kExpected[i]) {
      if (bad++ < 20) printf("result %zu: 0x%x, the card gave 0x%x\n", i, got[i], kExpected[i]);
    }
  printf("%zu results\n", got.size());
  puts(bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
