// Formatted surface stores (sust.p), checked against an RTX 3060 (sm_86): the 32-bit values of R, G, B
// and A converted to the surface's channel format -- unsigned and signed integer channels clamped to
// their range, 32-bit floats stored as they are, 16-bit floats rounded toward zero -- for every format
// CUDA gives a surface, with one to four values, on 1D, 2D and 3D surfaces, with .clamp and .zero
// out-of-range coordinates. Each case's surface contents are hashed and compared with the card's,
// recorded in surface_formatted_expected.inc (`surface_formatted --print` prints them). Prints PASS
// on the last line, and runs the same on a GPU.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

struct Expected { const char* tag; unsigned long long hash; };
static const Expected kExpected[] = {
#include "surface_formatted_expected.inc"
};
static bool g_print = false;
static int g_fails = 0;

// One thread's sust.p, of NV values, with policy POL (0 .trap, 1 .clamp, 2 .zero) at (x, y, z).
template <int DIM, int NV, int POL>
__device__ void put(cudaSurfaceObject_t s, int x, int y, int z, unsigned a, unsigned b, unsigned c, unsigned d) {
#define SP(DIMS, VEC, POLS, COORD, VALS, ...) asm volatile("sust.p." DIMS "." VEC "b32." POLS " [%0, {" COORD "}], {" VALS "};" :: "l"(s), __VA_ARGS__)
  if (DIM == 1) {
    if (NV == 4 && POL == 0) SP("1d", "v4.", "trap", "%1", "%2,%3,%4,%5", "r"(x), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 4 && POL == 1) SP("1d", "v4.", "clamp", "%1", "%2,%3,%4,%5", "r"(x), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 4) SP("1d", "v4.", "zero", "%1", "%2,%3,%4,%5", "r"(x), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 2 && POL == 1) SP("1d", "v2.", "clamp", "%1", "%2,%3", "r"(x), "r"(a), "r"(b));
    else if (NV == 2) SP("1d", "v2.", "zero", "%1", "%2,%3", "r"(x), "r"(a), "r"(b));
    else if (POL == 1) SP("1d", "", "clamp", "%1", "%2", "r"(x), "r"(a));
    else SP("1d", "", "zero", "%1", "%2", "r"(x), "r"(a));
  } else if (DIM == 2) {
    if (NV == 4 && POL == 0) SP("2d", "v4.", "trap", "%1,%2", "%3,%4,%5,%6", "r"(x), "r"(y), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 4 && POL == 1) SP("2d", "v4.", "clamp", "%1,%2", "%3,%4,%5,%6", "r"(x), "r"(y), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 4) SP("2d", "v4.", "zero", "%1,%2", "%3,%4,%5,%6", "r"(x), "r"(y), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 2 && POL == 1) SP("2d", "v2.", "clamp", "%1,%2", "%3,%4", "r"(x), "r"(y), "r"(a), "r"(b));
    else if (NV == 2) SP("2d", "v2.", "zero", "%1,%2", "%3,%4", "r"(x), "r"(y), "r"(a), "r"(b));
    else if (POL == 1) SP("2d", "", "clamp", "%1,%2", "%3", "r"(x), "r"(y), "r"(a));
    else SP("2d", "", "zero", "%1,%2", "%3", "r"(x), "r"(y), "r"(a));
  } else {
    if (NV == 4 && POL == 0) SP("3d", "v4.", "trap", "%1,%2,%3,%3", "%4,%5,%6,%7", "r"(x), "r"(y), "r"(z), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 4 && POL == 1) SP("3d", "v4.", "clamp", "%1,%2,%3,%3", "%4,%5,%6,%7", "r"(x), "r"(y), "r"(z), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (NV == 4) SP("3d", "v4.", "zero", "%1,%2,%3,%3", "%4,%5,%6,%7", "r"(x), "r"(y), "r"(z), "r"(a), "r"(b), "r"(c), "r"(d));
    else if (POL == 1) SP("3d", "v2.", "clamp", "%1,%2,%3,%3", "%4,%5", "r"(x), "r"(y), "r"(z), "r"(a), "r"(b));
    else SP("3d", "v2.", "zero", "%1,%2,%3,%3", "%4,%5", "r"(x), "r"(y), "r"(z), "r"(a), "r"(b));
  }
#undef SP
}

// In range: every texel written once, by its own thread (several threads writing one texel would race).
template <int DIM, int NV, int POL> __global__ void store(cudaSurfaceObject_t s, const unsigned* in, int w, int h, int depth) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * (DIM >= 2 ? h : 1) * (DIM == 3 ? depth : 1)) return;
  put<DIM, NV, POL>(s, i % w, (i / w) % h, i / (w * h), in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]);
}
// Out of range, one thread after another: a coordinate past the surface is clamped to its edge or
// dropped, where the thread's order decides what a texel holds in the end.
template <int DIM, int NV, int POL> __global__ void store_outside(cudaSurfaceObject_t s, const unsigned* in, int w, int h, int depth) {
  const int xs[] = {-1, 0, w, w + 3, 2, -5, w - 1, w + 40}, ys[] = {0, -2, 1, h, h + 7, -9, h - 1, 3}, zs[] = {0, 0, 1, 2, 0, 5, -1, depth + 4};
  if (threadIdx.x) return;
  for (int j = 0; j < 8; ++j)
    put<DIM, NV, POL>(s, xs[j], DIM >= 2 ? ys[j] : 0, DIM == 3 ? zs[j] : 0, in[4 * (100 + j)], in[4 * (100 + j) + 1], in[4 * (100 + j) + 2], in[4 * (100 + j) + 3]);
}

struct Fmt { const char* name; cudaChannelFormatDesc cd; };

template <int DIM, int NV, int POL>
static void run_case(const Fmt& f, const std::vector<unsigned>& in, int w, int h, int depth) {
  cudaArray_t arr;
  cudaExtent ext = make_cudaExtent(w, DIM >= 2 ? h : 0, DIM == 3 ? depth : 0);
  if (cudaMalloc3DArray(&arr, &f.cd, ext, cudaArraySurfaceLoadStore) != cudaSuccess) { std::printf("cannot allocate %s\n", f.name); std::exit(2); }
  const size_t bytes = size_t(f.cd.x + f.cd.y + f.cd.z + f.cd.w) / 8, texels = size_t(w) * (DIM >= 2 ? h : 1) * (DIM == 3 ? depth : 1);
  std::vector<uint8_t> zeros(texels * bytes, 0xAB);
  cudaMemcpy3DParms up{};
  up.srcPtr = make_cudaPitchedPtr(zeros.data(), w * bytes, w, DIM >= 2 ? h : 1);
  up.dstArray = arr; up.extent = make_cudaExtent(w, DIM >= 2 ? h : 1, DIM == 3 ? depth : 1); up.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&up);
  cudaResourceDesc r{}; r.resType = cudaResourceTypeArray; r.res.array.array = arr;
  cudaSurfaceObject_t s; cudaCreateSurfaceObject(&s, &r);
  unsigned* din; cudaMalloc(&din, in.size() * 4); cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
  const int n = w * (DIM >= 2 ? h : 1) * (DIM == 3 ? depth : 1);
  store<DIM, NV, POL><<<(n + 63) / 64, 64>>>(s, din, w, h, depth);
  if (POL != 0) store_outside<DIM, NV, POL><<<1, 32>>>(s, din, w, DIM >= 2 ? h : 1, DIM == 3 ? depth : 1);
  const cudaError_t e = cudaDeviceSynchronize();
  std::vector<uint8_t> got(texels * bytes);
  cudaMemcpy3DParms dn{};
  dn.srcArray = arr; dn.dstPtr = make_cudaPitchedPtr(got.data(), w * bytes, w, DIM >= 2 ? h : 1);
  dn.extent = up.extent; dn.kind = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&dn);
  unsigned long long hsh = 1469598103934665603ull;
  for (uint8_t b : got) { hsh ^= b; hsh *= 1099511628211ull; }
  const char* oob[] = {"trap", "clamp", "zero"};
  const std::string tag = std::string(f.name) + " " + std::to_string(DIM) + "d v" + std::to_string(NV) + " " + oob[POL];
  if (g_print) std::printf("    {\"%s\", 0x%016llxull},\n", tag.c_str(), hsh);
  else {
    const Expected* x = nullptr;
    for (const Expected& k : kExpected) if (tag == k.tag) x = &k;
    const bool ok = x && x->hash == hsh && e == cudaSuccess;
    std::printf("%-26s %016llx %s\n", tag.c_str(), hsh, ok ? "ok" : x ? "MISMATCH" : "MISSING");
    g_fails += !ok;
  }
  cudaDestroySurfaceObject(s); cudaFreeArray(arr); cudaFree(din);
}

int main(int argc, char** argv) {
  g_print = argc > 1 && !std::strcmp(argv[1], "--print");
  const Fmt fmts[] = {
      {"u8x4", cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned)}, {"s8x4", cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned)},
      {"u16x4", cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned)}, {"s16x4", cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSigned)},
      {"u32x4", cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned)}, {"s32x4", cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned)},
      {"f32x4", cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat)}, {"f16x4", cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat)},
      {"u8x1", cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned)}, {"f32x1", cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat)},
      {"u16x2", cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned)}, {"f32x2", cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat)},
      {"s8x2", cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSigned)}, {"f16x2", cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindFloat)}};
  // Values: random integers and floats of every magnitude, and the edge cases of each conversion.
  const int w = 8, h = 8, depth = 2, n = 256;
  std::vector<unsigned> in(4 * n);
  uint32_t rs = 20261009;
  auto nx = [&]() { rs = rs * 1664525u + 1013904223u; return rs; };
  for (auto& v : in) {
    const int m = nx() % 6;
    if (m == 0) v = nx() & 0xff; else if (m == 1) v = nx() & 0xffff; else if (m == 2) v = nx();
    else if (m == 3) { const float f = (int(nx() % 40000) - 20000) / 37.0f; std::memcpy(&v, &f, 4); }
    else if (m == 4) { const float f = (int(nx() % 200) - 100) / 7.0f; std::memcpy(&v, &f, 4); }
    else { const float f = (nx() % 1000) * 1e5f; std::memcpy(&v, &f, 4); }
  }
  const unsigned special[] = {0x7fc00001u, 0xffc12345u, 0x7f800000u, 0xff800000u, 0x80000000u, 0x00000001u, 0x477fe000u, 0x477ff000u,
                              0x47800000u, 0x322bcc77u, 0xb22bcc77u, 0x33800000u, 0x33000000u, 0x38800000u, 0x387fe000u, 0xc77fe000u,
                              0x00000000u, 0x7fffffffu, 0x80000001u, 0xffffffffu, 0x000000ffu, 0x00000100u, 0x0000ffffu, 0x00010000u,
                              0x0000007fu, 0x00000080u, 0xffffff80u, 0xffffff7fu, 0xffff8000u, 0xffff7fffu, 0x7fff0000u, 0x80000000u,
                              0x7f800001u, 0x7fa00000u, 0x7f801000u, 0xff800001u, 0x7fbfffffu};
  for (size_t i = 0; i < sizeof special / sizeof *special; ++i) in[i] = special[i];   // the first texels' values
  for (const Fmt& f : fmts) {
    run_case<2, 4, 0>(f, in, w, h, depth);
    run_case<2, 2, 1>(f, in, w, h, depth);
    run_case<2, 1, 2>(f, in, w, h, depth);
    run_case<2, 4, 1>(f, in, w, h, depth);
    run_case<2, 4, 2>(f, in, w, h, depth);
    run_case<1, 4, 0>(f, in, w * h, 1, 1);
    run_case<1, 2, 1>(f, in, w * h, 1, 1);
    run_case<1, 1, 2>(f, in, w * h, 1, 1);
    run_case<3, 4, 0>(f, in, w, h, depth);
    run_case<3, 2, 1>(f, in, w, h, depth);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("CUDA error: %s\n", cudaGetErrorString(e)); ++g_fails; }
  if (g_print) return 0;
  std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
  return g_fails == 0 ? 0 : 1;
}
