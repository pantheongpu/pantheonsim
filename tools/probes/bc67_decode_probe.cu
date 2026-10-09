// Card probe: decode BC6H/BC7 blocks on the real texture unit and compare with the header's decoder.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include "vgpu/exec/block_compression.hpp"
using namespace vgpu::exec;

__global__ void fetch_uv(cudaTextureObject_t t, float4* out, int w, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < w * h) out[i] = tex2D<float4>(t, (i % w) + 0.5f, (i / w) + 0.5f);
}
static uint64_t g_lcg = 0x9E3779B97F4A7C15ull;
static uint8_t nb() { g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull; return (uint8_t)(g_lcg >> 56); }

static float half_to_float(uint16_t h) {
  int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = std::ldexp((float)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp((float)(m + 1024), e - 25);
  return s ? -v : v;
}

// Fetch all texels of blocks (bw x bh) of a format; returns floats [w*h*4].
static bool fetch(int kind, int bpx, int channels_bits[4], int bw, int bh, const std::vector<uint8_t>& blocks, bool srgb, std::vector<float>* res) {
  cudaChannelFormatDesc d = {channels_bits[0], channels_bits[1], channels_bits[2], channels_bits[3], cudaChannelFormatKindSigned};
  int k = kind; memcpy(&d.f, &k, 4);
  cudaArray_t a;
  if (cudaMallocArray(&a, &d, bw * 4, bh * 4) != cudaSuccess) { printf("array fail\n"); return false; }
  cudaMemcpy2DToArray(a, 0, 0, blocks.data(), bw * bpx, bw * bpx, bh, cudaMemcpyHostToDevice);
  cudaResourceDesc rd = {}; rd.resType = cudaResourceTypeArray; rd.res.array.array = a;
  cudaTextureDesc td = {}; td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp; td.filterMode = cudaFilterModePoint;
  bool norm = kind == 29 || kind == 30;
  td.readMode = norm ? cudaReadModeNormalizedFloat : cudaReadModeElementType; td.sRGB = srgb;
  cudaTextureObject_t t;
  cudaError_t e = cudaCreateTextureObject(&t, &rd, &td, nullptr);
  if (e) { printf("tex fail %d\n", (int)e); return false; }
  int n = bw * 4 * bh * 4;
  float4* dout; cudaMalloc(&dout, n * 16);
  fetch_uv<<<(n + 255) / 256, 256>>>(t, dout, bw * 4, bh * 4);
  std::vector<float4> out(n);
  cudaMemcpy(out.data(), dout, n * 16, cudaMemcpyDeviceToHost);
  res->resize(n * 4);
  for (int i = 0; i < n; ++i) { (*res)[i*4] = out[i].x; (*res)[i*4+1] = out[i].y; (*res)[i*4+2] = out[i].z; (*res)[i*4+3] = out[i].w; }
  cudaFree(dout); cudaDestroyTextureObject(t); cudaFreeArray(a);
  return true;
}

int main(int argc, char** argv) {
  std::string fmt = argc > 1 ? argv[1] : "bc7";
  int nblocks = argc > 2 ? atoi(argv[2]) : 4096;
  int force_mode = argc > 3 ? atoi(argv[3]) : -1;
  int seed = argc > 4 ? atoi(argv[4]) : 1;
  g_lcg ^= (uint64_t)seed * 0x1234567ull;
  cudaFree(0);
  int bw = 64, bh = (nblocks + 63) / 64; nblocks = bw * bh;
  std::vector<uint8_t> blocks(nblocks * 16);
  for (auto& x : blocks) x = nb();
  BlockFormat bf = fmt == "bc7" ? BlockFormat::BC7 : fmt == "bc6hu" ? BlockFormat::BC6HU : BlockFormat::BC6HS;
  for (int i = 0; i < nblocks; ++i) {
    uint8_t* p = &blocks[i * 16];
    int mode = force_mode;
    if (mode < 0) continue;
    if (bf == BlockFormat::BC7) { if (mode == 8) p[0] = 0; else p[0] = (uint8_t)((p[0] & ~((1 << (mode + 1)) - 1)) | (1 << mode)); }
    else {
      static const int codes[18] = {0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31};
      int c = codes[mode];
      if (c < 2) p[0] = (p[0] & ~3) | c; else p[0] = (p[0] & ~31) | c;
    }
  }
  std::vector<float> res;
  int bits[4] = {8,8,8,8};
  bool ok;
  if (bf == BlockFormat::BC7) ok = fetch(29, 16, bits, bw, bh, blocks, false, &res);
  else { int b3[4] = {16,16,16,0}; ok = fetch(bf == BlockFormat::BC6HU ? 27 : 28, 16, b3, bw, bh, blocks, false, &res); }
  if (!ok) return 1;
  long bad_blocks = 0, bad_texels = 0; int shown = 0;
  int W = bw * 4;
  for (int bi = 0; bi < nblocks; ++bi) {
    DecodedBlock db; decode_block(bf, &blocks[bi * 16], &db);
    bool badb = false;
    for (int t = 0; t < 16; ++t) {
      int x = (bi % bw) * 4 + t % 4, y = (bi / bw) * 4 + t / 4;
      const float* c = &res[(y * W + x) * 4];
      bool bad = false;
      for (int ch = 0; ch < 4; ++ch) {
        if (bf == BlockFormat::BC7) { int v = (int)std::lround(c[ch] * 255.0f); if (v != db.v[t][ch]) bad = true; }
        else { uint16_t h = db.v[t][ch]; float f = half_to_float(h); uint32_t a, b2; memcpy(&a, &f, 4); memcpy(&b2, &c[ch], 4); if (a != b2 && !(std::isnan(f) && std::isnan(c[ch]))) bad = true; }
      }
      if (bad) { badb = true; ++bad_texels;
        if (shown < 6) { ++shown; printf("block %d texel %d mode-bytes %02x: card", bi, t, blocks[bi*16]);
          for (int ch = 0; ch < 4; ++ch) { if (bf == BlockFormat::BC7) printf(" %d", (int)std::lround(c[ch]*255.0f)); else printf(" %g", c[ch]); }
          printf(" | ours");
          for (int ch = 0; ch < 4; ++ch) { if (bf == BlockFormat::BC7) printf(" %d", db.v[t][ch]); else printf(" %g", half_to_float(db.v[t][ch])); }
          printf("\n"); }
      }
    }
    if (badb) ++bad_blocks;
  }
  printf("%s mode %d: %d blocks, %ld differ (%ld texels)\n", fmt.c_str(), force_mode, nblocks, bad_blocks, bad_texels);
  return 0;
}
