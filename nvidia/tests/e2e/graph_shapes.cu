// Copies and fills with a shape, in graphs: what a node writes, and what a
// capture records.
//
// Two bugs are what this pins down. A fill node with 2- or 4-byte elements
// wrote the value's low byte into every byte -- a 4-byte fill of 2 gave
// 0x02020202 instead of 2 -- and the async copies that were not plain
// cudaMemcpyAsync (2D, to and from a symbol, between devices) ran the moment
// they were called on a capturing stream, so the graph came back without them
// and every replay skipped them. Both were silent: nothing failed, the answers
// were wrong.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

__device__ int dsym[64];

static cudaError_t count_nodes(cudaGraph_t g, size_t* n) { return cudaGraphGetNodes(g, nullptr, n); }

int main() {
  // ---- fills of 2- and 4-byte elements ---------------------------------------
  {
    const int n = 256;
    uint32_t* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(uint32_t)));
    cudaGraph_t g = nullptr;
    CK(cudaGraphCreate(&g, 0));
    cudaMemsetParams p{};
    p.dst = d;
    p.value = 0x12345678u;
    p.elementSize = 4;
    p.width = n;
    p.height = 1;
    cudaGraphNode_t fill = nullptr;
    CK(cudaGraphAddMemsetNode(&fill, g, nullptr, 0, &p));
    // It reads back as it was given: 4-byte elements, n of them.
    cudaMemsetParams back{};
    CK(cudaGraphMemsetNodeGetParams(fill, &back));
    CHECK(back.elementSize == 4 && back.width == static_cast<size_t>(n) && back.value == 0x12345678u);
    cudaGraphExec_t e = nullptr;
    CK(cudaGraphInstantiate(&e, g, 0));
    CK(cudaGraphLaunch(e, 0));
    CK(cudaDeviceSynchronize());
    uint32_t h[n];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      if (h[i] != 0x12345678u) {
        printf("FAIL 4-byte fill: element %d is 0x%08x\n", i, h[i]);
        return 1;
      }
    // Two-byte elements take the value's low two bytes and nothing else.
    cudaMemsetParams two = p;
    two.value = 0x1234abcdu;
    two.elementSize = 2;
    two.width = n * 2;
    CK(cudaGraphExecMemsetNodeSetParams(e, fill, &two));
    CK(cudaGraphLaunch(e, 0));
    CK(cudaDeviceSynchronize());
    uint16_t h16[n * 2];
    CK(cudaMemcpy(h16, d, sizeof h16, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n * 2; ++i)
      if (h16[i] != 0xabcdu) {
        printf("FAIL 2-byte fill: element %d is 0x%04x\n", i, h16[i]);
        return 1;
      }
    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    CK(cudaFree(d));
  }

  // ---- a 2D fill touches its rows and not the padding between them -----------
  {
    size_t pitch = 0;
    uint8_t* d = nullptr;
    const size_t width = 40, height = 6;
    CK(cudaMallocPitch(reinterpret_cast<void**>(&d), &pitch, width, height));
    CK(cudaMemset(d, 0x11, pitch * height));   // the sentinel the padding keeps
    cudaGraph_t g = nullptr;
    CK(cudaGraphCreate(&g, 0));
    cudaMemsetParams p{};
    p.dst = d;
    p.pitch = pitch;
    p.value = 0x7e;
    p.elementSize = 1;
    p.width = width;
    p.height = height;
    cudaGraphNode_t fill = nullptr;
    CK(cudaGraphAddMemsetNode(&fill, g, nullptr, 0, &p));
    cudaGraphExec_t e = nullptr;
    CK(cudaGraphInstantiate(&e, g, 0));
    // A 2D fill cannot be changed on the instantiated graph: the call documents
    // that both the old and the new operand must be 1D.
    cudaMemsetParams flat = p;
    flat.height = 1;
    WANT(cudaGraphExecMemsetNodeSetParams(e, fill, &flat), cudaErrorInvalidValue);
    CK(cudaGraphLaunch(e, 0));
    CK(cudaDeviceSynchronize());
    uint8_t h[4096];
    CHECK(pitch * height <= sizeof h);
    CK(cudaMemcpy(h, d, pitch * height, cudaMemcpyDeviceToHost));
    for (size_t y = 0; y < height; ++y)
      for (size_t x = 0; x < pitch; ++x) {
        const uint8_t want = x < width ? 0x7e : 0x11;
        if (h[y * pitch + x] != want) {
          printf("FAIL 2D fill: row %zu byte %zu is 0x%02x, expected 0x%02x\n", y, x,
                 h[y * pitch + x], want);
          return 1;
        }
      }
    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    CK(cudaFree(d));
  }

  // ---- a 3D copy of a sub-volume, positions and all -------------------------
  {
    // Source: 8 x 4 x 3 bytes at a 64-byte pitch; each byte is its coordinates.
    const size_t sx = 8, sy = 4, sz = 3, spitch = 64;
    uint8_t hsrc[spitch * sy * sz];
    for (size_t z = 0; z < sz; ++z)
      for (size_t y = 0; y < sy; ++y)
        for (size_t x = 0; x < spitch; ++x)
          hsrc[(z * sy + y) * spitch + x] = static_cast<uint8_t>(z * 64 + y * 16 + x);
    uint8_t *src = nullptr, *dst = nullptr;
    CK(cudaMalloc(&src, sizeof hsrc));
    CK(cudaMemcpy(src, hsrc, sizeof hsrc, cudaMemcpyHostToDevice));
    const size_t dpitch = 32, dy = 3, dz = 2;
    CK(cudaMalloc(&dst, dpitch * dy * dz));
    CK(cudaMemset(dst, 0, dpitch * dy * dz));
    // Copy 5 x 2 x 2 from (2, 1, 1) of the source to (3, 1, 0) of the destination.
    cudaMemcpy3DParms p{};
    p.srcPtr = cudaPitchedPtr{src, spitch, sx, sy};
    p.srcPos = cudaPos{2, 1, 1};
    p.dstPtr = cudaPitchedPtr{dst, dpitch, 16, dy};
    p.dstPos = cudaPos{3, 1, 0};
    p.extent = cudaExtent{5, 2, 2};
    p.kind = cudaMemcpyDeviceToDevice;
    cudaGraph_t g = nullptr;
    CK(cudaGraphCreate(&g, 0));
    cudaGraphNode_t copy = nullptr;
    CK(cudaGraphAddMemcpyNode(&copy, g, nullptr, 0, &p));
    cudaMemcpy3DParms back{};
    CK(cudaGraphMemcpyNodeGetParams(copy, &back));
    CHECK(back.srcPos.x == 2 && back.srcPos.z == 1 && back.dstPos.x == 3 && back.extent.depth == 2);
    CHECK(back.srcPtr.pitch == spitch && back.dstPtr.pitch == dpitch);
    cudaGraphExec_t e = nullptr;
    CK(cudaGraphInstantiate(&e, g, 0));
    // Nor can a 3D copy be swapped for another on the instantiated graph.
    WANT(cudaGraphExecMemcpyNodeSetParams(e, copy, &p), cudaErrorInvalidValue);
    CK(cudaGraphLaunch(e, 0));
    CK(cudaDeviceSynchronize());
    uint8_t h[32 * 3 * 2];
    CK(cudaMemcpy(h, dst, sizeof h, cudaMemcpyDeviceToHost));
    for (size_t z = 0; z < dz; ++z)
      for (size_t y = 0; y < dy; ++y)
        for (size_t x = 0; x < dpitch; ++x) {
          const bool inside = z < 2 && y >= 1 && y < 3 && x >= 3 && x < 8;
          const uint8_t want =
              inside ? static_cast<uint8_t>((z + 1) * 64 + (y - 1 + 1) * 16 + (x - 3 + 2)) : 0;
          if (h[(z * dy + y) * dpitch + x] != want) {
            printf("FAIL 3D copy at (%zu,%zu,%zu): 0x%02x, expected 0x%02x\n", x, y, z,
                   h[(z * dy + y) * dpitch + x], want);
            return 1;
          }
        }
    // A row wider than its pitch would overlap the next row, and is refused.
    cudaMemcpy3DParms wide = p;
    wide.extent.width = dpitch + 1;
    cudaGraphNode_t bad = nullptr;
    WANT(cudaGraphAddMemcpyNode(&bad, g, nullptr, 0, &wide), cudaErrorInvalidValue);
    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    CK(cudaFree(src));
    CK(cudaFree(dst));
  }

  // ---- what a capture records ----------------------------------------------
  //
  // A 2D copy, a 2D fill, a copy to and from a symbol, and a copy between two
  // devices, all on a capturing stream. None may happen at capture time; all
  // must happen when the graph runs, and again on the next launch.
  {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    CHECK(ndev >= 2);
    const size_t w = 16, rows = 4;
    size_t pa = 0, pb = 0;
    uint8_t *a = nullptr, *b = nullptr;
    CK(cudaMallocPitch(reinterpret_cast<void**>(&a), &pa, w, rows));
    CK(cudaMallocPitch(reinterpret_cast<void**>(&b), &pb, w, rows));
    CK(cudaMemset(a, 0x5a, pa * rows));
    CK(cudaMemset(b, 0x00, pb * rows));
    int host_in[64], host_out[64];
    for (int i = 0; i < 64; ++i) host_in[i] = 1000 + i;
    std::memset(host_out, 0, sizeof host_out);
    int* peer = nullptr;
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&peer, 64 * sizeof(int)));
    CK(cudaMemset(peer, 0, 64 * sizeof(int)));
    CK(cudaSetDevice(0));
    CK(cudaMemcpyToSymbol(dsym, host_in, sizeof host_in));   // a known starting value

    cudaStream_t s = nullptr;
    CK(cudaStreamCreate(&s));
    int fresh[64];
    for (int i = 0; i < 64; ++i) fresh[i] = 7 * i;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed));
    CK(cudaMemcpy2DAsync(b, pb, a, pa, w, rows, cudaMemcpyDeviceToDevice, s));   // a -> b
    CK(cudaMemset2DAsync(a, pa, 0x33, w, rows, s));                               // then a = 0x33
    CK(cudaMemcpyToSymbolAsync(dsym, fresh, sizeof fresh, 0, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyFromSymbolAsync(host_out, dsym, sizeof host_out, 0, cudaMemcpyDeviceToHost, s));
    int* dsym_addr = nullptr;
    CK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dsym_addr), dsym));
    CK(cudaMemcpyPeerAsync(peer, 1, dsym_addr, 0, 64 * sizeof(int), s));
    // Past the end of the symbol is refused, capture or not.
    WANT(cudaMemcpyToSymbolAsync(dsym, fresh, sizeof fresh, 4, cudaMemcpyHostToDevice, s),
         cudaErrorInvalidValue);
    cudaGraph_t g = nullptr;
    CK(cudaStreamEndCapture(s, &g));
    size_t nodes = 0;
    CK(count_nodes(g, &nodes));
    CHECK(nodes == 5);   // each operation is one node, as CUDA records them

    // Nothing happened yet.
    uint8_t hb[4096];
    CK(cudaMemcpy(hb, b, pb * rows, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < pb * rows; ++i) CHECK(hb[i] == 0x00);
    CHECK(host_out[0] == 0 && host_out[63] == 0);
    int got[64];
    CK(cudaMemcpyFromSymbol(got, dsym, sizeof got));
    CHECK(got[5] == 1005);

    cudaGraphExec_t e = nullptr;
    CK(cudaGraphInstantiate(&e, g, 0));
    CK(cudaGraphLaunch(e, s));
    CK(cudaStreamSynchronize(s));
    // b took a's rows before the fill; a is the fill; the symbol is the fresh
    // values; the host copy read them back; the peer copy took them to device 1.
    CK(cudaMemcpy(hb, b, pb * rows, cudaMemcpyDeviceToHost));
    for (size_t y = 0; y < rows; ++y)
      for (size_t x = 0; x < w; ++x) CHECK(hb[y * pb + x] == 0x5a);
    uint8_t ha[4096];
    CK(cudaMemcpy(ha, a, pa * rows, cudaMemcpyDeviceToHost));
    for (size_t y = 0; y < rows; ++y)
      for (size_t x = 0; x < w; ++x) CHECK(ha[y * pa + x] == 0x33);
    for (int i = 0; i < 64; ++i) CHECK(host_out[i] == 7 * i);
    int hp[64];
    CK(cudaMemcpy(hp, peer, sizeof hp, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 64; ++i) CHECK(hp[i] == 7 * i);

    // A second launch reads the buffers again: new fresh values arrive.
    for (int i = 0; i < 64; ++i) fresh[i] = -i;
    CK(cudaGraphLaunch(e, s));
    CK(cudaStreamSynchronize(s));
    for (int i = 0; i < 64; ++i) CHECK(host_out[i] == -i);
    CK(cudaMemcpy(hp, peer, sizeof hp, cudaMemcpyDeviceToHost));
    CHECK(hp[9] == -9);

    CK(cudaGraphExecDestroy(e));
    CK(cudaGraphDestroy(g));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(a));
    CK(cudaFree(b));
    CK(cudaFree(peer));
  }
  printf("PASS\n");
  return 0;
}
