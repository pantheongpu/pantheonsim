// NPP's stream-context functions inside a captured CUDA graph (see graph_capture_common.h): the
// stream is the context's, set with nppGetStreamContext and hStream. Each call is recorded and runs at
// each launch of the graph over new inputs. run_graph_capture.sh npp "nppc,nppial,..." --card runs the
// same program on NVIDIA's NPP.
#include <nppdefs.h>
#include <nppcore.h>
#include <npps_arithmetic_and_logical_operations.h>
#include <npps_statistics_functions.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <nppi_filtering_functions.h>
#include <nppi_color_conversion.h>
#include <nppi_statistics_functions.h>
#include <nppi_support_functions.h>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { NppStatus s_ = (x); if (s_ != NPP_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)s_); ok = false; } } while (0)

int main() {
  Runner r;
  NppStreamContext ctx{};   // filled in by hand: CUDA 13's NPP no longer has nppGetStreamContext
  ctx.hStream = r.st;
  cudaGetDevice(&ctx.nCudaDeviceId);
  cudaDeviceGetAttribute(&ctx.nMultiProcessorCount, cudaDevAttrMultiProcessorCount, ctx.nCudaDeviceId);
  cudaDeviceGetAttribute(&ctx.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, ctx.nCudaDeviceId);
  cudaDeviceGetAttribute(&ctx.nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, ctx.nCudaDeviceId);
  int shared = 0;
  cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlock, ctx.nCudaDeviceId);
  ctx.nSharedMemPerBlock = shared;
  cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, ctx.nCudaDeviceId);
  cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, ctx.nCudaDeviceId);
  unsigned flags = 0;
  cudaStreamGetFlags(r.st, &flags);
  ctx.nStreamFlags = flags;
  const int W = 32, H = 16;
  const NppiSize roi{W, H};
  const size_t px = (size_t)W * H;
  {
    float *a = r.alloc<float>(px), *b = r.alloc<float>(px), *c = r.alloc<float>(px);
    r.run("nppiAdd_32f_C1R_Ctx + nppiMulC_32f_C1R_Ctx (a constant read at the call)", [&] {
      bool ok = true;
      fill(r.st, r.counter, a, px);
      fill(r.st, r.counter, b, px, 0.25f, 0, 1);
      OK(nppiAdd_32f_C1R_Ctx(a, W * 4, b, W * 4, c, W * 4, roi, ctx));
      OK(nppiMulC_32f_C1R_Ctx(c, W * 4, 2.5f, c, W * 4, roi, ctx));
      return ok;
    }, {{c, px, Dt::F32}}, 1e-5);
  }
  {
    unsigned char *a = r.alloc<unsigned char>(px), *b = r.alloc<unsigned char>(px);
    r.run("nppiAddC_8u_C1RSfs_Ctx + nppiSet_8u_C1R_Ctx", [&] {
      bool ok = true;
      fill(r.st, r.counter, a, px, 10.0f);
      OK(nppiAddC_8u_C1RSfs_Ctx(a, W, 7, a, W, roi, 0, ctx));
      OK(nppiSet_8u_C1R_Ctx(200, b, W, roi, ctx));
      return ok;
    }, {{a, px, Dt::Bytes}, {b, px, Dt::Bytes}});
  }
  {
    unsigned char* a = r.alloc<unsigned char>(px * 3);
    float* f = r.alloc<float>(px);
    unsigned char* gray = r.alloc<unsigned char>(px);
    r.run("nppiRGBToGray_8u_C3C1R_Ctx + nppiConvert_8u32f_C1R_Ctx", [&] {
      bool ok = true;
      fill(r.st, r.counter, a, px * 3, 20.0f);
      OK(nppiRGBToGray_8u_C3C1R_Ctx(a, W * 3, gray, W, roi, ctx));
      OK(nppiConvert_8u32f_C1R_Ctx(gray, W, f, W * 4, roi, ctx));
      return ok;
    }, {{gray, px, Dt::Bytes}, {f, px, Dt::F32}});
  }
  {
    unsigned char *src = r.alloc<unsigned char>(px), *dst = r.alloc<unsigned char>(px / 4);
    r.run("nppiResize_8u_C1R_Ctx", [&] {
      bool ok = true;
      fill(r.st, r.counter, src, px, 10.0f);
      OK(nppiResize_8u_C1R_Ctx(src, W, NppiSize{W, H}, NppiRect{0, 0, W, H}, dst, W / 2, NppiSize{W / 2, H / 2}, NppiRect{0, 0, W / 2, H / 2},
                               NPPI_INTER_LINEAR, ctx));
      return ok;
    }, {{dst, px / 4, Dt::Bytes}});
  }
  {
    unsigned char *src = r.alloc<unsigned char>(px), *dst = r.alloc<unsigned char>(px);
    r.run("nppiFilterBox_8u_C1R_Ctx", [&] {
      bool ok = true;
      fill(r.st, r.counter, src, px, 10.0f);
      OK(nppiFilterBox_8u_C1R_Ctx(src + 2 * W + 2, W, dst + 2 * W + 2, W, NppiSize{W - 4, H - 4}, NppiSize{3, 3}, NppiPoint{1, 1}, ctx));
      return ok;
    }, {{dst, px, Dt::Bytes}});
  }
  {
    float *a = r.alloc<float>(256), *b = r.alloc<float>(256), *c = r.alloc<float>(256);
    r.run("nppsAdd_32f_Ctx + nppsMulC_32f_Ctx", [&] {
      bool ok = true;
      fill(r.st, r.counter, a, 256);
      fill(r.st, r.counter, b, 256, 0.25f, 0, 1);
      OK(nppsAdd_32f_Ctx(a, b, c, 256, ctx));
      OK(nppsMulC_32f_Ctx(c, 3.0f, c, 256, ctx));
      return ok;
    }, {{c, 256, Dt::F32}}, 1e-5);
  }
  {
    // A reduction: needs a scratch buffer (sized beforehand) and writes a device scalar.
    float *a = r.alloc<float>(256), *sum = r.alloc<float>(1);
    size_t bytes = 0;
    nppsSumGetBufferSize_32f_Ctx(256, &bytes, ctx);
    unsigned char* scratch = r.alloc<unsigned char>(bytes + 16);
    r.run("nppsSum_32f_Ctx (a scratch buffer)", [&] {
      bool ok = true;
      fill(r.st, r.counter, a, 256, 0.25f, 3.0f);
      OK(nppsSum_32f_Ctx(a, 256, sum, scratch, ctx));
      return ok;
    }, {{sum, 1, Dt::F32}}, 1e-4);
  }
  return finish();
}
