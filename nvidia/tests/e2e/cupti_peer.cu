// Copies between two devices that can reach each other's memory: what CUPTI
// records for them (CUPTI_ACTIVITY_KIND_MEMCPY2, the peer-to-peer copy kind)
// and what it records for the same copies once peer access is switched off
// (two copies through the host, as on devices with no path between them).
// Needs two GPUs with a peer path (NVLink, or PCIe peer-to-peer); where there
// is none the case says so and prints nothing else, which is what the two
// RTX 3060s of the machine this was developed on give. Run against NVIDIA's
// libcupti on such a pair it gives nvidia/tests/data/cupti_peer.expected.
#include "cupti_test_util.h"

using cupti_test::fmt;
using cupti_test::out;

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

#define CKE(x)                       \
  do {                               \
    cudaError_t e_ = (x);            \
    if (e_) return e_;               \
  } while (0)

namespace {

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    switch (static_cast<int>(r->kind)) {
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto* m = reinterpret_cast<const MemcpyRecord*>(r);
        out(fmt("  memcpy kind=%d src=%s dst=%s bytes=%llu device=%u context=#x%u stream=#s%u corr=#c%u order=%s",
                (int)m->copyKind, cupti_test::mem_kind(m->srcKind), cupti_test::mem_kind(m->dstKind),
                (unsigned long long)m->bytes, m->deviceId, m->contextId, m->streamId, m->correlationId,
                m->end >= m->start ? "ok" : "BAD"));
        break;
      }
      case 22: {   // CUPTI_ACTIVITY_KIND_MEMCPY2
        const auto* m = reinterpret_cast<const CUpti_ActivityMemcpyPtoP4*>(r);
        out(fmt("  peer copy kind=%d src=%s dst=%s flags=%u bytes=%llu device=%u src=%u/#x%u dst=%u/#x%u context=#x%u stream=#s%u corr=#c%u order=%s",
                (int)m->copyKind, cupti_test::mem_kind(m->srcKind), cupti_test::mem_kind(m->dstKind), (unsigned)m->flags,
                (unsigned long long)m->bytes, m->deviceId, m->srcDeviceId, m->srcContextId, m->dstDeviceId,
                m->dstContextId, m->contextId, m->streamId, m->correlationId, m->end >= m->start ? "ok" : "BAD"));
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        const char* name = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
        out(fmt("  runtime %s corr=#c%u", name ? name : "?", a->correlationId));
        break;
      }
      default: break;
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  CK(cudaGetDeviceCount(&devices));
  int ab = 0, ba = 0;
  if (devices >= 2) {
    CK(cudaDeviceCanAccessPeer(&ab, 0, 1));
    CK(cudaDeviceCanAccessPeer(&ba, 1, 0));
  }
  out(fmt("peer path between devices 0 and 1: %s", ab && ba ? "yes" : "no"));
  if (!(ab && ba)) {
    cupti_test::print_all();
    std::printf("# end\n");
    return 0;
  }
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  const int kinds[] = {CUPTI_ACTIVITY_KIND_MEMCPY, 22, CUPTI_ACTIVITY_KIND_RUNTIME};
  for (int k : kinds)
    out(fmt("enable kind %d: %s", k, cuptiActivityEnable(static_cast<CUpti_ActivityKind>(k)) == CUPTI_SUCCESS ? "yes" : "no"));

  const size_t keep = cupti_test::lines().size();
  float *a = nullptr, *b = nullptr;
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&a, 1 << 16));
  CK(cudaSetDevice(1));
  CK(cudaMalloc(&b, 1 << 16));
  CK(cudaSetDevice(0));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cuptiActivityFlushAll(0);
  cupti_test::lines().resize(keep);   // what the setup made is not under test

  const auto copies = [&](const char* title) -> cudaError_t {
    out(std::string("# ") + title);
    cudaError_t e = cudaMemcpyPeer(b, 1, a, 0, 4096);
    if (e) return e;
    if ((e = cudaMemcpyPeerAsync(b, 1, a, 0, 8192, s))) return e;
    if ((e = cudaStreamSynchronize(s))) return e;
    if ((e = cudaMemcpy(b, a, 12288, cudaMemcpyDefault))) return e;
    if ((e = cudaMemcpyAsync(b, a, 16384, cudaMemcpyDeviceToDevice, s))) return e;
    if ((e = cudaStreamSynchronize(s))) return e;
    // And the other way, from the stream of the device that receives.
    CKE(cudaSetDevice(1));
    cudaStream_t t;
    CKE(cudaStreamCreate(&t));
    if ((e = cudaMemcpyPeerAsync(a, 0, b, 1, 4096, t))) return e;
    if ((e = cudaStreamSynchronize(t))) return e;
    CKE(cudaStreamDestroy(t));
    CKE(cudaSetDevice(0));
    cuptiActivityFlushAll(0);
    return cudaSuccess;
  };

  if (cudaError_t e = copies("without peer access enabled")) { std::fprintf(stderr, "FAIL copies: %s\n", cudaGetErrorString(e)); return 1; }
  CK(cudaSetDevice(0));
  CK(cudaDeviceEnablePeerAccess(1, 0));
  CK(cudaSetDevice(1));
  CK(cudaDeviceEnablePeerAccess(0, 0));
  CK(cudaSetDevice(0));
  if (cudaError_t e = copies("with peer access enabled")) { std::fprintf(stderr, "FAIL copies: %s\n", cudaGetErrorString(e)); return 1; }

  // A 3-D copy between devices, and one that only one side may reach.
  out("# a 3-D copy between devices");
  {
    cudaMemcpy3DPeerParms p = {};
    p.srcPtr = make_cudaPitchedPtr(a, 256, 64, 4);
    p.srcDevice = 0;
    p.dstPtr = make_cudaPitchedPtr(b, 256, 64, 4);
    p.dstDevice = 1;
    p.extent = make_cudaExtent(256, 4, 2);
    CK(cudaMemcpy3DPeer(&p));
    CK(cudaDeviceSynchronize());
    cuptiActivityFlushAll(0);
  }
  CK(cudaSetDevice(0));
  CK(cudaDeviceDisablePeerAccess(1));
  CK(cudaSetDevice(1));
  CK(cudaDeviceDisablePeerAccess(0));
  CK(cudaSetDevice(0));
  if (cudaError_t e = copies("peer access disabled again")) { std::fprintf(stderr, "FAIL copies: %s\n", cudaGetErrorString(e)); return 1; }

  for (int k : kinds) cuptiActivityDisable(static_cast<CUpti_ActivityKind>(k));
  CK(cudaStreamDestroy(s));
  CK(cudaFree(a));
  CK(cudaSetDevice(1));
  CK(cudaFree(b));
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
