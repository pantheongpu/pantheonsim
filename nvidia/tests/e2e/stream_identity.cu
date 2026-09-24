// Streams are distinct: two created streams are two streams, a capture on one
// does not take the work sent to another, and a stream answers questions about
// itself.
//
// Every cudaStreamCreate used to return the same handle -- 0x1, which is
// cudaStreamLegacy -- so a kernel launched on a second stream while the first
// was capturing was recorded into the first stream's graph and never ran. That
// is the first thing checked here.
#include <cuda_runtime.h>
#include <cstdio>
#include <thread>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

__global__ void set(int* p, int v) { *p = v; }

int main() {
  cudaStream_t a = nullptr, b = nullptr, c = nullptr;
  CK(cudaStreamCreate(&a));
  CK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
  CHECK(a != b);
  CHECK(a != cudaStreamLegacy && b != cudaStreamLegacy && a != nullptr);

  // ---- a capture belongs to one stream -----------------------------------
  int* x = nullptr;
  CK(cudaMalloc(&x, sizeof(int)));
  CK(cudaMemset(x, 0, sizeof(int)));
  CK(cudaStreamBeginCapture(a, cudaStreamCaptureModeRelaxed));
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusActive;
  CK(cudaStreamIsCapturing(b, &st));
  CHECK(st == cudaStreamCaptureStatusNone);
  set<<<1, 1, 0, b>>>(x, 7);            // runs now: b is not capturing
  CK(cudaStreamSynchronize(b));
  int h = -1;
  CK(cudaMemcpy(&h, x, sizeof h, cudaMemcpyDeviceToHost));
  CHECK(h == 7);
  set<<<1, 1, 0, a>>>(x, 9);            // recorded: a is
  cudaGraph_t g = nullptr;
  CK(cudaStreamEndCapture(a, &g));
  size_t nodes = 0;
  CK(cudaGraphGetNodes(g, nullptr, &nodes));
  CHECK(nodes == 1);                    // a's own launch, and only that
  CK(cudaMemcpy(&h, x, sizeof h, cudaMemcpyDeviceToHost));
  CHECK(h == 7);                        // and it has not run yet

  // ---- what a stream says about itself -------------------------------------
  unsigned flags = 99;
  CK(cudaStreamGetFlags(a, &flags));
  CHECK(flags == cudaStreamDefault);
  CK(cudaStreamGetFlags(b, &flags));
  CHECK(flags == cudaStreamNonBlocking);
  CK(cudaStreamGetFlags(0, &flags));
  CHECK(flags == cudaStreamDefault);
  WANT(cudaStreamCreateWithFlags(&c, 0x40u), cudaErrorInvalidValue);   // not a stream flag

  // Priorities are clamped to the device's range, which is what CUDA does.
  int least = 0, greatest = 0, prio = 12345;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  CK(cudaStreamCreateWithPriority(&c, cudaStreamNonBlocking, greatest - 5));
  CK(cudaStreamGetPriority(c, &prio));
  CHECK(prio == greatest);
  CK(cudaStreamGetFlags(c, &flags));
  CHECK(flags == cudaStreamNonBlocking);

  // cudaStreamGetDevice arrived in CUDA 12.8; older headers do not declare it.
#if CUDART_VERSION >= 12080
  int dev = -1;
  CK(cudaStreamGetDevice(a, &dev));
  CHECK(dev == 0);
#endif

  // Ids are unique, including between the default streams, and each host
  // thread's per-thread stream is its own.
  unsigned long long ia = 0, ib = 0, ilegacy = 0, ithis = 0, iother = 0;
  CK(cudaStreamGetId(a, &ia));
  CK(cudaStreamGetId(b, &ib));
  CK(cudaStreamGetId(cudaStreamLegacy, &ilegacy));
  CK(cudaStreamGetId(cudaStreamPerThread, &ithis));
  CHECK(ia != ib && ia != ilegacy && ib != ilegacy && ithis != ilegacy && ithis != ia);
  unsigned long long inull = 0;
  CK(cudaStreamGetId(0, &inull));
  CHECK(inull == ilegacy);              // 0 is the legacy stream
  cudaError_t other_rc = cudaSuccess;
  std::thread t([&] { other_rc = cudaStreamGetId(cudaStreamPerThread, &iother); });
  t.join();
  CK(other_rc);
  CHECK(iother != ithis);
  unsigned long long ithis_again = 0;
  CK(cudaStreamGetId(cudaStreamPerThread, &ithis_again));
  CHECK(ithis_again == ithis);          // and stable within a thread

  // ---- attributes read back as set, and copy between streams ---------------
  cudaStreamAttrValue v{};
  v.accessPolicyWindow.base_ptr = x;
  v.accessPolicyWindow.num_bytes = 4096;
  v.accessPolicyWindow.hitRatio = 0.5f;
  v.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  v.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  CK(cudaStreamSetAttribute(a, cudaStreamAttributeAccessPolicyWindow, &v));
  cudaStreamAttrValue got{};
  CK(cudaStreamGetAttribute(a, cudaStreamAttributeAccessPolicyWindow, &got));
  CHECK(got.accessPolicyWindow.base_ptr == x && got.accessPolicyWindow.num_bytes == 4096);
  CHECK(got.accessPolicyWindow.hitProp == cudaAccessPropertyPersisting);
  // Not set on b: it reads as unset.
  CK(cudaStreamGetAttribute(b, cudaStreamAttributeAccessPolicyWindow, &got));
  CHECK(got.accessPolicyWindow.base_ptr == nullptr && got.accessPolicyWindow.num_bytes == 0);
  CK(cudaStreamCopyAttributes(b, a));
  CK(cudaStreamGetAttribute(b, cudaStreamAttributeAccessPolicyWindow, &got));
  CHECK(got.accessPolicyWindow.base_ptr == x && got.accessPolicyWindow.num_bytes == 4096);
  // The priority attribute is the stream's priority.
  CK(cudaStreamGetAttribute(c, cudaStreamAttributePriority, &got));
  CHECK(got.priority == greatest);
  // A launch attribute that is not a stream attribute is refused.
  WANT(cudaStreamGetAttribute(a, cudaLaunchAttributeCooperative, &got), cudaErrorInvalidValue);

  // ---- the default streams are not a program's to destroy -------------------
  WANT(cudaStreamDestroy(0), cudaErrorInvalidResourceHandle);
  WANT(cudaStreamDestroy(cudaStreamPerThread), cudaErrorInvalidResourceHandle);

  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(a));
  CK(cudaStreamDestroy(b));
  CK(cudaStreamDestroy(c));
  CK(cudaFree(x));
  printf("PASS\n");
  return 0;
}
