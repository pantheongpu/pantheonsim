// A HIP program as hipcc builds it: device code inside the executable, kernels
// launched with the chevron syntax, and the runtime reached through the real
// HIP headers -- so every structure it hands the runtime is laid out the way
// those headers lay it out. It exercises what the pantheon workloads use:
// device properties, a launch, copies staged through host memory, a second
// device reached as a peer, and a graph captured from a stream and replayed.
//
// Built by build.sh, which needs ROCm; the executable is checked in beside it
// so the test that runs it needs nothing but the shim.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call)                                                                        \
  do {                                                                                     \
    hipError_t e_ = (call);                                                                \
    if (e_ != hipSuccess) {                                                                \
      std::printf("FAIL %s: %s (%s)\n", #call, hipGetErrorString(e_), hipGetErrorName(e_)); \
      return 1;                                                                            \
    }                                                                                      \
  } while (0)

__global__ void scale_add(const float* a, const float* b, float* out, float k, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * k + b[i];
}

__global__ void bump(int* slots, int by) { slots[threadIdx.x] += by; }

int main() {
  const int n = 3000;

  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, 0));
  std::printf("device %s %s warp %d CUs %d threads/CU %d L2 %d\n", props.name, props.gcnArchName, props.warpSize,
              props.multiProcessorCount, props.maxThreadsPerMultiProcessor, props.l2CacheSize);

  // A launch, with its inputs staged through pinned host memory.
  float *ha = nullptr, *hb = nullptr;
  CHECK(hipHostMalloc(reinterpret_cast<void**>(&ha), n * sizeof(float)));
  CHECK(hipHostMalloc(reinterpret_cast<void**>(&hb), n * sizeof(float)));
  for (int i = 0; i < n; ++i) {
    ha[i] = 0.5f * i;
    hb[i] = 3.0f - i;
  }
  float *a, *b, *out;
  CHECK(hipMalloc(&a, n * sizeof(float)));
  CHECK(hipMalloc(&b, n * sizeof(float)));
  CHECK(hipMalloc(&out, n * sizeof(float)));
  CHECK(hipMemcpy(a, ha, n * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(b, hb, n * sizeof(float), hipMemcpyHostToDevice));
  scale_add<<<(n + 255) / 256, 256>>>(a, b, out, 4.0f, n);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  std::vector<float> got(n);
  CHECK(hipMemcpy(got.data(), out, n * sizeof(float), hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; ++i) wrong += got[i] != ha[i] * 4.0f + hb[i];
  std::printf("chevron launch wrong %d of %d\n", wrong, n);

  // A graph: one launch captured from a stream, replayed three times. The
  // capture itself must not run the kernel, and each replay runs it once.
  int* slots;
  CHECK(hipMalloc(&slots, 64 * sizeof(int)));
  CHECK(hipMemset(slots, 0, 64 * sizeof(int)));
  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  hipGraph_t graph;
  hipGraphExec_t exec;
  CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  bump<<<1, 64, 0, stream>>>(slots, 2);
  CHECK(hipStreamEndCapture(stream, &graph));
  int after_capture = -1;
  CHECK(hipMemcpy(&after_capture, slots + 63, sizeof(int), hipMemcpyDeviceToHost));
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int r = 0; r < 3; ++r) CHECK(hipGraphLaunch(exec, stream));
  CHECK(hipStreamSynchronize(stream));
  int after_replay = -1;
  CHECK(hipMemcpy(&after_replay, slots + 63, sizeof(int), hipMemcpyDeviceToHost));
  std::printf("graph captured %d replayed %d\n", after_capture, after_replay);
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));

  // A second device, reached as a peer: the result copied across and back.
  int devices = 0;
  CHECK(hipGetDeviceCount(&devices));
  if (devices > 1) {
    int can = 0;
    CHECK(hipDeviceCanAccessPeer(&can, 0, 1));
    CHECK(hipDeviceEnablePeerAccess(1, 0));
    const hipError_t again = hipDeviceEnablePeerAccess(1, 0);
    CHECK(hipSetDevice(1));
    float* there;
    CHECK(hipMalloc(&there, n * sizeof(float)));
    CHECK(hipMemcpyPeerAsync(there, 1, out, 0, n * sizeof(float), 0));
    std::vector<float> back(n);
    CHECK(hipMemcpy(back.data(), there, n * sizeof(float), hipMemcpyDeviceToHost));
    std::printf("peer can %d, enabling twice says %s, copy intact %d\n", can, hipGetErrorName(again),
                std::memcmp(back.data(), got.data(), n * sizeof(float)) == 0);
    CHECK(hipFree(there));
    CHECK(hipSetDevice(0));
  }

  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipFree(out));
  CHECK(hipFree(slots));
  CHECK(hipHostFree(ha));
  CHECK(hipHostFree(hb));
  return wrong ? 1 : 0;
}
