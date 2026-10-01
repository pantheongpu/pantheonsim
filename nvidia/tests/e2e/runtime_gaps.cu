// Runtime API entry points CUDA's samples reach for, checked against what an
// RTX 3060's runtime answers: the deprecated 1D array copies, attaching
// managed memory to a stream, the generic cudaGraphAddNode, 3D peer copies
// with CUDA arrays, host memory flags, and the device attributes and
// properties the samples gate on. Every check passes on the card as well;
// where the simulated machine differs from that one by design (read-only
// registration is not supported) the check asks first.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

__global__ void add(float* p, int n, float v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += v;
}
static void CUDART_CB bump(void* user) { ++*static_cast<int*>(user); }

// ---- cudaMemcpyToArray and the rest: the array as its rows end to end ----
static void array_copies() {
  const cudaChannelFormatDesc fd = cudaCreateChannelDesc<float>();
  cudaArray_t arr, arr2, a1;
  IS(cudaMallocArray(&arr, &fd, 8, 4), cudaSuccess);
  IS(cudaMallocArray(&arr2, &fd, 8, 4), cudaSuccess);
  std::vector<float> src(64), back(32, -1.0f);
  for (int i = 0; i < 64; ++i) src[i] = static_cast<float>(i);
  std::vector<float> zero(32, 0.0f);
  IS(cudaMemcpy2DToArray(arr, 0, 0, zero.data(), 32, 32, 4, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpyToArray(arr, 0, 0, src.data(), 8 * 4, cudaMemcpyHostToDevice), cudaSuccess);
  // wOffset counts bytes: 4 is column 1.
  IS(cudaMemcpyToArray(arr, 4, 1, src.data(), 4 * 4, cudaMemcpyHostToDevice), cudaSuccess);
  // Longer than the rest of the row: it carries on into the next.
  IS(cudaMemcpyToArray(arr, 4, 2, src.data() + 16, 8 * 4, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpy2DFromArray(back.data(), 32, arr, 0, 0, 32, 4, cudaMemcpyDeviceToHost), cudaSuccess);
  check(back[0] == 0 && back[7] == 7 && back[8] == 0 && back[9] == 0 && back[12] == 3 && back[13] == 0,
        "cudaMemcpyToArray at a byte column and a row");
  check(back[16] == 0 && back[17] == 16 && back[23] == 22 && back[24] == 23 && back[25] == 0,
        "a copy past the end of a row carries on into the next");
  std::vector<float> fb(16, -1.0f);
  IS(cudaMemcpyFromArray(fb.data(), arr, 4, 1, 8 * 4, cudaMemcpyDeviceToHost), cudaSuccess);
  check(fb[0] == 0 && fb[3] == 3 && fb[4] == 0 && fb[7] == 0 && fb[8] == -1, "cudaMemcpyFromArray the same way");
  // Past the end of the array, the wrong direction, no array.
  IS(cudaMemcpyToArray(arr, 0, 3, src.data(), 16 * 4, cudaMemcpyHostToDevice), cudaErrorInvalidValue);
  IS(cudaMemcpyToArray(arr, 0, 4, src.data(), 4, cudaMemcpyHostToDevice), cudaErrorInvalidValue);
  IS(cudaMemcpyToArray(arr, 32, 0, src.data(), 4, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpyToArray(arr, 0, 0, src.data(), 0, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpyToArray(arr, 0, 0, src.data(), 4, cudaMemcpyDeviceToHost), cudaErrorInvalidMemcpyDirection);
  IS(cudaMemcpyToArray(arr, 0, 0, src.data(), 4, cudaMemcpyDefault), cudaSuccess);
  IS(cudaMemcpyFromArray(fb.data(), arr, 0, 3, 16 * 4, cudaMemcpyDeviceToHost), cudaErrorInvalidValue);
  IS(cudaMemcpyFromArray(fb.data(), arr, 0, 0, 4, cudaMemcpyHostToDevice), cudaErrorInvalidMemcpyDirection);
  IS(cudaMemcpyFromArray(nullptr, arr, 0, 0, 4, cudaMemcpyDeviceToHost), cudaErrorInvalidValue);
  IS(cudaMemcpyToArray(nullptr, 0, 0, src.data(), 4, cudaMemcpyHostToDevice), cudaErrorInvalidResourceHandle);
  cudaGetLastError();
  // Device memory both ways, the asynchronous forms, and array to array.
  float* dbuf = nullptr;
  IS(cudaMalloc(&dbuf, 256), cudaSuccess);
  IS(cudaMemcpy(dbuf, src.data(), 256, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpyToArray(arr, 0, 3, dbuf, 8 * 4, cudaMemcpyDeviceToDevice), cudaSuccess);
  IS(cudaMemcpyFromArray(dbuf + 32, arr, 0, 3, 8 * 4, cudaMemcpyDeviceToDevice), cudaSuccess);
  IS(cudaMemcpy(fb.data(), dbuf + 32, 32, cudaMemcpyDeviceToHost), cudaSuccess);
  check(fb[0] == 0 && fb[7] == 7, "device memory into an array and back");
  IS(cudaMemcpyToArrayAsync(arr, 0, 0, src.data() + 40, 8 * 4, cudaMemcpyHostToDevice, 0), cudaSuccess);
  IS(cudaMemcpyFromArrayAsync(fb.data(), arr, 0, 0, 8 * 4, cudaMemcpyDeviceToHost, 0), cudaSuccess);
  IS(cudaStreamSynchronize(0), cudaSuccess);
  check(fb[0] == 40 && fb[7] == 47, "the asynchronous forms");
  IS(cudaMemcpyArrayToArray(arr2, 0, 1, arr, 4, 0, 16, cudaMemcpyDeviceToDevice), cudaSuccess);
  IS(cudaMemcpyFromArray(fb.data(), arr2, 0, 1, 16, cudaMemcpyDeviceToHost), cudaSuccess);
  check(fb[0] == 41 && fb[3] == 44, "cudaMemcpyArrayToArray");
  IS(cudaMemcpyArrayToArray(arr2, 0, 3, arr, 0, 0, 64, cudaMemcpyDeviceToDevice), cudaErrorInvalidValue);
  cudaGetLastError();
  // A 1D array has one row.
  IS(cudaMallocArray(&a1, &fd, 16, 0), cudaSuccess);
  IS(cudaMemcpyToArray(a1, 0, 0, src.data(), 16 * 4, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpyToArray(a1, 0, 0, src.data(), 17 * 4, cudaMemcpyHostToDevice), cudaErrorInvalidValue);
  IS(cudaMemcpyToArray(a1, 0, 1, src.data(), 4, cudaMemcpyHostToDevice), cudaErrorInvalidValue);
  cudaGetLastError();
  cudaFreeArray(a1);
  cudaFreeArray(arr);
  cudaFreeArray(arr2);
  cudaFree(dbuf);
}

// ---- managed memory and streams, persisting L2 ----
static void attach() {
  float* m = nullptr;
  float* dm = nullptr;
  cudaStream_t s;
  IS(cudaMallocManaged(&m, 4096), cudaSuccess);
  IS(cudaMalloc(&dm, 4096), cudaSuccess);
  IS(cudaStreamCreate(&s), cudaSuccess);
  IS(cudaStreamAttachMemAsync(s, m, 0, cudaMemAttachSingle), cudaSuccess);
  IS(cudaStreamAttachMemAsync(s, m, 4096, cudaMemAttachGlobal), cudaSuccess);
  IS(cudaStreamAttachMemAsync(s, m, 1024, cudaMemAttachHost), cudaErrorInvalidValue);
  IS(cudaStreamAttachMemAsync(s, m + 16, 0, cudaMemAttachHost), cudaErrorInvalidValue);
  IS(cudaStreamAttachMemAsync(s, m, 0, 3), cudaErrorInvalidValue);
  IS(cudaStreamAttachMemAsync(0, m, 0, cudaMemAttachSingle), cudaErrorInvalidValue);
  IS(cudaStreamAttachMemAsync(0, m, 0, cudaMemAttachGlobal), cudaSuccess);
  IS(cudaStreamAttachMemAsync(s, dm, 0, cudaMemAttachGlobal), cudaErrorInvalidValue);
  IS(cudaStreamAttachMemAsync(s, nullptr, 0, cudaMemAttachGlobal), cudaErrorInvalidValue);
  cudaGetLastError();
  IS(cudaStreamSynchronize(s), cudaSuccess);
  IS(cudaCtxResetPersistingL2Cache(), cudaSuccess);
  cudaStreamDestroy(s);
  cudaFree(m);
  cudaFree(dm);
}

// CUDA 13 headers: cudaGraphNodeParams arrived in 12.2, and cudaGraphAddNode
// took its edge-data argument in 13.0, so older toolkits compile none of this.
#if CUDART_VERSION >= 13000
// ---- cudaGraphAddNode ----
static void add_node() {
  cudaGraph_t g, g2, g3;
  IS(cudaGraphCreate(&g, 0), cudaSuccess);
  float* d = nullptr;
  IS(cudaMalloc(&d, 64 * 4), cudaSuccess);
  std::vector<float> h(64, 1.0f);
  cudaGraphNode_t n1, n2, n3, n4, n5, n6, n7;
  cudaGraphNodeParams p{};
  p.type = cudaGraphNodeTypeMemcpy;
  p.memcpy.copyParams.srcPtr = make_cudaPitchedPtr(h.data(), 256, 64, 1);
  p.memcpy.copyParams.dstPtr = make_cudaPitchedPtr(d, 256, 64, 1);
  p.memcpy.copyParams.extent = make_cudaExtent(256, 1, 1);
  p.memcpy.copyParams.kind = cudaMemcpyHostToDevice;
  IS(cudaGraphAddNode(&n1, g, nullptr, nullptr, 0, &p), cudaSuccess);
  cudaGraphNodeParams k{};
  k.type = cudaGraphNodeTypeKernel;
  int n = 64;
  float v = 2.0f;
  void* args[] = {&d, &n, &v};
  k.kernel.func = (void*)add;
  k.kernel.gridDim = dim3(2);
  k.kernel.blockDim = dim3(32);
  k.kernel.kernelParams = args;
  IS(cudaGraphAddNode(&n2, g, &n1, nullptr, 1, &k), cudaSuccess);
  cudaGraphNodeParams ms{};
  ms.type = cudaGraphNodeTypeMemset;
  ms.memset.dst = d;
  ms.memset.elementSize = 4;
  ms.memset.width = 4;
  ms.memset.height = 1;
  IS(cudaGraphAddNode(&n3, g, &n2, nullptr, 1, &ms), cudaSuccess);
  int counter = 0;
  cudaGraphNodeParams hp{};
  hp.type = cudaGraphNodeTypeHost;
  hp.host.fn = bump;
  hp.host.userData = &counter;
  IS(cudaGraphAddNode(&n4, g, &n3, nullptr, 1, &hp), cudaSuccess);
  cudaGraphNodeParams ep{};
  ep.type = cudaGraphNodeTypeEmpty;
  IS(cudaGraphAddNode(&n5, g, &n4, nullptr, 1, &ep), cudaSuccess);
  cudaEvent_t ev;
  IS(cudaEventCreate(&ev), cudaSuccess);
  cudaGraphNodeParams er{};
  er.type = cudaGraphNodeTypeEventRecord;
  er.eventRecord.event = ev;
  IS(cudaGraphAddNode(&n6, g, &n5, nullptr, 1, &er), cudaSuccess);
  cudaGraphEdgeData ed{};
  IS(cudaGraphAddNode(&n7, g, &n6, &ed, 1, &ep), cudaSuccess);
  cudaGraphNodeType t;
  IS(cudaGraphNodeGetType(n2, &t), cudaSuccess);
  check(t == cudaGraphNodeTypeKernel, "a kernel node is a kernel node");
  // Only the default edge between these nodes.
  ed.type = cudaGraphDependencyTypeProgrammatic;
  check(cudaGraphAddNode(&n7, g, &n6, &ed, 1, &ep) != cudaSuccess, "a programmatic edge between empty nodes is refused");
  ed = {};
  ed.from_port = 1;
  check(cudaGraphAddNode(&n7, g, &n6, &ed, 1, &ep) != cudaSuccess, "so is an edge from port 1");
  cudaGetLastError();
  // Reserved fields are zero; the type is one there is.
  cudaGraphNodeParams bad{};
  bad.type = cudaGraphNodeTypeEmpty;
  bad.reserved0[0] = 1;
  IS(cudaGraphAddNode(&n7, g, nullptr, nullptr, 0, &bad), cudaErrorInvalidValue);
  bad = {};
  bad.type = cudaGraphNodeTypeEmpty;
  bad.reserved2 = 1;
  IS(cudaGraphAddNode(&n7, g, nullptr, nullptr, 0, &bad), cudaErrorInvalidValue);
  bad = {};
  bad.type = (cudaGraphNodeType)99;
  IS(cudaGraphAddNode(&n7, g, nullptr, nullptr, 0, &bad), cudaErrorInvalidValue);
  IS(cudaGraphAddNode(&n7, g, nullptr, nullptr, 0, nullptr), cudaErrorInvalidValue);
  IS(cudaGraphAddNode(nullptr, g, nullptr, nullptr, 0, &ep), cudaErrorInvalidValue);
  cudaGraphNodeParams mp = p;
  mp.memcpy.flags = 1;
  IS(cudaGraphAddNode(&n7, g, nullptr, nullptr, 0, &mp), cudaErrorInvalidValue);
  cudaGetLastError();
  // Allocation and free nodes; the address comes back in the parameters.
  IS(cudaGraphCreate(&g2, 0), cudaSuccess);
  cudaGraphNodeParams al{};
  al.type = cudaGraphNodeTypeMemAlloc;
  al.alloc.poolProps.allocType = cudaMemAllocationTypePinned;
  al.alloc.poolProps.location.type = cudaMemLocationTypeDevice;
  al.alloc.poolProps.location.id = 0;
  al.alloc.bytesize = 1024;
  cudaGraphNode_t na, nf, nc;
  IS(cudaGraphAddNode(&na, g2, nullptr, nullptr, 0, &al), cudaSuccess);
  check(al.alloc.dptr != nullptr, "an allocation node's address is handed back");
  cudaGraphNodeParams fr{};
  fr.type = cudaGraphNodeTypeMemFree;
  fr.free.dptr = al.alloc.dptr;
  IS(cudaGraphAddNode(&nf, g2, &na, nullptr, 1, &fr), cudaSuccess);
  // The first graph as a child of a third, and run.
  IS(cudaGraphCreate(&g3, 0), cudaSuccess);
  cudaGraphNodeParams cg{};
  cg.type = cudaGraphNodeTypeGraph;
  cg.graph.graph = g;
  IS(cudaGraphAddNode(&nc, g3, nullptr, nullptr, 0, &cg), cudaSuccess);
  IS(cudaGraphNodeGetType(nc, &t), cudaSuccess);
  check(t == cudaGraphNodeTypeGraph, "a child graph node");
  cudaGraphExec_t ex, ex2;
  IS(cudaGraphInstantiate(&ex, g3, 0), cudaSuccess);
  IS(cudaGraphLaunch(ex, 0), cudaSuccess);
  IS(cudaGraphInstantiate(&ex2, g2, 0), cudaSuccess);
  IS(cudaGraphLaunch(ex2, 0), cudaSuccess);
  IS(cudaStreamSynchronize(0), cudaSuccess);
  std::vector<float> out(64);
  IS(cudaMemcpy(out.data(), d, 256, cudaMemcpyDeviceToHost), cudaSuccess);
  check(out[0] == 0 && out[3] == 0 && out[4] == 3 && out[63] == 3 && counter == 1,
        "copy, kernel, memset and host nodes ran in order");
  cudaGraphExecDestroy(ex);
  cudaGraphExecDestroy(ex2);
  cudaGraphDestroy(g);
  cudaGraphDestroy(g2);
  cudaGraphDestroy(g3);
  cudaEventDestroy(ev);
  cudaFree(d);
}
#endif

// ---- 3D peer copies with arrays ----
static void peer_3d() {
  int count = 0;
  cudaGetDeviceCount(&count);
  const int other = count > 1 ? 1 : 0;
  const cudaChannelFormatDesc fd = cudaCreateChannelDesc<float>();
  const cudaExtent e = make_cudaExtent(8, 4, 2);   // elements, for an array
  std::vector<float> src(64), back(64, -1.0f), b(64, -1.0f);
  for (int i = 0; i < 64; ++i) src[i] = static_cast<float>(i);
  IS(cudaSetDevice(0), cudaSuccess);
  cudaArray_t a3;
  IS(cudaMalloc3DArray(&a3, &fd, e), cudaSuccess);
  cudaMemcpy3DParms up{};
  up.srcPtr = make_cudaPitchedPtr(src.data(), 32, 8, 4);
  up.dstArray = a3;
  up.extent = e;
  up.kind = cudaMemcpyHostToDevice;
  IS(cudaMemcpy3D(&up), cudaSuccess);
  IS(cudaSetDevice(other), cudaSuccess);
  cudaPitchedPtr pp;
  IS(cudaMalloc3D(&pp, make_cudaExtent(32, 4, 2)), cudaSuccess);
  check(pp.ptr && pp.pitch >= 32 && pp.xsize == 32 && pp.ysize == 4, "cudaMalloc3D");
  cudaMemcpy3DPeerParms pr{};
  pr.srcArray = a3;
  pr.srcDevice = 0;
  pr.dstPtr = pp;
  pr.dstDevice = other;
  pr.extent = e;
  IS(cudaMemcpy3DPeer(&pr), cudaSuccess);
  pr.extent = make_cudaExtent(32, 4, 2);   // bytes: wider than the array
  IS(cudaMemcpy3DPeer(&pr), cudaErrorInvalidValue);
  cudaGetLastError();
  cudaMemcpy3DParms dn{};
  dn.srcPtr = pp;
  dn.dstPtr = make_cudaPitchedPtr(back.data(), 32, 8, 4);
  dn.extent = make_cudaExtent(32, 4, 2);
  dn.kind = cudaMemcpyDeviceToHost;
  IS(cudaMemcpy3D(&dn), cudaSuccess);
  check(back[0] == 0 && back[9] == 9 && back[63] == 63, "an array to another device's pitched memory");
  // Array to array, at positions (which count elements on an array).
  cudaArray_t b3;
  IS(cudaMalloc3DArray(&b3, &fd, e), cudaSuccess);
  std::vector<float> zero(64, 0.0f);
  cudaMemcpy3DParms z{};
  z.srcPtr = make_cudaPitchedPtr(zero.data(), 32, 8, 4);
  z.dstArray = b3;
  z.extent = e;
  z.kind = cudaMemcpyHostToDevice;
  IS(cudaMemcpy3D(&z), cudaSuccess);
  cudaMemcpy3DPeerParms q{};
  q.srcArray = a3;
  q.srcDevice = 0;
  q.srcPos = make_cudaPos(1, 1, 0);
  q.dstArray = b3;
  q.dstDevice = other;
  q.dstPos = make_cudaPos(0, 0, 1);
  q.extent = make_cudaExtent(3, 2, 1);
  IS(cudaMemcpy3DPeer(&q), cudaSuccess);
  cudaMemcpy3DParms dn2{};
  dn2.srcArray = b3;
  dn2.dstPtr = make_cudaPitchedPtr(b.data(), 32, 8, 4);
  dn2.extent = e;
  dn2.kind = cudaMemcpyDeviceToHost;
  IS(cudaMemcpy3D(&dn2), cudaSuccess);
  check(b[32] == 9 && b[34] == 11 && b[35] == 0 && b[40] == 17 && b[0] == 0, "a box of one array into another");
  q.srcPos = make_cudaPos(6, 0, 0);   // 6 + 3 columns overruns 8
  q.extent = make_cudaExtent(3, 1, 1);
  IS(cudaMemcpy3DPeer(&q), cudaErrorInvalidValue);
  cudaGetLastError();
  // Pitched memory into an array.
  IS(cudaSetDevice(0), cudaSuccess);
  cudaPitchedPtr p0;
  IS(cudaMalloc3D(&p0, make_cudaExtent(32, 4, 2)), cudaSuccess);
  cudaMemcpy3DParms u0{};
  u0.srcPtr = make_cudaPitchedPtr(src.data(), 32, 8, 4);
  u0.dstPtr = p0;
  u0.extent = make_cudaExtent(32, 4, 2);
  u0.kind = cudaMemcpyHostToDevice;
  IS(cudaMemcpy3D(&u0), cudaSuccess);
  cudaMemcpy3DPeerParms r{};
  r.srcPtr = p0;
  r.srcDevice = 0;
  r.dstArray = b3;
  r.dstDevice = other;
  r.extent = e;
  IS(cudaMemcpy3DPeerAsync(&r, 0), cudaSuccess);
  IS(cudaDeviceSynchronize(), cudaSuccess);
  IS(cudaSetDevice(other), cudaSuccess);
  std::fill(b.begin(), b.end(), -1.0f);
  IS(cudaMemcpy3D(&dn2), cudaSuccess);
  check(b[0] == 0 && b[63] == 63, "pitched memory into another device's array");
  IS(cudaSetDevice(0), cudaSuccess);
  cudaFreeArray(a3);
  cudaFreeArray(b3);
  cudaFree(pp.ptr);
  cudaFree(p0.ptr);
}

// ---- host memory flags ----
static void host_flags() {
  const size_t N = 1 << 16;
  char* h = static_cast<char*>(std::aligned_alloc(4096, N));
  std::memset(h, 0x11, N);
  unsigned f = 99;
  IS(cudaHostGetFlags(&f, h), cudaErrorInvalidValue);
  cudaGetLastError();
  IS(cudaHostRegister(h, N, 0), cudaSuccess);
  IS(cudaHostGetFlags(&f, h), cudaSuccess);
  check(f == cudaHostAllocMapped, "registered memory is mapped");
  IS(cudaHostGetFlags(&f, h + 100), cudaSuccess);
  // The registered pointer is not something cudaMemset fills.
  IS(cudaMemset(h, 0x22, 16), cudaErrorInvalidValue);
  cudaGetLastError();
  check(h[0] == 0x11, "and it is untouched");
  IS(cudaHostUnregister(h), cudaSuccess);
  IS(cudaHostRegister(h, N, cudaHostRegisterPortable), cudaSuccess);
  IS(cudaHostGetFlags(&f, h), cudaSuccess);
  check(f == (cudaHostAllocMapped | cudaHostAllocPortable), "portable registered memory");
  IS(cudaHostUnregister(h), cudaSuccess);
  int ro = -1;
  IS(cudaDeviceGetAttribute(&ro, cudaDevAttrHostRegisterReadOnlySupported, 0), cudaSuccess);
  const cudaError_t r = cudaHostRegister(h, N, cudaHostRegisterReadOnly);
  check(ro ? r == cudaSuccess : r == cudaErrorNotSupported, "read-only registration as the device says");
  if (r == cudaSuccess) cudaHostUnregister(h);
  cudaGetLastError();
  std::free(h);
  void* p = nullptr;
  IS(cudaHostAlloc(&p, 64, cudaHostAllocPortable | cudaHostAllocWriteCombined), cudaSuccess);
  IS(cudaHostGetFlags(&f, static_cast<char*>(p) + 8), cudaSuccess);
  check(f == (cudaHostAllocPortable | cudaHostAllocWriteCombined | cudaHostAllocMapped),
        "pinned memory reports its flags and the mapping");
  cudaFreeHost(p);
  IS(cudaMallocHost(&p, 64), cudaSuccess);
  IS(cudaHostGetFlags(&f, p), cudaSuccess);
  check(f == cudaHostAllocMapped, "cudaMallocHost memory is mapped");
  cudaFreeHost(p);
}

// ---- what the samples gate on ----
static void capabilities() {
  int v = -1;
  IS(cudaDeviceGetAttribute(&v, cudaDevAttrGpuOverlap, 0), cudaSuccess);
  check(v == 1, "a copy overlaps a kernel");
  IS(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, 0), cudaSuccess);
  check(v == 1, "cooperative launch");
  IS(cudaDeviceGetAttribute(&v, cudaDevAttrHostRegisterSupported, 0), cudaSuccess);
  check(v == 1, "host registration");
  IS(cudaDeviceGetAttribute(&v, cudaDevAttrKernelExecTimeout, 0), cudaSuccess);
  IS(cudaDeviceGetAttribute(&v, cudaDevAttrCanUseHostPointerForRegisteredMem, 0), cudaSuccess);
  cudaDeviceProp p{};
  IS(cudaGetDeviceProperties(&p, 0), cudaSuccess);
  int a = -1;
  cudaDeviceGetAttribute(&a, cudaDevAttrAsyncEngineCount, 0);
  check(p.cooperativeLaunch == 1 && p.hostRegisterSupported == 1 && p.managedMemory == 1 &&
            p.asyncEngineCount == a && p.asyncEngineCount >= 1,
        "the properties agree with the attributes");
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);   // every line out, should a call crash
  if (cudaFree(nullptr) != cudaSuccess) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  array_copies();
  attach();
#if CUDART_VERSION >= 13000
  add_node();
#endif
  peer_3d();
  host_flags();
  capabilities();
  std::printf(failures ? "FAIL: %d runtime checks\n" : "PASS: every runtime check\n", failures);
  return failures ? 1 : 0;
}
