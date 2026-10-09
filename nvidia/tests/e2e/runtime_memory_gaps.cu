// Memory pools with handle types, the export and import calls around them,
// read-only host registration, and the pointer queries on memory exported for
// another process. Every check passes on an RTX 3060's runtime (driver 13.2,
// under WSL) as well as on this simulator. Where the answer depends on what the
// device supports -- pool handle types, read-only registration -- the check asks
// the device first and expects what its attribute says.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

__global__ void read_into(const int* p, int* out) { *out = p[threadIdx.x] + 1; }
__global__ void write_to(int* p) { p[threadIdx.x] = 42; }

// What the faulting write at the end of main uses.
static void* g_ro_pointer = nullptr;
static void* g_ro_host = nullptr;

static int attr(cudaDeviceAttr a) {
  int v = -1;
  cudaDeviceGetAttribute(&v, a, 0);
  return v;
}

static cudaMemPoolProps props(int handle_types) {
  cudaMemPoolProps pp = {};
  pp.allocType = cudaMemAllocationTypePinned;
  pp.location.type = cudaMemLocationTypeDevice;
  pp.location.id = 0;
  std::memcpy(&pp.handleTypes, &handle_types, sizeof handle_types);
  return pp;
}

// ---- pools: what cannot be made, and what cannot be shared ----
static void pool_handle_types() {
  const int supported = attr(cudaDevAttrMemoryPoolSupportedHandleTypes);
  cudaMemPool_t pool = nullptr;
  {
    cudaMemPoolProps pp = props(0);
    IS(cudaMemPoolCreate(&pool, &pp), cudaSuccess);
    if (pool) IS(cudaMemPoolDestroy(pool), cudaSuccess);
  }
  // An allocation type, a location or a device that does not exist, and the reserved bytes are invalid.
  {
    cudaMemPoolProps pp = props(0);
    pp.allocType = cudaMemAllocationTypeInvalid;
    IS(cudaMemPoolCreate(&pool, &pp), cudaErrorInvalidValue);
    pp = props(0);
    pp.location.type = cudaMemLocationTypeInvalid;
    IS(cudaMemPoolCreate(&pool, &pp), cudaErrorInvalidValue);
    pp = props(0);
    pp.location.id = 5;
    IS(cudaMemPoolCreate(&pool, &pp), cudaErrorInvalidValue);
    pp = props(0);
    pp.reserved[0] = 1;
    IS(cudaMemPoolCreate(&pool, &pp), cudaErrorInvalidValue);
    IS(cudaMemPoolCreate(&pool, nullptr), cudaErrorInvalidValue);
    IS(cudaMemPoolCreate(nullptr, &pp), cudaErrorInvalidValue);
  }
  if (supported == 0) {
    // The device offers no handle type, so asking for one is invalid, whichever it is.
    for (int t : {1, 2, 3, 4, 8, 0x10, 0x7fff}) {
      cudaMemPoolProps pp = props(t);
      char what[96];
      std::snprintf(what, sizeof what, "cudaMemPoolCreate with handle types %#x is invalid when the device supports none", t);
      check(cudaMemPoolCreate(&pool, &pp) == cudaErrorInvalidValue, what);
    }
    // A pool made without one cannot be exported, nor a pointer from it.
    cudaMemPool_t def;
    IS(cudaDeviceGetDefaultMemPool(&def, 0), cudaSuccess);
    int fd = -7;
    for (int t : {0, 1, 2, 8}) {
      cudaMemAllocationHandleType type;
      std::memcpy(&type, &t, sizeof t);
      char what[96];
      std::snprintf(what, sizeof what, "cudaMemPoolExportToShareableHandle (type %d) is invalid", t);
      check(cudaMemPoolExportToShareableHandle(&fd, def, type, 0) == cudaErrorInvalidValue, what);
    }
    IS(cudaMemPoolExportToShareableHandle(nullptr, def, cudaMemHandleTypePosixFileDescriptor, 0), cudaErrorInvalidValue);
    IS(cudaMemPoolExportToShareableHandle(&fd, nullptr, cudaMemHandleTypePosixFileDescriptor, 0), cudaErrorInvalidValue);
    check(fd == -7, "an export that failed wrote no descriptor");
    // Importing: the pointers and flags first, then the type -- the file descriptor, Win32 and fabric types
    // the device does not support, anything else invalid.
    cudaMemPool_t imported = nullptr;
    int h = 3;
    IS(cudaMemPoolImportFromShareableHandle(&imported, &h, cudaMemHandleTypePosixFileDescriptor, 0), cudaErrorNotSupported);
    IS(cudaMemPoolImportFromShareableHandle(&imported, &h, cudaMemHandleTypeWin32, 0), cudaErrorNotSupported);
    IS(cudaMemPoolImportFromShareableHandle(&imported, &h, cudaMemHandleTypeNone, 0), cudaErrorInvalidValue);
    IS(cudaMemPoolImportFromShareableHandle(&imported, &h, cudaMemHandleTypePosixFileDescriptor, 5), cudaErrorInvalidValue);
    IS(cudaMemPoolImportFromShareableHandle(nullptr, &h, cudaMemHandleTypePosixFileDescriptor, 0), cudaErrorInvalidValue);
    IS(cudaMemPoolImportFromShareableHandle(&imported, nullptr, cudaMemHandleTypePosixFileDescriptor, 0), cudaErrorInvalidValue);
    void* p = nullptr;
    IS(cudaMallocFromPoolAsync(&p, 4096, def, 0), cudaSuccess);
    IS(cudaStreamSynchronize(0), cudaSuccess);
    cudaMemPoolPtrExportData ed;
    std::memset(&ed, 0, sizeof ed);
    IS(cudaMemPoolExportPointer(&ed, p), cudaErrorInvalidValue);
    IS(cudaMemPoolExportPointer(nullptr, p), cudaErrorInvalidValue);
    void* q = nullptr;
    IS(cudaMemPoolImportPointer(&q, def, &ed), cudaErrorInvalidValue);
    IS(cudaFreeAsync(p, 0), cudaSuccess);
    IS(cudaStreamSynchronize(0), cudaSuccess);
    // A graph allocation node asks for the same thing in its pool properties.
    cudaGraph_t g;
    IS(cudaGraphCreate(&g, 0), cudaSuccess);
    cudaGraphNode_t n;
    cudaMemAllocNodeParams ap = {};
    ap.bytesize = 4096;
    ap.poolProps = props(1);
    IS(cudaGraphAddMemAllocNode(&n, g, nullptr, 0, &ap), cudaErrorInvalidValue);
    ap.poolProps = props(0);
    IS(cudaGraphAddMemAllocNode(&n, g, nullptr, 0, &ap), cudaSuccess);
    IS(cudaGraphDestroy(g), cudaSuccess);
  }
}

// ---- registration flags and read-only registration ----
static void host_register() {
  const size_t N = 1 << 16;
  void* h = nullptr;
  if (posix_memalign(&h, 4096, N)) return;
  IS(cudaHostRegister(h, N, 0x10), cudaErrorInvalidValue);
  IS(cudaHostRegister(h, N, 0x100), cudaErrorInvalidValue);
  IS(cudaHostRegister(h, N, 0x80), cudaErrorInvalidValue);
  IS(cudaHostRegister(h, 0, 0), cudaErrorInvalidValue);
  IS(cudaHostRegister(nullptr, N, 0), cudaErrorInvalidValue);
  if (attr(cudaDevAttrHostRegisterReadOnlySupported) == 0) {
    IS(cudaHostRegister(h, N, cudaHostRegisterReadOnly), cudaErrorNotSupported);
    std::free(h);
    return;
  }
  int* ints = static_cast<int*>(h);
  for (int i = 0; i < 64; ++i) ints[i] = i;
  IS(cudaHostRegister(h, N, cudaHostRegisterReadOnly), cudaSuccess);
  IS(cudaHostRegister(h, N, cudaHostRegisterReadOnly), cudaErrorHostMemoryAlreadyRegistered);
  unsigned flags = 0;
  IS(cudaHostGetFlags(&flags, h), cudaSuccess);
  check((flags & cudaHostAllocMapped) != 0, "a registered range reports itself mapped");
  void* dp = nullptr;
  IS(cudaHostGetDevicePointer(&dp, h, 0), cudaSuccess);
  int* out = nullptr;
  IS(cudaMalloc(&out, 4), cudaSuccess);
  // The device reads it.
  cudaGetLastError();
  read_into<<<1, 1>>>(static_cast<const int*>(dp), out);
  IS(cudaGetLastError(), cudaSuccess);
  int r = -1;
  IS(cudaMemcpy(&r, out, 4, cudaMemcpyDeviceToHost), cudaSuccess);
  check(r == 1, "a kernel reads read-only registered memory");
  // A copy into it, from the device or the host, is refused and writes nothing.
  IS(cudaMemcpy(out, h, 4, cudaMemcpyHostToDevice), cudaSuccess);
  IS(cudaMemcpy(h, out, 4, cudaMemcpyDeviceToHost), cudaErrorInvalidValue);
  IS(cudaMemcpyAsync(h, out, 4, cudaMemcpyDeviceToHost, 0), cudaErrorInvalidValue);
  IS(cudaStreamSynchronize(0), cudaSuccess);
  check(ints[0] == 0 && ints[63] == 63, "a refused copy wrote nothing");
  cudaGetLastError();   // the refused copies above left the thread's last error set
  // Unregistering is allowed; the faulting write comes last, in main, because on the card nothing works after it.
  IS(cudaHostUnregister(h), cudaSuccess);
  IS(cudaHostRegister(h, N, cudaHostRegisterReadOnly), cudaSuccess);
  IS(cudaHostGetDevicePointer(&dp, h, 0), cudaSuccess);
  g_ro_pointer = dp;
  g_ro_host = h;
}

// ---- memory exported for another process is still memory ----
static void ipc_pointer() {
  void* d = nullptr;
  IS(cudaMalloc(&d, 256), cudaSuccess);
  cudaPointerAttributes a;
  IS(cudaPointerGetAttributes(&a, d), cudaSuccess);
  cudaIpcMemHandle_t handle;
  const cudaError_t e = cudaIpcGetMemHandle(&handle, d);
  if (e == cudaSuccess) {
    IS(cudaPointerGetAttributes(&a, d), cudaSuccess);
    check(a.type == cudaMemoryTypeDevice, "an exported allocation is still device memory");
    IS(cudaMemset(d, 0, 256), cudaSuccess);
  } else {
    std::printf("note: cudaIpcGetMemHandle -> %d here; the export checks are skipped\n", (int)e);
  }
  IS(cudaFree(d), cudaSuccess);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (cudaFree(nullptr) != cudaSuccess) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  pool_handle_types();
  ipc_pointer();
  host_register();
  if (g_ro_pointer) {
    // A kernel that writes read-only registered memory faults and the context ends: the sync after it is
    // cudaErrorLaunchFailure, the memory is untouched, and an RTX 3060 under WSL can do nothing more, so
    // this is the last thing the program does.
    int* ints = static_cast<int*>(g_ro_host);
    write_to<<<1, 1>>>(static_cast<int*>(g_ro_pointer));
    cudaGetLastError();
    IS(cudaDeviceSynchronize(), cudaErrorLaunchFailure);
    check(ints[0] == 0, "the faulting kernel changed nothing");
    IS(cudaGetLastError(), cudaErrorLaunchFailure);
    std::free(g_ro_host);
  }
  std::printf(failures ? "FAIL: %d runtime checks\n" : "PASS: every runtime check\n", failures);
  return failures ? 1 : 0;
}
