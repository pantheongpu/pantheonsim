// The driver API's contexts and what hangs off them, checked against what an
// RTX 3060's driver (13.2, under WSL) answers: calls made with no current
// context, the context stack, context flags and ids, the primary context's
// state and flags, execution affinity, context limits, memory handles exported
// as file descriptors, and multicast objects. Every check passes on the card as
// well as on this simulator; where the answer depends on what the device
// supports, the check asks the device first.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static int attr(CUdevice_attribute a, CUdevice dev = 0) {
  int v = -1;
  cuDeviceGetAttribute(&v, a, dev);
  return v;
}
static CUcontext current() {
  CUcontext c = reinterpret_cast<CUcontext>(0x1);
  cuCtxGetCurrent(&c);
  return c;
}
static unsigned flags_of_current() {
  unsigned f = 0xdead;
  cuCtxGetFlags(&f);
  return f;
}
static CUresult make_ctx(CUcontext* c, unsigned flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
  CUctxCreateParams p{};
  return cuCtxCreate(c, &p, flags, dev);
#else
  return cuCtxCreate(c, flags, dev);
#endif
}
static CUresult make_ctx_affinity(CUcontext* c, CUexecAffinityParam* ap, int n, unsigned flags, CUdevice dev) {
#if CUDA_VERSION >= 13000
  CUctxCreateParams p{};
  p.execAffinityParams = ap;
  p.numExecAffinityParams = n;
  return cuCtxCreate(c, &p, flags, dev);
#else
  return cuCtxCreate_v3(c, ap, n, flags, dev);
#endif
}

// ---- calls with no current context ----
static void no_context() {
  IS(cuCtxGetCurrent(nullptr), CUDA_ERROR_INVALID_VALUE);
  check(current() == nullptr, "no context is current at first");
  CUdeviceptr p = 0;
  void* hp = nullptr;
  CUstream st;
  CUevent ev;
  CUmodule mod;
  size_t lim = 0;
  unsigned f = 0;
  unsigned long long id = 0;
  unsigned ver = 0;
  int lo = 0, hi = 0;
  IS(cuMemAlloc(&p, 64), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuMemAllocHost(&hp, 64), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuMemHostAlloc(&hp, 64, 0), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuMemAllocManaged(&p, 64, CU_MEM_ATTACH_GLOBAL), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuStreamCreate(&st, 0), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuEventCreate(&ev, 0), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuModuleLoadData(&mod, ""), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxSynchronize(), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxGetLimit(&lim, CU_LIMIT_STACK_SIZE), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 1024), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxGetFlags(&f), CUDA_ERROR_INVALID_CONTEXT);
#if CUDA_VERSION >= 12010  // cuCtxSetFlags arrived in CUDA 12.1
  IS(cuCtxSetFlags(0), CUDA_ERROR_INVALID_CONTEXT);
#endif
  IS(cuCtxGetId(nullptr, &id), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxGetApiVersion(nullptr, &ver), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxGetStreamPriorityRange(&lo, &hi), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuMemsetD8(0, 0, 1), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuMemcpyHtoD(0, &p, 4), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuStreamSynchronize(nullptr), CUDA_ERROR_INVALID_CONTEXT);
  // Freeing nothing is fine, and so is asking about a device.
  IS(cuMemFree(0), CUDA_SUCCESS);
  IS(cuMemFreeHost(nullptr), CUDA_SUCCESS);
  size_t total = 0;
  IS(cuDeviceTotalMem(&total, 0), CUDA_SUCCESS);
  IS(cuDeviceTotalMem(nullptr, 0), CUDA_ERROR_INVALID_VALUE);
  int count = 0;
  IS(cuDeviceGetCount(&count), CUDA_SUCCESS);
}

// ---- the stack ----
// SetCurrent(NULL) takes the current context off the thread's stack, as a pop does; a context destroyed
// below the top stays on the stack until it is popped.
static void stack() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext a, b, c, t;
  IS(make_ctx(&a, 0, dev), CUDA_SUCCESS);
  IS(make_ctx(&b, 0, dev), CUDA_SUCCESS);
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  check(current() == nullptr, "popping what creation pushed leaves no context current");
  IS(cuCtxPopCurrent(&t), CUDA_ERROR_INVALID_CONTEXT);
  IS(make_ctx(&c, 0, dev), CUDA_SUCCESS);   // the stack is [c]
  IS(cuCtxPushCurrent(a), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(b), CUDA_SUCCESS);
  check(current() == b, "a push makes the context current");
  IS(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  check(current() == a, "SetCurrent(NULL) removes the current context from the stack, as a pop does");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  check(t == a && current() == c, "and the next pop returns the one below");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  check(t == c && current() == nullptr, "the last pop leaves nothing current");
  IS(cuCtxPopCurrent(&t), CUDA_ERROR_INVALID_CONTEXT);
  IS(cuCtxPushCurrent(nullptr), CUDA_ERROR_INVALID_VALUE);
  // SetCurrent replaces the top.
  IS(cuCtxSetCurrent(a), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(b), CUDA_SUCCESS);
  IS(cuCtxSetCurrent(c), CUDA_SUCCESS);
  check(current() == c, "SetCurrent(c) puts c where b was");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  check(t == c && current() == a, "the context below is untouched");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  // Destroying the current context makes the one below it current.
  IS(cuCtxSetCurrent(a), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(b), CUDA_SUCCESS);
  IS(cuCtxDestroy(b), CUDA_SUCCESS);
  check(current() == a, "destroying the current context makes the one below current");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  // Destroying one lower down leaves it on the stack.
  IS(make_ctx(&b, 0, dev), CUDA_SUCCESS);
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(a), CUDA_SUCCESS);
  IS(cuCtxPushCurrent(b), CUDA_SUCCESS);
  IS(cuCtxDestroy(a), CUDA_SUCCESS);
  check(current() == b, "destroying a context below the current one changes nothing current");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  check(t == b, "popped b");
  IS(cuCtxPopCurrent(&t), CUDA_SUCCESS);
  check(t == a, "and the destroyed context is still on the stack to be popped");
  IS(cuCtxDestroy(b), CUDA_SUCCESS);
  IS(cuCtxDestroy(c), CUDA_SUCCESS);
}

// ---- flags and ids of contexts ----
static void flags_and_ids() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  // Valid at creation: one scheduling mode (0, SPIN 1, YIELD 2, BLOCKING_SYNC 4) with any of MAP_HOST 8,
  // LMEM_RESIZE_TO_MAX 0x10 and SYNC_MEMOPS 0x80; reported back as given.
  for (unsigned f : {0u, 1u, 2u, 4u, 8u, 0x10u, 0x80u, 0xcu, 0x9u, 0xau, 0x14u, 0x98u, 0x99u, 0x9au}) {
    CUcontext c = nullptr;
    char what[96];
    std::snprintf(what, sizeof what, "a context made with flags %#x reports them", f);
    const CUresult r = make_ctx(&c, f, dev);
    check(r == CUDA_SUCCESS && flags_of_current() == f, what);
    if (c) cuCtxDestroy(c);
  }
  for (unsigned f : {3u, 5u, 6u, 7u, 0x100u, 0x2000u, 0xffffu, 0xfeu}) {
    CUcontext c = nullptr;
    char what[96];
    std::snprintf(what, sizeof what, "flags %#x make no context", f);
    check(make_ctx(&c, f, dev) == CUDA_ERROR_INVALID_VALUE && c == nullptr, what);
  }
  // cuCtxSetFlags changes the scheduling mode, LMEM_RESIZE_TO_MAX and SYNC_MEMOPS; MAP_HOST and the core-dump
  // flags are accepted and not kept.
  CUcontext c;
  IS(make_ctx(&c, 0, dev), CUDA_SUCCESS);
#if CUDA_VERSION >= 12010  // cuCtxSetFlags arrived in CUDA 12.1
  IS(cuCtxSetFlags(1), CUDA_SUCCESS);
  check(flags_of_current() == 1, "SetFlags(SPIN) is reported");
  IS(cuCtxSetFlags(4), CUDA_SUCCESS);
  check(flags_of_current() == 4, "SetFlags(BLOCKING_SYNC) replaces it");
  IS(cuCtxSetFlags(8), CUDA_SUCCESS);
  check(flags_of_current() == 0, "SetFlags(MAP_HOST) is taken and not kept");
  IS(cuCtxSetFlags(0x11), CUDA_SUCCESS);
  check(flags_of_current() == 0x11, "SPIN with LMEM_RESIZE_TO_MAX is kept whole");
  IS(cuCtxSetFlags(0x80), CUDA_SUCCESS);
  check(flags_of_current() == 0x80, "SYNC_MEMOPS is kept");
  IS(cuCtxSetFlags(3), CUDA_ERROR_INVALID_VALUE);
  IS(cuCtxSetFlags(0x100), CUDA_ERROR_INVALID_VALUE);
  check(flags_of_current() == 0x80, "a refused SetFlags changes nothing");
#endif
  IS(cuCtxGetFlags(nullptr), CUDA_ERROR_INVALID_VALUE);
  // Ids count up from 1 in the order contexts are made.
  unsigned long long id1 = 0, id2 = 0, idnull = 0;
  IS(cuCtxGetId(c, &id1), CUDA_SUCCESS);
  IS(cuCtxGetId(nullptr, &idnull), CUDA_SUCCESS);
  check(id1 == idnull && id1 > 0, "a NULL context means the current one");
  CUcontext d;
  IS(make_ctx(&d, 0, dev), CUDA_SUCCESS);
  IS(cuCtxGetId(d, &id2), CUDA_SUCCESS);
  check(id2 > id1, "a later context has a larger id");
  IS(cuCtxGetId(d, nullptr), CUDA_ERROR_INVALID_VALUE);
  unsigned ver = 0;
  IS(cuCtxGetApiVersion(d, &ver), CUDA_SUCCESS);
  check(ver >= 3000, "a context has an API version");
  IS(cuCtxGetApiVersion(nullptr, &ver), CUDA_SUCCESS);
  IS(cuCtxGetApiVersion(d, nullptr), CUDA_ERROR_INVALID_VALUE);
  int lo = 7, hi = 7;
  IS(cuCtxGetStreamPriorityRange(&lo, &hi), CUDA_SUCCESS);
  check(lo == 0 && hi < 0, "the stream priority range runs from 0 to a negative number");
  IS(cuCtxGetStreamPriorityRange(nullptr, nullptr), CUDA_SUCCESS);
  cuCtxDestroy(d);
  cuCtxDestroy(c);
  cuCtxSetCurrent(nullptr);
}

// ---- the primary context's flags and state ----
static void primary() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  unsigned fl = 77;
  int act = 7;
  IS(cuDevicePrimaryCtxGetState(dev, &fl, &act), CUDA_SUCCESS);
  check(fl == 0 && act == 0, "a primary context nobody retained has no flags and is inactive");
  IS(cuDevicePrimaryCtxGetState(dev, nullptr, &act), CUDA_ERROR_INVALID_VALUE);
  IS(cuDevicePrimaryCtxGetState(dev, &fl, nullptr), CUDA_ERROR_INVALID_VALUE);
  IS(cuDevicePrimaryCtxGetState(9, &fl, &act), CUDA_ERROR_INVALID_DEVICE);
  // Flags that can be set: a scheduling mode with any of LMEM_RESIZE_TO_MAX, the core-dump flags and
  // SYNC_MEMOPS; not MAP_HOST (a primary context always has it), nor two modes, nor beyond 0xff.
  for (unsigned f : {0u, 1u, 2u, 4u, 0x10u, 0x11u, 0x80u, 0x81u, 0x90u, 0x30u, 0xf0u}) {
    char what[96];
    std::snprintf(what, sizeof what, "primary flags %#x are taken and reported", f);
    const CUresult r = cuDevicePrimaryCtxSetFlags(dev, f);
    cuDevicePrimaryCtxGetState(dev, &fl, &act);
    check(r == CUDA_SUCCESS && fl == f, what);
  }
  for (unsigned f : {8u, 5u, 0xcu, 0x18u, 3u, 0xffu, 0x100u}) {
    char what[96];
    std::snprintf(what, sizeof what, "primary flags %#x are refused and change nothing", f);
    cuDevicePrimaryCtxSetFlags(dev, 0x90);
    const CUresult r = cuDevicePrimaryCtxSetFlags(dev, f);
    cuDevicePrimaryCtxGetState(dev, &fl, &act);
    check(r == CUDA_ERROR_INVALID_VALUE && fl == 0x90, what);
  }
  IS(cuDevicePrimaryCtxSetFlags(9, 0), CUDA_ERROR_INVALID_DEVICE);
  cuDevicePrimaryCtxSetFlags(dev, 0);
  CUcontext pc;
  IS(cuDevicePrimaryCtxRetain(&pc, dev), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxGetState(dev, &fl, &act), CUDA_SUCCESS);
  check(act == 1, "a retained primary context is active");
  IS(cuCtxSetCurrent(pc), CUDA_SUCCESS);
  check(flags_of_current() == 0x8, "its flags are MAP_HOST alone");
  // Setting them while it is active is accepted and shows at once.
  for (unsigned f : {1u, 4u, 0x10u, 0x80u, 0u}) {
    char what[96];
    std::snprintf(what, sizeof what, "primary flags %#x set while active are reported with MAP_HOST", f);
    const CUresult r = cuDevicePrimaryCtxSetFlags(dev, f);
    check(r == CUDA_SUCCESS && flags_of_current() == (f | 0x8), what);
  }
  // A reset takes it out of the active state and clears the flags; the retain count survives it.
  cuDevicePrimaryCtxSetFlags(dev, 1);
  IS(cuDevicePrimaryCtxReset(dev), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxGetState(dev, &fl, &act), CUDA_SUCCESS);
  check(fl == 0 && act == 0, "after a reset: no flags, inactive");
  IS(cuDevicePrimaryCtxReset(9), CUDA_ERROR_INVALID_DEVICE);
  IS(cuCtxSetCurrent(nullptr), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxRelease(dev), CUDA_SUCCESS);
  IS(cuDevicePrimaryCtxRelease(dev), CUDA_ERROR_INVALID_CONTEXT);
}

// ---- execution affinity ----
static void exec_affinity() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  int support = -1;
  IS(cuDeviceGetExecAffinitySupport(&support, CU_EXEC_AFFINITY_TYPE_SM_COUNT, dev), CUDA_SUCCESS);
  IS(cuDeviceGetExecAffinitySupport(nullptr, CU_EXEC_AFFINITY_TYPE_SM_COUNT, dev), CUDA_ERROR_INVALID_VALUE);
  IS(cuDeviceGetExecAffinitySupport(&support, CU_EXEC_AFFINITY_TYPE_SM_COUNT, 9), CUDA_ERROR_INVALID_DEVICE);
  IS(cuDeviceGetExecAffinitySupport(&support, CU_EXEC_AFFINITY_TYPE_SM_COUNT, dev), CUDA_SUCCESS);
  CUcontext c = nullptr;
  CUexecAffinityParam ap;
  std::memset(&ap, 0, sizeof ap);
  ap.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
  ap.param.smCount.val = 8;
  if (support == 0) {
    IS(make_ctx_affinity(&c, &ap, 1, 0, dev), CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY);
    ap.param.smCount.val = 0;
    IS(make_ctx_affinity(&c, &ap, 1, 0, dev), CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY);
    ap.param.smCount.val = 100000;
    IS(make_ctx_affinity(&c, &ap, 1, 0, dev), CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY);
    check(c == nullptr, "a refused affinity makes no context");
  }
  // Parameters named and none counted is invalid.
  IS(make_ctx_affinity(&c, &ap, 0, 0, dev), CUDA_ERROR_INVALID_VALUE);
  // The affinity of an ordinary context is all the multiprocessors.
  IS(make_ctx(&c, 0, dev), CUDA_SUCCESS);
  CUexecAffinityParam got;
  std::memset(&got, 0xff, sizeof got);
  IS(cuCtxGetExecAffinity(&got, CU_EXEC_AFFINITY_TYPE_SM_COUNT), CUDA_SUCCESS);
  check(got.type == CU_EXEC_AFFINITY_TYPE_SM_COUNT &&
            static_cast<int>(got.param.smCount.val) == attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        "an ordinary context's SM count is the device's");
  IS(cuCtxGetExecAffinity(nullptr, CU_EXEC_AFFINITY_TYPE_SM_COUNT), CUDA_ERROR_INVALID_VALUE);
  if (support == 0) IS(cuCtxGetExecAffinity(&got, static_cast<CUexecAffinityType>(9)), CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY);
  cuCtxDestroy(c);
}

// ---- limits ----
static size_t limit(CUlimit l) {
  size_t v = 12345;
  cuCtxGetLimit(&v, l);
  return v;
}
static void limits() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext c;
  IS(make_ctx(&c, 0, dev), CUDA_SUCCESS);
  size_t v = 0;
  // Numbers 7 and 8 are limits that can be read and not set, 9 is not supported, beyond that there are none.
  IS(cuCtxGetLimit(&v, static_cast<CUlimit>(7)), CUDA_SUCCESS);
  check(v > 0, "the shared memory limit (7) has a value");
  IS(cuCtxGetLimit(&v, static_cast<CUlimit>(8)), CUDA_SUCCESS);
  check(v == 0, "CIG is not enabled (8)");
  IS(cuCtxSetLimit(static_cast<CUlimit>(7), 4096), CUDA_ERROR_NOT_PERMITTED);
  IS(cuCtxSetLimit(static_cast<CUlimit>(8), 1), CUDA_ERROR_NOT_PERMITTED);
  IS(cuCtxGetLimit(&v, static_cast<CUlimit>(9)), CUDA_ERROR_UNSUPPORTED_LIMIT);
  IS(cuCtxSetLimit(static_cast<CUlimit>(9), 1), CUDA_ERROR_UNSUPPORTED_LIMIT);
  for (int l : {10, 15, 256, -1}) {
    char what[96];
    std::snprintf(what, sizeof what, "limit %d does not exist", l);
    check(cuCtxGetLimit(&v, static_cast<CUlimit>(l)) == CUDA_ERROR_INVALID_VALUE &&
              cuCtxSetLimit(static_cast<CUlimit>(l), 1) == CUDA_ERROR_INVALID_VALUE,
          what);
  }
  IS(cuCtxGetLimit(nullptr, CU_LIMIT_STACK_SIZE), CUDA_ERROR_INVALID_VALUE);
  // The stack: whole elements of 16 bytes, no more than a thread's local memory.
  IS(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 4096), CUDA_SUCCESS);
  check(limit(CU_LIMIT_STACK_SIZE) == 4096, "a stack size is kept");
  IS(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 100), CUDA_SUCCESS);
  check(limit(CU_LIMIT_STACK_SIZE) == 112, "and rounded up to a whole element");
  IS(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 1 << 20), CUDA_ERROR_INVALID_VALUE);
  check(limit(CU_LIMIT_STACK_SIZE) == 112, "a refused size changes nothing");
  // The printf buffer and malloc heap have smallest sizes; the pending launch count has one too.
  IS(cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 1), CUDA_SUCCESS);
  check(limit(CU_LIMIT_PRINTF_FIFO_SIZE) == 393216, "a printf buffer of 1 byte becomes the smallest, 384 KiB");
  IS(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, 1), CUDA_SUCCESS);
  check(limit(CU_LIMIT_MALLOC_HEAP_SIZE) == 4u << 20, "a malloc heap of 1 byte becomes the smallest, 4 MiB");
  IS(cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, 0), CUDA_SUCCESS);
  check(limit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT) == 32, "no pending launches means the smallest count, 32");
  // The L2 fetch granularity goes to 128 bytes.
  IS(cuCtxSetLimit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY, 100), CUDA_SUCCESS);
  check(limit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY) == 100, "a fetch granularity is kept");
  IS(cuCtxSetLimit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY, 1024), CUDA_ERROR_INVALID_VALUE);
  if (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) < 9) {
    IS(cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, 3), CUDA_SUCCESS);
    IS(cuCtxSetLimit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, 100), CUDA_ERROR_INVALID_VALUE);
    check(limit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH) == 3, "a sync depth beyond the largest is refused");
  }
  CUevent ev = nullptr;
  IS(cuEventCreate(&ev, 0), CUDA_SUCCESS);
  IS(cuEventDestroy(ev), CUDA_SUCCESS);
  IS(cuEventSynchronize(nullptr), CUDA_ERROR_INVALID_HANDLE);
  IS(cuEventQuery(nullptr), CUDA_ERROR_INVALID_HANDLE);
  cuCtxDestroy(c);
}

// ---- memory handles exported as file descriptors ----
static CUresult make_handle(CUmemGenericAllocationHandle* h, size_t size, int types) {
  CUmemAllocationProp prop;
  std::memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  std::memcpy(&prop.requestedHandleTypes, &types, sizeof types);
  return cuMemCreate(h, size, &prop, 0);
}
static void exported_memory() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext c;
  IS(make_ctx(&c, 0, dev), CUDA_SUCCESS);
  const size_t size = 2u << 20;
  CUmemGenericAllocationHandle h = 0;
  // Win32, Win32 KMT and fabric handles, alone or with another, are invalid; a bit CUDA has no type for is
  // ignored.
  for (int t : {2, 4, 8, 3}) {
    char what[96];
    std::snprintf(what, sizeof what, "cuMemCreate with handle types %#x is invalid", t);
    check(make_handle(&h, size, t) == CUDA_ERROR_INVALID_VALUE, what);
  }
  IS(make_handle(&h, size, 0), CUDA_SUCCESS);
  int fd = -5;
  IS(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), CUDA_ERROR_INVALID_VALUE);
  check(fd == -5, "a failed export wrote no descriptor");
  IS(cuMemRelease(h), CUDA_SUCCESS);
  IS(make_handle(&h, size, 0x10), CUDA_SUCCESS);
  IS(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemRelease(h), CUDA_SUCCESS);
  IS(cuMemExportToShareableHandle(nullptr, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), CUDA_ERROR_INVALID_VALUE);
  IS(cuMemExportToShareableHandle(&fd, 0, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), CUDA_ERROR_INVALID_VALUE);

  unsigned long long imp = 0;
  IS(cuMemImportFromShareableHandle(&imp, reinterpret_cast<void*>(static_cast<intptr_t>(-1)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
     CUDA_ERROR_INVALID_DEVICE);
  IS(cuMemImportFromShareableHandle(nullptr, reinterpret_cast<void*>(3), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), CUDA_ERROR_INVALID_VALUE);
#if CUDA_VERSION >= 12030  // fabric handles arrived in CUDA 12.3
  IS(cuMemImportFromShareableHandle(&imp, reinterpret_cast<void*>(3), CU_MEM_HANDLE_TYPE_FABRIC), CUDA_ERROR_NOT_SUPPORTED);
#endif
  IS(cuMemImportFromShareableHandle(&imp, reinterpret_cast<void*>(3), static_cast<CUmemAllocationHandleType>(3)), CUDA_ERROR_INVALID_VALUE);

  if (attr(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED) == 1) {
    // Exported, a handle is a descriptor: another handle made from it names the same memory.
    IS(make_handle(&h, size, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), CUDA_SUCCESS);
    CUmemAllocationProp prop;
    IS(cuMemGetAllocationPropertiesFromHandle(&prop, h), CUDA_SUCCESS);
    check(prop.requestedHandleTypes == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, "the handle reports the type it was made with");
    IS(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), CUDA_SUCCESS);
    check(fd >= 0, "the export is a descriptor");
    int fd2 = -5;
    IS(cuMemExportToShareableHandle(&fd2, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 1), CUDA_ERROR_INVALID_VALUE);
    IS(cuMemExportToShareableHandle(&fd2, h, CU_MEM_HANDLE_TYPE_NONE, 0), CUDA_ERROR_INVALID_VALUE);
    IS(cuMemImportFromShareableHandle(&imp, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR), CUDA_SUCCESS);
    // Map both, write through one, read through the other.
    CUdeviceptr a = 0, b = 0;
    IS(cuMemAddressReserve(&a, size, 0, 0, 0), CUDA_SUCCESS);
    IS(cuMemAddressReserve(&b, size, 0, 0, 0), CUDA_SUCCESS);
    IS(cuMemMap(a, size, 0, h, 0), CUDA_SUCCESS);
    IS(cuMemMap(b, size, 0, imp, 0), CUDA_SUCCESS);
    CUmemAccessDesc ad;
    std::memset(&ad, 0, sizeof ad);
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad.location.id = 0;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    IS(cuMemSetAccess(a, size, &ad, 1), CUDA_SUCCESS);
    IS(cuMemSetAccess(b, size, &ad, 1), CUDA_SUCCESS);
    std::vector<int> src(64), dst(64, -1);
    for (int i = 0; i < 64; ++i) src[i] = 1000 + i;
    IS(cuMemcpyHtoD(a + 4096, src.data(), 64 * sizeof(int)), CUDA_SUCCESS);
    IS(cuMemcpyDtoH(dst.data(), b + 4096, 64 * sizeof(int)), CUDA_SUCCESS);
    check(dst[0] == 1000 && dst[63] == 1063, "memory written through one handle is read through the other");
    IS(cuMemcpyHtoD(b, src.data(), 4), CUDA_SUCCESS);
    int back = -1;
    IS(cuMemcpyDtoH(&back, a, 4), CUDA_SUCCESS);
    check(back == 1000, "and the other way");
    IS(cuMemUnmap(a, size), CUDA_SUCCESS);
    IS(cuMemUnmap(b, size), CUDA_SUCCESS);
    IS(cuMemAddressFree(a, size), CUDA_SUCCESS);
    IS(cuMemAddressFree(b, size), CUDA_SUCCESS);
    IS(cuMemRelease(imp), CUDA_SUCCESS);
    IS(cuMemRelease(h), CUDA_SUCCESS);
    close(fd);
  }
  cuCtxDestroy(c);
}

// ---- multicast ----
static void multicast() {
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext c;
  IS(make_ctx(&c, 0, dev), CUDA_SUCCESS);
#if CUDA_VERSION >= 12010
  if (attr(CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED) == 0) {
    CUmulticastObjectProp mp;
    std::memset(&mp, 0, sizeof mp);
    mp.numDevices = 2;
    mp.size = 2u << 20;
    mp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle mh = 0;
    size_t g = 0;
    IS(cuMulticastCreate(&mh, &mp), CUDA_ERROR_NOT_SUPPORTED);
    IS(cuMulticastGetGranularity(&g, &mp, CU_MULTICAST_GRANULARITY_MINIMUM), CUDA_ERROR_NOT_SUPPORTED);
    IS(cuMulticastAddDevice(mh, dev), CUDA_ERROR_NOT_SUPPORTED);
    IS(cuMulticastBindMem(mh, 0, 0, 0, 2u << 20, 0), CUDA_ERROR_NOT_SUPPORTED);
    IS(cuMulticastUnbind(mh, dev, 0, 2u << 20), CUDA_ERROR_NOT_SUPPORTED);
  }
#endif
  cuCtxDestroy(c);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);   // every line out, should a call crash
  if (cuInit(0) != CUDA_SUCCESS) {
    std::printf("FAIL: no device\n");
    return 1;
  }
  no_context();
  stack();
  flags_and_ids();
  primary();
  exec_affinity();
  limits();
  exported_memory();
  multicast();
  std::printf(failures ? "FAIL: %d driver checks\n" : "PASS: every driver check\n", failures);
  return failures ? 1 : 0;
}
