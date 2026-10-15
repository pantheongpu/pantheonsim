// The driver functions only the CUDA 13.2 cuda.h declares (tests/lint/check_header_exports.py against a 13.2
// toolkit), except the graph ones, which are in driver_graph.cpp with the rest of the graph calls. Included by
// driver_api.cpp after driver_sweep.inc, whose helpers they use. Every answer is an RTX 3060's under driver 596.36
// (CUDA 13.2), measured by nvidia/tests/e2e/exports_sweep_driver.cpp (the [13.2] lines of
// nvidia/tests/data/exports_sweep_driver.rtx3060.expected, made by run_exports_sweep.sh driver --card --update with a
// 13.2 nvcc); what the card answers and this simulator refuses on purpose is said where it is. The library is built with no
// toolkit, so nothing here needs a 13.2 header and these are exported whatever toolkit built it.

// ---- how many parameters a kernel takes ----------------------------------------------------------------------------------------
// A CUfunction and a CUkernel are not interchangeable here: the wrong kind of handle is CUDA_ERROR_INVALID_HANDLE, a
// null function or kernel CUDA_ERROR_INVALID_VALUE (as for the parameter info calls), and so is a null result.
static CUresult cuFuncGetParamCount_impl(CUfunction func, size_t* paramCount);
VGPU_EXPORT CUresult cuFuncGetParamCount(CUfunction func, size_t* paramCount) { return traced("cuFuncGetParamCount", cuFuncGetParamCount_impl, func, paramCount); }
static CUresult cuFuncGetParamCount_impl(CUfunction func, size_t* paramCount) {
  return api("cuFuncGetParamCount", true, false, [&](ShimState& s) -> CUresult {
    if (!paramCount || !func) return CUDA_ERROR_INVALID_VALUE;
    if ((reinterpret_cast<uintptr_t>(func) & 7) == kTagKernel) return CUDA_ERROR_INVALID_HANDLE;
    const auto it = s.functions.find(reinterpret_cast<uintptr_t>(func));
    if (it == s.functions.end() || !it->second.fn) return CUDA_ERROR_INVALID_VALUE;
    *paramCount = it->second.fn->params.size();
    return CUDA_SUCCESS;
  });
}

static CUresult cuKernelGetParamCount_impl(void* kernel, size_t* paramCount);
VGPU_EXPORT CUresult cuKernelGetParamCount(void* kernel, size_t* paramCount) { return traced("cuKernelGetParamCount", cuKernelGetParamCount_impl, kernel, paramCount); }
static CUresult cuKernelGetParamCount_impl(void* kernel, size_t* paramCount) {
  return api("cuKernelGetParamCount", true, false, [&](ShimState& s) -> CUresult {
    if (!paramCount || !kernel) return CUDA_ERROR_INVALID_VALUE;
    const uintptr_t kh = reinterpret_cast<uintptr_t>(kernel);
    if ((kh & 7) != kTagKernel) return CUDA_ERROR_INVALID_HANDLE;
    const auto it = s.kernels.find(kh);
    if (it == s.kernels.end()) return CUDA_ERROR_INVALID_VALUE;
    const int dev = ctx_stack().empty() ? 0 : current_device(s);
    const uint64_t mid = library_module_on(s, it->second.library, dev);
    const vgpu::ptx::EntryFn* fn = s.rt->device(dev).get_function(mid, it->second.name);
    if (!fn) return CUDA_ERROR_INVALID_VALUE;
    *paramCount = fn->params.size();
    return CUDA_SUCCESS;
  });
}

// ---- a host function, with its synchronization mode ----------------------------------------------------------------------------------
// As cuLaunchHostFunc, with CU_HOST_TASK_BLOCKING (0) or CU_HOST_TASK_SPINWAIT (1) for how the thread that runs it
// waits for the next task: both run the function once the stream's earlier work is done, which here is at once. Any other
// mode is CUDA_ERROR_INVALID_VALUE (checked first). A null function is CUDA_SUCCESS on the card and runs nothing.
static CUresult cuLaunchHostFunc_v2_impl(CUstream stream, CUhostFn fn, void* user, unsigned int syncMode);
VGPU_EXPORT CUresult cuLaunchHostFunc_v2(CUstream stream, CUhostFn fn, void* user, unsigned int syncMode) { return traced("cuLaunchHostFunc_v2", cuLaunchHostFunc_v2_impl, stream, fn, user, syncMode); }
static CUresult cuLaunchHostFunc_v2_impl(CUstream st, CUhostFn fn, void* user, unsigned int syncMode) {
  if (syncMode > 1) return CUDA_ERROR_INVALID_VALUE;
  if (!fn) return CUDA_SUCCESS;
  VGPU_CAPTURE_NODE(st, kNodeHost, fn(user));
  fn(user);
  return CUDA_SUCCESS;
}

// ---- one copy, with attributes -------------------------------------------------------------------------------------------------------
// cuMemcpyWithAttributesAsync and cuMemcpy3DWithAttributesAsync are a batch of one: a null attribute, a size or extent of
// 0, an access order that is invalid or past 3, flags past bit 0 and the legacy default stream are all
// CUDA_ERROR_INVALID_VALUE, as in the batch calls (measured for these two too).
static CUresult cuMemcpyWithAttributesAsync_impl(CUdeviceptr dst, CUdeviceptr src, size_t size, void* attr, CUstream hStream);
VGPU_EXPORT CUresult cuMemcpyWithAttributesAsync(CUdeviceptr dst, CUdeviceptr src, size_t size, void* attr, CUstream hStream) { return traced("cuMemcpyWithAttributesAsync", cuMemcpyWithAttributesAsync_impl, dst, src, size, attr, hStream); }
static CUresult cuMemcpyWithAttributesAsync_impl(CUdeviceptr dst, CUdeviceptr src, size_t size, void* attr, CUstream hStream) {
  unsigned long long d = dst, s = src;
  size_t z = size, idx = 0;
  return cuMemcpyBatchAsync_v2_impl(&d, &s, &z, 1, attr, &idx, 1, hStream);
}

static CUresult cuMemcpy3DWithAttributesAsync_impl(void* op, unsigned long long flags, CUstream hStream);
VGPU_EXPORT CUresult cuMemcpy3DWithAttributesAsync(void* op, unsigned long long flags, CUstream hStream) { return traced("cuMemcpy3DWithAttributesAsync", cuMemcpy3DWithAttributesAsync_impl, op, flags, hStream); }
static CUresult cuMemcpy3DWithAttributesAsync_impl(void* op, unsigned long long flags, CUstream hStream) {
  return cuMemcpy3DBatchAsync_v2_impl(1, op, flags, hStream);
}

// ---- GPU core dump callbacks ---------------------------------------------------------------------------------------------------------
// A callback registered is a handle; it is called when a GPU core dump starts or completes, which never happens here
// (a fault in a kernel is reported as the error it is, and no dump is written). A null callback is
// CUDA_ERROR_INVALID_VALUE and a null result handle CUDA_SUCCESS (the callback is registered all the same), as the
// card has it; deregistering a null handle is CUDA_ERROR_INVALID_VALUE. The card crashes on a handle deregistered twice or
// of the other kind; those are CUDA_ERROR_INVALID_VALUE here.
namespace {
std::mutex g_coredump_cb_mu;
std::set<uintptr_t> g_coredump_cbs;
uintptr_t g_next_coredump_cb = 0x1000;

CUresult coredump_register(const char* api_name, const void* callback, void** handle_out) {
  if (!callback) return CUDA_ERROR_INVALID_VALUE;
  (void)api_name;
  std::lock_guard<std::mutex> lock(g_coredump_cb_mu);
  const uintptr_t h = g_next_coredump_cb += 0x10;
  g_coredump_cbs.insert(h);
  if (handle_out) *handle_out = reinterpret_cast<void*>(h);
  return CUDA_SUCCESS;
}
CUresult coredump_deregister(void* handle) {
  if (!handle) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_coredump_cb_mu);
  return g_coredump_cbs.erase(reinterpret_cast<uintptr_t>(handle)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
}  // namespace

static CUresult cuCoredumpRegisterStartCallback_impl(void* callback, void* userData, void** callbackOut);
VGPU_EXPORT CUresult cuCoredumpRegisterStartCallback(void* callback, void* userData, void** callbackOut) { return traced("cuCoredumpRegisterStartCallback", cuCoredumpRegisterStartCallback_impl, callback, userData, callbackOut); }
static CUresult cuCoredumpRegisterStartCallback_impl(void* callback, void* userData, void** callbackOut) {
  (void)userData;
  return api("cuCoredumpRegisterStartCallback", true, false, [&](ShimState&) -> CUresult { return coredump_register("cuCoredumpRegisterStartCallback", callback, callbackOut); });
}
static CUresult cuCoredumpRegisterCompleteCallback_impl(void* callback, void* userData, void** callbackOut);
VGPU_EXPORT CUresult cuCoredumpRegisterCompleteCallback(void* callback, void* userData, void** callbackOut) { return traced("cuCoredumpRegisterCompleteCallback", cuCoredumpRegisterCompleteCallback_impl, callback, userData, callbackOut); }
static CUresult cuCoredumpRegisterCompleteCallback_impl(void* callback, void* userData, void** callbackOut) {
  (void)userData;
  return api("cuCoredumpRegisterCompleteCallback", true, false, [&](ShimState&) -> CUresult { return coredump_register("cuCoredumpRegisterCompleteCallback", callback, callbackOut); });
}
static CUresult cuCoredumpDeregisterStartCallback_impl(void* callback);
VGPU_EXPORT CUresult cuCoredumpDeregisterStartCallback(void* callback) { return traced("cuCoredumpDeregisterStartCallback", cuCoredumpDeregisterStartCallback_impl, callback); }
static CUresult cuCoredumpDeregisterStartCallback_impl(void* callback) {
  return api("cuCoredumpDeregisterStartCallback", true, false, [&](ShimState&) -> CUresult { return coredump_deregister(callback); });
}
static CUresult cuCoredumpDeregisterCompleteCallback_impl(void* callback);
VGPU_EXPORT CUresult cuCoredumpDeregisterCompleteCallback(void* callback) { return traced("cuCoredumpDeregisterCompleteCallback", cuCoredumpDeregisterCompleteCallback_impl, callback); }
static CUresult cuCoredumpDeregisterCompleteCallback_impl(void* callback) {
  return api("cuCoredumpDeregisterCompleteCallback", true, false, [&](ShimState&) -> CUresult { return coredump_deregister(callback); });
}

// ---- multicast objects, the second form ----------------------------------------------------------------------------------------------
// An RTX 3060 has no multicast objects, and both bind calls say so (CUDA_ERROR_NOT_SUPPORTED), as the first forms do.
static CUresult cuMulticastBindMem_v2_impl(unsigned long long, CUdevice, size_t, unsigned long long, size_t, size_t, unsigned long long);
VGPU_EXPORT CUresult cuMulticastBindMem_v2(unsigned long long a0, CUdevice a1, size_t a2, unsigned long long a3, size_t a4, size_t a5, unsigned long long a6) { return traced("cuMulticastBindMem_v2", cuMulticastBindMem_v2_impl, a0, a1, a2, a3, a4, a5, a6); }
static CUresult cuMulticastBindMem_v2_impl(unsigned long long, CUdevice, size_t, unsigned long long, size_t, size_t, unsigned long long) {
  return multicast_not_supported("cuMulticastBindMem_v2");
}
static CUresult cuMulticastBindAddr_v2_impl(unsigned long long, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long);
VGPU_EXPORT CUresult cuMulticastBindAddr_v2(unsigned long long a0, CUdevice a1, size_t a2, CUdeviceptr a3, size_t a4, unsigned long long a5) { return traced("cuMulticastBindAddr_v2", cuMulticastBindAddr_v2_impl, a0, a1, a2, a3, a4, a5); }
static CUresult cuMulticastBindAddr_v2_impl(unsigned long long, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long) {
  return multicast_not_supported("cuMulticastBindAddr_v2");
}

// ---- CUDA in Graphics ---------------------------------------------------------------------------------------------------------------------
// Capturing a stream into a command list of a graphics client (Direct3D 12, which this simulator has none of). On the
// card: a null parameter block is CUDA_ERROR_INVALID_VALUE and any other is CUDA_ERROR_INVALID_CONTEXT (the context
// is not a CIG context); ending a capture that was never begun is CUDA_ERROR_INVALID_VALUE.
static CUresult cuStreamBeginCaptureToCig_impl(CUstream hStream, void* params);
VGPU_EXPORT CUresult cuStreamBeginCaptureToCig(CUstream hStream, void* params) { return traced("cuStreamBeginCaptureToCig", cuStreamBeginCaptureToCig_impl, hStream, params); }
static CUresult cuStreamBeginCaptureToCig_impl(CUstream, void* params) {
  return api("cuStreamBeginCaptureToCig", true, false, [&](ShimState&) -> CUresult {
    return params ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_INVALID_VALUE;
  });
}
static CUresult cuStreamEndCaptureToCig_impl(CUstream hStream);
VGPU_EXPORT CUresult cuStreamEndCaptureToCig(CUstream hStream) { return traced("cuStreamEndCaptureToCig", cuStreamEndCaptureToCig_impl, hStream); }
static CUresult cuStreamEndCaptureToCig_impl(CUstream) {
  return api("cuStreamEndCaptureToCig", true, false, [&](ShimState&) -> CUresult { return CUDA_ERROR_INVALID_VALUE; });
}

// ---- green contexts: the resources of a stream, and a split of SMs ----------------------------------------------------------------
// Refused with the rest of the green-context machinery (above): the card answers these, this simulator has no
// SM partitions to hand out. Not part of the card-checked cases of the sweep for that reason.
static CUresult cuStreamGetDevResource_impl(CUstream hStream, void* resource, int type);
VGPU_EXPORT CUresult cuStreamGetDevResource(CUstream hStream, void* resource, int type) { return traced("cuStreamGetDevResource", cuStreamGetDevResource_impl, hStream, resource, type); }
static CUresult cuStreamGetDevResource_impl(CUstream, void*, int) {
  return api("cuStreamGetDevResource", true, false, [&](ShimState&) -> CUresult {
    return sweep_refuse("cuStreamGetDevResource", "green contexts and device resources are not simulated");
  });
}
static CUresult cuDevSmResourceSplit_impl(void* result, unsigned int nbGroups, const void* input, void* remainder, unsigned int flags, void* groupParams);
VGPU_EXPORT CUresult cuDevSmResourceSplit(void* result, unsigned int nbGroups, const void* input, void* remainder, unsigned int flags, void* groupParams) { return traced("cuDevSmResourceSplit", cuDevSmResourceSplit_impl, result, nbGroups, input, remainder, flags, groupParams); }
static CUresult cuDevSmResourceSplit_impl(void*, unsigned int, const void*, void*, unsigned int, void*) {
  return api("cuDevSmResourceSplit", true, false, [&](ShimState&) -> CUresult {
    return sweep_refuse("cuDevSmResourceSplit", "green contexts and device resources are not simulated");
  });
}
