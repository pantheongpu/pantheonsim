// libcuda's graph, stream-capture and user-object entry points. Each is a
// thin call into libcudart, which owns the graph machinery (vgpu_bridge.h):
// the driver and the runtime see one set of graphs, nodes and executable
// graphs, as NVIDIA's do. Included by driver_api.cpp after driver_bridge.inc.

// One export per entry: the driver call checks what every driver call checks
// (initialised, a live context) and hands the arguments to the runtime.
#define VGPU_DRIVER_GRAPH_EXPORT(name, params, args)                                              \
  VGPU_EXPORT CUresult name params {                                                              \
    return api(#name, true, false, [&](ShimState&) -> CUresult {                                  \
      return Bridge<decltype(&vgpu_##name)>::call("vgpu_" #name, VGPU_EXPAND args);               \
    });                                                                                           \
  }
VGPU_GRAPH_BRIDGE_LIST(VGPU_DRIVER_GRAPH_EXPORT)
#undef VGPU_DRIVER_GRAPH_EXPORT

// ---- the names the bridge list does not spell the way the driver does ---------

// cuGraphInstantiate came before the flags did, and kept its name: the
// original takes the node that failed and a log, and so does the later _v2.
VGPU_EXPORT CUresult cuGraphInstantiate(void** exec, void* g, void** errNode, char* log, size_t logSize) {
  return cuGraphInstantiate_v2(exec, g, errNode, log, logSize);
}
VGPU_EXPORT CUresult cuGraphInstantiateWithParams_ptsz(void** exec, void* g, void* params) {
  return cuGraphInstantiateWithParams(exec, g, params);
}

// cuStreamBeginCapture's original has no mode (global); the _v2 has one.
VGPU_EXPORT CUresult cuStreamBeginCapture(CUstream stream) { return cuStreamBeginCapture_v2(stream, 0); }
VGPU_EXPORT CUresult cuStreamGetCaptureInfo(CUstream stream, int* status, unsigned long long* id) {
  return cuStreamGetCaptureInfo_v3(stream, status, id, nullptr, nullptr, nullptr, nullptr);
}

// cuStreamGetCaptureInfo: the CUDA 10 form says whether and which capture; the
// CUDA 11.3 form (_v2) adds the graph and the dependencies; the CUDA 12.3 form
// (_v3) the edge data.
VGPU_EXPORT CUresult cuStreamGetCaptureInfo_v2(CUstream stream, int* status, unsigned long long* id, void** g,
                                               const void* const** deps, size_t* n) {
  return cuStreamGetCaptureInfo_v3(stream, status, id, g, deps, nullptr, n);
}
