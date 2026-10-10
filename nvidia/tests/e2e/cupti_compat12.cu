// The entry points CUDA 12.x spells with a _v2 (or _v3), as a CUDA 12 program
// calls them and a CUDA 12 profiler hears them: the edge-data graph calls
// (12.3), the location-taking prefetch and advise (12.2), the capture queries
// with edge data, the event-kind elapsed time (12.8) and the second device
// properties. CUDA 13 gave these the plain names, so this is only a program
// of the 12.3-12.9 headers, run against a runtime shim built against them; with
// any other it prints SKIP and nothing else.
//
// What a CUDA 12.8 CUPTI reported for each on an RTX 3060 is in
// nvidia/tests/data/cupti_compat12.expected: the function name carries the
// _v2 (cudaGraphAddNode_v2, not cudaGraphAddNode), and so does the callback
// id's name (cudaGraphAddNode_v2_v12030). The entry points are looked up by
// name at run time so the program links against any runtime, and skips where the
// runtime has none of them (a shim built against CUDA 13 does not export them).
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

#if CUDART_VERSION < 12030 || CUDART_VERSION >= 13000

int main() {
  std::printf("SKIP: this needs the CUDA 12.3 to 12.9 headers (the _v2 entry points)\n");
  return 0;
}

#else

#include <dlfcn.h>

#include <cupti.h>

namespace {

// The types of the four entry points whose declarations give trailing parameters a default
// argument. They are spelled out because nvcc's front end (12.0 to 12.9) writes
// decltype(&function) back out with those defaults, and the host compiler then refuses it.
using CaptureInfoV3Fn = cudaError_t (*)(cudaStream_t, cudaStreamCaptureStatus*, unsigned long long*, cudaGraph_t*,
                                        const cudaGraphNode_t**, const cudaGraphEdgeData**, size_t*);
using CaptureInfoV2Fn = cudaError_t (*)(cudaStream_t, cudaStreamCaptureStatus*, unsigned long long*, cudaGraph_t*,
                                        const cudaGraphNode_t**, size_t*);
using UpdateCaptureV2Fn = cudaError_t (*)(cudaStream_t, cudaGraphNode_t*, const cudaGraphEdgeData*, size_t, unsigned int);
using PrefetchV2Fn = cudaError_t (*)(const void*, size_t, cudaMemLocation, unsigned int, cudaStream_t);

std::vector<std::string> g_lines;

template <class T>
std::string field(const char* name, const T& v) {
  char buf[96];
  if constexpr (std::is_pointer_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%s", name, v ? "set" : "null");
  } else if constexpr (std::is_enum_v<T> || std::is_integral_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%lld", name, static_cast<long long>(v));
  } else if constexpr (std::is_floating_point_v<T>) {
    std::snprintf(buf, sizeof buf, " %s=%g", name, static_cast<double>(v));
  } else {
    std::snprintf(buf, sizeof buf, " %s=<%zu bytes>", name, sizeof(T));
  }
  return buf;
}

#define F(n) field(#n, p->n)
std::string params_of(CUpti_CallbackId id, const void* d) {
  switch (id) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddNode_v2_v12030: {
      const auto* p = static_cast<const cudaGraphAddNode_v2_v12030_params*>(d);
      return F(pGraphNode) + F(graph) + F(pDependencies) + F(dependencyData) + F(numDependencies) + F(nodeParams);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphAddDependencies_v2_v12030: {
      const auto* p = static_cast<const cudaGraphAddDependencies_v2_v12030_params*>(d);
      return F(graph) + F(from) + F(to) + F(edgeData) + F(numDependencies);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphRemoveDependencies_v2_v12030: {
      const auto* p = static_cast<const cudaGraphRemoveDependencies_v2_v12030_params*>(d);
      return F(graph) + F(from) + F(to) + F(edgeData) + F(numDependencies);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphGetEdges_v2_v12030: {
      const auto* p = static_cast<const cudaGraphGetEdges_v2_v12030_params*>(d);
      return F(graph) + F(from) + F(to) + F(edgeData) + F(numEdges);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependencies_v2_v12030: {
      const auto* p = static_cast<const cudaGraphNodeGetDependencies_v2_v12030_params*>(d);
      return F(node) + F(pDependencies) + F(edgeData) + F(pNumDependencies);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphNodeGetDependentNodes_v2_v12030: {
      const auto* p = static_cast<const cudaGraphNodeGetDependentNodes_v2_v12030_params*>(d);
      return F(node) + F(pDependentNodes) + F(edgeData) + F(pNumDependentNodes);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v3_v12030: {
      const auto* p = static_cast<const cudaStreamGetCaptureInfo_v3_v12030_params*>(d);
      return F(stream) + F(captureStatus_out) + F(id_out) + F(graph_out) + F(dependencies_out) + F(edgeData_out) +
             F(numDependencies_out);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamGetCaptureInfo_v2_v11030: {
      const auto* p = static_cast<const cudaStreamGetCaptureInfo_v2_v11030_params*>(d);
      return F(stream) + F(captureStatus_out) + F(id_out) + F(graph_out) + F(dependencies_out) + F(numDependencies_out);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v2_v12030: {
      const auto* p = static_cast<const cudaStreamUpdateCaptureDependencies_v2_v12030_params*>(d);
      return F(stream) + F(dependencies) + F(dependencyData) + F(numDependencies) + F(flags);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamUpdateCaptureDependencies_v11030: {
      const auto* p = static_cast<const cudaStreamUpdateCaptureDependencies_v11030_params*>(d);
      return F(stream) + F(dependencies) + F(numDependencies) + F(flags);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemPrefetchAsync_v2_v12020: {
      const auto* p = static_cast<const cudaMemPrefetchAsync_v2_v12020_params*>(d);
      return F(devPtr) + F(count) + F(location) + F(flags) + F(stream);
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemAdvise_v2_v12020: {
      const auto* p = static_cast<const cudaMemAdvise_v2_v12020_params*>(d);
      return F(devPtr) + F(count) + F(advice) + F(location);
    }
#if CUDART_VERSION >= 12080
    case CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v2_v12080: {
      const auto* p = static_cast<const cudaEventElapsedTime_v2_v12080_params*>(d);
      return F(ms) + F(start) + F(end);
    }
#endif
    case CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v2_v12000: {
      const auto* p = static_cast<const cudaGetDeviceProperties_v2_v12000_params*>(d);
      return F(prop) + F(device);
    }
    default: return "";
  }
}
#undef F

void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;
  const auto* cb = static_cast<const CUpti_CallbackData*>(data);
  const bool enter = cb->callbackSite == CUPTI_API_ENTER;
  const char* id_name = nullptr;
  cuptiGetCallbackName(domain, cbid, &id_name);
  std::string line = std::string("CB ") + cb->functionName + " [" + (id_name ? id_name : "?") + "]" +
                     (enter ? " ENTER" : " EXIT");
  if (enter) {
    line += params_of(cbid, cb->functionParams);
  } else if (cb->functionReturnValue) {
    line += " ret=" + std::to_string(static_cast<int>(*static_cast<const cudaError_t*>(cb->functionReturnValue)));
  }
  g_lines.push_back(line);
}

template <class Fn>
Fn lookup(const char* name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, name));
}

}  // namespace

int main() {
  const auto add_dependencies_v2 = lookup<decltype(&cudaGraphAddDependencies_v2)>("cudaGraphAddDependencies_v2");
  const auto remove_dependencies_v2 =
      lookup<decltype(&cudaGraphRemoveDependencies_v2)>("cudaGraphRemoveDependencies_v2");
  const auto get_edges_v2 = lookup<decltype(&cudaGraphGetEdges_v2)>("cudaGraphGetEdges_v2");
  const auto get_dependencies_v2 =
      lookup<decltype(&cudaGraphNodeGetDependencies_v2)>("cudaGraphNodeGetDependencies_v2");
  const auto get_dependents_v2 =
      lookup<decltype(&cudaGraphNodeGetDependentNodes_v2)>("cudaGraphNodeGetDependentNodes_v2");
  const auto add_node_v2 = lookup<decltype(&cudaGraphAddNode_v2)>("cudaGraphAddNode_v2");
  const auto capture_info_v3 = lookup<CaptureInfoV3Fn>("cudaStreamGetCaptureInfo_v3");
  const auto capture_info_v2 = lookup<CaptureInfoV2Fn>("cudaStreamGetCaptureInfo_v2");
  const auto update_capture_v2 =
      lookup<UpdateCaptureV2Fn>("cudaStreamUpdateCaptureDependencies_v2");
  const auto prefetch_v2 = lookup<PrefetchV2Fn>("cudaMemPrefetchAsync_v2");
  const auto advise_v2 = lookup<decltype(&cudaMemAdvise_v2)>("cudaMemAdvise_v2");
  const auto properties_v2 = lookup<decltype(&cudaGetDeviceProperties_v2)>("cudaGetDeviceProperties_v2");
#if CUDART_VERSION >= 12080
  const auto elapsed_v2 = lookup<decltype(&cudaEventElapsedTime_v2)>("cudaEventElapsedTime_v2");
#else
  const void* elapsed_v2 = reinterpret_cast<const void*>(1);
#endif
  if (!add_dependencies_v2 || !remove_dependencies_v2 || !get_edges_v2 || !get_dependencies_v2 || !get_dependents_v2 ||
      !add_node_v2 || !capture_info_v3 || !capture_info_v2 || !update_capture_v2 || !prefetch_v2 || !advise_v2 ||
      !properties_v2 || !elapsed_v2) {
    std::printf("SKIP: the runtime does not export the CUDA 12 _v2 entry points\n");
    return 0;
  }

  CUpti_SubscriberHandle sub;
  if (cuptiSubscribe(&sub, on_callback, nullptr) != CUPTI_SUCCESS) return 1;
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API);

  // Warm-up: make the context, so lazy creation is not in the output.
  cudaSetDevice(0);
  cudaFree(nullptr);
  g_lines.clear();

  // ---- graph edges with edge data ----
  cudaGraph_t g;
  cudaGraphCreate(&g, 0);
  cudaGraphNode_t a, b, c;
  cudaGraphAddEmptyNode(&a, g, nullptr, 0);
  cudaGraphAddEmptyNode(&b, g, nullptr, 0);
  add_dependencies_v2(g, &a, &b, nullptr, 1);
  size_t edges = 0;
  get_edges_v2(g, nullptr, nullptr, nullptr, &edges);
  cudaGraphNode_t from[2], to[2];
  cudaGraphEdgeData data[2];
  edges = 2;
  get_edges_v2(g, from, to, data, &edges);
  size_t n = 0;
  get_dependencies_v2(b, nullptr, nullptr, &n);
  n = 0;
  get_dependents_v2(a, nullptr, nullptr, &n);
  // The node parameters are a union with constructors in the headers: a zeroed
  // buffer of the right size and alignment is the portable way to fill one.
  alignas(cudaGraphNodeParams) unsigned char storage[sizeof(cudaGraphNodeParams)];
  std::memset(storage, 0, sizeof storage);
  auto* np = reinterpret_cast<cudaGraphNodeParams*>(storage);
  np->type = cudaGraphNodeTypeEmpty;
  add_node_v2(&c, g, &b, nullptr, 1, np);
  remove_dependencies_v2(g, &a, &b, nullptr, 1);
  cudaGraphDestroy(g);

  // ---- capture ----
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  cudaStreamCaptureStatus status;
  unsigned long long id;
  cudaGraph_t cg;
  const cudaGraphNode_t* deps;
  const cudaGraphEdgeData* edge_data;
  size_t ndeps;
  capture_info_v3(s, &status, &id, &cg, &deps, &edge_data, &ndeps);
  capture_info_v2(s, &status, &id, &cg, &deps, &ndeps);
  update_capture_v2(s, nullptr, nullptr, 0, cudaStreamAddCaptureDependencies);
  cudaStreamUpdateCaptureDependencies(s, nullptr, 0, cudaStreamAddCaptureDependencies);
  cudaGraph_t captured;
  cudaStreamEndCapture(s, &captured);
  cudaGraphDestroy(captured);

  // ---- managed memory with a location ----
  void* managed = nullptr;
  cudaMallocManaged(&managed, 4096);
  cudaMemLocation where;
  std::memset(&where, 0, sizeof where);
  where.type = cudaMemLocationTypeDevice;
  where.id = 0;
  advise_v2(managed, 4096, cudaMemAdviseSetReadMostly, where);
  // A prefetch to a device is refused by the card under WSL (no concurrent
  // managed access) and not by a native Linux one, so the location it is asked
  // for is one every machine refuses.
  cudaMemLocation nowhere;
  std::memset(&nowhere, 0, sizeof nowhere);
  nowhere.type = cudaMemLocationTypeInvalid;
  prefetch_v2(managed, 4096, nowhere, 0, s);
  cudaFree(managed);

  // ---- events and properties ----
  cudaEvent_t e1, e2;
  cudaEventCreate(&e1);
  cudaEventCreate(&e2);
  cudaEventRecord(e1, s);
  cudaEventRecord(e2, s);
  cudaStreamSynchronize(s);
  float ms = 0;
  cudaEventElapsedTime(&ms, e1, e2);
#if CUDART_VERSION >= 12080
  elapsed_v2(&ms, e1, e2);
#endif
  cudaEventDestroy(e1);
  cudaEventDestroy(e2);
  cudaStreamDestroy(s);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  properties_v2(&prop, 0);

  cuptiUnsubscribe(sub);
  for (const std::string& l : g_lines) std::printf("%s\n", l.c_str());
  std::printf("# end\n");
  return 0;
}

#endif
