// What the resource callbacks and the context, stream and function records
// tell a profiler about a program's code and its streams. A program launches
// kernels of its own module, queries one, makes streams and sets their
// attributes, loads a module of the driver's through a cubin, makes and
// destroys a second context, and resets the device while a stream and a module
// are alive; the subscriber prints every resource callback and the activity
// records that name the context, its own streams and the functions it loaded.
// Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_resource.expected (an RTX 3060, CUDA 13.0).
//
// What the card does, and the shim reproduces:
//  * a program's code is loaded as one module when its first kernel is
//    launched, with the module's cubin (the very image of the fatbin that runs),
//    and every launch of one of its kernels afterwards says the module is being
//    profiled -- whether or not anything profiles it;
//  * a context is made, with eight streams of the driver's own told beside it,
//    when the device is first set, and cudaDeviceReset ends it: the context is
//    told first, then its streams, then its modules;
//  * setting a stream attribute is told, naming no context, with the id of the
//    attribute and the value passed;
//  * loading a module through the driver API tells the module at once, with its
//    cubin when it was given one.
// Not reproduced: the cubin of a module loaded from PTX text, which the card
// shows as the driver's own compilation of it and this engine cannot hand over
// (it gives none).
#include "cupti_test_util.h"

#include <cuda.h>

#include <initializer_list>

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
#define CU(x)                                                                          \
  do {                                                                                 \
    CUresult r_ = (x);                                                                 \
    if (r_) {                                                                          \
      std::fprintf(stderr, "FAIL %s:%d: driver error %d\n", __FILE__, __LINE__, (int)r_); \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

namespace {

bool wanted_runtime(const char* n) {
  static const char* const names[] = {"cudaLaunchKernel", "cudaStreamCreate", "cudaStreamDestroy",
                                      "cudaStreamSetAttribute", "cudaStreamCopyAttributes", "cudaDeviceReset",
                                      "cudaFuncGetAttributes", "cudaSetDevice", "cudaFree", "cudaMalloc"};
  for (const char* w : names)
    if (std::strcmp(n, w) == 0) return true;
  return false;
}

std::string cubin_info(const CUpti_ModuleResourceData* m) {
  if (!m) return "no module data";
  std::string s = fmt("module=#m%u cubin=%s", m->moduleId, m->pCubin ? "set" : "none");
  if (m->pCubin && m->cubinSize >= 4)
    s += fmt(" magic=%02x%02x%02x%02x size=%zu", (unsigned char)m->pCubin[0], (unsigned char)m->pCubin[1],
             (unsigned char)m->pCubin[2], (unsigned char)m->pCubin[3], m->cubinSize);
  return s;
}

void CUPTIAPI on_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    const auto* cb = static_cast<const CUpti_CallbackData*>(data);
    if (wanted_runtime(cb->functionName))
      out(fmt("CB runtime %s %s", cb->functionName, cb->callbackSite == CUPTI_API_ENTER ? "ENTER" : "EXIT"));
    return;
  }
  if (domain != CUPTI_CB_DOMAIN_RESOURCE) return;
  if (!data) {
    out(fmt("RES %d no data", (int)cbid));
    return;
  }
  const auto* r = static_cast<const CUpti_ResourceData*>(data);
  switch (cbid) {
    case CUPTI_CBID_RESOURCE_MODULE_LOADED:
    case CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING:
    case CUPTI_CBID_RESOURCE_MODULE_PROFILED:
      out(fmt("RES %d context=#x%llu %s", (int)cbid, (unsigned long long)reinterpret_cast<uintptr_t>(r->context),
              cubin_info(static_cast<const CUpti_ModuleResourceData*>(r->resourceDescriptor)).c_str()));
      break;
#if CUPTI_API_VERSION >= 22
    case 21: {   // CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED
      const auto* a = static_cast<const CUpti_StreamAttrData*>(r->resourceDescriptor);
      out(fmt("RES %d context=%s stream=#s%llu attribute=%d value=%s", (int)cbid, r->context ? "set" : "none",
              (unsigned long long)reinterpret_cast<uintptr_t>(a->stream), (int)a->attr, a->value ? "set" : "null"));
      break;
    }
#endif
    default:
      out(fmt("RES %d context=#x%llu stream=#s%llu", (int)cbid,
              (unsigned long long)reinterpret_cast<uintptr_t>(r->context),
              (unsigned long long)reinterpret_cast<uintptr_t>(r->resourceHandle.stream)));
  }
}

#if CUPTI_API_VERSION >= 130000
using ContextRecord = CUpti_ActivityContext3;
#elif CUPTI_API_VERSION >= 24
using ContextRecord = CUpti_ActivityContext3;
#else
using ContextRecord = CUpti_ActivityContext2;
#endif

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    switch (static_cast<int>(r->kind)) {
      case CUPTI_ACTIVITY_KIND_STREAM: {
        const auto* t = reinterpret_cast<const CUpti_ActivityStream*>(r);
        out(fmt("REC stream id=#s%u context=#x%u flag=%d priority=%d corr=#c%u", t->streamId, t->contextId,
                (int)t->flag, (int)t->priority, t->correlationId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_CONTEXT: {
        const auto* c = reinterpret_cast<const ContextRecord*>(r);
        out(fmt("REC context id=#x%u device=%u nullstream=#s%u", c->contextId, c->deviceId, (unsigned)c->nullStreamId));
        break;
      }
      case 26: {   // CUPTI_ACTIVITY_KIND_FUNCTION
        const auto* f = reinterpret_cast<const CUpti_ActivityFunction*>(r);
        out(fmt("REC function name=%s module=#m%u context=#x%u", f->name, f->moduleId, f->contextId));
        break;
      }
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        const auto* a = reinterpret_cast<const CUpti_ActivityAPI*>(r);
        const char* name = nullptr;
        cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &name);
        if (wanted_runtime(name ? std::string(name).substr(0, std::string(name).rfind("_v")).c_str() : ""))
          out(fmt("REC runtime %s corr=#c%u", name ? name : "?", a->correlationId));
        break;
      }
      default: break;
    }
  }
}

}  // namespace

__global__ void kernel_one(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1.f;
}
__global__ void kernel_two(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= 2.f;
}
__global__ void kernel_three(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] -= 1.f;
}

int main() {
  CUpti_SubscriberHandle sub;
  cuptiSubscribe(&sub, on_callback, nullptr);
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE);
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API);
  cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed);
  for (int kind : std::initializer_list<int>{CUPTI_ACTIVITY_KIND_STREAM, CUPTI_ACTIVITY_KIND_CONTEXT,
                                              26 /* FUNCTION */, CUPTI_ACTIVITY_KIND_RUNTIME})
    cuptiActivityEnable(static_cast<CUpti_ActivityKind>(kind));

  out("# context");
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  cuptiActivityFlushAll(0);

  // ---- the program's own module ----
  out("# launches");
  float* d = nullptr;
  CK(cudaMalloc(&d, 4096));
  kernel_one<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  kernel_one<<<1, 32>>>(d, 32);
  kernel_two<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  cudaFuncAttributes fa;
  CK(cudaFuncGetAttributes(&fa, reinterpret_cast<const void*>(kernel_three)));
  kernel_three<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  cuptiActivityFlushAll(0);

  // ---- streams and their attributes ----
  out("# streams");
  cudaStream_t s1, s2;
  CK(cudaStreamCreate(&s1));
  CK(cudaStreamCreate(&s2));
  cudaStreamAttrValue v;
  std::memset(&v, 0, sizeof v);
  v.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  v.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  CK(cudaStreamSetAttribute(s1, cudaStreamAttributeAccessPolicyWindow, &v));
  cudaStreamAttrValue sync;
  std::memset(&sync, 0, sizeof sync);
  sync.syncPolicy = cudaSyncPolicyAuto;
  CK(cudaStreamSetAttribute(s1, cudaStreamAttributeSynchronizationPolicy, &sync));
  CK(cudaStreamCopyAttributes(s2, s1));
  CK(cudaStreamDestroy(s2));
  cuptiActivityFlushAll(0);

  // ---- a module through the driver API ----
  out("# driver module");
  const char* path = std::getenv("CUPTI_TEST_CUBIN");
  if (path && path[0]) {
    std::vector<char> image;
    if (FILE* f = std::fopen(path, "rb")) {
      char buf[4096];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) image.insert(image.end(), buf, buf + n);
      std::fclose(f);
    }
    if (image.empty()) {
      std::fprintf(stderr, "FAIL: cannot read %s\n", path);
      return 1;
    }
    CUmodule mod;
    CU(cuModuleLoadData(&mod, image.data()));
    CUfunction fill;
    CU(cuModuleGetFunction(&fill, mod, "module_fill"));
    int n = 32;
    void* kargs[] = {&d, &n};
    CU(cuLaunchKernel(fill, 1, 1, 1, 32, 1, 1, 0, nullptr, kargs, nullptr));
    CU(cuLaunchKernel(fill, 1, 1, 1, 32, 1, 1, 0, nullptr, kargs, nullptr));
    CK(cudaDeviceSynchronize());
    CU(cuModuleUnload(mod));
  }

  // ---- a second context of the driver's, made and destroyed ----
  out("# second context");
  {
    CUdevice dev;
    CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
#if CUDA_VERSION >= 13000
    CU(cuCtxCreate(&ctx, nullptr, 0, dev));
#else
    CU(cuCtxCreate(&ctx, 0, dev));
#endif
    CU(cuCtxDestroy(ctx));
  }
  cuptiActivityFlushAll(0);

  // ---- reset, with a stream and a module alive ----
  out("# reset");
  CK(cudaDeviceReset());
  CK(cudaFree(nullptr));
  out("# after reset");
  CK(cudaMalloc(&d, 4096));
  kernel_one<<<1, 32>>>(d, 32);
  CK(cudaDeviceSynchronize());
  cuptiActivityFlushAll(0);

  cuptiUnsubscribe(sub);
  cupti_test::print_all();
  std::printf("# end\n");
  return 0;
}
