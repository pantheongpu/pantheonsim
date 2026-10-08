// A program that marks itself with NVTX while CUPTI listens, printing the
// callbacks and the activity records it gets, for comparison with what NVIDIA's
// CUPTI printed for the same calls on an RTX 3060. NVTX is header-only and
// does nothing until a tool is injected; the script names the CUPTI under test
// in NVTX_INJECTION64_PATH, as a profiler does.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cupti.h>
#include <nvtx3/nvToolsExt.h>

namespace {
std::vector<std::string> g_out;

void CUPTIAPI request(uint8_t** b, size_t* s, size_t* m) {
  static uint8_t storage[1 << 20] __attribute__((aligned(8)));
  *b = storage;
  *s = sizeof storage;
  *m = 0;
}

void CUPTIAPI complete(CUcontext, uint32_t, uint8_t* b, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  char line[256];
  while (cuptiActivityGetNextRecord(b, valid, &r) == CUPTI_SUCCESS) {
    if (r->kind == CUPTI_ACTIVITY_KIND_MARKER) {
      const auto* m = reinterpret_cast<const CUpti_ActivityMarker2*>(r);
      std::snprintf(line, sizeof line, "MARKER flags=0x%x id=%u object=%d name=%s domain=%s", (unsigned)m->flags,
                    m->id, (int)m->objectKind, m->name ? m->name : "(null)", m->domain ? m->domain : "(null)");
    } else if (r->kind == CUPTI_ACTIVITY_KIND_MARKER_DATA) {
      const auto* m = reinterpret_cast<const CUpti_ActivityMarkerData*>(r);
      std::snprintf(line, sizeof line, "MARKER_DATA flags=0x%x id=%u color=0x%x category=%u payloadKind=%d payload=%llu",
                    (unsigned)m->flags, m->id, m->color, m->category, (int)m->payloadKind,
                    (unsigned long long)m->payload.metricValueUint64);
    } else if (r->kind == CUPTI_ACTIVITY_KIND_NAME) {
      const auto* m = reinterpret_cast<const CUpti_ActivityName*>(r);
      std::snprintf(line, sizeof line, "NAME object=%d name=%s", (int)m->objectKind, m->name ? m->name : "(null)");
    } else {
      continue;
    }
    g_out.push_back(line);
  }
}

std::vector<std::string> g_callbacks;
void CUPTIAPI on_callback(void*, CUpti_CallbackDomain d, CUpti_CallbackId id, const void* data) {
  if (d != CUPTI_CB_DOMAIN_NVTX) return;
  const auto* n = static_cast<const CUpti_NvtxData*>(data);
  g_callbacks.push_back(std::string("CB nvtx ") + std::to_string(id) + " " + n->functionName);
}

nvtxEventAttributes_t attrib(const char* msg, uint32_t color, uint32_t category) {
  nvtxEventAttributes_t a;
  std::memset(&a, 0, sizeof a);
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  a.colorType = color ? NVTX_COLOR_ARGB : NVTX_COLOR_UNKNOWN;
  a.color = color;
  a.messageType = NVTX_MESSAGE_TYPE_ASCII;
  a.message.ascii = msg;
  a.category = category;
  return a;
}
}  // namespace

int main() {
  CUpti_SubscriberHandle sub;
  if (cuptiSubscribe(&sub, on_callback, nullptr) != CUPTI_SUCCESS) return 1;
  cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_NVTX);
  cuptiActivityRegisterCallbacks(request, complete);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME);
  cudaFree(nullptr);

  nvtxMarkA("plain mark");
  nvtxMarkW(L"wide mark");
  nvtxRangePushA("outer");
  nvtxRangePushA("inner");
  nvtxRangePop();
  nvtxRangePop();
  nvtxRangeId_t r = nvtxRangeStartA("started");
  nvtxRangeEnd(r);
  nvtxEventAttributes_t colored = attrib("coloured", 0xff336699u, 7);
  r = nvtxRangeStartEx(&colored);
  nvtxRangeEnd(r);
  nvtxEventAttributes_t plain = attrib("no colour", 0, 0);
  nvtxMarkEx(&plain);
  nvtxEventAttributes_t with_double = attrib("double payload", 0, 1);
  with_double.payloadType = NVTX_PAYLOAD_TYPE_DOUBLE;
  with_double.payload.dValue = 2.5;
  nvtxMarkEx(&with_double);
  nvtxEventAttributes_t with_int = attrib("int64 payload", 0, 1);
  with_int.payloadType = NVTX_PAYLOAD_TYPE_INT64;
  with_int.payload.llValue = -7;
  nvtxMarkEx(&with_int);
  nvtxEventAttributes_t with_uint = attrib("uint64 payload", 0, 1);
  with_uint.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  with_uint.payload.ullValue = 42;
  nvtxMarkEx(&with_uint);
  nvtxNameOsThreadA(1234, "worker");
  // The wide-character spellings: whether a tool is told of them is part of
  // what is being compared.
  nvtxRangeId_t wr = nvtxRangeStartW(L"wide range");
  nvtxRangeEnd(wr);
  nvtxRangePushW(L"wide push");
  nvtxRangePop();
  nvtxNameOsThreadW(4321, L"wide worker");

  nvtxDomainHandle_t dom = nvtxDomainCreateA("mydomain");
  nvtxStringHandle_t reg = nvtxDomainRegisterStringA(dom, "registered text");
  nvtxEventAttributes_t named = attrib(nullptr, 0xff00ff00u, 3);
  named.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  named.message.registered = reg;
  nvtxDomainMarkEx(dom, &named);
  nvtxDomainRangePushEx(dom, &colored);
  nvtxDomainRangePop(dom);
  nvtxRangeId_t dr = nvtxDomainRangeStartEx(dom, &plain);
  nvtxDomainRangeEnd(dom, dr);
  nvtxDomainDestroy(dom);

  cuptiActivityFlushAll(0);
  cuptiUnsubscribe(sub);
  std::printf("# callbacks\n");
  for (const auto& l : g_callbacks) std::printf("%s\n", l.c_str());
  std::printf("# activity\n");
  for (const auto& l : g_out) std::printf("%s\n", l.c_str());
  return 0;
}
