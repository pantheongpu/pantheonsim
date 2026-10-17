// What Nsight Systems asks of CUPTI through the public API, one case. `nsys`
// loads its own CUPTI and does not reach this library (nvidia/docs/cupti.md,
// "Nsight Systems"), but while finding out why, two things it calls turned out
// to be missing here, and both are plain parts of cupti_activity.h that any
// consumer may use:
//  * cuptiSetThreadIdType / cuptiGetThreadIdType, which choose the number that
//    names a thread in the records: the POSIX thread handle (the default) or
//    the system thread id;
//  * the device record's `uuid`, which a consumer identifies a GPU by.
// Run against NVIDIA's libcupti on a card it gives
// nvidia/tests/data/cupti_nsys.expected; the shim must print the same.
#include "cupti_test_util.h"

#include <cuda.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <vector>

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

#if CUPTI_API_VERSION >= 130200
using DeviceRecord = CUpti_ActivityDevice6;
#elif CUPTI_API_VERSION >= 26
using DeviceRecord = CUpti_ActivityDevice5;
#else
using DeviceRecord = CUpti_ActivityDevice4;
#endif

namespace {

const char* result_name(CUptiResult r) {
  const char* s = "?";
  cuptiGetResultString(r, &s);
  return s;
}

std::vector<uint32_t> g_thread_ids;     // threadId of every runtime-API record
std::vector<CUuuid> g_device_uuids;     // uuid of the device-0 record(s)

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
  CUpti_Activity* r = nullptr;
  while (cuptiActivityGetNextRecord(buffer, valid, &r) == CUPTI_SUCCESS) {
    if (r->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
      g_thread_ids.push_back(reinterpret_cast<const CUpti_ActivityAPI*>(r)->threadId);
    } else if (r->kind == CUPTI_ACTIVITY_KIND_DEVICE) {
      const auto* d = reinterpret_cast<const DeviceRecord*>(r);
      if (d->id == 0) g_device_uuids.push_back(d->uuid);
    }
  }
}

// Whether every runtime record of the last batch names this thread by `want`.
bool all_ids_are(uint32_t want) {
  if (g_thread_ids.empty()) return false;
  for (uint32_t id : g_thread_ids)
    if (id != want) return false;
  return true;
}

// Reads the type into `t` before printing it: the arguments of a call are not
// evaluated in a fixed order.
void print_type(const char* what) {
  CUpti_ActivityThreadIdType t = CUPTI_ACTIVITY_THREAD_ID_TYPE_FORCE_INT;
  const CUptiResult r = cuptiGetThreadIdType(&t);
  out(fmt("%s -> %s, type %d", what, result_name(r), static_cast<int>(t)));
}

int run_batch(const char* label) {
  g_thread_ids.clear();
  void* p = nullptr;
  CK(cudaMalloc(&p, 256));
  CK(cudaFree(p));
  if (cuptiActivityFlushAll(0) != CUPTI_SUCCESS) return 1;
  const uint32_t posix = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_self()));
  const uint32_t sys = static_cast<uint32_t>(syscall(SYS_gettid));
  out(fmt("%s: runtime records seen: %d; all carry pthread_self: %d; all carry gettid: %d",
          label, g_thread_ids.empty() ? 0 : 1, all_ids_are(posix) ? 1 : 0, all_ids_are(sys) ? 1 : 0));
  return 0;
}

}  // namespace

int main() {
  print_type("get default");
  out(fmt("get(NULL) -> %s", result_name(cuptiGetThreadIdType(nullptr))));

  if (cuptiActivityRegisterCallbacks(cupti_test::request_buffer, buffer_completed) != CUPTI_SUCCESS) return 1;
  if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME) != CUPTI_SUCCESS) return 1;
  if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE) != CUPTI_SUCCESS) return 1;
  CK(cudaSetDevice(0));
  CK(cudaFree(nullptr));
  if (cuptiActivityFlushAll(0) != CUPTI_SUCCESS) return 1;

  // The device record's uuid is the one the runtime reports.
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  CUuuid zero;
  std::memset(&zero, 0, sizeof zero);
  int same = 0, nonzero = 0;
  for (const CUuuid& u : g_device_uuids) {
    if (std::memcmp(u.bytes, prop.uuid.bytes, sizeof u.bytes) == 0) ++same;
    if (std::memcmp(u.bytes, zero.bytes, sizeof u.bytes) != 0) ++nonzero;
  }
  out(fmt("device record for device 0: %s; uuid equals cudaDeviceProp.uuid: %d; uuid not all zero: %d",
          g_device_uuids.empty() ? "none" : "seen", same > 0 ? 1 : 0, nonzero > 0 ? 1 : 0));

  if (run_batch("default type")) return 1;

  out(fmt("set SYSTEM -> %s", result_name(cuptiSetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM))));
  print_type("get");
  if (run_batch("system type")) return 1;

  out(fmt("set DEFAULT -> %s", result_name(cuptiSetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT))));
  print_type("get");
  if (run_batch("default again")) return 1;

  out(fmt("set 7 -> %s", result_name(cuptiSetThreadIdType(static_cast<CUpti_ActivityThreadIdType>(7)))));
  print_type("get");

  cupti_test::print_all();
  return 0;
}
