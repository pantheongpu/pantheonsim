// hipDeviceGetAttribute and the PCI bus calls answer from each device's properties, made at the first question and kept
// (properties_of in hip_api.cpp). Two devices differ in their bus, so a cache that mixed them up, or kept the first answer for
// every ordinal, gives one device the other's. The answers must also be what the full structure says (hipGetDeviceProperties
// fills it afresh), and the same on every ask.
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
int hipDeviceGetAttribute(int*, int, int);
int hipDeviceGetPCIBusId(char*, int, int);
int hipDeviceGetByPCIBusId(int*, const char*);
int hipGetDeviceCount(int*);
}

namespace {
constexpr int kMaxThreadsPerBlock = 56, kMultiprocessorCount = 63, kPciBusId = 67, kPciDeviceId = 68, kWarpSize = 87,
              kTotalGlobalMem = 0;
int failures = 0;
void check(bool ok, const char* what, int device = -1) {
  if (!ok) {
    std::printf("FAIL %s (device %d)\n", what, device);
    ++failures;
  }
}
int attr(int which, int device) {
  int v = -12345;
  if (hipDeviceGetAttribute(&v, which, device) != 0) return -99999;
  return v;
}
}  // namespace

int main() {
  setenv("VGPU_DEVICE_COUNT", "3", 1);
  int count = 0;
  check(hipGetDeviceCount(&count) == 0 && count == 3, "three devices");
  (void)kTotalGlobalMem;
  const int first_threads = attr(kMaxThreadsPerBlock, 0), first_cus = attr(kMultiprocessorCount, 0);
  // Every ordinal, in an order that asks each after each of the others; the answer for one never depends on what was asked before.
  const int order[] = {0, 1, 2, 2, 1, 0, 1, 0, 2, 0, 2, 1};
  for (int d : order) {
    check(attr(kPciBusId, d) == d + 1, "the bus is the ordinal + 1", d);
    check(attr(kPciDeviceId, d) == 0, "the device number is 0", d);
    check(attr(kWarpSize, d) > 0 && attr(kMaxThreadsPerBlock, d) == first_threads, "the same kind of device answers alike", d);
    check(attr(kMultiprocessorCount, d) == first_cus, "the multiprocessor count", d);
    char id[32];
    char want[32];
    std::snprintf(want, sizeof want, "0000:%02x:00.0", d + 1);
    check(hipDeviceGetPCIBusId(id, sizeof id, d) == 0 && std::strcmp(id, want) == 0, "the bus id string", d);
    int back = -1;
    check(hipDeviceGetByPCIBusId(&back, want) == 0 && back == d, "the ordinal back from the bus id", d);
  }
  check(attr(kPciBusId, 3) == -99999, "a device that is not there is an error", 3);
  std::printf(failures ? "FAILED\n" : "ok\n");
  return failures ? 1 : 0;
}
