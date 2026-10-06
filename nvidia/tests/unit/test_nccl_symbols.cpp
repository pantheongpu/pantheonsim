// The NCCL 2.29+ entry points the simulator does not implement are still
// exported: PyTorch for CUDA 13 (2.14, built against NCCL 2.30) binds
// libtorch_cuda to every one of them at load time, so a library without them
// fails `import torch` with "undefined symbol: ncclCommResume". Each one must
// exist under its real name, and must say it is unsupported when called,
// never succeed quietly.
#include <dlfcn.h>

#include <cstdint>

#include "nccl.h"
#include "vtest.hpp"

// nccl_device/core.h is not vendored: the device API's host entry points, as it declares them.
extern "C" {
ncclResult_t ncclDevCommCreate(ncclComm_t comm, const void* reqs, void* outDevComm);
ncclResult_t ncclDevCommDestroy(ncclComm_t comm, const void* devComm);
ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr);
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr);
}

namespace {
const char* const kNames[] = {
    "ncclCommRevoke",   "ncclCommGetUniqueId", "ncclCommGrow",
    "ncclCommSuspend",  "ncclCommResume",      "ncclCommMemStats",
    "ncclPutSignal",    "ncclSignal",          "ncclWaitSignal",
    "ncclDevCommCreate", "ncclDevCommDestroy", "ncclGetLsaMultimemDevicePointer",
    "ncclGetPeerDevicePointer"};
}  // namespace

VTEST(every_symbol_libtorch_cuda_binds_is_exported) {
  void* self = dlopen(nullptr, RTLD_NOW);
  VCHECK(self != nullptr);
  for (const char* n : kNames) {
    void* sym = dlsym(self, n);
    if (!sym) std::fprintf(stderr, "missing export: %s\n", n);
    VCHECK(sym != nullptr);
  }
}

VTEST(each_one_answers_not_supported_when_called) {
  ncclComm_t comm = nullptr;
  ncclUniqueId id{};
  ncclComm_t out = nullptr;
  uint64_t value = 0;
  void* ptr = nullptr;
  VCHECK_EQ(ncclCommRevoke(comm, 0), ncclInvalidUsage);
  VCHECK_EQ(ncclCommGetUniqueId(comm, &id), ncclInvalidUsage);
  VCHECK_EQ(ncclCommGrow(comm, 2, &id, 0, &out, nullptr), ncclInvalidUsage);
  VCHECK_EQ(ncclCommSuspend(comm, 0), ncclInvalidUsage);
  VCHECK_EQ(ncclCommResume(comm), ncclInvalidUsage);
  VCHECK_EQ(ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(0), &value), ncclInvalidUsage);
  VCHECK_EQ(ncclSignal(0, 0, 0, 0, comm, nullptr), ncclInvalidUsage);
  VCHECK_EQ(ncclWaitSignal(0, nullptr, comm, nullptr), ncclInvalidUsage);
  VCHECK_EQ(ncclPutSignal(nullptr, 0, ncclFloat32, 0, nullptr, 0, 0, 0, 0, comm, nullptr), ncclInvalidUsage);
  VCHECK_EQ(ncclDevCommCreate(comm, nullptr, nullptr), ncclInvalidUsage);
  VCHECK_EQ(ncclDevCommDestroy(comm, nullptr), ncclInvalidUsage);
  VCHECK_EQ(ncclGetLsaMultimemDevicePointer(nullptr, 0, &ptr), ncclInvalidUsage);
  VCHECK_EQ(ncclGetPeerDevicePointer(nullptr, 0, 0, &ptr), ncclInvalidUsage);
}

VTEST_MAIN
