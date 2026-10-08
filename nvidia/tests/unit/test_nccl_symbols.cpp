// The NCCL 2.28+ entry points are exported under their real names: PyTorch for
// CUDA 13 (2.14, built against NCCL 2.30) binds libtorch_cuda to every one of
// them at load time, so a library without them fails `import torch` with
// "undefined symbol: ncclCommResume". They are implemented now (see the
// nccl_comm_ops e2e program for their behaviour on live communicators); this
// unit test pins the names and the answer to a NULL communicator or window,
// which needs no transport.
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

VTEST(a_null_communicator_or_window_is_an_invalid_argument) {
  ncclComm_t comm = nullptr;
  ncclUniqueId id{};
  uint64_t value = 0;
  void* ptr = nullptr;
  VCHECK_EQ(ncclCommGetUniqueId(comm, &id), ncclInvalidArgument);
  VCHECK_EQ(ncclCommSuspend(comm, 0), ncclInvalidArgument);
  VCHECK_EQ(ncclCommResume(comm), ncclInvalidArgument);
  VCHECK_EQ(ncclCommMemStats(comm, static_cast<ncclCommMemStat_t>(0), &value), ncclInvalidArgument);
  VCHECK_EQ(ncclSignal(0, 0, 0, 0, comm, nullptr), ncclInvalidArgument);
  VCHECK_EQ(ncclWaitSignal(0, nullptr, comm, nullptr), ncclInvalidArgument);
  VCHECK_EQ(ncclPutSignal(nullptr, 0, ncclFloat32, 0, nullptr, 0, 0, 0, 0, comm, nullptr), ncclInvalidArgument);
  VCHECK_EQ(ncclDevCommCreate(comm, nullptr, nullptr), ncclInvalidArgument);
  VCHECK_EQ(ncclDevCommDestroy(comm, nullptr), ncclInvalidArgument);
  VCHECK_EQ(ncclGetLsaMultimemDevicePointer(nullptr, 0, &ptr), ncclInvalidArgument);
  VCHECK_EQ(ncclGetPeerDevicePointer(nullptr, 0, 0, &ptr), ncclInvalidArgument);
}

VTEST_MAIN
