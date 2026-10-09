// Which of the machine's devices a CUDA program is shown: CUDA_VISIBLE_DEVICES
// and CUDA_DEVICE_ORDER, read as NVIDIA's driver reads them. The rules below
// were measured on two RTX 3060s (driver 13.0), not guessed:
//
//   unset                    every device, in PCI order
//   "1,0"                    those, in that order; indices are decimal, read as
//                            far as a number goes ("01" is 1, "0x1" is 0, "1.0"
//                            is 1), spaces around an element are ignored
//   "GPU-31f5e719"           a device by its UUID, or any prefix of it that
//                            names exactly one ("GPU-" alone names all, so
//                            names none)
//   an element that names no device (past the last, negative, not a number, a
//   UUID that matches none or several) ends the list there: "0,5" shows
//   device 0, "0,abc,1" shows device 0, and "5" or "abc" or "" shows none
//   a list is all indices or all UUIDs: the first element says which, and one
//   of the other kind ends the list ("GPU-<uuid of 1>,0" shows device 1 only)
//   a device named twice: an error (cudaErrorInvalidDevice), even if one UUID
//   is a prefix of the other
//   CUDA_DEVICE_ORDER        FASTEST_FIRST (the default) or PCI_BUS_ID; any
//   other value is cudaErrorInvalidDevice. The simulated devices of a machine
//   are identical, so the two orders agree.
//
// Every CUDA call of a program with no device shown answers
// cudaErrorNoDevice (100), and cuInit does; with an invalid list, 101.
#pragma once

#include <vector>

#include "vgpu/profile.hpp"

namespace vgpu::runtime {

struct VisibleDevices {
  // The machine's devices a program sees, in the order it sees them.
  std::vector<int> physical;
  // 0, or the CUDA error number every call answers (100: no device,
  // 101: invalid device).
  int error = 0;
};

// `machine` is how many devices the machine has; `visible` and `order` are
// the variables' values, null when unset.
VisibleDevices cuda_visible_devices(const DeviceProfile& profile, int machine, const char* visible,
                                    const char* order);

}  // namespace vgpu::runtime
