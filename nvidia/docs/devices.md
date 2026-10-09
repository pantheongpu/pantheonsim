# What a program is told about its device

CUDA answers questions about a device three ways -- `cudaDeviceGetAttribute`,
`cuDeviceGetAttribute` and `cudaDeviceProp` -- and gives it an identity four
(UUID, PCI address, bus-id string, enumeration order). They come from one place
here (`nvidia/src/device_attributes.cpp`, `vgpu/cuda_attributes.hpp`), so they
cannot disagree, and from the same place NVML's identity does
(`telemetry::describe_device`), so a tool that correlates the two finds one
device.

## Where a value comes from

| Kind | Examples | From |
| --- | --- | --- |
| A fact of the card | L2 size, memory bus width, boost clock, copy engines, persisting-L2 maximum | the profile's `limits:` and `cuda:` sections; each profile's comment says whether a value was read from a card (the RTX 3060's all were) or is its datasheet's |
| A rule of the compute capability | texture and surface limits, the access policy window, clusters and the tensor map from 9.0, the FP64 ratio | the CUDA C Programming Guide's technical-specification tables, and the 3060 where it says (they agree) |
| A capability | sparse arrays, compressible memory, host memory pools, 64-bit stream memory operations | **yes only if the simulator implements it.** The card says 1 to several the simulator does not; a program that sees the capability takes the path that uses it, so they read 0 |
| Where the card sits | PCI bus, UUID, subsystem id | the simulated machine (device *n* is bus *n+1*), identical in NVML |

A profile that does not say a value (`memory_bus_width_bits`, `l2_cache_bytes`)
reports zero for it rather than a number made up for it: B200 and B300 do not
publish an L2 size CUDA would report, so theirs is unset.

`device_attributes --dump` prints every attribute, `cudaDeviceProp` member and
the stream priority range; `nvidia/tests/data/cuda_attributes_rtx3060.card.txt`
is an RTX 3060's output (driver 13.0). `e2e_device_attributes` holds the
simulated 3060 to it except for the lines it names: where the card sits, what
its WSL display driver does (a watchdog, no concurrent managed access), and the
capabilities above. It also runs on every profile, checking the three views
against each other, and passes against the real driver.

## Stream priorities

The range is `[0, -5]` (`cudaDeviceGetStreamPriorityRange` on an RTX 3060);
both APIs clamp to it and read a stream's priority back. Streams run in order,
so a priority changes nothing else.

## CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER

`VGPU_DEVICE_COUNT` says how many devices the machine has; `CUDA_VISIBLE_DEVICES`
picks which a program sees, and in what order. NVML, `nvidia-smi` and the
machine's telemetry still list them all, as on a real host. The rules are the
driver's, measured on two RTX 3060s (`e2e_cuda_visible_devices` runs the same
table against the real driver with `--card`):

* unset: every device. A list is decimal indices read as far as a number goes
  (`01` is 1, `0x1` is 0, `1.0` is 1), spaces ignored, or UUIDs (`GPU-` and any
  prefix that names exactly one device), not both: an element of the other kind
  ends the list.
* An element that names no device (past the last, negative, not a number, an
  unknown or ambiguous UUID) ends the list there: `0,5` shows device 0. A list
  that shows nothing -- `5`, `abc`, an empty value -- is `cudaErrorNoDevice` (100)
  from every runtime call and from `cuInit`, after which `cuDeviceGetCount` says
  `CUDA_ERROR_NOT_INITIALIZED`.
* A device named twice is `cudaErrorInvalidDevice` (101), as is any
  `CUDA_DEVICE_ORDER` but `FASTEST_FIRST` and `PCI_BUS_ID` (the simulated
  devices of a machine are identical, so the two agree).

A device keeps its identity whatever it is called: with `CUDA_VISIBLE_DEVICES=1`,
device 0 is the machine's second device, with its UUID and PCI address.
