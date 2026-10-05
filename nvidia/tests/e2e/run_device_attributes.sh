#!/usr/bin/env bash
# What a program is told about its device (nvidia/tests/e2e/device_attributes.cu):
# the runtime, the driver and cudaDeviceProp agree, a device has one identity
# for CUDA and NVML, and stream priorities are a range, kept and clamped. Run on
# every NVIDIA profile, with two devices (so identities can differ), and then
# on the simulated RTX 3060 against what the card says.
#
# The card's answers are in nvidia/tests/data/cuda_attributes_rtx3060.card.txt:
# every attribute, property member and the stream priority range of an RTX 3060
# (driver 13.0), as `device_attributes --dump` prints them. Each simulated
# value must equal the card's but for the few below that are not a fact of the
# card, or that name a capability the simulator does not have.
# Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/device_attributes.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_device_attributes_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++14 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda
trap 'rm -f "$out"' EXIT
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
# Both libraries carry the simulator's core, which a sanitizer build reports as
# an ODR violation when one program loads the two.
if [[ -n "$(shim_sanitizer "$shim")" ]]; then echo "SKIP: a sanitizer build loads two copies of the core"; exit 0; fi

fail=0
for profile in "$root"/nvidia/profiles/*.yaml; do
  gpu="nvidia/$(basename "$profile" .yaml)"
  result="$(VGPU_QUIET=1 VGPU_GPU="$gpu" VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
  if [[ "$result" == "PASS" ]]; then echo "ok    $gpu"; else echo "FAIL  $gpu: $result"; fail=1; fi
done

# The simulated RTX 3060 against the card. A line the simulator prints that
# the card's file does not have (an attribute newer than its toolkit) is not compared.
python3 - "$root/nvidia/tests/data/cuda_attributes_rtx3060.card.txt" \
  <(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" --dump 2>&1) <<'PY' || fail=1
import re, sys

# Not facts of the card, or capabilities the simulator does not implement:
# the card answers 1 where this answers 0 so that a program that sees the
# capability does not take a path that needs it.
different = {
  # where the card sits and what its host does
  "rt PciBusId", "drv PCI_BUS_ID", "prop pciBusID", "prop uuid",
  "rt GpuPciSubsystemId", "drv GPU_PCI_SUBSYSTEM_ID", "prop gpuPciSubsystemID",
  "rt KernelExecTimeout", "drv KERNEL_EXEC_TIMEOUT",                   # the card's display driver (WSL) has a watchdog
  "rt ConcurrentManagedAccess", "drv CONCURRENT_MANAGED_ACCESS", "prop concurrentManagedAccess",   # WSL answers 0, Linux 1
  # not implemented: sparse and deferred-mapped arrays, compressible memory,
  # read-only host registration, host memory pools and VMM, the 64-bit and NOR stream memory operations, POSIX FD handles
  "rt SparseCudaArraySupported", "drv SPARSE_CUDA_ARRAY_SUPPORTED", "prop sparseCudaArraySupported",
  "rt DeferredMappingCudaArraySupported", "drv DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED", "prop deferredMappingCudaArraySupported",
  "drv GENERIC_COMPRESSION_SUPPORTED",
  "rt HostRegisterReadOnlySupported", "drv READ_ONLY_HOST_REGISTER_SUPPORTED", "prop hostRegisterReadOnlySupported",
  "rt HostMemoryPoolsSupported", "drv HOST_MEMORY_POOLS_SUPPORTED",
  "rt HostNumaMemoryPoolsSupported", "drv HOST_NUMA_MEMORY_POOLS_SUPPORTED",
  "drv HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", "drv HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED",
  "rt Reserved122", "rt Reserved123", "rt Reserved141", "rt Reserved145",   # the runtime's names for those four
  "drv CAN_USE_64_BIT_STREAM_MEM_OPS", "drv CAN_USE_STREAM_WAIT_VALUE_NOR",
  "drv HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED",
}

def table(lines):
    d = {}
    for line in lines:
        m = re.match(r"^(rt|drv|prop|meminfo|stream_priority)\s+(\S+)\s*(.*)$", line.strip())
        if m: d[(m.group(1), m.group(2))] = m.group(3)
    return d

card = table(open(sys.argv[1]))
sim = table(open(sys.argv[2]))
bad = []
compared = 0
for key, want in sorted(card.items()):
    if key not in sim or f"{key[0]} {key[1]}" in different: continue
    compared += 1
    if sim[key] != want: bad.append((key, want, sim[key]))
for (kind, name), want, got in bad:
    print(f"FAIL  {kind} {name}: card {want}, simulator {got}")
print(("ok    " if not bad else "FAIL  ") + f"{compared - len(bad)} of {compared} values the card gave match the simulated RTX 3060")
sys.exit(1 if bad else 0)
PY
exit $fail
