#!/usr/bin/env bash
# AMD's Python package for AMD SMI (amdsmi), unmodified, on VirtualGPU's
# libamd_smi -- what vLLM asks it before anything else: whether this is a
# ROCm machine (any GPUs), each GPU's target and name, its memory and UUID,
# and whether the GPUs are fully connected by XGMI. The package is the one
# vLLM's ROCm wheels install (amdsmi 26.2.2); it binds every function of the
# library when imported, so importing it at all checks the library exports
# them.
#
# The package is taken from $VGPU_AMDSMI_WHEEL, or downloaded once into the
# build directory; without either (no network) the test skips.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
shim="$build/shim"
[[ -e "$shim/libamd_smi.so" ]] || { echo "SKIP: no libamd_smi in $shim"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
# ctypes loads the library into python, which a sanitizer runtime refuses.
. "$root/tests/shim_guard.sh"
san="$(shim_sanitizer "$shim")"
[[ -z "$san" ]] || { echo "SKIP: $san library cannot load into python"; exit 0; }

url="https://wheels.vllm.ai/rocm/ced6857afa0ea7b2e3f0846a62e1394e90f15607/amdsmi-26.2.2%2Bc2d9476115-py3-none-any.whl"
sha=c711cdf0a23684735b258a9410b177e68b1d249e31aa558813efa3ebce5ebb63
pkg="$build/amdsmi-26.2.2"
if [[ ! -d "$pkg/amdsmi" ]]; then
  wheel="${VGPU_AMDSMI_WHEEL:-$build/amdsmi-26.2.2.whl}"
  [[ -f "$wheel" ]] || curl -sfL --max-time 120 -o "$wheel" "$url" || { rm -f "$wheel"; echo "SKIP: cannot download amdsmi"; exit 0; }
  echo "$sha  $wheel" | sha256sum -c --quiet - || { rm -f "$wheel"; echo "FAIL  the amdsmi wheel is not the one expected"; exit 1; }
  rm -rf "$pkg.tmp" && mkdir -p "$pkg.tmp" && python3 -m zipfile -e "$wheel" "$pkg.tmp" && mv "$pkg.tmp" "$pkg"
fi
# The package looks in $ROCM_HOME/$ROCM_PATH before the library path, and
# falls back to the copy of AMD's library it carries; the simulator's must be
# the one found.
rm -f "$pkg/amdsmi/libamd_smi.so"

work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_amdsmi.XXXXXX")"
trap 'rm -rf "$work"' EXIT
fail=0
for g in amd/mi300x amd/rx7900xtx; do
  mkdir -p "$work/$g"
  out=$(env -u ROCM_PATH -u ROCM_HOME VGPU_GPU=$g VGPU_DEVICE_COUNT=2 VGPU_QUIET=1 VGPU_TELEMETRY_PATH="$work/$g" \
        LD_LIBRARY_PATH="$shim" PYTHONPATH="$pkg" python3 - 2>&1 <<'EOF'
import amdsmi
from amdsmi import amdsmi_wrapper
import os
print("library", next(os.path.realpath(l.split()[-1]) for l in open("/proc/self/maps") if "libamd_smi" in l))
amdsmi.amdsmi_init()
h = amdsmi.amdsmi_get_processor_handles()
print("gpus", len(h))
a = amdsmi.amdsmi_get_gpu_asic_info(h[0])
print("target", a["target_graphics_version"], a["device_id"], a["market_name"], a["num_compute_units"])
print("memory", amdsmi.amdsmi_get_gpu_memory_total(h[0], amdsmi.AmdSmiMemoryType.VRAM) > 0)
print("uuid", len(amdsmi.amdsmi_get_gpu_device_uuid(h[0])))
print("numa", amdsmi.amdsmi_topo_get_numa_node_number(h[0]))
# vLLM's is_fully_connected: one XGMI hop (type 2) between every pair.
l = amdsmi.amdsmi_topo_get_link_type(h[0], h[1])
print("link", l["hops"], l["type"])
print("bdf", amdsmi.amdsmi_get_gpu_device_bdf(h[1]))
try:
    amdsmi.amdsmi_get_gpu_vbios_info(h[0])
    print("vbios answered")
except amdsmi.AmdSmiLibraryException as e:
    print("vbios", e.get_error_code())
amdsmi.amdsmi_shut_down()
EOF
)
  check() {
    if grep -qx -- "$2" <<<"$out"; then echo "ok    $1 ($g)"; else
      echo "FAIL  $1 ($g): wanted '$2'"; echo "$out" | sed 's/^/      /'; fail=1; fi
  }
  if [[ $g == amd/mi300x ]]; then
    check "amdsmi finds the GPUs: vLLM's test for a ROCm machine" "gpus 2"
    check "each GPU's target, device ID, name and compute units" "target gfx942 0x74a1 AMD Instinct MI300X 304"
    check "Instinct GPUs are one XGMI hop apart: vLLM's fully-connected test" "link 1 2"
  else
    check "each GPU's target, device ID, name and compute units" "target gfx1100 0x744c AMD Radeon RX 7900 XTX 96"
    check "Radeon GPUs meet over PCI Express" "link 2 1"
  fi
  grep -qx "library $(realpath "$shim/libamd_smi.so")" <<<"$out" && echo "ok    the simulator's library is the one loaded ($g)" || {
    echo "FAIL  another libamd_smi was loaded ($g)"; echo "$out" | head -3 | sed 's/^/      /'; fail=1; }
  check "its memory" "memory True"
  check "its UUID" "uuid 36"
  check "its NUMA node" "numa 0"
  check "its PCI address" "bdf 0000:02:00.0"
  check "what is not modelled is refused as a card refuses it" "vbios 2"
done
exit $fail
