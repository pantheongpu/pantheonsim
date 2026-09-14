#!/usr/bin/env bash
# Two memory totals, and the rule that relates them.
#
# A CUDA program sees totalGlobalMem; nvidia-smi and NVML report the whole
# framebuffer, which on a real datacenter card is a few hundred MiB more -- the
# driver's reserve. The simulator used to report totalGlobalMem to both, so its
# nvidia-smi was 430-770 MiB short on every verified card; the profiles now
# carry the framebuffer measured on real hardware. Two failures this pins:
#   * the two totals must differ by the card's reserve, and stay consistent
#     when VGPU_VRAM_MB resizes the card -- the idle nvidia-smi path used to
#     ignore the override and show 80 GB to a session running on 4;
#   * a VGPU_VRAM_MB that is not a number is ignored, not read as zero, which
#     used to leave a card with no memory at all.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"

# Loading an instrumented shim into the system Python aborts ("ASan runtime
# does not come first"), so under a sanitizer build this defers to the C++ unit
# tests, which check the same logic with the instrumentation on.
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: shim is built with $san, and the Python interpreter is not instrumented"; exit 0; fi
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
rt=$(ls "$build"/shim/libcudart.so.* 2>/dev/null | head -1)
[[ -n "$rt" ]] || { echo "SKIP: no runtime shim"; exit 0; }

totals() {  # env options then assignments ("-u X" first) -> "<cuda MiB> <nvidia-smi MiB>"
  local cuda smi
  cuda=$(env "$@" VGPU_QUIET=1 python3 -c "
import ctypes; rt=ctypes.CDLL('$rt'); f=ctypes.c_size_t(); t=ctypes.c_size_t()
rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)); print(t.value // 2**20)")
  smi=$(env "$@" VGPU_TELEMETRY_PATH=/nonexistent-so-idle "$build/vgpu" smi \
        --query-gpu=memory.total --format=csv,noheader,nounits)
  echo "$cuda $smi"
}
fail=0
expect() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }

read -r c s <<< "$(totals -u VGPU_VRAM_MB VGPU_GPU=nvidia/h100)"
expect "H100: nvidia-smi shows the real framebuffer (81559), CUDA the real totalGlobalMem (81089)" \
       '[[ $s -eq 81559 && $c -eq 81089 ]]'
read -r c4 s4 <<< "$(totals VGPU_GPU=nvidia/h100 VGPU_VRAM_MB=4096)"
expect "VGPU_VRAM_MB=4096: CUDA sees 4096 and nvidia-smi agrees, reserve kept (got $c4 / $s4)" \
       '[[ $c4 -eq 4096 && $((s4 - c4)) -ge 469 && $((s4 - c4)) -le 470 ]]'
read -r cj sj <<< "$(totals VGPU_GPU=nvidia/h100 VGPU_VRAM_MB=junk)"
expect "VGPU_VRAM_MB=junk is ignored rather than read as 0 (got $cj / $sj)" \
       '[[ $cj -eq 81089 && $sj -eq 81559 ]]'
read -r cr sr <<< "$(totals -u VGPU_VRAM_MB VGPU_GPU=nvidia/rtx3060)"
expect "RTX 3060: consumer card, no meaningful reserve (got $cr / $sr, real nvidia-smi 12288)" \
       '[[ $sr -eq 12288 && $((sr - cr)) -le 1 ]]'
exit $fail
