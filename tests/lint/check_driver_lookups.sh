#!/usr/bin/env bash
# No shim but the driver looks a driver function up with dlsym(RTLD_DEFAULT, "...").
#
# `vgpu run --preload` loads the driver privately (RTLD_LOCAL, #597): its functions are not in the global scope, and such a
# lookup finds nothing there. The runtime asks the private handle first (driver_dlsym in runtime_api.cpp); the other shims ask
# the runtime (vgpu_driver_symbol, nvidia/src/driver_symbol.hpp). CUPTI's current context and nvshmemx_cumodule_init /
# nvshmemx_culibrary_init were found doing it, and so were the runtime's own vgpu_driver_has_managed_v1 probe before an
# asynchronous copy. A driver function is a cu* name or one of the driver shim's vgpu_driver_* hand-over entry points; the
# driver's own sources (driver_*.cpp, driver_*.inc) may look up whatever they like.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
hits=$(grep -rnE 'dlsym\(RTLD_DEFAULT, *"(cu[A-Z][A-Za-z0-9_]*|vgpu_driver_[A-Za-z0-9_]*)"' \
         --include='*.cpp' --include='*.hpp' --include='*.inc' --include='*.h' nvidia/src src include |
       grep -vE '^nvidia/src/driver_[^/]*:|^nvidia/src/driver_symbol\.hpp:' || true)
if [[ -n $hits ]]; then
  echo "FAIL: a driver function looked up in the global scope only (use driver_dlsym in the runtime, vgpu_driver_symbol elsewhere):"
  sed 's/^/    /' <<< "$hits"
  exit 1
fi
echo "PASS: no shim looks the driver up in the global scope only"
