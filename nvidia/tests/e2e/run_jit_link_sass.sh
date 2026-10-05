#!/usr/bin/env bash
# nvJitLink linking machine code (nvjitlink_sass.cpp): sass_link_main.cu and
# sass_link_lib.cu compiled by nvcc -rdc for SASS only -- cubins, a host
# object and a static library -- linked at run time and run on the
# simulator, for every generation this nvcc compiles that a profile stands
# for. On the RTX 3060 the same program passes with NVIDIA's nvJitLink, and
# with this one, whose cubin NVIDIA's driver then runs.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
e2e="$root/nvidia/tests/e2e"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
for lib in nvJitLink cuda; do
  shopt -s nullglob
  have=("$shim"/lib"$lib".so.[0-9]*)
  shopt -u nullglob
  if (( ${#have[@]} == 0 )); then
    echo "SKIP: the $lib shim is not built"; exit 0
  fi
done
nvcc_host_compiler_fix
tmp="${TMPDIR:-/tmp}/vgpu_jit_link_sass_$$"
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT

# As run_jit_link.sh: NVIDIA's nvJitLink header where the toolkit has a
# library of this shim's major, VirtualGPU's declarations otherwise.
name=nvjitlink_sass
build() {
  nvcc -std=c++17 -cudart none -Wno-deprecated-gpu-targets -Xcompiler -Wno-deprecated-declarations \
       $(shim_sanitizer_nvcc_flags "$shim") "$e2e/$name.cpp" -o "$tmp/$name" "$@" -lcuda
}
shim_has_its_nvjitlink() {
  local need
  need="$(objdump -p "$tmp/$name" 2>/dev/null | awk '/NEEDED/ && /libnvJitLink/ {print $2}')"
  [[ -n "$need" && -e "$shim/$need" ]]
}
if ! build -lnvJitLink 2>"$tmp/build.log" || ! shim_has_its_nvjitlink; then
  echo "note: this toolkit's nvJitLink is missing or another major; building against VirtualGPU's declarations"
  build -DVGPU_OWN_NVJITLINK_H "$shim/libnvJitLink.so" || { echo "FAIL: does not compile"; exit 1; }
fi
require_shim_libs "$shim" "$tmp/$name" || exit 0

fails=0
ran=0
# arch:profile:another architecture whose SASS cannot run on it
for entry in 75:t4:90 80:a100:90 86:rtx3060:90 89:l4:90 90:h100:100 100:b200:120 120:rtx5090:100; do
  IFS=: read -r sm profile other <<< "$entry"
  dir="$tmp/sm_$sm"
  mkdir -p "$dir"
  flags=(-rdc=true -Wno-deprecated-gpu-targets -gencode "arch=compute_$sm,code=sm_$sm")
  if ! nvcc -cubin "${flags[@]}" "$e2e/sass_link_main.cu" -o "$dir/main.cubin" 2>"$dir/log"; then
    if grep -qiE "unsupported gpu architecture|not defined for option|is not supported" "$dir/log"; then
      echo "note: this nvcc cannot target sm_$sm"; continue
    fi
    cat "$dir/log"; echo "FAIL: sm_$sm: does not compile"; fails=1; continue
  fi
  nvcc -cubin "${flags[@]}" "$e2e/sass_link_lib.cu" -o "$dir/lib.cubin" &&
    nvcc -dc "${flags[@]}" "$e2e/sass_link_lib.cu" -o "$dir/lib.o" &&
    ar rcs "$dir/lib.a" "$dir/lib.o" || { echo "FAIL: sm_$sm: the library does not compile"; fails=1; continue; }
  args=(sm_"$sm" "$dir/main.cubin" "$dir/lib.cubin")
  if nvcc -cubin -rdc=true -Wno-deprecated-gpu-targets -gencode "arch=compute_$other,code=sm_$other" \
          "$e2e/sass_link_lib.cu" -o "$dir/other.cubin" 2>/dev/null; then
    args+=("$dir/other.cubin" "$dir/lib.o" "$dir/lib.a")
  fi
  ran=$((ran + 1))
  log="$(VGPU_QUIET=1 VGPU_GPU="nvidia/$profile" LD_LIBRARY_PATH="$shim" "$tmp/$name" "${args[@]}" 2>&1)"
  rc=$?
  echo "$log" | sed "s/^/    sm_$sm: /"
  if [[ $rc != 0 || "$(tail -n 1 <<< "$log")" != PASS ]] || grep -qE '^FAIL|VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$log"; then
    echo "FAIL: sm_$sm on $profile"; fails=1
  fi
done
(( ran > 0 )) || { echo "SKIP: this nvcc targets none of the architectures"; exit 0; }
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: nvJitLink SASS links"
exit $fails
