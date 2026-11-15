#!/usr/bin/env bash
# Builds the programs gen.py wrote, four ways each: -O0 and -O3, with and without fast math (the paper's optimization axis).
#
#   tools/numerics/varity/build.sh nvidia <gen-dir> <bin-dir> [sm]     nvcc, -arch=sm_<sm> (default 75)
#   tools/numerics/varity/build.sh amd    <gen-dir> <bin-dir> [gfx]    hipcc, --offload-arch=<gfx> (default: hipcc's own choice)
#
# <bin-dir>/<program>.<O0|O3|O0_fm|O3_fm>; a program that does not build is listed in <bin-dir>/build-failed.txt. JOBS=4 compiles at once.
#
# hipcc -ffast-math assumes no NaN and no infinity, and Varity's programs divide zero by zero: clang then compiles some kernels to
# nothing at all (a function of size 0, no s_endpgm), and a card that is sent one runs off the end of its code. Those are not kept:
# they are listed in <bin-dir>/empty-kernel.txt and run.sh prints "<program>.all EMPTY-KERNEL" for them. The check needs
# llvm-objcopy, clang-offload-bundler and llvm-objdump (ROCm's lib/llvm/bin); without them no fast-math program is kept for AMD.
set -uo pipefail
kind="${1:?usage: $0 nvidia|amd <gen-dir> <bin-dir> [arch]}"; gen="$(cd "${2:?}" && pwd)"; mkdir -p "${3:?}"; bin="$(cd "$3" && pwd)"; arch="${4:-}"
inc="$(cd "$(dirname "$0")/.." && pwd)"
case "$kind" in
  nvidia) cc=(nvcc -std=c++17 -cudart shared -arch="sm_${arch:-75}" -Wno-deprecated-gpu-targets); fm=-use_fast_math ;;
  amd)    cc=(hipcc -std=c++17 -w); [[ -n "$arch" ]] && cc+=(--offload-arch="$arch"); fm=-ffast-math ;;
  *) echo "nvidia or amd"; exit 2 ;;
esac
: > "$bin/build-failed.txt"
one() {   # <program> <variant>
  local p="$1" v="$2" flags=()
  case "$v" in O0) flags=(-O0) ;; O3) flags=(-O3) ;; O0_fm) flags=(-O0 "$fm") ;; O3_fm) flags=(-O3 "$fm") ;; esac
  "${cc[@]}" "${flags[@]}" -I "$inc" "$gen/$p.cu" -o "$bin/$p.$v" > "$bin/$p.$v.err" 2>&1 && rm -f "$bin/$p.$v.err" || { echo "$p.$v" >> "$bin/build-failed.txt"; return; }
  if [[ $kind == amd && $v == *_fm ]] && ! has_kernel_code "$bin/$p.$v"; then rm -f "$bin/$p.$v"; echo "$p.$v" >> "$bin/empty-kernel.txt"; fi
}
# Whether an executable hipcc built carries a kernel that ends (an s_endpgm in one of its device code objects).
has_kernel_code() {
  local t; t="$(mktemp -d "${TMPDIR:-/tmp}/varity_chk.XXXXXX")" || return 1
  local ok=1 target
  if "$llvm/llvm-objcopy" --dump-section .hip_fatbin="$t/fat" "$1" 2> /dev/null; then
    for target in $("$llvm/clang-offload-bundler" --type=o --input="$t/fat" --list 2> /dev/null | grep amdgcn); do
      "$llvm/clang-offload-bundler" --type=o --targets="$target" --input="$t/fat" --output="$t/co" --unbundle 2> /dev/null || continue
      "$llvm/llvm-objdump" -d "$t/co" 2> /dev/null | grep -q s_endpgm && ok=0
    done
  fi
  rm -rf "$t"
  return $ok
}
llvm=""
if [[ $kind == amd ]]; then
  llvm="$(dirname "$(readlink -f "$(command -v hipcc)")")/../lib/llvm/bin"
  [[ -x "$llvm/llvm-objdump" ]] || llvm="$(dirname "$(command -v llvm-objdump 2> /dev/null || echo /nonexistent/x)")"
fi
: > "$bin/empty-kernel.txt"
export -f one has_kernel_code; export gen bin inc fm kind llvm; export CC_CMD="$(declare -p cc)"
for f in "$gen"/*.cu; do p="$(basename "$f" .cu)"; for v in O0 O3 O0_fm O3_fm; do echo "$p $v"; done; done |
  xargs -P "${JOBS:-4}" -n 2 bash -c 'eval "$CC_CMD"; one "$0" "$1"'
echo "built $(ls "$bin" | grep -c -E '\.(O0|O3|O0_fm|O3_fm)$') programs, $(wc -l < "$bin/build-failed.txt") failed, $(wc -l < "$bin/empty-kernel.txt") with an empty kernel left out"
