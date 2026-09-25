#!/usr/bin/env bash
# The decoder against ROCm's own assembler, on code ROCm's compiler built.
#
# The code object amd/tests/hipcc/ops.gfx942.o was built by hipcc, and its
# listing by the same toolchain's llvm-objdump; test_amd_hipcc_ops_decode
# checks the decoder against them everywhere. This checks that the files are
# in the repository, and, wherever ROCm is installed, rebuilds the object
# from source and checks the decoder against that too -- a newer ROCm emits
# instructions an older one did not, and this is where that shows first.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
test_bin="$build/test_amd_gcn"
[[ -x "$test_bin" ]] || { echo "SKIP: no $test_bin"; exit 0; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
  untracked=""
  for f in amd/tests/hipcc/ops.hip amd/tests/hipcc/ops.gfx942.o amd/tests/hipcc/ops.gfx942.dis \
           amd/tests/hipcc/wmma.cpp amd/tests/hipcc/wmma.gfx942.o amd/tests/hipcc/wmma.gfx942.dis \
           amd/tests/hipcc/chevron.cpp amd/tests/hipcc/chevron.gfx942 \
           amd/tests/hipcc/chevron.O0.gfx942 amd/tests/hipcc/chevron.O0.gfx942.o amd/tests/hipcc/chevron.O0.gfx942.dis \
           amd/tests/hipcc/printf.cpp amd/tests/hipcc/printf.gfx942 amd/tests/hipcc/printf.gfx942.o \
           amd/tests/hipcc/printf.gfx942.dis amd/tests/hipcc/cooperative.cpp amd/tests/hipcc/cooperative.gfx942 \
           amd/tests/hipcc/cooperative.gfx942.o amd/tests/hipcc/cooperative.gfx942.dis \
           amd/tests/hipcc/runtime.cpp amd/tests/hipcc/runtime.gfx942; do
    git -C "$root" ls-files --error-unmatch "$f" >/dev/null 2>&1 || untracked="$untracked $f"
  done
  expect "every hipcc fixture is in the repository" "" "$untracked"
fi

rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/bin/hipcc" && -x "$c/lib/llvm/bin/llvm-objdump" ]] && { rocm=$c; break; }
done
if [[ -z "$rocm" ]]; then
  echo "note  no ROCm found (set VGPU_ROCM_PATH): the checked-in object was checked, a fresh build was not"
  exit $fail
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
objdump="$rocm/lib/llvm/bin/llvm-objdump"
listing() { "$objdump" -d --mcpu=gfx942 "$1" | sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g'; }

expect "the checked-in object and listing are of the same build" "" \
  "$(diff <(listing "$root/amd/tests/hipcc/ops.gfx942.o") "$root/amd/tests/hipcc/ops.gfx942.dis" | head -5)"

export ROCM_PATH=$rocm HIP_PATH=$rocm HIP_CLANG_PATH=$rocm/lib/llvm/bin HIP_DEVICE_LIB_PATH=$rocm/amdgcn/bitcode
if "$rocm/bin/hipcc" -O3 -std=c++17 --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
     -c "$root/amd/tests/hipcc/ops.hip" -o "$tmp/fresh.o" 2>"$tmp/err"; then
  listing "$tmp/fresh.o" > "$tmp/fresh.dis"
  out=$(VGPU_GCN_OBJECT="$tmp/fresh.o" VGPU_GCN_LISTING="$tmp/fresh.dis" "$test_bin" 2>&1)
  expect "every instruction $rocm's hipcc emits decodes as its llvm-objdump prints it" "yes" \
    "$(grep -q '^\[ PASS \] every_instruction_decodes_as_the_assembler_wrote_it' <<< "$out" && echo yes ||
       grep -m1 -v '^\[' <<< "$out")"
else
  expect "hipcc builds the fixture" "yes" "no: $(head -2 "$tmp/err")"
fi
exit $fail
