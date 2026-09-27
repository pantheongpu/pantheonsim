#!/usr/bin/env bash
# The AMD kernel debugger (vgpu debug), driven by a file of commands over
# hipcc/chevron.gfx942 on a simulated MI300X: a breakpoint stops the first
# wave where it should, and disas, print, step and set say and do what gdb's
# would. Then the program runs to its end with its results right. A second
# session quits at the kernel's first instruction, and the launch fails.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
vgpu="$build/vgpu"
exe="$root/amd/tests/hipcc/chevron.gfx942"
[[ -x "$vgpu" && -e "$build/shim/libamdhip64.so.7" ]] || { echo "SKIP: no vgpu or HIP shim in $build"; exit 0; }
fail=0
expect() {   # expect NAME WANT GOT
  if [[ "$2" == "$3" ]]; then echo "ok    $1"
  else echo "FAIL  $1: wanted [$2], got [$3]"; fail=1; fi
}
has() {      # has NAME LINE
  if grep -qxF -- "$2" <<< "$out"; then echo "ok    $1"
  else echo "FAIL  $1: no line [$2]"; fail=1; fi
}

out=$(LD_LIBRARY_PATH="$build/shim" "$vgpu" debug -x "$root/amd/tests/debugger/session.txt" --gpu amd/mi300x --quiet -- "$exe" 2>&1)
status=$?
echo "$out" | sed 's/^/      /' | head -40
expect "the program runs to its end under the debugger" "0" "$status"
has "the breakpoint is set where asked" "Breakpoint 1 at scale_add+0x20"
has "the first wave stops there" "Breakpoint 1, _Z9scale_addPKfS0_Pffi+0x20, work-group (0,0,0), wave 0, exec 0xffffffffffffffff"
has "disas lists from the stop" "=> 0x20:  v_add_u32_e32 v0, s2, v0"
has "print shows one lane's value" "v0[3] = 3"
has "step runs two instructions and stops" "_Z9scale_addPKfS0_Pffi+0x28, work-group (0,0,0), wave 0, exec 0xffffffffffffffff"
has "set changes a register, and print shows it" "v200[5] = 42"
has "info registers shows EXEC" "exec   0xffffffffffffffff"
expect "the results are right after the session" "chevron launch wrong 0 of 3000" \
  "$(grep -o 'chevron launch wrong [0-9]* of [0-9]*' <<< "$out")"

out=$(LD_LIBRARY_PATH="$build/shim" "$vgpu" debug -x "$root/amd/tests/debugger/quit.txt" --gpu amd/mi300x -- "$exe" 2>&1)
status=$?
expect "quit stops the launch, which fails" "yes" "$([[ $status != 0 ]] && grep -q 'told to quit' <<< "$out" && echo yes || echo no)"
exit $fail
