#!/usr/bin/env bash
# The decoder against the assembler, on code built here and now.
#
# The unit tests read a code object checked into amd/tests/data. This builds
# one with the real compiler instead -- a different clang emits different
# code, and a newer one emits instructions the decoder has not seen -- and
# checks that every instruction decodes as llvm-objdump prints it, and that
# the loader reads the kernels the toolchain says are there.
#
# Skips where there is no clang with the amdgcn target, as the CUDA tests skip
# without nvcc.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
test_bin="$build/test_amd_gcn"
[[ -x "$test_bin" ]] || { echo "SKIP: no $test_bin"; exit 0; }
clang=${VGPU_CLANG:-clang}
command -v "$clang" >/dev/null || { echo "SKIP: no clang, so nothing to build a code object with"; exit 0; }
"$clang" --print-targets 2>/dev/null | grep -q amdgcn || { echo "SKIP: this clang has no amdgcn target"; exit 0; }
objdump=${VGPU_LLVM_OBJDUMP:-$(dirname "$(readlink -f "$(command -v "$clang")")")/llvm-objdump}
[[ -x "$objdump" ]] || objdump=$(command -v llvm-objdump)
[[ -n "$objdump" && -x "$objdump" ]] || { echo "SKIP: no llvm-objdump beside $clang"; exit 0; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
fixtures="vector_add ops math memory globals grid bytes int64 atomics mixed half calls builtins"
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

# The kernels with the wider instruction mix, so a clang that emits something
# new for them fails here rather than in a kernel's results.
cp "$root"/amd/tests/data/*.c "$tmp/"
# One object of everything the fixtures use, which is the widest set of
# instructions this has to decode.
( cd "$tmp" && cat ops.c math.c memory.c globals.c grid.c bytes.c int64.c atomics.c mixed.c half.c calls.c builtins.c > both.c &&
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx942 -nogpulib -O2 -c both.c -o fresh.o ) 2>"$tmp/clang.err"
if [[ ! -s "$tmp/fresh.o" ]]; then
  echo "SKIP: this clang could not build for gfx942: $(head -2 "$tmp/clang.err")"; exit 0
fi
"$objdump" -d --mcpu=gfx942 "$tmp/fresh.o" |
  sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > "$tmp/fresh.dis"
expect "the toolchain disassembles the object it built" "yes" \
  "$([[ -s "$tmp/fresh.dis" ]] && echo yes || echo no)"

# The decoder and the loader, against that object rather than the fixture.
out=$(VGPU_GCN_OBJECT="$tmp/fresh.o" VGPU_GCN_LISTING="$tmp/fresh.dis" "$test_bin" 2>&1)
echo "$out" | sed 's/^/      /'
expect "every instruction it emitted decodes as it prints it" "yes" \
  "$(grep -q "^\[ PASS \] every_instruction_decodes_as_the_assembler_wrote_it" <<< "$out" && echo yes || echo no)"
expect "and the rest of the decoder's checks hold on it" "0" \
  "$(grep -c "^\[ FAIL \]" <<< "$out")"

# Every fixture the tests read has to be in the repository. .gitignore covers
# *.o, so a code object is only there because something asked for it
# explicitly: without this check the tests pass here and fail for everyone who
# clones, which is exactly what happened once.
if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
  untracked=""
  for name in $fixtures; do
    for f in "amd/tests/data/$name.gfx942.o" "amd/tests/data/$name.gfx942.dis"; do
      git -C "$root" ls-files --error-unmatch "$f" >/dev/null 2>&1 || untracked="$untracked $f"
    done
  done
  expect "every fixture the tests read is in the repository" "" "$untracked"
fi

# The version that is checked in has to stay the version the tests read: a
# code object whose listing was regenerated without the object, or the other
# way round, would pass here and fail for everyone else.
for name in $fixtures; do
  committed=$("$objdump" -d --mcpu=gfx942 "$root/amd/tests/data/$name.gfx942.o" |
    sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g')
  expect "the checked-in $name object and listing are of the same build" "" \
    "$(diff <(echo "$committed") "$root/amd/tests/data/$name.gfx942.dis" | head -5)"
done
exit $fail
