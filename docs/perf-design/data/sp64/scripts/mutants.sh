#!/bin/bash
# mutants.sh <src> <build> <work> <name> <hpp|cpp> <sed expression>   (or: mutants.sh <src> <build> <work> pristine)
# Round sp64's mutation check, one mutant a call (each under the machine lock). A mutant is a copy of one source file with one change (sed),
# compiled OUTSIDE the tree with the build's own command line (no dependency files written):
#   hpp: a copy of src/sass/jit.hpp first on the include path of the test's own object (the predicate it checks is inline there);
#   cpp: src/sass/jit.cpp compiled from the copy and linked into the test before libvgpu.a, whose jit.cpp.o is then not pulled in.
# The test is test_jit_threads; rc 0 means it passed (the mutant survived).
set -u
src=$1; b=$2; w=$3; name=$4
t=test_jit_threads
mkdir -p "$w"
if [ "$name" = pristine ]; then
  TMPDIR=$w timeout 600 "$b/$t" > "$w/pristine.out" 2>&1; echo "pristine rc=$? | $(tail -1 "$w/pristine.out")"; exit 0
fi
kind=$5; expr=$6; d="$w/$name"
rm -rf "$d"; mkdir -p "$d/sass"
strip_dep() { sed -E 's# -MD##; s# -MT [^ ]+##; s# -MF [^ ]+##'; }
cc=$(ninja -C "$b" -t commands $t | grep -E -- "-c [^ ]*tests/unit/$t.cpp" | tail -1 | strip_dep)
ld=$(ninja -C "$b" -t commands $t | tail -1)
jc=$(ninja -C "$b" -t commands vgpu | grep -E -- "-c [^ ]*src/sass/jit.cpp$" | tail -1 | strip_dep)
if [ "$kind" = hpp ]; then
  sed "$expr" "$src/src/sass/jit.hpp" > "$d/sass/jit.hpp"
  cmp -s "$d/sass/jit.hpp" "$src/src/sass/jit.hpp" && { echo "mutant=$name NOT APPLIED"; exit 0; }
  c=$(echo "$cc" | sed -E "s#^([^ ]+) #\1 -I$d #; s#-o [^ ]+$t.cpp.o#-o $d/t.o#")
  l=$(echo "$ld" | sed "s#CMakeFiles/$t.dir/tests/unit/$t.cpp.o#$d/t.o#; s# -o $t # -o $d/$t #")
else
  sed "$expr" "$src/src/sass/jit.cpp" > "$d/jit.cpp"
  cmp -s "$d/jit.cpp" "$src/src/sass/jit.cpp" && { echo "mutant=$name NOT APPLIED"; exit 0; }
  c=$(echo "$jc" | sed -E "s#^([^ ]+) #\1 -I$src/src/sass #; s#-o [^ ]+jit.cpp.o#-o $d/jit.o#; s#-c [^ ]+src/sass/jit.cpp#-c $d/jit.cpp#")
  l=$(echo "$ld" | sed "s#CMakeFiles/$t.dir/tests/unit/$t.cpp.o#CMakeFiles/$t.dir/tests/unit/$t.cpp.o $d/jit.o#; s# -o $t # -o $d/$t #")
fi
(cd "$b" && eval "$c") > "$d/cc.err" 2>&1 || { echo "mutant=$name compile failed: $(grep -m1 error "$d/cc.err")"; exit 0; }
(cd "$b" && eval "$l") > "$d/ld.err" 2>&1 || { echo "mutant=$name link failed: $(grep -m1 -i error "$d/ld.err")"; exit 0; }
TMPDIR=$w timeout 600 "$d/$t" > "$d/out" 2>&1; rc=$?
echo "mutant=$name rc=$rc | $(grep -m1 -E 'FAIL' "$d/out" | cut -c1-150)"
rm -f "$d/jit.o" "$d/t.o"
