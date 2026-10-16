#!/usr/bin/env bash
# check_odr_globals.py finds what it is for: two shared libraries built from one
# common object, the way libcuda and libcudart share libvgpu.a, with a strong
# global in it (must fail), then with that global made static, hidden, or inline
# (must pass), then allow-listed (must pass).
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lint="$here/check_odr_globals.py"
cxx="${CXX:-c++}"
command -v "$cxx" >/dev/null || { echo "SKIP: no C++ compiler"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <wanted exit status> <got>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else echo "FAIL  $1 (exit $3, wanted $2)"; fail=1; fi
}
cat > "$tmp/own.cpp" <<'S'
int own_a(int x) { return x + 1; }
S
cat > "$tmp/own2.cpp" <<'S'
int own_b(int x) { return x + 2; }
S
build() {   # build <declaration of the shared global>
  printf '%s\nint touch(int x) { return x + shared_table[0]; }\n' "$1" > "$tmp/common.cpp"
  "$cxx" -std=c++17 -O1 -fPIC -c "$tmp/common.cpp" -o "$tmp/common.o" &&
  "$cxx" -std=c++17 -fPIC -c "$tmp/own.cpp" -o "$tmp/own.o" &&
  "$cxx" -std=c++17 -fPIC -c "$tmp/own2.cpp" -o "$tmp/own2.o" &&
  "$cxx" -shared "$tmp/own.o" "$tmp/common.o" -o "$tmp/liba.so" &&
  "$cxx" -shared "$tmp/own2.o" "$tmp/common.o" -o "$tmp/libb.so" || { echo "FAIL  could not build the probe libraries"; exit 1; }
}
check() { python3 "$lint" --lib "$tmp/liba.so" --lib "$tmp/libb.so" --obj "$tmp/common.o" --allow "${1:-/nonexistent}" >"$tmp/out" 2>&1; echo $?; }

build 'int shared_table[4] = {1, 2, 3, 4};'
expect "a strong global in shared code fails" 1 "$(check)"
grep -q shared_table "$tmp/out" && echo "ok    and names it" || { echo "FAIL  the report does not name it"; fail=1; }
build 'extern const int shared_table[4]; const int shared_table[4] = {1, 2, 3, 4};'
expect "a strong const table fails too" 1 "$(check)"
build 'static int shared_table[4] = {1, 2, 3, 4};'
expect "static passes" 0 "$(check)"
build '__attribute__((visibility("hidden"))) int shared_table[4] = {1, 2, 3, 4};'
expect "hidden passes" 0 "$(check)"
build 'inline int shared_table[4] = {1, 2, 3, 4};'
expect "inline (COMDAT) passes" 0 "$(check)"
build 'int shared_table[4] = {1, 2, 3, 4};'
printf '^shared_table$   # a probe: shared on purpose\n' > "$tmp/allow"
expect "an allow-listed one passes" 0 "$(check "$tmp/allow")"
printf '^shared_table$\n' > "$tmp/allow"
expect "an allow-list entry without a reason is refused" 1 "$(check "$tmp/allow" 2>/dev/null)"
exit $fail
