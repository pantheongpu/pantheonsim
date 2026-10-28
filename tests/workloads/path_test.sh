#!/usr/bin/env bash
# Checks path_without (path.sh), which keeps the GitHub Action's drop-in nvcc off the PATH
# the workload runner starts `vgpu shell` with, under bash and the plain sh the runner may
# get. Needs no GPU, no simulator and no pantheon.
set -u

here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/build/bin" "$tmp/usr/bin" "$tmp/with space/bin" "$tmp/other"
ln -s "$tmp/build/bin" "$tmp/link-to-bin"
failed=0

# check <shell> <name> <dir to drop> <PATH value> <wanted result>
check() {
  shell="$1"; name="$2"; drop="$3"; list="$4"; want="$5"
  got="$("$shell" -c ". '$here/path.sh'; path_without \"\$1\" \"\$2\"" _ "$drop" "$list")"
  if [ "$got" != "$want" ]; then
    echo "FAIL [$shell] $name: wanted '$want', got '$got'"
    failed=1
  fi
}

for shell in bash sh; do
  command -v "$shell" >/dev/null || continue
  b="$tmp/build/bin"

  check $shell "the entry goes, the rest keep their order" "$b" "$b:$tmp/usr/bin:/bin" "$tmp/usr/bin:/bin"
  check $shell "in the middle" "$b" "/usr/bin:$b:/bin" "/usr/bin:/bin"
  check $shell "at the end" "$b" "/usr/bin:/bin:$b" "/usr/bin:/bin"
  check $shell "a PATH without it is unchanged" "$b" "/usr/bin:/bin" "/usr/bin:/bin"
  check $shell "a trailing slash is the same directory" "$b" "$b/:/bin" "/bin"
  check $shell "a symlink to it is the same directory" "$b" "$tmp/link-to-bin:/bin" "/bin"
  check $shell "it twice" "$b" "$b:/bin:$b" "/bin"
  check $shell "empty entries go" "$b" ":/bin::$tmp/usr/bin:" "/bin:$tmp/usr/bin"
  check $shell "a path with a space survives" "$b" "$tmp/with space/bin:$b" "$tmp/with space/bin"
  check $shell "a directory that does not exist drops only an exact match" "$tmp/nonexistent" "$tmp/nonexistent:/bin" "/bin"
  check $shell "only it is left: the result is empty" "$b" "$b" ""
done

if [ "$failed" -eq 0 ]; then
  echo "path_without: all checks passed"
fi
exit "$failed"
