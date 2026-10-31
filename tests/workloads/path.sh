# POSIX sh. Read by run_pantheon_workloads.sh and by path_test.sh.
#
#   path_without <dir> <PATH value>
#
# prints the PATH value without the entry <dir>, spelled as given or reached by another
# route (a symlink, a trailing slash), and without empty entries. Order is kept.
#
# Why the runner needs it: the GitHub Action puts its drop-in nvcc and nvidia-smi
# (<build>/bin) on PATH. `vgpu shell` writes drop-ins of its own and, when it starts,
# looks for "the real nvcc" by searching PATH. With the Action's copy first, that search
# finds the Action's script, so the session's nvcc runs the Action's, which searches PATH
# for the next nvcc that is not itself and finds the session's, and the two exec each
# other for ever. Pantheon's CI sat in its first `make` for 42 minutes on exactly this.
path_without() {
  drop="$1"
  list="$2"
  drop_real=$(cd "$drop" 2>/dev/null && pwd -P)
  out=""
  old_ifs="$IFS"
  IFS=:
  for entry in $list; do
    [ -n "$entry" ] || continue
    [ "$entry" = "$drop" ] && continue
    if [ -n "$drop_real" ] && [ "$(cd "$entry" 2>/dev/null && pwd -P)" = "$drop_real" ]; then
      continue
    fi
    out="${out:+$out:}$entry"
  done
  IFS="$old_ifs"
  printf '%s' "$out"
}
