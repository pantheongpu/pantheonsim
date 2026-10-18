#!/usr/bin/env bash
# The default number of simulator threads for `make run` and ./run: half the CPUs
# this process may use, at least 1. Prints nothing when the person already chose:
# VGPU_THREADS is set, or the settings file has a `threads:` line. Flags
# (`--threads`, make THREADS=) are not seen here; they beat the environment anyway.
#
#   scripts/half-threads.sh [settings-file]
#
# The library's own default (one thread per core) is unchanged; this only sets
# VGPU_THREADS for the two front doors, so CI and the playground behave as before.
set -u
[[ -z ${VGPU_THREADS:-} ]] || exit 0
xdg=${XDG_CONFIG_HOME:-${HOME:-/nonexistent}/.config}/pantheonsim/config.yaml
cfg=${1:-${VGPU_CONFIG:-}}
if [[ -z $cfg ]]; then
  if [[ -e pantheonsim.yaml ]]; then cfg=pantheonsim.yaml; elif [[ -e $xdg ]]; then cfg=$xdg; fi
fi
if [[ -n $cfg && -r $cfg ]] && grep -q -E '^[[:space:]]*threads[[:space:]]*:' "$cfg"; then exit 0; fi
n=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
echo $(( (n + 1) / 2 ))
