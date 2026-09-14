#!/usr/bin/env bash
# `vgpu shell` driven the way scripts drive it: the versions it defaults to, the
# flags it refuses, the identity it reports and the status it exits with.
#
# Every check is a way the session once disagreed with what was asked for:
#   the default was CUDA 12.4 / driver 550 on a CUDA 13 build, so the session's
#   driver was older than the runtime shim it loaded and frameworks refused to
#   start; --hostname reached only `uname -a`; SHELL=/bin/dash could not run -c
#   ("Illegal option --"); a flag with its value missing was taken as "", which
#   for -c is an interactive shell waiting on stdin; numbers went through atoi,
#   so `--count abc` was one GPU; and a command killed by a signal exited 1.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_shell_cli.XXXXXX")"
trap 'rm -rf "$work"' EXIT
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_CUDA_VERSION VGPU_DRIVER_VERSION VGPU_NVML_VERSION VGPU_SESSION
export VGPU_QUIET=1 VGPU_TELEMETRY_PATH="$work/telemetry"
mkdir -p "$VGPU_TELEMETRY_PATH"

fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
# A one-shot session with no prompt and no namespace. `timeout` makes a session
# that waits on stdin a failure instead of a hung test.
sess() { timeout 60 "$vgpu" shell -y --no-isolate "$@" </dev/null 2>&1; }

# --- the defaults are the build's own ---
outside=$("$build/bin/nvidia-smi" --version 2>&1)
expect "the session's default versions are the build's (nvidia-smi --version, inside = outside)" \
  "$outside" "$(sess -c 'nvidia-smi --version')"
cuda=$(sed -n 's/^CUDA Version *: //p' <<< "$outside")
expect "the session's nvcc reports that CUDA release" \
  "Cuda compilation tools, release $cuda (VirtualGPU session)" "$(sess -c 'nvcc --version' | sed -n 2p)"
expect "--cuda and --driver still override" "12.6 560.35.03" \
  "$(sess --cuda 12.6 --driver 560.35.03 -c 'echo $VGPU_CUDA_VERSION $VGPU_DRIVER_VERSION')"
shopt -s nullglob; carts=("$build/shim"/libcudart.so.[0-9]*); shopt -u nullglob
if [[ ${#carts[@]} -gt 0 ]] && command -v python3 >/dev/null && [[ -z "$(shim_sanitizer "$build/shim")" ]]; then
  cat > "$work/versions.py" <<'PY'
import ctypes, sys
rt = ctypes.CDLL(sys.argv[1])
drv, run = ctypes.c_int(), ctypes.c_int()
rt.cudaDriverGetVersion(ctypes.byref(drv)); rt.cudaRuntimeGetVersion(ctypes.byref(run))
print("ok" if drv.value >= run.value else "driver %d is older than runtime %d" % (drv.value, run.value))
PY
  expect "a default session's driver is not older than the runtime it loads" "ok" \
    "$(sess -c "python3 '$work/versions.py' '${carts[0]}'")"
fi

# --- hostname and uname ---
expect "--hostname and --kernel reach hostname and every uname form" \
  $'sim-node.example\nsim-node\nsim-node.example\nLinux 6.1.0-test\nLinux 6.1.0-test\n6.1.0-test\nLinux\nLinux sim-node.example 6.1.0-test #1 SMP x86_64 x86_64 x86_64 GNU/Linux' \
  "$(sess --hostname sim-node.example --kernel 6.1.0-test \
       -c 'hostname; hostname -s; uname -n; uname -sr; uname -r -s; uname --kernel-release; uname; uname -a')"
sess -c 'uname -z' >/dev/null; expect "uname rejects a flag it does not have" "1" "$?"
if unshare -r -m -u true 2>/dev/null; then
  host_before=$(cat /proc/sys/kernel/hostname)
  expect "an isolated session sets the kernel's own hostname" "iso-node" \
    "$(timeout 60 "$vgpu" shell -y --hostname iso-node -c 'cat /proc/sys/kernel/hostname' </dev/null 2>/dev/null)"
  expect "and the host's hostname is untouched" "$host_before" "$(cat /proc/sys/kernel/hostname)"
else
  echo "skip  isolated hostname: user namespaces are unavailable here"
fi

# --- shells other than bash ---
if [[ -x /bin/dash ]]; then
  expect "SHELL=/bin/dash runs -c" "hi" "$(SHELL=/bin/dash sess -c 'echo hi')"
fi

# --- exit status ---
sess -c 'exit 7' >/dev/null; expect "-c exits with the command's status" "7" "$?"
sess -c 'kill -9 $$' >/dev/null; expect "a command killed by SIGKILL exits 128+9" "137" "$?"

# --- a flag with no value is an error naming the flag, everywhere ---
missing() {  # missing <flag> -- <command...>
  local flag="$1"; shift 2
  local out rc; out=$(timeout 20 "$@" </dev/null 2>&1); rc=$?
  if [[ $rc -eq 2 && "$out" == *"$flag needs a value"* ]]; then echo "ok    missing value: ${*:2}"
  else echo "FAIL  missing value: ${*:2} -> exit $rc: $out"; fail=1; fi
}
missing -c -- "$vgpu" shell -y -c
missing --gpu -- "$vgpu" shell -y --gpu
missing --gpu -- "$vgpu" run --gpu
missing --gpu -- "$vgpu" info --gpu
missing --count -- "$vgpu" serve --count
missing --load -- "$vgpu" serve --gpu nvidia/t4 --load
missing --gpus -- "$vgpu" test --matrix --gpus
missing -n -- "$vgpu" demo vectoradd -n

# --- numbers and versions are parsed whole, in range ---
rejects() {  # rejects <command...>
  local out rc; out=$(timeout 20 "$@" </dev/null 2>&1); rc=$?
  if [[ $rc -eq 2 && "$out" == *"needs "* ]]; then echo "ok    rejected: ${*:2}"
  else echo "FAIL  not rejected: ${*:2} -> exit $rc: $out"; fail=1; fi
}
for bad in "--count 0" "--count abc" "--count 17" "--vram-mb -1" "--vram-mb 1x" "--load 5" \
           "--load abc" "--cuda 13" "--cuda 12.4.1" "--driver abc" "--rocm 6.x" "--hostname bad/name"; do
  # shellcheck disable=SC2086
  rejects "$vgpu" shell -y --no-isolate $bad -c true
done
rejects "$vgpu" run --threads -3 /bin/true
rejects "$vgpu" run --count 0 /bin/true
rejects "$vgpu" run --vram-mb 1x /bin/true
rejects "$vgpu" serve --load abc
rejects "$vgpu" serve --count 0
rejects "$vgpu" demo vectoradd -n abc
rejects "$vgpu" demo vectoradd -n 99999999999999999999999

# --- run --quiet says what it does ---
expect "run --help says --quiet silences errors and warnings too" "yes" \
  "$("$vgpu" run --help | grep -q 'errors and' && echo yes || echo no)"

exit $fail
