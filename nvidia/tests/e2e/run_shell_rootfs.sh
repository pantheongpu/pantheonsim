#!/usr/bin/env bash
# `vgpu shell --rootfs`: the session's own root, with host directories added by -v
# the way `docker run -v` does.
#
# What the checks are for, one per way the private root could quietly stop being
# private or stop working:
#   * a host file is not there unless it was mounted (the whole point);
#   * -v is read-write by default and :ro really is read-only;
#   * /root, /home, /tmp and the rest start empty, and --home keeps /root;
#   * the host's /usr still runs programs, read-only;
#   * the GPU surfaces (/proc/driver, /etc/os-release, nvidia-smi, /sys/class/drm,
#     the AMD device files) are the simulated ones inside the new root;
#   * a bad -v, --home or flag combination fails before a session starts;
#   * where namespaces are refused the session does not start on the host's files.
# The checks that need user namespaces are left out (with a note) where the host
# refuses them, as in run_shell_cli.sh; the argument and refusal checks run everywhere.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/session_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_shell_rootfs.XXXXXX")"
trap 'rm -rf "$work"' EXIT
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_CUDA_VERSION VGPU_DRIVER_VERSION VGPU_NVML_VERSION VGPU_SESSION VGPU_TELEMETRY_PATH
export VGPU_QUIET=1

fail=0
ok() { echo "ok    $1"; }
bad() { echo "FAIL  $1"; shift; for l in "$@"; do echo "      $l"; done; fail=1; }
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then ok "$1"; else bad "$1" "expected: $2" "actual:   $3"; fi
}
contains() {  # contains <name> <needle> <haystack>
  if [[ "$3" == *"$2"* ]]; then ok "$1"; else bad "$1" "wanted: $2" "got:    $3"; fi
}
lacks() {  # lacks <name> <needle> <haystack>
  if [[ "$3" != *"$2"* ]]; then ok "$1"; else bad "$1" "must not contain: $2" "got: $3"; fi
}
# One-shot private-root session; stderr merged. `timeout` turns a session that waits into a failure.
rs() { timeout 120 "$vgpu" shell -y --rootfs "$@" </dev/null 2>&1; }
code() {  # code <command...>: the exit status only
  timeout 120 "$@" </dev/null >/dev/null 2>&1; echo $?
}
# The session's `ls -A` of a directory, less what the simulator itself has to put there:
# its own session directory, and the next part of its install path.
own() {  # own <dir>: filters `ls -A <dir>` of the session
  local d=$1 skip="-"
  [[ "$build/" == "$d/"* ]] && { skip=${build#"$d"/}; skip=${skip%%/*}; }
  # (in /opt: the host's toolkits, which the session has where a real machine does)
  grep -vx -e "$skip" | grep -v '^vgpu-session-' | { [[ $d == /opt ]] && grep -v -e '^rocm' -e '^cuda' || cat; } || true
}

# ---------------------------------------------------------------- argument checks (no namespaces needed)
mkdir -p "$work/host" "$work/rw" "$work/ro"
echo secret > "$work/host/secret"
refuses() {  # refuses <name> <exit> <message part> <command...>
  local name=$1 want=$2 part=$3; shift 3
  local out rc; out=$(timeout 30 "$@" </dev/null 2>&1); rc=$?
  if [[ $rc -eq $want && "$out" == *"$part"* ]]; then ok "$name"; else bad "$name" "exit $rc (wanted $want): $out"; fi
}
refuses "-v without --rootfs is an error that says to add it" 2 "needs --rootfs" \
  "$vgpu" shell -y -v "$work/rw:/data" -c true
refuses "--home without --rootfs is an error that says to add it" 2 "needs --rootfs" \
  "$vgpu" shell -y --home "$work/rw" -c true
refuses "-v with a missing host path" 2 "does not exist" \
  "$vgpu" shell -y --rootfs -v "$work/nope:/data" -c 'echo started'
refuses "-v with a relative destination" 2 "absolute" \
  "$vgpu" shell -y --rootfs -v "$work/rw:data" -c 'echo started'
refuses "-v onto /proc" 2 "/proc or /sys" "$vgpu" shell -y --rootfs -v "$work/rw:/proc/x" -c true
refuses "-v onto /sys" 2 "/proc or /sys" "$vgpu" shell -y --rootfs -v "$work/rw:/sys" -c true
refuses "-v onto the root" 2 "below the root" "$vgpu" shell -y --rootfs -v "$work/rw:/" -c true
refuses "-v with an unknown mode" 2 "'ro' or 'rw'" "$vgpu" shell -y --rootfs -v "$work/rw:/d:zz" -c true
refuses "-v with one part" 2 "HOST:DEST" "$vgpu" shell -y --rootfs -v "$work/rw" -c true
refuses "-v with no value" 2 "needs a value" "$vgpu" shell -y --rootfs -v
refuses "--home that is not a directory" 2 "not an existing directory" \
  "$vgpu" shell -y --rootfs --home "$work/host/secret" -c true
refuses "--rootfs with --no-isolate" 2 "contradict" "$vgpu" shell -y --rootfs --no-isolate -c true
out=$(timeout 30 "$vgpu" shell -y --rootfs -v "$work/nope:/data" -c "touch $work/started" </dev/null 2>&1)
[[ ! -e "$work/started" ]] && ok "a refused -v starts no session" || bad "a refused -v started the command"

# Where namespaces are refused the session does not start, and never on the host's files.
out=$(VGPU_SHELL_NO_NAMESPACES=1 timeout 30 "$vgpu" shell -y --rootfs -c "touch $work/ran" </dev/null 2>&1); rc=$?
expect "no namespaces: --rootfs exits 1" 1 "$rc"
contains "no namespaces: the message says why and what to do" "needs user namespaces" "$out"
contains "no namespaces: it names the way back" "drop --rootfs" "$out"
[[ ! -e "$work/ran" ]] && ok "no namespaces: the command did not run on the host" || bad "no namespaces: the command ran on the host"
out=$(VGPU_SHELL_NO_NAMESPACES=1 timeout 30 "$vgpu" shell -y --no-isolate -c 'echo plain' </dev/null 2>&1)
expect "the namespace hook does not touch a session without --rootfs" "plain" "$out"

# ---------------------------------------------------------------- everything below needs namespaces
if ! session_isolation_ok; then
  # Like the other session tests: the checks above ran, the rest cannot on this host.
  echo "note: this host refuses the user namespaces and mounts an isolated session needs; the isolated checks did not run"
  exit $fail
fi

# --- default behaviour is unchanged: the host's files are visible without --rootfs
out=$(timeout 60 "$vgpu" shell -y -c "cat $work/host/secret" </dev/null 2>&1)
expect "without --rootfs the session still sees the host's files" "secret" "$out"
out=$(timeout 60 "$vgpu" shell -y </dev/null 2>&1)
lacks "without --rootfs the banner has no private-root section" "Root     : private" "$out"

# --- the host's files are hidden
out=$(rs -c "cat $work/host/secret; echo rc=\$?")
contains "a host file is not visible without -v" "No such file" "$out"
out=$(rs -c "test -e $work/host && echo yes || echo no")
expect "the host's temporary directory is not there" "no" "$out"
out=$(HOME=/home/someone USER=someone LOGNAME=someone rs -c 'echo $HOME $USER $LOGNAME; pwd; id -u')
expect "HOME, USER and cwd are the session's, and you are root" $'/root root root\n/root\n0' "$out"
for d in /root /work /opt /var/tmp /mnt /srv; do
  out=$(rs -c "ls -A $d" | own $d)
  expect "$d starts empty" "" "$out"
done
out=$(rs -c "ls -A /home" | own /home)
expect "/home starts empty" "" "$out"
out=$(rs -c "ls -A /tmp" | own /tmp)
expect "/tmp starts empty (but for the simulator's own session directory)" "" "$out"
out=$(rs -c 'echo hello > /tmp/x; echo hello > /root/x; echo hello > /work/x; echo ok')
expect "the session can write to its own directories" "ok" "$out"
out=$(rs -c 'ls /dev | tr "\n" " "')
allowed=" null zero full random urandom tty pts shm ptmx fd stdin stdout stderr kfd dri "
extra=""
for n in $out; do [[ "$allowed" == *" $n "* ]] || extra+="$n "; done
expect "/dev holds only the minimal nodes" "" "$extra"

out=$(rs -c 'head -c 4 /dev/urandom | wc -c; echo x > /dev/null && echo null; head -c 3 /dev/zero | wc -c; ls /dev/pts')
expect "/dev/urandom, /dev/null, /dev/zero and a devpts work" $'4\nnull\n3\nptmx' "$out"
out=$(rs -c 'echo shm > /dev/shm/f && cat /dev/shm/f')
expect "/dev/shm is writable" "shm" "$out"

# --- the system still works, read-only
out=$(rs -c 'ls /usr/bin | head -1 >/dev/null && sh -c "echo from-sh" && cat /etc/hostname 2>/dev/null; true')
contains "ls and sh run from the host's /usr" "from-sh" "$out"
out=$(rs -c 'touch /usr/x 2>&1; echo x > /etc/passwd 2>&1; touch /lib/x 2>&1' )
[[ $(grep -c "Read-only file system" <<< "$out") -eq 3 ]] && ok "/usr, /lib and the host's /etc entries are read-only" \
  || bad "/usr, /lib and /etc/passwd are read-only" "$out"
out=$(rs -c 'getent passwd root | cut -d: -f1; getent hosts localhost | wc -l')
contains "the passwd and nsswitch entries resolve names" "root" "$out"
if command -v python3 >/dev/null; then
  out=$(rs -c 'python3 -c "import json,sys;print(json.dumps([1]))"')
  expect "the host's python3 runs" "[1]" "$out"
fi
if [[ -s /etc/resolv.conf ]]; then
  out=$(rs -c 'test -s /etc/resolv.conf && echo yes')
  expect "resolv.conf is there" "yes" "$out"
fi
if [[ -d /etc/ssl/certs ]]; then
  out=$(rs -c 'test -d /etc/ssl/certs && echo yes')
  expect "/etc/ssl/certs is there" "yes" "$out"
fi
rs -c 'exit 7' >/dev/null; expect "the command's exit status comes through" 7 "$?"

# --- toolkits where a real machine has them, and only if the host has them
for t in /usr/local/cuda /opt/rocm; do
  host=no; [[ -e $t ]] && host=yes
  out=$(rs -c "test -e $t && echo yes || echo no")
  expect "$t is in the session exactly when the host has it" "$host" "$out"
done
for t in /usr/local/cuda /opt/rocm; do   # where the host has a toolkit: its files, read-only
  [[ -d $t ]] || continue
  expect "$t lists what the host's does" "$(ls "$t")" "$(rs -c "ls $t")"
  out=$(rs -c "touch $t/vgpu-rootfs-probe 2>&1")
  contains "$t is read-only" "Read-only file system" "$out"
done
out=$(timeout 60 "$vgpu" shell -y --rootfs </dev/null 2>&1)
contains "the banner says the root is private" "Root     : private" "$out"
contains "the banner lists the session's own directories" "the session's own" "$out"
contains "the banner says what happened to /usr/local/cuda" "/usr/local/cuda :" "$out"
contains "the banner says what happened to /opt/rocm" "/opt/rocm" "$out"
out=$(timeout 60 "$vgpu" shell -y --rootfs -c 'echo quiet' </dev/null 2>&1)
expect "-c prints no banner" "quiet" "$out"

# --- -v: read-write by default, :ro read-only
out=$(rs -v "$work/rw:/data" -c 'echo written > /data/f; cat /data/f')
expect "-v is read-write: the write is seen inside" "written" "$out"
expect "-v is read-write: the write is on the host" "written" "$(cat "$work/rw/f" 2>/dev/null)"
echo kept > "$work/ro/f"
out=$(rs -v "$work/ro:/data:ro" -c 'echo no > /data/f 2>&1; echo no > /data/g 2>&1; cat /data/f')
[[ $(grep -c "Read-only file system" <<< "$out") -eq 2 && "$out" == *kept ]] && ok "-v :ro refuses writes with EROFS and the content is readable" \
  || bad "-v :ro" "$out"
expect "-v :ro: the host is untouched" "kept" "$(cat "$work/ro/f")"
[[ ! -e "$work/ro/g" ]] && ok "-v :ro: no file appeared on the host" || bad "-v :ro: a file appeared on the host"
out=$(rs -v "$work/rw:/data:rw" -c 'echo explicit > /data/e; cat /data/e')
expect "-v :rw is the default spelled out" "explicit" "$out"
out=$(rs -v "$work/rw:/a/b/c/d" -c 'cat /a/b/c/d/f')
expect "the destination is created with its parents" "written" "$out"
echo one > "$work/file1"
out=$(rs -v "$work/file1:/opt/one.txt:ro" -c 'cat /opt/one.txt; echo x > /opt/one.txt 2>&1 | head -1')
contains "a single file can be mounted" "one" "$out"
mkdir -p "$work/inner"; echo inner > "$work/inner/i"
out=$(rs -v "$work/rw:/m" -v "$work/inner:/m/sub" -c 'cat /m/sub/i; ls /m')
contains "mounts nest, in the order given" "inner" "$out"
echo mine > "$work/hosts"
out=$(rs -v "$work/hosts:/etc/hosts:ro" -c 'cat /etc/hosts')
expect "a -v over a system file replaces it" "mine" "$out"
out=$(rs -v "$work/rw:/usr/local/vgpu-new" -c 'echo started'); rc=$?
[[ "$out" == *"cannot create mount point"* && "$out" != *started* ]] && ok "a destination that cannot be created in the read-only system is an error, not a fallback" \
  || bad "unwritable destination" "$out"
mkdir -p "$work/symhost"; ln -s "$work/symhost" "$work/link"
echo viasym > "$work/symhost/f"
out=$(rs -v "$work/link:/viasym" -c 'cat /viasym/f')
expect "a symlink as HOST is followed" "viasym" "$out"

# --- :ro reaches a filesystem mounted inside the directory too
mkdir -p "$work/tree/sub"; echo deep > "$work/tree/sub/f"
out=$(unshare -r -m sh -c "mount -t tmpfs t '$work/tree/sub' && echo deep > '$work/tree/sub/f' &&
  VGPU_QUIET=1 timeout 120 '$vgpu' shell -y --rootfs -v '$work/tree:/t:ro' -c 'echo x > /t/sub/g 2>&1; cat /t/sub/f'" </dev/null 2>&1)
[[ "$out" == *"Read-only file system"* && "$out" == *deep ]] && ok "-v :ro covers a mount inside the directory" \
  || bad "-v :ro covers a mount inside the directory" "$out"

# --- --home keeps /root
mkdir -p "$work/home"
rs --home "$work/home" -c 'echo remembered > /root/note; echo "alias ll=true" > /root/.bashrc' >/dev/null
out=$(rs --home "$work/home" -c 'cat /root/note')
expect "--home persists /root between two runs" "remembered" "$out"
expect "--home is the host directory" "remembered" "$(cat "$work/home/note")"
out=$(rs -c 'ls -A /root')
expect "without --home /root is empty again" "" "$out"
out=$(rs -c 'echo gone > /root/note'; rs -c 'cat /root/note')
contains "without --home nothing survives the session" "No such file" "$out"

# --- the GPU surfaces are the simulated ones
out=$(rs --gpu nvidia/h100 --driver 580.65.06 -c 'cat /proc/driver/nvidia/version')
contains "/proc/driver/nvidia/version reports the simulated driver" "580.65.06" "$out"
out=$(rs --gpu nvidia/h100 --count 2 -c 'nvidia-smi -L')
[[ $(grep -c '^GPU ' <<< "$out") -eq 2 && "$out" == *H100* ]] && ok "nvidia-smi -L lists the simulated GPUs" || bad "nvidia-smi -L" "$out"
out=$(rs --os rocky:9 -c 'cat /etc/os-release')
contains "/etc/os-release is the simulated OS" "Rocky Linux" "$out"
out=$(rs --gpu nvidia/h100 --count 2 -c 'ls /sys/class/drm')
contains "/sys/class/drm lists the simulated cards" "card1" "$out"
out=$(rs --hostname rootfs-node -c 'hostname; cat /proc/sys/kernel/hostname')
expect "the hostname is the simulated one" $'rootfs-node\nrootfs-node' "$out"
out=$(rs -c 'echo "$PATH" | cut -d: -f1; echo "$LD_LIBRARY_PATH" | cut -d: -f1')
contains "the session's tools are first on PATH" "/vgpu-session-" "$out"
contains "the shim directory is first on LD_LIBRARY_PATH" "/shim" "$out"
out=$(rs -c 'vgpu --version 2>&1 | head -1; test -x "$(command -v nvidia-smi)" && echo tool')
contains "the simulator's own binary and tools run inside" "tool" "$out"
out=$(rs --gpu amd/mi300x -c 'test -e /dev/kfd && echo kfd; ls /dev/dri | head -1; head -c0 /dev/kfd && echo opens; rocm_agent_enumerator | head -3')
contains "an AMD session has /dev/kfd that opens" "opens" "$out"
contains "an AMD session has /dev/dri/card0" "card0" "$out"
contains "an AMD session lists its agent" "gfx" "$out"
out=$(rs --gpu amd/mi300x -c 'cat /sys/module/amdgpu/initstate; command -v nvidia-smi >/dev/null && echo has-nvidia-smi')
contains "an AMD session has the amdgpu module loaded" "live" "$out"

# --- the session directory does not outlive the session
dir=$(rs -c 'echo $VGPU_SESSION')
[[ -n "$dir" && ! -e "$dir" ]] && ok "the session directory is removed from the host's /tmp" || bad "the session directory $dir is still there"

exit $fail
