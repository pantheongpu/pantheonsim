# Whether `vgpu shell`'s isolation can work on this host, for the tests that
# check an isolated session (sourced; defines session_isolation_ok).
#
# The shell re-executes itself in `unshare -r -m -u` and bind-mounts the
# session's files over /etc, /proc/driver, /sys and /dev. A host can allow the
# namespaces and still refuse the mounts -- a Docker container without
# privileges (its seccomp and AppArmor profiles), or a kernel that restricts
# unprivileged user namespaces -- and then the shell runs without isolation,
# so the isolated checks would fail for a reason that is the host's, not the
# simulator's. So this tries what the shell does, in a throwaway namespace:
# make the mount tree private, bind-mount a file, and recursively bind /sys.
# Checking for a container instead would be wrong both ways.
session_isolation_ok() {
  command -v unshare >/dev/null 2>&1 || return 1
  unshare -r -m -u sh -c '
    d=$(mktemp -d) || exit 1
    trap "rm -rf \"$d\"" EXIT
    mkdir "$d/sys" && : > "$d/f" &&
      mount --make-rprivate / &&
      mount --bind /etc/hostname "$d/f" &&
      mount --rbind /sys "$d/sys"' >/dev/null 2>&1
}
