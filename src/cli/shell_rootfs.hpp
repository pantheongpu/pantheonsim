// `vgpu shell --rootfs`: the session's own root filesystem.
//
// Without it the session sees the host's files (only the GPU surfaces are
// replaced). With it, inside the mount namespace `vgpu shell` already creates,
// the session builds a new root on a tmpfs and pivots into it. See
// docs/machine-simulator.md, "A private root".
#pragma once

#include <string>
#include <vector>

namespace vgpu::cli {

// One `-v HOST:DEST[:ro]`.
struct Volume {
  std::string host;   // absolute, symlinks resolved, exists
  std::string dest;   // absolute, normalized, inside the session
  bool ro = false;
};

// Parses and validates one `-v` argument. Checks everything that can be checked
// before a session starts: the host path exists, the destination is absolute
// and not under /proc, /sys or the root itself. On failure returns false and
// sets *err (no leading "vgpu shell:").
bool parse_volume(const std::string& spec, Volume* out, std::string* err);

// `--home DIR`: must be an existing directory; returns it absolute.
bool parse_home(const std::string& dir, std::string* out, std::string* err);

// True when user namespaces cannot be created here (or the test hook
// VGPU_SHELL_NO_NAMESPACES=1 says so). Runs `unshare -r -m -u true`.
bool namespaces_unavailable();

struct RootfsInput {
  std::string session_dir;                 // /tmp/vgpu-session-PID, bound at the same path
  std::string session_root;                // its root/ (the staged /dev files)
  std::vector<std::string> install_dirs;   // the simulator's own directories, bound read-only at their host paths
  std::vector<Volume> volumes;
  std::string home;                        // empty: /root is a tmpfs
};

struct RootfsResult {
  std::string error;                       // set when the root could not be built
  std::vector<std::string> summary;        // what the session now has, one line each (the banner)
};

// Builds the root and pivots into it. Must run as the namespace's root, after
// the GPU surfaces are overlaid. Returns false (with result->error) on any
// failure; the process is then still in the host's view, and must exit.
bool enter_rootfs(const RootfsInput& in, RootfsResult* result);

// At the end of the session: lets go of the session directory's mount in the new
// root and removes the directory itself from the host's /tmp, which the new root
// cannot otherwise reach. Call after the directory's contents are removed.
void rootfs_release_session_dir(const std::string& dir);

}  // namespace vgpu::cli
