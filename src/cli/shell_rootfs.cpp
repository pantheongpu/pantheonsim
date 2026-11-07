// `vgpu shell --rootfs`: see shell_rootfs.hpp.
#include "shell_rootfs.hpp"

#include <fcntl.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace vgpu::cli {
namespace {

std::string errno_text(const std::string& what) { return what + ": " + std::strerror(errno); }

bool under(const std::string& path, const std::string& dir) {
  return path == dir || path.rfind(dir + "/", 0) == 0;
}

// The host trees the session reads through a read-only bind (or, on a merged-/usr
// host, through the same symlink the host has).
const char* const kSystemDirs[] = {"usr", "lib", "lib32", "lib64", "libx32", "bin", "sbin"};

bool visible_as_system(const std::string& path) {
  for (const char* d : kSystemDirs)
    if (under(path, std::string("/") + d)) return true;
  return false;
}

// What an entry of the host's /etc is needed for: loading programs, resolving
// names, certificates, users, the shell. Nothing else of /etc comes through.
const char* const kEtcEntries[] = {
    "ld.so.cache", "ld.so.conf", "ld.so.conf.d", "resolv.conf", "hosts", "host.conf", "nsswitch.conf",
    "gai.conf", "passwd", "group", "ssl", "ca-certificates", "ca-certificates.conf", "pki",
    "crypto-policies", "alternatives", "bash.bashrc", "profile", "profile.d", "shells", "localtime",
    "terminfo", "inputrc", "mime.types", "protocols", "services", "environment"};

bool make_dirs(const std::string& path, mode_t mode = 0755) {
  std::string cur;
  for (size_t i = 1; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      cur = path.substr(0, i);
      if (::mkdir(cur.c_str(), mode) != 0 && errno != EEXIST) return false;
    }
  }
  return true;
}

bool touch(const std::string& path) {
  struct stat st{};
  if (::stat(path.c_str(), &st) == 0) return true;   // already there (a file mounted over, say)
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0 && errno != EISDIR) return false;
  if (fd >= 0) ::close(fd);
  return true;
}

bool is_dir(const std::string& path) {
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string unescape_mountinfo(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 3 < s.size() + 0 && std::isdigit(static_cast<unsigned char>(s[i + 1]))) {
      out.push_back(static_cast<char>(std::stoi(s.substr(i + 1, 3), nullptr, 8)));
      i += 3;
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

// Makes the mount at `point` read-only, keeping the flags the mount already has
// (nosuid, nodev, noexec, atime): in a user namespace a mount inherited from the
// host cannot have those cleared, and a remount that omits them is refused.
bool remount_ro_one(const std::string& point) {
  struct statvfs sv{};
  if (::statvfs(point.c_str(), &sv) != 0) return false;
  unsigned long keep = sv.f_flag & (ST_NOSUID | ST_NODEV | ST_NOEXEC | ST_NOATIME | ST_NODIRATIME | ST_RELATIME);
  return ::mount(nullptr, point.c_str(), nullptr, MS_BIND | MS_REMOUNT | MS_RDONLY | keep, nullptr) == 0;
}

// Read-only for the mount at `dir` and for every mount below it: a read-only
// bind does not make the mounts inside the tree read-only (/usr/local on its own
// filesystem would stay writable).
bool remount_ro_tree(const std::string& dir, std::string* err) {
  std::vector<std::string> points;
  std::ifstream info("/proc/self/mountinfo");
  for (std::string line; std::getline(info, line);) {
    std::istringstream f(line);
    std::string id, parent, dev, root, point;
    f >> id >> parent >> dev >> root >> point;
    point = unescape_mountinfo(point);
    if (under(point, dir)) points.push_back(point);
  }
  for (const std::string& p : points)
    if (!remount_ro_one(p)) {
      *err = errno_text("cannot make " + p + " read-only");
      return false;
    }
  return true;
}

// Binds `src` (a directory or a file) at `dst`, creating the mount point.
bool bind_at(const std::string& src, const std::string& dst, bool ro, std::string* err) {
  const bool dir = is_dir(src);
  if (dir ? !make_dirs(dst) : !(make_dirs(std::filesystem::path(dst).parent_path().string()) && touch(dst))) {
    *err = errno_text("cannot create mount point " + dst);
    return false;
  }
  if (::mount(src.c_str(), dst.c_str(), nullptr, MS_BIND | (dir ? MS_REC : 0), nullptr) != 0) {
    *err = errno_text("cannot mount " + src + " at " + dst);
    return false;
  }
  return !ro || remount_ro_tree(dst, err);
}

bool symlink_at(const std::string& target, const std::string& dst) {
  if (!make_dirs(std::filesystem::path(dst).parent_path().string())) return false;
  return ::symlink(target.c_str(), dst.c_str()) == 0 || errno == EEXIST;
}

std::string real_path(const std::string& p) {
  char buf[PATH_MAX];
  return ::realpath(p.c_str(), buf) ? std::string(buf) : std::string();
}

std::string read_link(const std::string& p) {
  char buf[PATH_MAX];
  ssize_t n = ::readlink(p.c_str(), buf, sizeof buf - 1);
  return n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
}

bool is_symlink(const std::string& p) {
  struct stat st{};
  return ::lstat(p.c_str(), &st) == 0 && S_ISLNK(st.st_mode);
}

bool exists(const std::string& p) {
  struct stat st{};
  return ::lstat(p.c_str(), &st) == 0;
}

int g_session_parent_fd = -1;

}  // namespace

bool parse_volume(const std::string& spec, Volume* out, std::string* err) {
  std::vector<std::string> parts;
  size_t at = 0;
  while (true) {
    size_t colon = spec.find(':', at);
    parts.push_back(spec.substr(at, colon == std::string::npos ? std::string::npos : colon - at));
    if (colon == std::string::npos) break;
    at = colon + 1;
  }
  if (parts.size() < 2 || parts.size() > 3 || parts[0].empty() || parts[1].empty()) {
    *err = "-v '" + spec + "': expected HOST:DEST or HOST:DEST:ro (paths cannot contain ':')";
    return false;
  }
  Volume v;
  if (parts.size() == 3) {
    if (parts[2] == "ro") v.ro = true;
    else if (parts[2] != "rw") {
      *err = "-v '" + spec + "': the mode after the second ':' is 'ro' or 'rw', got '" + parts[2] + "'";
      return false;
    }
  }
  if (parts[1][0] != '/') {
    *err = "-v '" + spec + "': the destination '" + parts[1] + "' must be an absolute path";
    return false;
  }
  std::string dest = std::filesystem::path(parts[1]).lexically_normal().string();
  while (dest.size() > 1 && dest.back() == '/') dest.pop_back();
  if (dest == "/" || dest.rfind("/..", 0) == 0) {
    *err = "-v '" + spec + "': the destination '" + parts[1] + "' is not a directory below the root";
    return false;
  }
  if (under(dest, "/proc") || under(dest, "/sys")) {
    *err = "-v '" + spec + "': the destination cannot be under /proc or /sys (the session's own)";
    return false;
  }
  v.host = real_path(parts[0]);
  if (v.host.empty()) {
    *err = "-v '" + spec + "': the host path '" + parts[0] + "' does not exist";
    return false;
  }
  v.dest = dest;
  *out = v;
  return true;
}

bool parse_home(const std::string& dir, std::string* out, std::string* err) {
  const std::string real = real_path(dir);
  if (real.empty() || !is_dir(real)) {
    *err = "--home '" + dir + "' is not an existing directory";
    return false;
  }
  *out = real;
  return true;
}

bool namespaces_unavailable() {
  // The hook: a host with namespaces cannot otherwise test the refusal.
  if (const char* force = std::getenv("VGPU_SHELL_NO_NAMESPACES"); force && std::strcmp(force, "1") == 0) return true;
  pid_t pid = ::fork();
  if (pid < 0) return true;
  if (pid == 0) {
    int null = ::open("/dev/null", O_RDWR);
    if (null >= 0) {
      ::dup2(null, 1);
      ::dup2(null, 2);
    }
    ::execlp("unshare", "unshare", "-r", "-m", "-u", "true", static_cast<char*>(nullptr));
    _exit(127);
  }
  int status = 0;
  ::waitpid(pid, &status, 0);
  return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

bool enter_rootfs(const RootfsInput& in, RootfsResult* result) {
  std::string& err = result->error;
  std::vector<std::string>& say = result->summary;
  if (::mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) {
    err = errno_text("cannot make the mount tree private");
    return false;
  }
  const std::string root = in.session_dir + "/newroot";
  if (!make_dirs(root) || ::mount("tmpfs", root.c_str(), "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755") != 0) {
    err = errno_text("cannot create the session's root at " + root);
    return false;
  }
  for (const char* d : {"proc", "sys", "dev", "etc", "run", "var", "root", "home", "opt", "tmp", "work", "mnt", "srv", "var/tmp"})
    if (!make_dirs(root + "/" + d)) {
      err = errno_text(std::string("cannot create /") + d);
      return false;
    }
  ::chmod((root + "/tmp").c_str(), 01777);
  ::chmod((root + "/var/tmp").c_str(), 01777);
  ::chmod((root + "/root").c_str(), 0700);

  // The session's own directory (its tools, telemetry and generated files) at the
  // path every variable and script already names. First, while the new root has
  // nothing mounted in it to copy; the copy of the new root that the recursive
  // bind carries along is dropped at once.
  if (!bind_at(in.session_dir, root + in.session_dir, false, &err)) return false;
  ::umount2((root + in.session_dir + "/newroot").c_str(), MNT_DETACH);
  say.push_back("session files   : " + in.session_dir + " (the simulator's own)");

  // The system, from the host, read-only. A merged-/usr host's /bin, /lib and
  // /sbin are links into /usr, and are links here.
  std::string listed;
  for (const char* d : kSystemDirs) {
    const std::string host = std::string("/") + d;
    if (!exists(host)) continue;
    if (is_symlink(host)) {
      if (!symlink_at(read_link(host), root + host)) {
        err = errno_text("cannot link " + host);
        return false;
      }
    } else if (!bind_at(host, root + host, true, &err)) {
      return false;
    }
    listed += std::string(listed.empty() ? "" : " ") + host;
  }
  say.push_back("host, read-only : " + listed);

  // /etc: the entries a dynamic loader, a resolver, TLS and a shell need.
  std::string etc;
  auto etc_entry = [&](const std::string& name, bool force_bind) -> bool {
    const std::string host = "/etc/" + name;
    if (!exists(host)) return true;
    std::string src = host;
    if (is_symlink(host)) {
      const std::string real = real_path(host);
      if (real.empty()) return true;   // dangling on the host
      if (!force_bind && visible_as_system(real)) {
        if (!symlink_at(read_link(host), root + host)) {
          err = errno_text("cannot link " + host);
          return false;
        }
        etc += (etc.empty() ? "" : " ") + name;
        return true;
      }
      src = real;   // resolv.conf is a link into /run, which the session does not have
    }
    if (!bind_at(src, root + host, true, &err)) return false;
    etc += (etc.empty() ? "" : " ") + name;
    return true;
  };
  // The synthesized one, as the overlay left it.
  if (!etc_entry("os-release", true)) return false;
  for (const char* e : kEtcEntries)
    if (!etc_entry(e, false)) return false;
  symlink_at("/proc/self/mounts", root + "/etc/mtab");
  symlink_at("/run", root + "/var/run");
  say.push_back("/etc, read-only : " + etc);

  // /dev: the few nodes a program opens, a fresh devpts, and the simulated
  // AMD device files where the session has them (they open as /dev/null).
  if (!make_dirs(root + "/dev/pts") || !make_dirs(root + "/dev/shm", 01777)) {
    err = errno_text("cannot create /dev");
    return false;
  }
  ::chmod((root + "/dev/shm").c_str(), 01777);
  for (const char* n : {"null", "zero", "full", "random", "urandom", "tty"})
    if (exists(std::string("/dev/") + n) && !bind_at(std::string("/dev/") + n, root + "/dev/" + n, false, &err)) return false;
  for (const char* n : {"kfd"})
    if (exists(in.session_root + "/dev/" + n) && !bind_at(in.session_root + "/dev/" + n, root + "/dev/" + n, false, &err)) return false;
  const std::string dri = in.session_root + "/dev/dri";
  if (is_dir(dri))
    for (const auto& e : std::filesystem::directory_iterator(dri))
      if (!bind_at(e.path().string(), root + "/dev/dri/" + e.path().filename().string(), false, &err)) return false;
  ::mount("tmpfs", (root + "/dev/shm").c_str(), "tmpfs", MS_NOSUID | MS_NODEV, "mode=1777");
  if (::mount("devpts", (root + "/dev/pts").c_str(), "devpts", MS_NOSUID | MS_NOEXEC, "newinstance,ptmxmode=0666,mode=0620") == 0)
    symlink_at("pts/ptmx", root + "/dev/ptmx");
  symlink_at("/proc/self/fd", root + "/dev/fd");
  symlink_at("/proc/self/fd/0", root + "/dev/stdin");
  symlink_at("/proc/self/fd/1", root + "/dev/stdout");
  symlink_at("/proc/self/fd/2", root + "/dev/stderr");
  say.push_back("/dev            : null zero full random urandom tty pts shm" +
                std::string(exists(in.session_root + "/dev/kfd") ? " kfd dri (simulated)" : ""));

  // /proc and /sys as the session has them now: the host's, with the
  // synthesized /proc/driver, /sys/class/drm, hwmon and PCI trees over them.
  if (!bind_at("/proc", root + "/proc", false, &err) || !bind_at("/sys", root + "/sys", false, &err)) return false;
  say.push_back("/proc, /sys     : as the session has them (GPU surfaces simulated)");

  // The simulator's own install, where its variables and scripts name it.
  std::string install;
  std::vector<std::string> bound_install;
  for (const std::string& d : in.install_dirs) {
    if (d.empty() || d == "/" || visible_as_system(d)) continue;
    bool covered = false;
    for (const std::string& done : bound_install) covered = covered || under(d, done);
    if (covered) continue;
    if (!bind_at(d, root + d, true, &err)) return false;
    bound_install.push_back(d);
    install += (install.empty() ? "" : " ") + d;
  }
  if (!install.empty()) say.push_back("simulator       : " + install + " (read-only, at its host path)");

  // The toolkits, where a real machine has them. Never invented: with none on the
  // host there is nothing here.
  {
    const std::string cuda = real_path("/usr/local/cuda");
    if (cuda.empty()) {
      say.push_back("/usr/local/cuda : none on the host, not created");
    } else {
      if (!visible_as_system(cuda) && !bind_at(cuda, root + cuda, true, &err)) return false;
      say.push_back("/usr/local/cuda : the host's toolkit " + cuda + " (read-only)");
    }
    const std::string rocm = real_path("/opt/rocm");
    if (rocm.empty()) {
      say.push_back("/opt/rocm       : none on the host, not created");
    } else {
      if (!visible_as_system(rocm) && !bind_at(rocm, root + rocm, true, &err)) return false;
      if (is_symlink("/opt/rocm") && !symlink_at(read_link("/opt/rocm"), root + "/opt/rocm")) {
        err = errno_text("cannot link /opt/rocm");
        return false;
      }
      say.push_back("/opt/rocm       : the host's ROCm " + rocm + " (read-only)");
    }
  }

  // /root: the session's, or a host directory that outlives it.
  if (!in.home.empty()) {
    if (!bind_at(in.home, root + "/root", false, &err)) return false;
    say.push_back("/root           : " + in.home + " (the host's, read-write, kept)");
  } else {
    say.push_back("/root           : empty, the session's own (discarded on exit)");
  }
  say.push_back("/home /opt /tmp /work /var/tmp /run /mnt /srv : empty, the session's own (discarded on exit)");

  // The mounts asked for, in order: a later one over an earlier one wins, over
  // the system trees too.
  for (const Volume& v : in.volumes) {
    if (!bind_at(v.host, root + v.dest, v.ro, &err)) {
      err = "-v " + v.host + ":" + v.dest + ": " + err;
      return false;
    }
    say.push_back("-v " + v.host + " -> " + v.dest + (v.ro ? " (read-only)" : " (read-write)"));
  }

  // Keep a handle on the host directory the session directory lives in: the
  // root's own path to it is a mount that has to go before the directory can.
  const std::filesystem::path sd(in.session_dir);
  g_session_parent_fd = ::open(sd.parent_path().c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);

  // Pivot into it, and let the host's tree go.
  const std::string old = root + "/.oldroot";
  if (!make_dirs(old) || ::chdir(root.c_str()) != 0 ||
      ::syscall(SYS_pivot_root, ".", ".oldroot") != 0) {
    err = errno_text("cannot switch to the session's root");
    return false;
  }
  if (::chdir("/") != 0 || ::umount2("/.oldroot", MNT_DETACH) != 0) {
    err = errno_text("cannot detach the host's root");
    return false;
  }
  ::rmdir("/.oldroot");
  if (::chdir("/root") != 0 && ::chdir("/") != 0) {
    err = errno_text("cannot enter the new root");
    return false;
  }
  return true;
}

void rootfs_release_session_dir(const std::string& dir) {
  ::umount2(dir.c_str(), MNT_DETACH);
  if (g_session_parent_fd >= 0)
    ::unlinkat(g_session_parent_fd, std::filesystem::path(dir).filename().c_str(), AT_REMOVEDIR);
}

}  // namespace vgpu::cli
