// See vgpu/leak_check.hpp.
#include "vgpu/leak_check.hpp"

#include <dirent.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vgpu/exec/worker_pool.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace vgpu::runtime {
namespace {

enum class Mode { Off, Report, Strict };

Mode mode() {
  static const Mode m = [] {
    const char* e = std::getenv("VGPU_LEAK_CHECK");
    if (!e || !e[0] || (e[0] == '0' && !e[1])) return Mode::Off;
    return std::strcmp(e, "strict") == 0 ? Mode::Strict : Mode::Report;
  }();
  return m;
}

struct Counts {
  long fds = 0, threads = 0, maps = 0;
  uint64_t rss_kb = 0;
  std::vector<std::string> fd_targets;
};

Counts count_now() {
  Counts c;
  if (DIR* d = opendir("/proc/self/fd")) {
    const int self = dirfd(d);
    while (const dirent* e = readdir(d)) {
      if (e->d_name[0] == '.') continue;
      const int fd = std::atoi(e->d_name);
      if (fd == self) continue;
      char link[64], target[512];
      std::snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
      const ssize_t n = readlink(link, target, sizeof target - 1);
      target[n > 0 ? n : 0] = 0;
      ++c.fds;
      c.fd_targets.push_back(std::to_string(fd) + "=" + target);
    }
    closedir(d);
  }
  {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.compare(0, 8, "Threads:") == 0) c.threads = std::strtol(line.c_str() + 8, nullptr, 10);
      if (line.compare(0, 6, "VmRSS:") == 0) c.rss_kb = std::strtoull(line.c_str() + 6, nullptr, 10);
    }
  }
  {
    std::ifstream f("/proc/self/maps");
    std::string line;
    while (std::getline(f, line)) ++c.maps;
  }
  // The worker pool's threads are the simulator's, kept between launches; they are not growth.
  c.threads -= static_cast<long>(exec::worker_pool_stats().created);
  return c;
}

Counts g_base;
bool g_armed = false;

}  // namespace

void leak_check_arm() {
  if (mode() == Mode::Off || g_armed) return;
  g_base = count_now();
  g_armed = true;
}

std::string leak_check_line(Runtime* rt, bool* leaked) {
  if (leaked) *leaked = false;
  if (!g_armed) return {};
  uint64_t bytes = 0, allocs = 0;
  if (rt)
    for (int i = 0; i < rt->device_count(); ++i) {
      MemoryManager& mm = rt->device(i).memory();
      bytes += mm.used();
      allocs += mm.live_allocations();
    }
  uint64_t pinned_bytes = 0, pinned = 0;
  if (rt)
    for (const auto& [p, r] : rt->host_allocations()) {
      (void)p;
      pinned_bytes += r.size;
      ++pinned;
    }
  const Counts now = count_now();
  const long dfd = now.fds - g_base.fds, dth = now.threads - g_base.threads, dmap = now.maps - g_base.maps;
  std::string what;
  auto add = [&](bool bad, const char* name) {
    if (!bad) return;
    what += what.empty() ? "" : ",";
    what += name;
  };
  add(bytes != 0, "device_bytes");
  add(allocs != 0, "device_allocs");
  add(pinned != 0, "pinned_allocs");
  add(dfd > 0, "fds");
  add(dth > 0, "threads");
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "vgpu-leak-check: device_bytes=%llu device_allocs=%llu pinned_allocs=%llu pinned_bytes=%llu fds=%+ld threads=%+ld "
                "maps=%+ld rss_kb=%llu -> ",
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(allocs),
                static_cast<unsigned long long>(pinned), static_cast<unsigned long long>(pinned_bytes), dfd, dth, dmap,
                static_cast<unsigned long long>(now.rss_kb));
  std::string line = buf;
  line += what.empty() ? "clean" : "LEAK " + what;
  if (dfd > 0) {
    // Which: the descriptors open now that were not open then.
    std::string open;
    for (const auto& t : now.fd_targets) {
      bool known = false;
      for (const auto& b : g_base.fd_targets) known = known || b == t;
      if (!known) open += " [" + t + "]";
    }
    line += "\nvgpu-leak-check: new descriptors:" + open;
  }
  if (leaked) *leaked = !what.empty();
  return line;
}

void leak_check_finish(Runtime* rt) {
  if (mode() == Mode::Off || !g_armed) return;
  bool leaked = false;
  const std::string line = leak_check_line(rt, &leaked);
  std::FILE* f = stderr;
  bool close = false;
  if (const char* p = std::getenv("VGPU_LEAK_CHECK_FILE"); p && *p) {
    if (std::FILE* o = std::fopen(p, "a")) {
      f = o;
      close = true;
    }
  }
  std::fprintf(f, "%s\n", line.c_str());
  std::fflush(f);
  if (close) std::fclose(f);
  g_armed = false;
  if (leaked && mode() == Mode::Strict) {
    std::fflush(nullptr);
    _exit(70);
  }
}

}  // namespace vgpu::runtime
