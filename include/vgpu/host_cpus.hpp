#pragma once
// How many host CPUs this process can actually run on at once: the ones its
// affinity mask allows, capped by its cgroup's CPU quota.
//
// std::thread::hardware_concurrency() counts every online CPU, which inside a
// container with a quota (docker --cpus 1, a Kubernetes limit, a pantheonsim.com
// playground session) is the host's count, not the container's. Spreading a
// grid over that many workers on a one-CPU quota makes them take turns being
// throttled: Ollama on a simulated T4 in a one-CPU cgroup on a 20-core host ran
// 1.5 times slower with 20 workers than with one.
//
// Computed once. The quota is the tightest one on the way from this process's
// cgroup (/proc/self/cgroup) up to the root: cgroup v2's cpu.max ("<quota>
// <period>" or "max <period>"), or v1's cpu.cfs_quota_us and cpu.cfs_period_us,
// with a fractional quota rounded up.
#include <sched.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

namespace vgpu {

inline unsigned host_cpus() {
  static const unsigned n = [] {
    unsigned cpus = std::thread::hardware_concurrency();
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
      const int c = CPU_COUNT(&set);
      if (c > 0) cpus = cpus ? std::min(cpus, static_cast<unsigned>(c)) : static_cast<unsigned>(c);
    }
    // The tightest quota between this process's own cgroup and the root.
    // Inside a container with its own cgroup namespace the path is "/" and
    // /sys/fs/cgroup is the container's; outside one it is the full path
    // (a systemd scope, say) under the host's hierarchy.
    auto limit_at = [](const std::string& dir, bool v2) -> unsigned {
      long long quota = -1, period = 0;
      if (v2) {
        std::ifstream f(dir + "/cpu.max");
        std::string q;
        if (!(f >> q >> period) || q == "max") return 0;
        quota = std::atoll(q.c_str());
      } else {
        std::ifstream fq(dir + "/cpu.cfs_quota_us"), fp(dir + "/cpu.cfs_period_us");
        if (!(fq >> quota) || !(fp >> period)) return 0;
      }
      if (quota <= 0 || period <= 0) return 0;
      return static_cast<unsigned>(std::max(1LL, (quota + period - 1) / period));
    };
    std::ifstream self("/proc/self/cgroup");
    std::string line;
    while (std::getline(self, line)) {
      // v2: "0::<path>"; v1: "<id>:<controllers>:<path>" with cpu among them.
      const size_t c1 = line.find(':'), c2 = line.find(':', c1 == std::string::npos ? 0 : c1 + 1);
      if (c1 == std::string::npos || c2 == std::string::npos) continue;
      const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1);
      std::string rel = line.substr(c2 + 1);
      const bool v2 = line.compare(0, 3, "0::") == 0;
      bool cpu = v2;
      for (size_t at = 0; !cpu && at <= controllers.size();) {
        const size_t comma = controllers.find(',', at);
        cpu = controllers.substr(at, comma == std::string::npos ? std::string::npos : comma - at) == "cpu";
        if (comma == std::string::npos) break;
        at = comma + 1;
      }
      if (!cpu) continue;
      const std::string root = v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/cpu";
      while (true) {
        if (const unsigned lim = limit_at(root + (rel == "/" ? "" : rel), v2))
          cpus = cpus ? std::min(cpus, lim) : lim;
        if (rel.empty() || rel == "/") break;
        const size_t slash = rel.find_last_of('/');
        rel = slash == 0 || slash == std::string::npos ? "/" : rel.substr(0, slash);
      }
      if (v2) break;
    }
    return cpus ? cpus : 1u;
  }();
  return n;
}

}  // namespace vgpu
