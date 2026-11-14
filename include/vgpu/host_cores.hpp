#pragma once
// How many physical cores this process can run on at once: the distinct cores among the CPUs its affinity mask allows, capped by
// host_cpus() (which also knows the cgroup's CPU quota).
//
// The AMD engine spreads a dispatch's work-groups over this many host threads when VGPU_THREADS is unset. Two hardware threads
// of one core share its vector units and caches, and the interpreter's work is vector arithmetic on register files: on an
// 8-core, 16-thread host a second thread per core bought nothing in wall time and doubled the CPU time (inq's solvers::poisson:
// 63.3 s and 183 CPU-s on 8 threads, 63.2 s and 324 CPU-s on 16; Round sp70), which is what the other test processes of a
// `ctest -j4` would have used.
//
// A core is told by (physical_package_id, core_id) under /sys/devices/system/cpu/cpuN/topology. Where that cannot be read (no
// sysfs, another system), every CPU counts as a core and the answer is host_cpus(). Computed once.
#include <sched.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <utility>

#include "vgpu/host_cpus.hpp"

namespace vgpu {

// The distinct cores among the CPUs of `set`, reading the topology under `sysfs_cpu` ("/sys/devices/system/cpu"; a test gives its
// own tree). 0 when the topology of any of them cannot be read.
inline unsigned cores_in(const cpu_set_t& set, const std::string& sysfs_cpu) {
  std::set<std::pair<long, long>> cores;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &set)) continue;
    const std::string dir = sysfs_cpu + "/cpu" + std::to_string(cpu) + "/topology/";
    long package = -1, core = -1;
    std::ifstream fp(dir + "physical_package_id"), fc(dir + "core_id");
    if (!(fp >> package) || !(fc >> core)) return 0;
    cores.emplace(package, core);
  }
  return static_cast<unsigned>(cores.size());
}

inline unsigned host_cores() {
  static const unsigned n = [] {
    const unsigned cpus = host_cpus();
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return cpus;
    const unsigned cores = cores_in(set, "/sys/devices/system/cpu");
    return cores ? std::min(cores, cpus) : cpus;
  }();
  return n;
}

}  // namespace vgpu
