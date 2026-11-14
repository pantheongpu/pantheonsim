// vgpu::cores_in and vgpu::host_cores (include/vgpu/host_cores.hpp): the physical cores among the CPUs a process may use, which is
// how many host threads the AMD engine spreads a dispatch over when VGPU_THREADS is unset.
//
// cores_in is given topology trees made here (a directory a CPU, as /sys/devices/system/cpu has them), so the test says the same
// on any host: two hardware threads of a core count once, the same core number in another package is another core, and a CPU
// whose topology cannot be read makes the answer 0 ("unknown": host_cores then falls back to host_cpus). host_cores itself is
// held to its bounds and to this host's own sysfs.
// Framework-free: prints what failed and exits non-zero.
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "vgpu/host_cores.hpp"
#include "vgpu/host_cpus.hpp"

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  std::printf("%s  %s\n", ok ? "ok   " : "FAIL ", what);
  if (!ok) ++failures;
}

// root/cpu<N>/topology/{physical_package_id,core_id}
void add_cpu(const std::string& root, int cpu, int package, int core) {
  const std::string dir = root + "/cpu" + std::to_string(cpu);
  ::mkdir(dir.c_str(), 0755);
  ::mkdir((dir + "/topology").c_str(), 0755);
  std::ofstream(dir + "/topology/physical_package_id") << package << "\n";
  std::ofstream(dir + "/topology/core_id") << core << "\n";
}

cpu_set_t set_of(std::initializer_list<int> cpus) {
  cpu_set_t s;
  CPU_ZERO(&s);
  for (int c : cpus) CPU_SET(c, &s);
  return s;
}

}  // namespace

int main() {
  char tmpl[] = "/tmp/vgpu-host-cores-XXXXXX";
  const char* made = ::mkdtemp(tmpl);
  if (!made) {
    std::printf("SKIP: no temporary directory\n");
    return 0;
  }
  const std::string root = made;
  // Two packages of two cores, two hardware threads a core, numbered as Linux numbers them on such a host: the second threads
  // after all the first ones. CPUs 0-3 are the first threads of (0,0) (0,1) (1,0) (1,1), CPUs 4-7 their siblings.
  for (int cpu = 0; cpu < 8; ++cpu) add_cpu(root, cpu, (cpu % 4) / 2, cpu % 2);

  check(vgpu::cores_in(set_of({0, 1, 2, 3, 4, 5, 6, 7}), root) == 4, "eight hardware threads on four cores are four cores");
  check(vgpu::cores_in(set_of({0, 4}), root) == 1, "the two hardware threads of one core are one core");
  check(vgpu::cores_in(set_of({0, 1}), root) == 2, "two cores of one package are two cores");
  check(vgpu::cores_in(set_of({0, 2}), root) == 2, "the same core number in two packages is two cores");
  check(vgpu::cores_in(set_of({0, 4, 5}), root) == 2, "a core with both threads and one with one are two cores");
  check(vgpu::cores_in(set_of({}), root) == 0, "no CPU is no core");
  check(vgpu::cores_in(set_of({0, 9}), root) == 0, "a CPU without a topology makes the count unknown (0)");
  check(vgpu::cores_in(set_of({0, 1}), root + "/nowhere") == 0, "no topology tree at all is unknown (0)");

  // This host: at least one, never more than the CPUs the process may use, and what its own sysfs says when that can be read.
  const unsigned cores = vgpu::host_cores(), cpus = vgpu::host_cpus();
  check(cores >= 1 && cores <= cpus, "host_cores is between 1 and host_cpus");
  cpu_set_t mine;
  CPU_ZERO(&mine);
  if (sched_getaffinity(0, sizeof mine, &mine) == 0) {
    const unsigned here = vgpu::cores_in(mine, "/sys/devices/system/cpu");
    check(cores == (here ? std::min(here, cpus) : cpus), "host_cores is this host's cores, capped by host_cpus (or host_cpus where sysfs has no topology)");
    std::printf("      this host: %u core(s), %u CPU(s) the process may use\n", cores, cpus);
  }

  const std::string rm = "rm -rf '" + root + "'";
  if (std::system(rm.c_str()) != 0) std::printf("      (could not remove %s)\n", root.c_str());
  return failures ? 1 : 0;
}
