// read_launch_env (vgpu/exec/launch_env.hpp) finds in one pass over the environment what the launch used to ask getenv for, one variable at a time.
// It must answer what getenv answers: the first entry of the name, the whole name (VGPU_THREADS2, VGPU_LOCKSTEP_LOG and VGPU_TIER_X are not
// VGPU_THREADS, VGPU_LOCKSTEP and a tier variable), an empty value as an empty string, a variable that is not there as null. The environment is
// replaced by arrays made here, so entries that setenv would never make (the same name twice) are covered.
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/exec/launch_env.hpp"
#include "vtest.hpp"

namespace {

using vgpu::exec::LaunchEnv;

struct Replaced {
  char** saved;
  std::vector<char*> entries;
  explicit Replaced(std::vector<const char*> list) : saved(environ) {
    for (const char* s : list) entries.push_back(const_cast<char*>(s));
    entries.push_back(nullptr);
    environ = entries.data();
  }
  ~Replaced() { environ = saved; }
};

struct Names {
  const char* fastpath = std::getenv("VGPU_FASTPATH");
  const char* tc_exact = std::getenv("VGPU_TC_EXACT");
  const char* sass_trace = std::getenv("VGPU_SASS_TRACE");
  const char* tier_stats = std::getenv("VGPU_TIER_STATS");
  const char* tier_chain = std::getenv("VGPU_TIER_CHAIN");
  const char* threads = std::getenv("VGPU_THREADS");
  const char* lockstep = std::getenv("VGPU_LOCKSTEP");
};

// What the one pass finds and what getenv finds in the same environment: the same pointers.
void same_as_getenv(const char* what) {
  const LaunchEnv e = vgpu::exec::read_launch_env();
  const Names n;
  if (e.fastpath != n.fastpath) throw vtest::Failure(std::string(what) + ": VGPU_FASTPATH differs");
  if (e.tc_exact != n.tc_exact) throw vtest::Failure(std::string(what) + ": VGPU_TC_EXACT differs");
  if (e.sass_trace != n.sass_trace) throw vtest::Failure(std::string(what) + ": VGPU_SASS_TRACE differs");
  if (e.tier_stats != n.tier_stats) throw vtest::Failure(std::string(what) + ": VGPU_TIER_STATS differs");
  if (e.tier_chain != n.tier_chain) throw vtest::Failure(std::string(what) + ": VGPU_TIER_CHAIN differs");
  if (e.threads != n.threads) throw vtest::Failure(std::string(what) + ": VGPU_THREADS differs");
  if (e.lockstep != n.lockstep) throw vtest::Failure(std::string(what) + ": VGPU_LOCKSTEP differs");
}

}  // namespace

VTEST(every_variable_the_launch_reads_is_found_with_its_value) {
  Replaced env({"PATH=/usr/bin", "VGPU_FASTPATH=0", "VGPU_TC_EXACT=0", "VGPU_SASS_TRACE=1", "VGPU_TIER_STATS=2", "VGPU_TIER_CHAIN=0",
                "VGPU_THREADS=4", "VGPU_LOCKSTEP=all", "HOME=/root"});
  const LaunchEnv e = vgpu::exec::read_launch_env();
  VCHECK(e.fastpath && std::string(e.fastpath) == "0");
  VCHECK(e.tc_exact && std::string(e.tc_exact) == "0");
  VCHECK(e.sass_trace && std::string(e.sass_trace) == "1");
  VCHECK(e.tier_stats && std::string(e.tier_stats) == "2");
  VCHECK(e.tier_chain && std::string(e.tier_chain) == "0");
  VCHECK(e.threads && std::string(e.threads) == "4");
  VCHECK(e.lockstep && std::string(e.lockstep) == "all");
  same_as_getenv("all set");
}

VTEST(a_variable_that_is_not_there_is_null_and_an_empty_one_is_an_empty_string) {
  {
    Replaced env({"PATH=/usr/bin", "VGPU_QUIET=1"});
    const LaunchEnv e = vgpu::exec::read_launch_env();
    VCHECK(!e.fastpath && !e.tc_exact && !e.sass_trace && !e.tier_stats && !e.tier_chain && !e.threads && !e.lockstep);
    same_as_getenv("none set");
  }
  {
    Replaced env({"VGPU_THREADS=", "VGPU_FASTPATH="});
    const LaunchEnv e = vgpu::exec::read_launch_env();
    VCHECK(e.threads && e.threads[0] == '\0');
    VCHECK(e.fastpath && e.fastpath[0] == '\0');
    same_as_getenv("empty values");
  }
  {
    Replaced env({});
    const LaunchEnv e = vgpu::exec::read_launch_env();
    VCHECK(!e.threads && !e.lockstep);
  }
}

VTEST(only_the_whole_name_counts_and_the_first_entry_wins) {
  Replaced env({"VGPU_THREADS2=9", "VGPU_THREAD=9", "VGPU_LOCKSTEP_LOG=/tmp/x", "VGPU_TIER_X=1", "VGPU_TIER=0", "VGPU_TIER_STATS_FILE=/tmp/y", "VGPU_TC=1",
                "VGPU_SASS_TRACE_BLOCK=3", "VGPU_FASTPATHS=0", "XVGPU_THREADS=7", "vgpu_threads=7", "VGPU_THREADS=3", "VGPU_THREADS=5", "VGPU_LOCKSTEP=off",
                "VGPU_LOCKSTEP=all"});
  const LaunchEnv e = vgpu::exec::read_launch_env();
  VCHECK(e.threads && std::string(e.threads) == "3");
  VCHECK(e.lockstep && std::string(e.lockstep) == "off");
  VCHECK(!e.tier_stats && !e.tier_chain && !e.sass_trace && !e.fastpath && !e.tc_exact);
  same_as_getenv("near misses and a repeated name");
}

VTEST_MAIN
