// Growth over many repeats of one workload: what a single exit-time line cannot tell from the library's own warm-up. The cycle of
// leak_cycle.h (runtime, driver, cuBLAS: every kind of handle, each freed) runs WARM times, the process is measured, it runs
// CYCLES times more, and it is measured again. Nothing may have grown:
//   device_free  cudaMemGetInfo's free bytes, which the simulator takes from its allocator's own accounting: not a byte less
//   fds          open file descriptors: not one more
//   threads      host threads: not one more
//   rss          resident memory: at most 4 MiB more (a cycle that kept 40 KiB shows over 400 cycles; the allocator's arenas
//                and the library's caches settle during the warm-up and stay)
//   maps         lines of /proc/self/maps: at most 24 more
// (SOAK_NO_RSS=1 leaves out the last two, for a sanitizer build.)
// "--leak=<what>" keeps one thing a cycle on purpose, which is how the check is shown to bite (run_leak_soak.sh runs each and wants
// the right resource named): device (a cudaMalloc), host (a malloc touched), pinned (a cudaMallocHost touched), fd, thread.
// Prints "PASS", or "LEAK <resources>" and exits 1.
#include "leak_cycle.h"

#include <dirent.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Sample {
  size_t device_free = 0;
  long fds = 0, threads = 0, maps = 0, rss_kb = 0;
};

static Sample measure() {
  Sample s;
  size_t total = 0;
  cudaMemGetInfo(&s.device_free, &total);
  if (DIR* d = opendir("/proc/self/fd")) {
    const int self = dirfd(d);
    while (const dirent* e = readdir(d))
      if (e->d_name[0] != '.' && std::atoi(e->d_name) != self) ++s.fds;
    closedir(d);
  }
  std::ifstream st("/proc/self/status");
  std::string line;
  while (std::getline(st, line)) {
    if (line.compare(0, 8, "Threads:") == 0) s.threads = std::strtol(line.c_str() + 8, nullptr, 10);
    if (line.compare(0, 6, "VmRSS:") == 0) s.rss_kb = std::strtol(line.c_str() + 6, nullptr, 10);
  }
  if (std::getenv("SOAK_THREAD_NAMES")) {   // (for finding a thread that appeared: its name is the thread's own or the program's)
    if (DIR* d = opendir("/proc/self/task")) {
      while (const dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        std::ifstream c(std::string("/proc/self/task/") + e->d_name + "/comm");
        std::string name;
        std::getline(c, name);
        std::printf("  thread %s %s\n", e->d_name, name.c_str());
      }
      closedir(d);
    }
  }
  std::ifstream mp("/proc/self/maps");
  while (std::getline(mp, line)) ++s.maps;
  return s;
}

static std::atomic<bool> g_stop{false};

static void leak_one(const char* what) {
  if (!what) return;
  if (!std::strcmp(what, "device")) {
    void* p = nullptr;
    WANT_OK(cudaMalloc(&p, 64 << 10));
  } else if (!std::strcmp(what, "host")) {
    char* p = static_cast<char*>(std::malloc(128 << 10));
    std::memset(p, 1, 128 << 10);
  } else if (!std::strcmp(what, "pinned")) {
    void* p = nullptr;
    WANT_OK(cudaMallocHost(&p, 128 << 10));
    std::memset(p, 1, 128 << 10);
  } else if (!std::strcmp(what, "fd")) {
    if (open("/dev/null", O_RDONLY) < 0) ++fails;
  } else if (!std::strcmp(what, "thread")) {
    new std::thread([] { while (!g_stop) usleep(1000); });
  }
}

int main(int argc, char** argv) {
  const char* leak = nullptr;
  for (int i = 1; i < argc; ++i)
    if (!std::strncmp(argv[i], "--leak=", 7)) leak = argv[i] + 7;
  const int warm = 30;
  const int cycles = std::getenv("SOAK_CYCLES") ? std::atoi(std::getenv("SOAK_CYCLES")) : 400;
  const long rss_slack_kb = 4 * 1024, maps_slack = 24;
  const bool judge_memory = !std::getenv("SOAK_NO_RSS");   // (a sanitizer's quarantine holds freed memory: resident memory and mappings grow with it)
  WANT_OK(cuInit(0));
  for (int i = 0; i < warm; ++i) {
    everything_and_give_it_back();
    leak_one(leak);   // (the warm-up leaks too, so that the growth is the cycles')
  }
  const Sample a = measure();
  for (int i = 0; i < cycles; ++i) {
    everything_and_give_it_back();
    leak_one(leak);
  }
  const Sample b = measure();
  std::string bad;
  const auto add = [&](bool cond, const char* name) {
    if (cond) bad += (bad.empty() ? "" : ",") + std::string(name);
  };
  add(b.device_free < a.device_free, "device_free");
  add(b.fds > a.fds, "fds");
  add(b.threads > a.threads, "threads");
  add(judge_memory && b.rss_kb - a.rss_kb > rss_slack_kb, "rss");
  add(judge_memory && b.maps - a.maps > maps_slack, "maps");
  std::printf("after %d warm and %d measured cycles: device_free %zu -> %zu, fds %ld -> %ld, threads %ld -> %ld, rss %ld -> %ld kB, maps %ld -> %ld\n",
              warm, cycles, a.device_free, b.device_free, a.fds, b.fds, a.threads, b.threads, a.rss_kb, b.rss_kb, a.maps, b.maps);
  g_stop = true;
  if (fails) bad += (bad.empty() ? "" : ",") + std::string("cuda-errors");
  if (!bad.empty()) {
    std::printf("LEAK %s\n", bad.c_str());
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
