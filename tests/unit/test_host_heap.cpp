// The host memory a simulator process runs on (#551): device memory comes from arenas the kernel is asked to back with
// huge pages (madvise MADV_HUGEPAGE, which a runner whose transparent huge pages are `madvise` honours), and making a
// device fixes glibc's heap thresholds (backing::tune_host_heap), so a program and the host-computed libraries that
// allocate and free buffers of a few hundred KiB on every call keep their heap instead of giving it back to the kernel
// and faulting it in again. Each case runs in a child process, which has a heap of its own and makes its device once.
#include <malloc.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vgpu/memory_backing.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define VGPU_TEST_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define VGPU_TEST_SANITIZER 1
#endif
#endif

namespace {

long minflt() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_minflt;
}

// The VmFlags of the mapping of /proc/self/smaps that holds `p`, or "" when none does.
std::string vm_flags_at(const void* p) {
  const auto a = reinterpret_cast<uintptr_t>(p);
  std::ifstream f("/proc/self/smaps");
  bool inside = false;
  for (std::string line; std::getline(f, line);) {
    unsigned long lo = 0, hi = 0;
    if (std::sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2) {   // a mapping's first line (no field line parses so)
      inside = a >= lo && a < hi;
      continue;
    }
    if (inside && line.rfind("VmFlags:", 0) == 0) return line;
  }
  return "";
}

// What the heap does in this process: the page faults of 32 rounds of forty 100 KiB buffers, touched and freed (the
// shape of a host-computed library call), and whether a 4 MiB allocation is its own mapping.
struct Heap {
  long churn_faults;
  bool big_is_mapped;
};
Heap measure_heap() {
  Heap h{};
  const long f0 = minflt();
  for (int round = 0; round < 32; ++round) {
    std::vector<char*> bufs;
    for (int i = 0; i < 40; ++i) {
      char* b = static_cast<char*>(std::malloc(100 << 10));
      std::memset(b, round + 1, 100 << 10);
      bufs.push_back(b);
    }
    for (char* b : bufs) std::free(b);
  }
  h.churn_faults = minflt() - f0;
  const size_t blocks = mallinfo2().hblks;
  char* big = static_cast<char*>(std::malloc(4 << 20));
  big[0] = 1;
  h.big_is_mapped = mallinfo2().hblks > blocks;
  std::free(big);
  return h;
}

// Runs `body` in a child with `env` set (name=value, or name alone to unset it); its exit status.
template <class F> int in_child(const std::vector<std::string>& env, F body) {
  std::fflush(stdout);
  const pid_t pid = fork();
  if (pid == 0) {
    for (const std::string& e : env) {
      const auto eq = e.find('=');
      if (eq == std::string::npos) unsetenv(e.c_str());
      else setenv(e.substr(0, eq).c_str(), e.c_str() + eq + 1, 1);
    }
    std::_Exit(body());
  }
  int st = 0;
  waitpid(pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 100;
}

const std::vector<std::string> kClean = {"VGPU_HOST_HEAP", "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_TOP_PAD_",
                                         "GLIBC_TUNABLES", "VGPU_QUIET=1"};
std::vector<std::string> with(std::vector<std::string> v, const std::string& e) {
  v.push_back(e);
  return v;
}

// A device made, then the heap measured: 0 when it behaved as `tuned` says, else which check failed.
int device_then_heap(bool tuned) {
  vgpu::runtime::Device dev(vgpu::load_gpu("nvidia/h100"), 0, nullptr);
  if (vgpu::backing::host_heap_tuned() != tuned) return 1;
  const Heap h = measure_heap();
  std::printf("  %s: %ld page faults over 32 rounds of 40 x 100 KiB, a 4 MiB block %s\n", tuned ? "tuned" : "glibc's defaults",
              h.churn_faults, h.big_is_mapped ? "is its own mapping" : "comes from the heap");
  std::fflush(stdout);
  if (tuned) {
    if (h.churn_faults > 3000) return 2;   // the first round's 1000 pages, and none after
    if (h.big_is_mapped) return 3;
  } else {
    if (h.churn_faults < 10000) return 4;   // glibc gives the 4 MB back every round, and faults it in again
    if (!h.big_is_mapped) return 5;
  }
  return 0;
}

}  // namespace

VTEST(device_memory_arenas_are_madvised_for_huge_pages) {
#ifdef VGPU_TEST_SANITIZER
  std::printf("  skipped: under a sanitizer device memory is plain heap\n");
#else
  std::ifstream thp("/sys/kernel/mm/transparent_hugepage/enabled");
  if (!thp) {
    std::printf("  skipped: this kernel has no transparent huge pages\n");
    return;
  }
  uint8_t* chunk = vgpu::backing::acquire();
  const std::string flags = vm_flags_at(chunk);
  std::printf("  the chunk's mapping: %s\n", flags.c_str());
  VCHECK(!flags.empty());
  VCHECK(flags.find(" hg") != std::string::npos);   // VM_HUGEPAGE: MADV_HUGEPAGE was given
  vgpu::backing::release(chunk);
#endif
}

VTEST(making_a_device_keeps_the_heap_from_churning) {
#ifdef VGPU_TEST_SANITIZER
  std::printf("  skipped: a sanitizer's allocator is not glibc's\n");
#else
  VCHECK_EQ(in_child(kClean, [] { return device_then_heap(true); }), 0);
#endif
}

VTEST(glibc_defaults_churn_and_the_opt_outs_keep_them) {
#ifdef VGPU_TEST_SANITIZER
  std::printf("  skipped: a sanitizer's allocator is not glibc's\n");
#else
  // What the tuning is for: with glibc's own thresholds the same rounds fault the buffers in every time.
  VCHECK_EQ(in_child(with(kClean, "VGPU_HOST_HEAP=0"), [] { return device_then_heap(false); }), 0);
  // A threshold the environment sets is the user's, and is left as it is.
  for (const char* e : {"MALLOC_TRIM_THRESHOLD_=131072", "MALLOC_MMAP_THRESHOLD_=131072", "GLIBC_TUNABLES=glibc.malloc.trim_threshold=131072"})
    VCHECK_EQ(in_child(with(kClean, e), [] {
                vgpu::runtime::Device dev(vgpu::load_gpu("nvidia/h100"), 0, nullptr);
                return vgpu::backing::host_heap_tuned() ? 1 : 0;
              }),
              0);
  // Another malloc tunable is not a threshold: the tuning still happens.
  VCHECK_EQ(in_child(with(kClean, "GLIBC_TUNABLES=glibc.malloc.hugetlb=1"), [] {
              vgpu::runtime::Device dev(vgpu::load_gpu("nvidia/h100"), 0, nullptr);
              return vgpu::backing::host_heap_tuned() ? 0 : 1;
            }),
            0);
#endif
}

VTEST_MAIN
