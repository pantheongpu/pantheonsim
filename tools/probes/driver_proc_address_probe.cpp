// What a real driver's cuGetProcAddress_v2 answers for every name NVIDIA's typedef headers give
// (nvidia/src/driver_proc_versions.inc), at CUDA 11.4, 12.0, 12.4 and 13.0, with the default flags and with the
// per-thread default stream: the result, the status, and the export it handed back, named by dladdr (the
// driver's own libcuda exports every versioned symbol by name). The checked-in recording is
// nvidia/tests/data/driver_proc_address.rtx3060.txt, which test_driver_proc_address compares the shim with.
//
//   g++ -std=c++17 -O1 tools/probes/driver_proc_address_probe.cpp -o /tmp/pa -ldl
//   /tmp/pa nvidia/src/driver_proc_versions.inc > nvidia/tests/data/driver_proc_address.rtx3060.txt
//
// Run it on a machine with an NVIDIA GPU and its driver (not under the simulator).
#include <dlfcn.h>

#include <cstdio>
#include <fstream>
#include <regex>
#include <set>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <driver_proc_versions.inc>\n", argv[0]);
    return 2;
  }
  void* h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) {
    std::fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
  using Init = int (*)(unsigned);
  using Version = int (*)(int*);
  using GetProc = int (*)(const char*, void**, int, unsigned long long, int*);
  auto init = reinterpret_cast<Init>(dlsym(h, "cuInit"));
  auto version = reinterpret_cast<Version>(dlsym(h, "cuDriverGetVersion"));
  auto get = reinterpret_cast<GetProc>(dlsym(h, "cuGetProcAddress_v2"));
  if (!init || !version || !get || init(0) != 0) {
    std::fprintf(stderr, "no usable driver\n");
    return 1;
  }
  std::set<std::string> names;
  std::ifstream in(argv[1]);
  const std::regex row(R"re(^\{"(cu\w+)", \d+, \d\},$)re");
  for (std::string line; std::getline(in, line);) {
    std::smatch m;
    if (std::regex_match(line, m, row)) names.insert(m[1]);
  }
  int v = 0;
  version(&v);
  std::printf("# cuGetProcAddress_v2 on a real driver (cuDriverGetVersion %d), by tools/probes/driver_proc_address_probe.cpp\n", v);
  std::printf("# name cudaVersion flags result status symbol ('-': no function)\n");
  for (unsigned long long flags : {0ull, 2ull})
    for (const auto& n : names)
      for (int ver : {11040, 12000, 12040, 13000}) {
        void* fn = nullptr;
        int status = -1;
        const int r = get(n.c_str(), &fn, ver, flags, &status);
        Dl_info d{};
        const char* sym = "-";
        if (fn) sym = dladdr(fn, &d) && d.dli_sname && d.dli_saddr == fn ? d.dli_sname : "?";
        std::printf("%s %d %llu %d %d %s\n", n.c_str(), ver, flags, r, status, sym);
      }
  return 0;
}
