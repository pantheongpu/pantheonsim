// Issue #636: Tensile (rocBLAS, hipBLASLt) finds a GEMM kernel by asking each module it has loaded, in turn, with
// hipModuleGetFunction, takes hipErrorNotFound from the ones that do not hold it as "ask the next", and clears the error
// with hipGetLastError (Tensile's SolutionAdapter::getKernel). ROCm's HIP says nothing on such a miss; the simulator wrote
// "the code object has no kernel named ..." to stderr for every probe, which read as a failed GEMM in a run whose GEMMs
// were right. A miss is now said only when no loaded module has the kernel (a name the program got wrong); the error
// code and hipGetLastError are as before. The code objects are amd/tests/data/vector_add.gfx942.hsaco (vector_add,
// reduce_sum) and linked.gfx942.hsaco (scalar, vector, group_z); no ROCm library is needed.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#include <unistd.h>

#include "vgpu_hip.h"

namespace {
int failures = 0;
void check(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

std::string fixture(const char* name) {
  std::ifstream in(std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/" + name, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), {});
}

// What `call` writes to stderr, through an unnamed temporary file.
template <typename F>
std::string stderr_of(F call) {
  std::fflush(stderr);
  FILE* tmp = std::tmpfile();
  if (!tmp) return "<no temporary file>";
  const int saved = dup(2);
  dup2(fileno(tmp), 2);
  call();
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::string out;
  std::rewind(tmp);
  for (int c; (c = std::fgetc(tmp)) != EOF;) out.push_back(static_cast<char>(c));
  std::fclose(tmp);
  return out;
}
}  // namespace

int main() {
  setenv("VGPU_QUIET", "0", 1);   // the message under test is the one VGPU_QUIET silences
  const std::string first = fixture("vector_add.gfx942.hsaco"), second = fixture("linked.gfx942.hsaco");
  if (first.empty() || second.empty()) {
    std::printf("FAIL no code object fixtures\n");
    return 1;
  }
  hipModule_t a = nullptr, b = nullptr;
  if (hipModuleLoadData(&a, first.data()) != hipSuccess || hipModuleLoadData(&b, second.data()) != hipSuccess) {
    std::printf("FAIL could not load the code objects\n");
    return 1;
  }

  // Tensile's loop: each module in load order until one has the kernel.
  hipFunction_t f = nullptr;
  hipError_t miss = hipSuccess, cleared = hipSuccess, after = hipErrorUnknown, hit = hipErrorUnknown;
  const std::string probe = stderr_of([&] {
    miss = hipModuleGetFunction(&f, a, "group_z");
    cleared = hipGetLastError();
    after = hipGetLastError();
    hit = hipModuleGetFunction(&f, b, "group_z");
  });
  check(miss == hipErrorNotFound, "a module without the kernel answers hipErrorNotFound");
  check(cleared == hipErrorNotFound && after == hipSuccess, "which hipGetLastError reports once and clears");
  check(hit == hipSuccess && f != nullptr, "the next module has it");
  check(probe.empty(), "and the probe of the first module says nothing on stderr");
  if (!probe.empty()) std::printf("  stderr was: %s", probe.c_str());

  // A kernel no loaded module has is a name the program got wrong, and is still said.
  hipError_t wrong = hipSuccess;
  const std::string said = stderr_of([&] { wrong = hipModuleGetFunction(&f, a, "no_such_kernel"); });
  check(wrong == hipErrorNotFound, "a kernel in no module answers hipErrorNotFound");
  check(said.find("no kernel named no_such_kernel") != std::string::npos, "and is said on stderr");
  (void)hipGetLastError();

  // Once the module that has it is unloaded, a miss is said again.
  check(hipModuleUnload(b) == hipSuccess, "the second module unloads");
  const std::string gone = stderr_of([&] { wrong = hipModuleGetFunction(&f, a, "group_z"); });
  check(wrong == hipErrorNotFound && gone.find("no kernel named group_z") != std::string::npos,
        "a kernel whose module was unloaded is said missing");
  (void)hipGetLastError();
  check(hipModuleUnload(a) == hipSuccess, "the first module unloads");

  std::printf("%s\n", failures ? "FAILED" : "PASSED");
  return failures ? 1 : 0;
}
