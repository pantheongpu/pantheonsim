// A dispatch while the host has floating-point traps enabled (feenableexcept, gfortran -ffpe-trap): the AMD engine's
// arithmetic must not raise SIGFPE (pantheonsim #605, which the CUDA side hit first), and the host's traps, rounding
// mode and sticky flags are as they were afterwards. The kernel is amd/tests/data/vector_add.c, out[i] = a[i] + b[i],
// with inf + -inf (invalid) and FLT_MAX + FLT_MAX (overflow), on one host thread and on several. Without the fix the
// program is killed by SIGFPE, which CTest reports as a failure.
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

constexpr int kTraps = FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW;

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/vector_add." + vtest::amd_target() + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

uint64_t put(MemoryManager& mem, const void* p, size_t bytes) {
  const uint64_t a = mem.alloc(bytes);
  mem.write(a, p, bytes);
  return a;
}

void run(const char* threads) {
  setenv("VGPU_THREADS", threads, 1);
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  if (!k) throw vtest::Failure("no kernel named vector_add");
  MemoryManager mem(64ull << 20);
  constexpr uint32_t kGroups = 8, n = kGroups * 256;
  const float inf = std::numeric_limits<float>::infinity(), big = std::numeric_limits<float>::max();
  std::vector<float> a(n), b(n);
  for (uint32_t i = 0; i < n; ++i) {
    switch (i % 3) {
      case 0: a[i] = inf, b[i] = -inf; break;   // NaN: invalid
      case 1: a[i] = big, b[i] = big; break;    // inf: overflow
      default: a[i] = 1.5f, b[i] = 0.25f; break;
    }
  }
  const uint64_t pa = put(mem, a.data(), n * 4ull), pb = put(mem, b.data(), n * 4ull), out = mem.alloc(n * 4ull);
  std::vector<uint8_t> args(k->kernarg_size, 0);
  std::memcpy(&args[k->args[0].offset], &pa, 8);
  std::memcpy(&args[k->args[1].offset], &pb, 8);
  std::memcpy(&args[k->args[2].offset], &out, 8);
  std::memcpy(&args[k->args[3].offset], &n, 4);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = put(mem, args.data(), args.size());
  d.groups[0] = kGroups;
  d.group_size[0] = 256;

  std::feclearexcept(FE_ALL_EXCEPT);
  std::feraiseexcept(FE_INEXACT);   // (not trapped) a flag the host had before, which it must still have after
  const int round = std::fegetround();
  feenableexcept(kTraps);
  std::string err;
  try {
    amd::execute(d, mem);
  } catch (const std::exception& e) {
    err = e.what();
  }
  const int traps_after = fegetexcept(), flags_after = std::fetestexcept(FE_ALL_EXCEPT), round_after = std::fegetround();
  fedisableexcept(FE_ALL_EXCEPT);
  std::feclearexcept(FE_ALL_EXCEPT);
  unsetenv("VGPU_THREADS");
  if (!err.empty()) throw vtest::Failure(err);
  VCHECK_EQ(traps_after, kTraps);
  VCHECK_EQ(flags_after, FE_INEXACT);
  VCHECK_EQ(round_after, round);
  std::vector<float> r(n);
  mem.read(out, r.data(), n * 4ull);
  for (uint32_t i = 0; i < n; ++i) {
    if (i % 3 == 0) VCHECK(std::isnan(r[i]));
    else if (i % 3 == 1) VCHECK_EQ(r[i], inf);
    else VCHECK_EQ(r[i], 1.75f);
  }
}

}  // namespace

VTEST(dispatch_under_host_fp_traps_one_thread) { run("1"); }

VTEST(dispatch_under_host_fp_traps_worker_threads) { run("4"); }

VTEST_MAIN
