// The rest of what a kernel does with doubles, run and checked against what
// the C means: comparing them, rounding them, the smallest and largest of
// two, and the conversions to and from them.
//
// The kernels are amd/tests/data/doubles.c, compiled for gfx942. All of this
// is exact -- a double holds every one of these results without rounding
// anything away -- so the answers have to match to the last bit, the square
// root included: the compiler builds that one out of a reciprocal square root
// and a refinement, and the refinement lands on the correctly rounded answer
// whether the reciprocal it started from was a card's table or this model's
// exact one.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/doubles.gfx942.o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

template <typename T>
uint64_t upload(MemoryManager& mem, const std::vector<T>& v) {
  const uint64_t p = mem.alloc(v.size() * sizeof(T));
  mem.write(p, v.data(), v.size() * sizeof(T));
  return p;
}

template <typename T>
std::vector<T> download(MemoryManager& mem, uint64_t p, size_t n) {
  std::vector<T> out(n);
  mem.read(p, out.data(), n * sizeof(T));
  return out;
}

uint64_t kernargs(MemoryManager& mem, const amd::Kernel& k, const std::vector<uint64_t>& values) {
  std::vector<uint8_t> buf(k.kernarg_size, 0);
  for (size_t i = 0; i < values.size() && i < k.args.size(); ++i) {
    const amd::KernelArg& a = k.args[i];
    for (uint32_t b = 0; b < a.size && b < 8; ++b) buf[a.offset + b] = static_cast<uint8_t>(values[i] >> (8 * b));
  }
  const uint64_t p = mem.alloc(buf.empty() ? 1 : buf.size());
  if (!buf.empty()) mem.write(p, buf.data(), buf.size());
  return p;
}

void run(const amd::CodeObject& o, const char* name, MemoryManager& mem, const std::vector<uint64_t>& args) {
  const amd::Kernel* k = amd::find_kernel(o, name);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + name);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = kernargs(mem, *k, args);
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);
}

const int kN = 64;

}  // namespace

VTEST(comparing_two_doubles_including_what_a_nan_answers) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<double> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<double>(i % 9) - 4.0;
    b[i] = static_cast<double>(i % 7) - 3.0;
  }
  // One pair with a NaN on each side: every ordered comparison answers no,
  // and "not equal", which is written as a negation, answers yes.
  a[11] = std::nan("");
  b[23] = std::nan("");
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 4);
  run(o, "cmp", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<int32_t> out = download<int32_t>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const int32_t want = (a[i] < b[i]) + 2 * (a[i] >= b[i]) + 4 * (a[i] == b[i]) + 8 * (a[i] != b[i]);
    if (out[i] != want)
      throw vtest::Failure("cmp[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(rounding_a_double_and_choosing_between_two) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<double> a(kN), b(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<double>(i) * 0.5 - 16.0;    // halves, where the tie-break shows
    b[i] = static_cast<double>(32 - i) * 0.25;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(kN * 8);
  run(o, "pick", mem, {pa, pb, pout, static_cast<uint64_t>(kN)});

  const std::vector<double> out = download<double>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const double want = std::fmin(a[i], b[i]) + std::fmax(a[i], b[i]) + std::fabs(a[i]) + std::floor(b[i]) +
                        std::trunc(a[i]) + std::ceil(b[i]) + std::rint(a[i]);
    if (out[i] != want)
      throw vtest::Failure("pick[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST(between_a_double_and_the_narrower_numbers) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<double> a(kN);
  std::vector<float> f(kN);
  std::vector<int32_t> i32(kN);
  std::vector<uint32_t> u32(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<double>(i) * 1234.5678 - 30000.0;
    f[i] = static_cast<float>(i) * 0.1f - 3.2f;   // not a double exactly, so widening has to be exact
    i32[i] = i * 33554432 - 1000000000;           // large enough that a float could not hold them
    u32[i] = 4000000000u - static_cast<uint32_t>(i) * 55555555u;
  }
  const uint64_t pa = upload(mem, a), pf = upload(mem, f), pi = upload(mem, i32), pu = upload(mem, u32),
                 pout = mem.alloc(kN * 8), pof = mem.alloc(kN * 4);
  run(o, "conv", mem, {pa, pf, pi, pu, pout, pof, static_cast<uint64_t>(kN)});

  const std::vector<double> out = download<double>(mem, pout, kN);
  const std::vector<float> of = download<float>(mem, pof, kN);
  for (int i = 0; i < kN; ++i) {
    const double want = static_cast<double>(f[i]) + static_cast<double>(i32[i]) + static_cast<double>(u32[i]);
    if (out[i] != want)
      throw vtest::Failure("conv[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
    VCHECK_EQ(of[i], static_cast<float>(a[i]));
  }
}

VTEST(the_square_root_of_a_double) {
  const amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  std::vector<double> a(kN);
  for (int i = 0; i < kN; ++i) a[i] = static_cast<double>(i) * 7.25 - 100.0;   // both signs; the kernel folds them
  const uint64_t pa = upload(mem, a), pout = mem.alloc(kN * 8);
  run(o, "roots", mem, {pa, pout, static_cast<uint64_t>(kN)});

  const std::vector<double> out = download<double>(mem, pout, kN);
  for (int i = 0; i < kN; ++i) {
    const double want = std::sqrt(std::fabs(a[i]));
    // The compiler builds this out of a reciprocal square root and a
    // refinement. A card's reciprocal square root is a table good to about
    // one unit in the last place, where this one is exact, and the
    // refinement around it lands on the correctly rounded root either way --
    // which is why this can ask for the answer exactly.
    if (out[i] != want)
      throw vtest::Failure("roots[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
