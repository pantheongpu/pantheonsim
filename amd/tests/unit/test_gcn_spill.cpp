// A kernel with more values than it has registers for.
//
// The kernel is amd/tests/data/spill.c, compiled for gfx942. The machine has
// a second bank of vector registers, the accumulation registers, and the
// compiler moves what will not fit in the first bank there and back rather
// than to memory. They are numbered in the same space as the vector
// registers, and which bank an instruction means is the instruction itself,
// so a model that read the wrong bank would quietly take the wrong values.
// The kernel also calls a function, so it has to be placed before it runs.
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/error.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/spill.gfx942.o";
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

// Places the module's memory on the device, as a loader does.
uint64_t place(amd::CodeObject& o, MemoryManager& mem) {
  const uint64_t base = mem.alloc(o.data.empty() ? 1 : o.data.size());
  if (!o.data.empty()) mem.write(base, o.data.data(), o.data.size());
  amd::place_globals(o, base);
  return base;
}

}  // namespace

VTEST(a_kernel_keeps_what_will_not_fit_in_the_other_bank) {
  amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  place(o, mem);

  const int n = 64;
  std::vector<int> a(n);
  for (int i = 0; i < n; ++i) a[i] = i * 271 - 5000;
  const uint64_t pa = upload(mem, a), pout = mem.alloc(n * 4);

  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "user");
  VCHECK(d.kernel != nullptr);
  d.kernarg = kernargs(mem, *d.kernel, {pa, pout, static_cast<uint64_t>(n)});
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);

  // The forty values the function keeps, added up in an order that reads all
  // of them, plus the work-item's own number.
  int local[40];
  for (int i = 0; i < 40; ++i) local[i] = a[i % n] + i;
  int s = 0;
  for (int i = 0; i < 40; ++i) s += local[(i * 7) % 40];

  const std::vector<int> out = download<int>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const int want = s + i;
    if (out[i] != want)
      throw vtest::Failure("user[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
