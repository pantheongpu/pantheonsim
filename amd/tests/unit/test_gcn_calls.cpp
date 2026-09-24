// A kernel that calls a function, which the compiler did not inline.
//
// The kernels are amd/tests/data/calls.c, compiled for gfx942. A call is two
// instructions and a relocation: the code works out where it is with
// s_getpc_b64, adds the distance to the function the loader filled in, and
// jumps with s_swappc_b64, which leaves the address to come back to in a
// register pair; the function returns by jumping to that pair. The distance
// is in the code as a hole until the module is placed, the same way a
// global's address is, so a kernel that calls a function has to be placed
// before it will run.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/calls.gfx942.o";
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

VTEST(a_call_is_a_hole_in_the_code_until_the_module_is_placed) {
  const amd::CodeObject o = object();
  // The distance to the function, in two halves: the code works it out once
  // and both calls jump to it.
  VCHECK(o.relocations.size() >= 2);
  bool to_a_function = false;
  for (const amd::Relocation& r : o.relocations) to_a_function = to_a_function || r.in_text;
  VCHECK(to_a_function);
  VCHECK(!o.placed);
}

VTEST(a_kernel_calls_a_function_and_gets_its_answer_back) {
  amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  place(o, mem);

  const int n = 64;
  std::vector<int> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = i * 37 - 500;
    b[i] = (i % 9) * 11 + 1;
  }
  const uint64_t pa = upload(mem, a), pb = upload(mem, b), pout = mem.alloc(n * 4);

  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "caller");
  VCHECK(d.kernel != nullptr);
  d.kernarg = kernargs(mem, *d.kernel, {pa, pb, pout, static_cast<uint64_t>(n)});
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);

  // out[t] = helper(a, b) + helper(b, 3), with helper(x, y) = x * y + (x ^ y).
  const auto helper = [](int x, int y) { return x * y + (x ^ y); };
  const std::vector<int> out = download<int>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const int want = helper(a[i], b[i]) + helper(b[i], 3);
    if (out[i] != want)
      throw vtest::Failure("caller[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
}

VTEST_MAIN
