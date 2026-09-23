// A kernel that reaches variables of its own: what a __device__ global
// compiles to.
//
// The module has memory beside its code -- the variables it declares, the
// initialised ones and the zeroed ones -- and the code reaches them by adding
// a constant to the program counter. That constant is not in the object: the
// compiler leaves a hole and a relocation saying which variable it meant, and
// a loader fills it in once it knows where the module's memory is. These
// tests place the memory, run the kernel, and read the variables back.
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
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/globals.gfx942.o";
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

VTEST(a_module_carries_its_variables_and_what_they_start_as) {
  const amd::CodeObject o = object();
  // int counters[64] and float scale = 3.5f: one zeroed, one initialised.
  const amd::GlobalVar* counters = amd::find_global(o, "counters");
  const amd::GlobalVar* scale = amd::find_global(o, "scale");
  VCHECK(counters != nullptr);
  VCHECK(scale != nullptr);
  VCHECK_EQ(counters->size, 64u * 4u);
  VCHECK_EQ(scale->size, 4u);
  VCHECK(o.data.size() >= counters->offset + counters->size);

  float started_as = 0;
  std::memcpy(&started_as, o.data.data() + scale->offset, 4);
  VCHECK_EQ(started_as, 3.5f);
  for (uint64_t i = 0; i < counters->size; ++i) VCHECK_EQ(o.data[counters->offset + i], 0);

  // The code cannot know where those will be, so it carries relocations.
  VCHECK(o.relocations.size() >= 2);
  VCHECK(!o.placed);
}

VTEST(a_kernel_reads_and_writes_its_own_variables) {
  amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const uint64_t base = place(o, mem);

  const int n = 64;
  std::vector<int> in(n);
  for (int i = 0; i < n; ++i) in[i] = i * 3 + 1;
  const uint64_t pin = upload(mem, in), pout = mem.alloc(n * 4);

  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "use_global");
  d.kernarg = kernargs(mem, *d.kernel, {pin, pout, static_cast<uint64_t>(n)});
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);

  // out[i] = (int)(counters[i & 63] * scale), with counters[i] just written.
  const std::vector<int> out = download<int>(mem, pout, n);
  for (int i = 0; i < n; ++i) {
    const int want = static_cast<int>(static_cast<float>(in[i]) * 3.5f);
    if (out[i] != want)
      throw vtest::Failure("out[" + std::to_string(i) + "] is " + std::to_string(out[i]) + ", not " +
                           std::to_string(want));
  }
  // And the kernel's writes landed in the module's own memory, where the
  // host can read them.
  const amd::GlobalVar* counters = amd::find_global(o, "counters");
  const std::vector<int> seen = download<int>(mem, base + counters->offset, n);
  for (int i = 0; i < n; ++i) VCHECK_EQ(seen[i], in[i]);
}

VTEST(the_host_can_write_a_variable_before_the_kernel_reads_it) {
  amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  const uint64_t base = place(o, mem);
  const amd::GlobalVar* scale = amd::find_global(o, "scale");

  const float two = 2.0f;
  mem.write(base + scale->offset, &two, 4);

  const int n = 64;
  std::vector<int> in(n, 10);
  const uint64_t pin = upload(mem, in), pout = mem.alloc(n * 4);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = amd::find_kernel(o, "use_global");
  d.kernarg = kernargs(mem, *d.kernel, {pin, pout, static_cast<uint64_t>(n)});
  d.groups[0] = 1;
  d.group_size[0] = 64;
  amd::execute(d, mem);

  const std::vector<int> out = download<int>(mem, pout, n);
  for (int i = 0; i < n; ++i) VCHECK_EQ(out[i], 20);   // 10 * 2.0, not 10 * 3.5
}

VTEST(placing_a_module_twice_in_two_places_is_refused) {
  amd::CodeObject o = object();
  MemoryManager mem(64ull << 20);
  place(o, mem);
  std::string what;
  try {
    amd::place_globals(o, o.data_base + 4096);
  } catch (const std::exception& e) {
    what = e.what();
  }
  VCHECK_CONTAINS(what, "already at another address");
}

VTEST_MAIN
