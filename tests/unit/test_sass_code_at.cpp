// Module::code_at (vgpu/sass/exec.hpp): the binary search over the code sections' start addresses against the
// scan over the sections it replaced (code_at_linear), for every address that matters -- below the first section,
// on and around each section's first and last byte, in the padding between sections, past the last, and at the
// ends of the address space -- on modules shaped like the ones load() makes (consecutive 256-aligned placements,
// empty sections included) and on ones it never makes (out of order, overlapping), which keep no index and scan.
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/sass/exec.hpp"
#include "vtest.hpp"

using vgpu::sass::Code;
using vgpu::sass::Module;

namespace {

struct Rng {
  uint64_t s = 0x2545F4914F6CDD1Dull;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

void add(Module& m, uint64_t base, size_t count) {
  Code c;
  c.section = ".text." + std::to_string(m.code.size());
  c.base = base;
  c.count = count;
  m.code.push_back(std::move(c));
}

// Every address of interest against the scan; `indexed` is whether the module is one that keeps an index.
void compare(const Module& m, bool indexed) {
  VCHECK_EQ(m.code_bases.empty(), !indexed);
  std::vector<uint64_t> probes = {0, 1, 15, 16, 0xff, 0x100, ~uint64_t{0}, ~uint64_t{0} - 1, uint64_t{1} << 63};
  for (const Code& c : m.code) {
    const uint64_t end = c.base + 16 * c.count;
    for (uint64_t d : {uint64_t{0}, uint64_t{1}, uint64_t{15}, uint64_t{16}, uint64_t{255}, uint64_t{256}, uint64_t{257}}) {
      probes.push_back(c.base + d);
      probes.push_back(c.base - d);
      probes.push_back(end + d);
      probes.push_back(end - d);
    }
  }
  Rng rng;
  const uint64_t top = m.code.empty() ? 0x1000 : m.code.back().base + 16 * m.code.back().count + 0x400;
  for (int i = 0; i < 4000; ++i) probes.push_back(rng.below(top + 1));
  for (uint64_t a : probes) {
    const Code* want = m.code_at_linear(a);
    const Code* got = m.code_at(a);
    if (want != got)
      throw vtest::Failure("code_at(" + std::to_string(a) + ") is section " +
                           (got ? std::to_string(got - m.code.data()) : std::string("none")) + ", the scan's is " +
                           (want ? std::to_string(want - m.code.data()) : std::string("none")));
  }
}

constexpr uint64_t kBase = 0x7000'0000ull;   // (a code window's start: any would do)

}  // namespace

VTEST(code_at_matches_the_scan_on_modules_load_places) {
  Rng rng;
  for (int iter = 0; iter < 300; ++iter) {
    Module m;
    uint64_t va = kBase + (rng.below(2) ? 0 : rng.below(1 << 20) * 256);
    const unsigned n = static_cast<unsigned>(rng.below(rng.below(2) ? 4 : 120));   // (a cubin of many kernels, or none)
    for (unsigned i = 0; i < n; ++i) {
      const size_t count = rng.below(5) == 0 ? 0 : 1 + rng.below(rng.below(2) ? 4 : 600);
      add(m, va, count);
      va += (16 * count + 0xff) & ~uint64_t{0xff};   // exactly load()'s placement
    }
    m.index_code();
    compare(m, !m.code.empty());
  }
}

VTEST(code_at_empty_module) {
  Module m;
  m.index_code();
  VCHECK(m.code_at(0) == nullptr);
  VCHECK(m.code_at(kBase) == nullptr);
}

// A module whose sections are out of order, overlapping, or wrapping the address space keeps no index and the
// lookup is the scan.
VTEST(code_at_without_an_index_scans) {
  {
    Module m;
    add(m, kBase + 0x1000, 8);
    add(m, kBase, 8);   // out of order
    m.index_code();
    compare(m, false);
  }
  {
    Module m;
    add(m, kBase, 100);
    add(m, kBase + 0x100, 8);   // overlaps the first
    m.index_code();
    compare(m, false);
  }
  {
    Module m;
    add(m, ~uint64_t{0} - 31, 8);   // wraps
    add(m, ~uint64_t{0} - 15, 1);
    m.index_code();
    compare(m, false);
  }
  {
    Module m;   // never indexed at all
    add(m, kBase, 8);
    add(m, kBase + 0x100, 8);
    compare(m, false);
    VCHECK(m.code_at(kBase + 0x104) == &m.code[1]);
  }
}

VTEST_MAIN
