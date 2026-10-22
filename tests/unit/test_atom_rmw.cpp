// Round 10: the PTX interpreter's global atomics, whole kernels with VGPU_FASTPATH on and off.
//
// The single-lookup read-modify-write behind them (MemoryManager::rmw_scalar) has its own memory-level test, test_memrmw.cpp.
// These run 13 atom/red forms as PTX kernels, on memory in each of its states (untouched, uniform, real), and require the same
// old values, the same memory afterwards, the same resident bytes and the same error text with the fast path on and off.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

#define VT_ASSERT(cond) do { if (!(cond)) throw vtest::Failure("not covered enough: " #cond); } while (0)

namespace {

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
  uint64_t below(uint64_t n) { return next() % n; }
};

// The memory in one of its four states: untouched (zero), uniform (a memset), real (written), mixed.
void prepare(MemoryManager& m, uint64_t base, uint64_t bytes, int state, Rng& rng) {
  const uint8_t pats[] = {0x00, 0x5A, 0xFF};
  switch (state) {
    case 0: break;
    case 1: {
      const uint8_t p = pats[rng.below(3)];
      m.fill(base, &p, 1, bytes);
      break;
    }
    case 2: {
      std::vector<uint8_t> v(bytes);
      for (auto& b : v) b = static_cast<uint8_t>(rng.next());
      m.write(base, v.data(), bytes);
      break;
    }
    default: {   // some chunks uniform, some real, some untouched
      const uint8_t p = 0x11;
      for (uint64_t off = 0; off < bytes; off += kChunkSize) {
        const uint64_t n = std::min<uint64_t>(kChunkSize, bytes - off);
        if (off / kChunkSize % 3 == 0) m.fill(base + off, &p, 1, n);
        else if (off / kChunkSize % 3 == 1) {
          std::vector<uint8_t> v(n);
          for (auto& b : v) b = static_cast<uint8_t>(rng.next());
          m.write(base + off, v.data(), n);
        }
      }
    }
  }
}

}  // namespace

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct KernelResult {
  std::string error;
  std::vector<uint64_t> old;
  std::vector<uint8_t> memory;
  uint64_t resident = 0;
  bool operator==(const KernelResult& o) const {
    return error == o.error && old == o.old && memory == o.memory && resident == o.resident;
  }
};

// 64 threads each run `body` (an atom/red on [%rd7] with %r10 / %rd10 operands) at base + tid * stride + offset; the old values
// are stored to a result buffer.
KernelResult run_atoms(const std::string& body, bool wide, uint32_t stride, uint32_t offset, int state, uint32_t span) {
  std::string ptx = R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 po)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [pa];
    ld.param.u64 %rd2, [po];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd5, %r1, )" + std::to_string(stride) + R"(;
    add.u64 %rd7, %rd1, %rd5;
    add.u64 %rd7, %rd7, )" + std::to_string(offset) + R"(;
    add.u32 %r10, %r1, 3;
    cvt.u64.u32 %rd10, %r10;
    mul.lo.u32 %r11, %r10, 7;
    cvt.rn.f32.u32 %f1, %r10;
    )" + body + R"(
    mul.wide.u32 %rd8, %r1, 8;
    add.u64 %rd9, %rd2, %rd8;
    )" + (wide ? "st.global.u64 [%rd9], %rd11;" : "cvt.u64.u32 %rd11, %r12;\n    st.global.u64 [%rd9], %rd11;") + R"(
    ret;
}
)";
  KernelResult res;
  MemoryManager mem{1 << 26};
  DeviceProfile prof = load_gpu("nvidia/a10");
  auto m = ptx::parse(ptx);
  const uint64_t bytes = span;
  const uint64_t pa = mem.alloc(bytes), po = mem.alloc(64 * 8);
  Rng r{12345};
  prepare(mem, pa, bytes, state, r);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  try {
    exec::launch(m.entries[0], cfg, {arg_u64(pa), arg_u64(po)}, mem, prof);
  } catch (const Error& e) {
    std::string t = e.what();
    char hex[40];
    std::snprintf(hex, sizeof hex, "%llx", static_cast<unsigned long long>(pa));
    for (size_t p; (p = t.find(hex)) != std::string::npos;) t.replace(p, std::strlen(hex), "BASE");
    res.error = t;
  }
  res.old.resize(64);
  mem.read(po, res.old.data(), 64 * 8);
  res.memory.resize(bytes);
  mem.read(pa, res.memory.data(), bytes);
  res.resident = mem.resident_bytes();
  return res;
}

}  // namespace

VTEST(ptx_global_atomics_match_the_two_access_path) {
  struct Form { const char* body; bool wide; uint32_t width; };
  const Form forms[] = {
      {"atom.global.add.u32 %r12, [%rd7], %r10;", false, 4},
      {"atom.global.exch.b32 %r12, [%rd7], %r11;", false, 4},
      {"atom.global.min.s32 %r12, [%rd7], %r11;", false, 4},
      {"atom.global.max.u32 %r12, [%rd7], %r11;", false, 4},
      {"atom.global.and.b32 %r12, [%rd7], 0;", false, 4},
      {"atom.global.xor.b32 %r12, [%rd7], 0;", false, 4},
      {"atom.global.or.b32 %r12, [%rd7], 0xffffffff;", false, 4},
      {"atom.global.cas.b32 %r12, [%rd7], 0, %r11;", false, 4},
      {"atom.global.inc.u32 %r12, [%rd7], 5;", false, 4},
      {"atom.global.add.u64 %rd11, [%rd7], %rd10;", true, 8},
      {"atom.global.exch.b64 %rd11, [%rd7], %rd10;", true, 8},
      {"atom.global.add.f32 %f2, [%rd7], %f1; mov.b32 %r12, %f2;", false, 4},
      {"red.global.add.u32 [%rd7], %r10; mov.u32 %r12, 0;", false, 4},
  };
  int ran = 0, failed = 0;
  const uint32_t span = 3 * 65536;
  for (const Form& f : forms)
    for (uint32_t stride : {0u, 4u, 8u, 12u, 260u, 1024u})
      for (uint32_t offset : {0u, 2u, 64u, span - 64 * 4 - 4, span - 16})   // misaligned, near the end, running off it
        for (int state : {0, 1, 2, 3}) {
          if (f.width == 8 && stride % 8 && stride != 0) continue;
          setenv("VGPU_FASTPATH", "1", 1);
          const KernelResult fast = run_atoms(f.body, f.wide, stride, offset, state, span);
          setenv("VGPU_FASTPATH", "0", 1);
          const KernelResult slow = run_atoms(f.body, f.wide, stride, offset, state, span);
          unsetenv("VGPU_FASTPATH");
          ++ran;
          failed += !fast.error.empty();
          if (!(fast == slow)) {
            std::string what = std::string("'") + f.body + "' stride " + std::to_string(stride) + " offset " + std::to_string(offset) +
                               " state " + std::to_string(state) + ": ";
            if (fast.error != slow.error) what += "error '" + fast.error + "' vs '" + slow.error + "'";
            else if (fast.old != slow.old) what += "returned values differ";
            else if (fast.memory != slow.memory) what += "memory differs";
            else what += "resident bytes " + std::to_string(fast.resident) + " vs " + std::to_string(slow.resident);
            throw vtest::Failure(what);
          }
        }
  VT_ASSERT(ran > 1000 && failed > 20);   // some of the cases are errors, and the same ones
}

VTEST_MAIN
