// Round 10: the single-lookup read-modify-write behind the PTX interpreter's global atomics against the two-access code it replaces.
//
// A device atomic used to be load_scalar then store_scalar (two table lookups); now MemoryManager::rmw_begin / rmw_commit
// serve the ordinary case with one, and decline -- changing nothing -- whenever the two accesses could differ. These run
// every operation both ways, on memory in each of its states (untouched, uniform, real), and require the same value, the same memory afterwards, the same resident
// bytes and the same error text, for accesses inside an allocation, misaligned, past its end, with a fault armed on
// loads or on stores, and for whole PTX kernels with VGPU_FASTPATH on and off.
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

// A fault that changes what a load returns / what a store writes, so a path that skipped the hook would show.
struct Faults : MemoryManager::AccessFault {
  uint64_t loads_armed = 0, stores_armed = 0;
  int load_calls = 0, store_calls = 0;
  Faults() { load_pending = &loads_armed, store_pending = &stores_armed; }
  uint64_t on_load(uint64_t, uint32_t, uint64_t value) override { ++load_calls; return value ^ 0xFF; }
  uint64_t on_store(uint64_t, uint32_t, uint64_t value) override { ++store_calls; return value ^ 0x100; }
  uint64_t on_shared_load(uint64_t, uint32_t, uint64_t value) override { return value; }
  void on_read(uint64_t, uint8_t*, uint64_t) override {}
  uint64_t on_alu(uint64_t value, uint32_t) override { return value; }
  void on_copy(uint64_t, uint8_t*, uint64_t) override {}
};

struct Outcome {
  bool failed = false;
  int code = -1;
  std::string text;
  uint64_t v = 0;
  bool operator==(const Outcome& o) const { return failed == o.failed && code == o.code && text == o.text && v == o.v; }
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

// What a device atomic did before: load_scalar, compute, store_scalar.
Outcome rmw_old(MemoryManager& m, uint64_t addr, uint32_t size, uint64_t operand) {
  Outcome o;
  try {
    o.v = m.load_scalar(addr, size);
    m.store_scalar(addr, size, o.v + operand);
  } catch (const Error& e) {
    o.failed = true, o.code = static_cast<int>(e.code()), o.text = e.what();
  }
  return o;
}

// What it does now (the interpreter's sequence).
Outcome rmw_new(MemoryManager& m, uint64_t addr, uint32_t size, uint64_t operand) {
  Outcome o;
  try {
    MemoryManager::ScalarRmw at;
    uint64_t old = 0;
    if (m.rmw_begin(addr, size, at, &old)) {
      o.v = old;
      m.rmw_commit(at, size, old + operand);
    } else {
      o.v = m.load_scalar(addr, size);
      m.store_scalar(addr, size, o.v + operand);
    }
  } catch (const Error& e) {
    o.failed = true, o.code = static_cast<int>(e.code()), o.text = e.what();
  }
  return o;
}

std::vector<uint8_t> dump(const MemoryManager& m, uint64_t base, uint64_t bytes) {
  std::vector<uint8_t> v(bytes);
  m.read(base, v.data(), bytes);
  return v;
}

}  // namespace

VTEST(rmw_matches_load_then_store_on_every_state_of_memory) {
  Rng rng{0x9E3779B97F4A7C15ull};
  int failed = 0, declined = 0, taken = 0;
  for (int iter = 0; iter < 4000; ++iter) {
    const uint64_t bytes = (1 + rng.below(3)) * kChunkSize;
    MemoryManager a{1ull << 30}, b{1ull << 30};
    const uint64_t pa = a.alloc(bytes), pb = b.alloc(bytes);
    Rng r1{static_cast<uint64_t>(iter) * 7919 + 1}, r2 = r1;
    const int state = iter % 4;
    prepare(a, pa, bytes, state, r1);
    prepare(b, pb, bytes, state, r2);
    Faults fa, fb;
    const int mode = static_cast<int>(rng.below(8));   // 0..4 no fault, 5 loads, 6 stores, 7 both
    if (mode >= 5) {
      a.set_access_fault(&fa), b.set_access_fault(&fb);
      if (mode != 6) fa.loads_armed = fb.loads_armed = 1;
      if (mode != 5) fa.stores_armed = fb.stores_armed = 1;
    } else if (mode == 4) {   // attached, not armed: the runtime's usual state
      a.set_access_fault(&fa), b.set_access_fault(&fb);
    }
    for (int op = 0; op < 24; ++op) {
      static const uint32_t sizes[] = {1, 2, 4, 8, 4, 8, 4, 3};
      const uint32_t size = sizes[rng.below(8)];
      uint64_t off = rng.below(bytes + 64);   // past the end now and then
      switch (rng.below(5)) {
        case 0: off &= ~uint64_t{size - 1}; break;                 // aligned
        case 1: off = (off & ~uint64_t{size - 1}) + (size > 1 ? 1 : 0); break;   // misaligned
        case 2: off = (rng.below(bytes / kChunkSize + 1)) * kChunkSize - size; break;   // at a chunk's end
        default: off &= ~uint64_t{size - 1}; break;
      }
      const uint64_t operand = rng.next() | (rng.below(3) == 0 ? 0 : 1);
      const Outcome x = rmw_old(a, pa + off, size, operand);
      const Outcome y = rmw_new(b, pb + off, size, operand);
      // (the allocation bases differ, so error text naming an address is compared with it removed)
      Outcome xs = x, ys = y;
      auto strip = [](std::string& s, uint64_t base) {
        char hex[40];
        std::snprintf(hex, sizeof hex, "%llx", static_cast<unsigned long long>(base));
        for (size_t p; (p = s.find(hex)) != std::string::npos;) s.replace(p, std::strlen(hex), "BASE");
      };
      strip(xs.text, pa), strip(ys.text, pb);
      if (!(xs == ys)) {
        throw vtest::Failure("iter " + std::to_string(iter) + " op " + std::to_string(op) + " size " + std::to_string(size) +
                             " off " + std::to_string(off) + ": old (" + std::to_string(x.failed) + ", " + std::to_string(x.v) +
                             ", '" + xs.text + "') vs new (" + std::to_string(y.failed) + ", " + std::to_string(y.v) + ", '" + ys.text + "')");
      }
      failed += x.failed;
      if (mode >= 5) ++declined;
      else ++taken;
    }
    if (dump(a, pa, bytes) != dump(b, pb, bytes)) throw vtest::Failure("iter " + std::to_string(iter) + ": memory differs afterwards");
    if (a.resident_bytes() != b.resident_bytes())
      throw vtest::Failure("iter " + std::to_string(iter) + ": resident bytes " + std::to_string(a.resident_bytes()) + " vs " +
                           std::to_string(b.resident_bytes()));
    if (fa.load_calls != fb.load_calls || fa.store_calls != fb.store_calls)
      throw vtest::Failure("iter " + std::to_string(iter) + ": fault hooks called " + std::to_string(fa.load_calls) + "/" +
                           std::to_string(fa.store_calls) + " times vs " + std::to_string(fb.load_calls) + "/" + std::to_string(fb.store_calls));
  }
  VT_ASSERT(failed > 100 && declined > 1000 && taken > 1000);   // the error paths and both ways were exercised
}

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
