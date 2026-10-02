// Host-level mirrors of behaviour that was otherwise checked only by the
// nvcc-built end-to-end programs in nvidia/tests/e2e. Those need a CUDA
// toolkit and a long build before a regression shows; these need neither --
// hand-written PTX, run through the runtime's Device API on the core library.
//
// Each test names the e2e program it mirrors. Expected values come from that
// program's own checks (an RTX 3060's output where the program records one) or
// from CUDA's documented semantics, never from running the simulator.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
std::vector<uint8_t> arg_u32(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

LaunchConfig block_of(uint32_t threads, uint32_t blocks = 1) {
  LaunchConfig cfg;
  cfg.grid = {blocks, 1, 1};
  cfg.block = {threads, 1, 1};
  return cfg;
}

// A module global's current address (and, optionally, its declared size).
uint64_t global_of(runtime::Device& dev, uint64_t mod, const std::string& name, uint64_t* size = nullptr) {
  uint64_t addr = 0, sz = 0;
  if (!dev.global(mod, name, &addr, &sz)) throw vtest::Failure("module has no global '" + name + "'");
  if (size) *size = sz;
  return addr;
}

uint32_t u32_at(runtime::Device& dev, uint64_t addr) {
  return static_cast<uint32_t>(dev.memory().load_scalar(addr, 4));
}
float f32_at(runtime::Device& dev, uint64_t addr) {
  const uint32_t bits = u32_at(dev, addr);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
double f64_at(runtime::Device& dev, uint64_t addr) {
  const uint64_t bits = dev.memory().load_scalar(addr, 8);
  double d;
  std::memcpy(&d, &bits, 8);
  return d;
}

// What `fn` writes to the process's stdout, where device printf goes.
template <class F>
std::string capture_stdout(F&& fn) {
  std::fflush(stdout);
  FILE* tmp = std::tmpfile();
  if (!tmp) throw vtest::Failure("tmpfile() failed");
  const int saved = dup(1);
  dup2(fileno(tmp), 1);
  auto restore = [&] {
    std::fflush(stdout);
    dup2(saved, 1);
    close(saved);
  };
  try {
    fn();
  } catch (...) {
    restore();
    std::fclose(tmp);
    throw;
  }
  restore();
  std::string text;
  std::rewind(tmp);
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) text.append(buf, n);
  std::fclose(tmp);
  return text;
}

// A printf argument buffer laid out as nvcc lays it: each argument at its
// natural alignment, ints (and promoted chars and shorts) as 4 bytes, floats
// promoted to double, pointers and long longs as 8.
struct VaList {
  std::vector<uint8_t> bytes;
  template <class T>
  VaList& add(T v) {
    const size_t at = (bytes.size() + sizeof(T) - 1) / sizeof(T) * sizeof(T);
    bytes.resize(at + sizeof(T));
    std::memcpy(bytes.data() + at, &v, sizeof(T));
    return *this;
  }
  VaList& i(int32_t v) { return add(v); }
  VaList& ll(int64_t v) { return add(v); }
  VaList& d(double v) { return add(v); }
  VaList& p(uint64_t v) { return add(v); }
};

// One vprintf call with the format and argument buffer the kernel is handed,
// and the character count it returns stored to `count`.
const char* kPrintfPtx = R"(
.extern .func (.param .b32 func_retval0) vprintf (.param .b64 p0, .param .b64 p1);
.visible .entry pf(.param .u64 fmt, .param .u64 va, .param .u64 count)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [fmt];
    ld.param.u64 %rd2, [va];
    ld.param.u64 %rd3, [count];
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd1;
    .param .b64 param1;
    st.param.b64 [param1+0], %rd2;
    .param .b32 retval0;
    call.uni (retval0), vprintf, (param0, param1);
    ld.param.b32 %r1, [retval0+0];
    }
    st.global.u32 [%rd3], %r1;
    ret;
}
)";

struct DevicePrintf {
  runtime::Runtime rt{load_gpu("nvidia/rtx3060")};
  runtime::Device& dev = rt.device(0);
  uint64_t mod = dev.load_module(std::string(kHeader) + kPrintfPtx);
  const ptx::EntryFn* fn = dev.get_function(mod, "pf");
  uint64_t count = dev.memory().alloc(4);

  uint64_t str(const std::string& s) {
    const uint64_t a = dev.memory().alloc(s.size() + 1);
    dev.memory().write(a, s.c_str(), s.size() + 1);
    return a;
  }
  // The text one device printf prints; also checks that vprintf returned its length.
  std::string operator()(const std::string& fmt, const VaList& va) {
    const uint64_t f = str(fmt);
    uint64_t v = 0;
    if (!va.bytes.empty()) {
      v = dev.memory().alloc(va.bytes.size());
      dev.memory().write(v, va.bytes.data(), va.bytes.size());
    }
    const std::string out = capture_stdout(
        [&] { dev.launch(*fn, LaunchConfig{}, {arg_u64(f), arg_u64(v), arg_u64(count)}, dev.symbols(mod)); });
    VCHECK_EQ(u32_at(dev, count), static_cast<uint32_t>(out.size()));
    return out;
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Mirrors printf_formats.cu / printf_formats.expected: device printf, line by
// line, against the text an RTX 3060 prints for the same calls.
VTEST(device_printf_formats_as_the_card_prints_them) {
  DevicePrintf pf;
  const uint64_t s = pf.str("hello");
  VCHECK_EQ(pf("a %a %A %.2a %10.1a %a|\n", VaList().d(3.0).d(-0.1).d(1.0).d(2.5).d(0.0)),
            std::string("a 0x1.8p+1 -0X1.999999999999AP-4 0x1.00p+0   0x1.4p+1 0x0p+0|\n"));
  VCHECK_EQ(pf("b %p %p %-12p|\n", VaList().p(0).p(0x1234).p(0x1234)),
            std::string("b (nil) 0x1234 0x1234      |\n"));
  VCHECK_EQ(pf("c %.2s|%8s|%-8s|%8.3s|%s|\n", VaList().p(s).p(s).p(s).p(s).p(0)),
            std::string("c he|   hello|hello   |     hel|(null)|\n"));
  VCHECK_EQ(pf("d %5c|%-5c|%c|\n", VaList().i('x').i('y').i(65)), std::string("d     x|y    |A|\n"));
  VCHECK_EQ(pf("e %*d|%-*d|%0*d|\n", VaList().i(5).i(1).i(5).i(2).i(5).i(3)),
            std::string("e     1|2    |00003|\n"));
  // A conversion the card does not know is printed as written and takes no argument.
  VCHECK_EQ(pf("f %k %d %q|\n", VaList().i(7)), std::string("f %k 7 %q|\n"));
  VCHECK_EQ(pf("i %hd %hu|\n", VaList().i(70000).i(70000)), std::string("i 4464 4464|\n"));
  VCHECK_EQ(pf("j %lld %llx %ld %lu|\n", VaList().ll(-1234567890123LL).ll(-1234567890123LL).ll(-5).ll(5)),
            std::string("j -1234567890123 fffffee08e04fb35 -5 5|\n"));
  // Six ints, then two doubles at the next 8-byte boundary.
  VCHECK_EQ(pf("k %+d % d %-6d| %06d %#x %#o %.0f %#.0f|\n",
               VaList().i(300).i(300).i(300).i(300).i(300).i(300).d(3.14).d(3.14)),
            std::string("k +300  300 300   | 000300 0x12c 0454 3 3.|\n"));
  VCHECK_EQ(pf("l %g %G %e %E %f %F|\n",
               VaList().d(3.14159265358979).d(3.14159265358979e20).d(3.14).d(3.14).d(2.5).d(-2.5)),
            std::string("l 3.14159 3.14159E+20 3.140000e+00 3.140000E+00 2.500000 -2.500000|\n"));
  const double inf = std::numeric_limits<double>::infinity();
  VCHECK_EQ(pf("m %f %f %e|\n", VaList().d(inf).d(-inf).d(-0.0)), std::string("m inf -inf -0.000000e+00|\n"));
  VCHECK_EQ(pf("n %lf %i %o %X %u|\n", VaList().d(2.5).i(-300).i(300).i(300).i(-300)),
            std::string("n 2.500000 -300 454 12C 4294966996|\n"));
}

// ---------------------------------------------------------------------------
// Mirrors device_intrinsics.cu (k_incdec, k_dec): atomicInc and atomicDec wrap
// against their operand rather than counting. CUDA: inc stores
// (old >= limit) ? 0 : old + 1; dec stores (old == 0 || old > limit) ? limit : old - 1.
VTEST(atomic_inc_and_dec_wrap_at_their_limit) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + R"(
.visible .entry incdec(.param .u64 out, .param .u32 limit, .param .u32 dec)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [limit];
    ld.param.u32 %r2, [dec];
    mov.u32 %r3, %tid.x;
    setp.ne.u32 %p1, %r2, 0;
    @%p1 bra DEC;
    atom.global.inc.u32 %r4, [%rd1], %r1;
    bra.uni STORE;
DEC:
    atom.global.dec.u32 %r4, [%rd1], %r1;
STORE:
    add.u32 %r5, %r3, 1;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r4;
    ret;
}
)");
  const ptx::EntryFn* fn = dev.get_function(mod, "incdec");
  const uint64_t out = dev.memory().alloc(64);
  auto run = [&](uint32_t start, uint32_t limit, bool dec, uint32_t threads) {
    dev.memory().store_scalar(out, 4, start);
    dev.launch(*fn, block_of(threads), {arg_u64(out), arg_u32(limit), arg_u32(dec ? 1 : 0)});
  };
  // Ten threads against a limit of 3 cycle 0,1,2,3,0,1,... (the e2e's check).
  run(0, 3, false, 10);
  for (uint32_t i = 0; i < 10; ++i) VCHECK_EQ(u32_at(dev, out + 4 * (1 + i)), i % 4);
  VCHECK_EQ(u32_at(dev, out), 2u);
  // From 5 with a limit of 5, three threads see 5, 4, 3.
  run(5, 5, true, 3);
  VCHECK_EQ(u32_at(dev, out + 4), 5u);
  VCHECK_EQ(u32_at(dev, out + 8), 4u);
  VCHECK_EQ(u32_at(dev, out + 12), 3u);
  VCHECK_EQ(u32_at(dev, out), 2u);
  // The wrap edges: dec from 0 and from above the limit both land on it, and
  // inc from above the limit lands on 0.
  run(0, 5, true, 1);
  VCHECK_EQ(u32_at(dev, out + 4), 0u);
  VCHECK_EQ(u32_at(dev, out), 5u);
  run(9, 5, true, 1);
  VCHECK_EQ(u32_at(dev, out + 4), 9u);
  VCHECK_EQ(u32_at(dev, out), 5u);
  run(7, 3, false, 1);
  VCHECK_EQ(u32_at(dev, out + 4), 7u);
  VCHECK_EQ(u32_at(dev, out), 0u);
}

// ---------------------------------------------------------------------------
// The device heap: malloc and free in a kernel.
namespace {
const char* kHeapDecls = R"(
.extern .func (.param .b64 func_retval0) malloc (.param .b64 size);
.extern .func free (.param .b64 ptr);
)";
}  // namespace

// Mirrors device_intrinsics.cu (k_malloc): a kernel mallocs 16 ints, fills
// them with i*3, sums them and frees the block.
VTEST(device_malloc_gives_usable_memory_and_free_returns_it) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + kHeapDecls + R"(
.visible .entry use_heap(.param .u64 out)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    mov.u64 %rd5, 64;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd5;
    .param .b64 retval0;
    call.uni (retval0), malloc, (param0);
    ld.param.b64 %rd2, [retval0+0];
    }
    setp.ne.s64 %p1, %rd2, 0;
    @%p1 bra GOT;
    mov.u32 %r5, -1;
    st.global.u32 [%rd1], %r5;
    ret;
GOT:
    mov.u32 %r1, 0;
FILL:
    mul.lo.s32 %r2, %r1, 3;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.u32 [%rd4], %r2;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p2, %r1, 16;
    @%p2 bra FILL;
    mov.u32 %r1, 0;
    mov.u32 %r3, 0;
SUM:
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.u32 %r4, [%rd4];
    add.s32 %r3, %r3, %r4;
    add.u32 %r1, %r1, 1;
    setp.lt.u32 %p2, %r1, 16;
    @%p2 bra SUM;
    st.global.u32 [%rd1], %r3;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd2;
    call.uni free, (param0);
    }
    ret;
}
)");
  const uint64_t out = dev.memory().alloc(4);
  const uint64_t used_before = dev.memory().used();
  dev.launch(*dev.get_function(mod, "use_heap"), LaunchConfig{}, {arg_u64(out)});
  VCHECK_EQ(u32_at(dev, out), 3u * (15 * 16 / 2));
  VCHECK_EQ(dev.memory().used(), used_before);   // the free gave it back
}

// Mirrors device_limits.cu ("the heap"): cudaLimitMallocHeapSize bounds what
// malloc in a kernel can hand out. Forty threads each ask for 1 MiB and hold it
// until all have asked: a 32 MiB heap satisfies 32 of them and gives the rest
// null, the default 8 MiB heap satisfies 8.
VTEST(device_malloc_is_bounded_by_the_heap_limit) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + kHeapDecls + R"(
.visible .entry grab(.param .u64 got, .param .u32 n)
{
    .reg .pred %p<4>;
    .reg .b16 %rs<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [got];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    mov.u64 %rd2, 0;
    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra SYNC;
    mov.u64 %rd3, 1048576;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd3;
    .param .b64 retval0;
    call.uni (retval0), malloc, (param0);
    ld.param.b64 %rd2, [retval0+0];
    }
    setp.ne.s64 %p2, %rd2, 0;
    selp.u32 %r6, 1, 0, %p2;
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    st.global.u32 [%rd5], %r6;
    @!%p2 bra SYNC;
    mov.u16 %rs1, 1;
    st.u8 [%rd2+1048575], %rs1;
SYNC:
    bar.sync 0;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd2;
    call.uni free, (param0);
    }
    ret;
}
)");
  const ptx::EntryFn* fn = dev.get_function(mod, "grab");
  constexpr uint32_t kThreads = 40;
  const uint64_t got = dev.memory().alloc(kThreads * 4);
  const uint64_t used_before = dev.memory().used();
  auto succeeded = [&](uint64_t heap_bytes) {
    const uint8_t zero = 0;
    dev.memory().fill(got, &zero, 1, kThreads * 4);
    LaunchConfig cfg = block_of(kThreads);
    cfg.device_heap_bytes = heap_bytes;
    dev.launch(*fn, cfg, {arg_u64(got), arg_u32(kThreads)});
    uint32_t n = 0;
    for (uint32_t i = 0; i < kThreads; ++i) n += u32_at(dev, got + 4 * i);
    return n;
  };
  VCHECK_EQ(succeeded(32ull << 20), 32u);
  VCHECK_EQ(succeeded(LaunchConfig{}.device_heap_bytes), 8u);   // the 8 MiB default
  VCHECK_EQ(LaunchConfig{}.device_heap_bytes, 8ull << 20);
  VCHECK_EQ(dev.memory().used(), used_before);   // every block was freed
}

// ---------------------------------------------------------------------------
// Mirrors cooperative_grid.cu: cg::this_grid().sync() as cooperative_groups
// compiles it. The workspace address comes from %envreg1 (high half) and
// %envreg2 (low half), reassembled with bfi; a null workspace traps. Thread 0
// of each block adds to the barrier word in the workspace -- block 0 adds
// 0x80000000 - (blocks - 1), every other block 1 -- and spins until bit 31 of
// the word differs from what its own add saw.
namespace {
const char* kGridSyncPtx = R"(
.visible .entry gridsync(.param .u64 x, .param .u64 y, .param .u32 n)
{
    .reg .pred %p<8>;
    .reg .b32 %r<24>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<16>;
    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [y];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p1, %r5, %r1;
    mul.wide.s32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    @%p1 bra SYNC;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f1, 0f3F800000;
    st.global.f32 [%rd4], %f2;
SYNC:
    mov.u32 %r6, %envreg1;
    mov.u32 %r7, %envreg2;
    cvt.u64.u32 %rd5, %r6;
    cvt.u64.u32 %rd6, %r7;
    bfi.b64 %rd7, %rd5, %rd6, 32, 32;
    setp.ne.s64 %p2, %rd7, 0;
    @%p2 bra VALID;
    trap;
VALID:
    bar.sync 0;
    setp.ne.u32 %p3, %r4, 0;
    @%p3 bra WAITED;
    add.s64 %rd8, %rd7, 4;
    mov.u32 %r9, 1;
    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra ARRIVE;
    mov.u32 %r8, %nctaid.x;
    mov.u32 %r10, -2147483647;
    sub.s32 %r9, %r10, %r8;
ARRIVE:
    membar.gl;
    atom.global.add.u32 %r11, [%rd8], %r9;
SPIN:
    ld.volatile.global.u32 %r12, [%rd8];
    xor.b32 %r13, %r12, %r11;
    and.b32 %r14, %r13, -2147483648;
    setp.eq.u32 %p5, %r14, 0;
    @%p5 bra SPIN;
    membar.gl;
WAITED:
    bar.sync 0;
    @%p1 bra DONE;
    add.s32 %r15, %r5, 64;
    rem.s32 %r16, %r15, %r1;
    mul.wide.s32 %rd9, %r16, 4;
    add.s64 %rd10, %rd1, %rd9;
    ld.global.f32 %f3, [%rd10];
    add.f32 %f4, %f3, %f3;
    add.s64 %rd11, %rd2, %rd3;
    st.global.f32 [%rd11], %f4;
DONE:
    ret;
}
)";
}  // namespace

VTEST(a_cooperative_launch_gives_grid_sync_its_workspace) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + kGridSyncPtx);
  const ptx::EntryFn* fn = dev.get_function(mod, "gridsync");
  constexpr uint32_t n = 256, threads = 64, blocks = n / threads;
  const uint64_t x = dev.memory().alloc(n * 4);
  const uint64_t y = dev.memory().alloc(n * 4);
  std::vector<float> h(n);
  for (uint32_t i = 0; i < n; ++i) h[i] = static_cast<float>(i);
  dev.memory().write(x, h.data(), n * 4);

  LaunchConfig cfg = block_of(threads, blocks);
  cfg.cooperative = true;
  dev.launch(*fn, cfg, {arg_u64(x), arg_u64(y), arg_u32(n)});
  std::vector<float> got(n);
  dev.memory().read(y, got.data(), n * 4);
  uint32_t wrong = 0;
  for (uint32_t i = 0; i < n; ++i)
    if (got[i] != (static_cast<float>((i + 64) % n) + 1.0f) * 2.0f) ++wrong;
  VCHECK_EQ(wrong, 0u);   // each thread read another block's update, after it was made

  // The same kernel launched ordinarily has no workspace: the generated code
  // traps rather than computing something else.
  cfg.cooperative = false;
  auto err = VCAPTURE(Error, dev.launch(*fn, cfg, {arg_u64(x), arg_u64(y), arg_u32(n)}));
  VCHECK(err.code() == Err::Trap);
}

// ---------------------------------------------------------------------------
// Mirrors symbols.cu: __constant__ and __device__ variables reached from the
// host by name (cudaMemcpyToSymbol / FromSymbol are this lookup plus a copy),
// read in the kernel with ld.const and updated with an atomic.
VTEST(constant_and_device_variables_are_reached_by_name) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + R"(
.const .align 4 .b8 g_coeff[64];
.const .align 4 .u32 g_scale;
.global .align 4 .u32 g_counter;
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    add.u32 %r2, %r1, 1;
    cvt.rn.f32.u32 %f1, %r2;
    mov.f32 %f2, 0f00000000;
    mov.u64 %rd2, g_coeff;
    mov.u32 %r3, 0;
LOOP:
    ld.const.f32 %f3, [%rd2];
    fma.rn.f32 %f2, %f3, %f1, %f2;
    add.s64 %rd2, %rd2, 4;
    add.u32 %r3, %r3, 1;
    setp.lt.u32 %p1, %r3, 16;
    @%p1 bra LOOP;
    ld.const.u32 %r4, [g_scale];
    cvt.rn.f32.s32 %f4, %r4;
    mul.f32 %f5, %f2, %f4;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3;
    st.global.f32 [%rd4], %f5;
    atom.global.add.u32 %r5, [g_counter], 1;
    ret;
}
)");
  uint64_t coeff_size = 0, scale_size = 0, counter_size = 0;
  const uint64_t coeff = global_of(dev, mod, "g_coeff", &coeff_size);
  const uint64_t scale = global_of(dev, mod, "g_scale", &scale_size);
  const uint64_t counter = global_of(dev, mod, "g_counter", &counter_size);
  VCHECK_EQ(coeff_size, uint64_t{64});
  VCHECK_EQ(scale_size, uint64_t{4});
  VCHECK_EQ(counter_size, uint64_t{4});
  VCHECK(!dev.global(mod, "no_such_symbol", nullptr, nullptr));

  float c[16];
  for (int i = 0; i < 16; ++i) c[i] = static_cast<float>(i + 1);
  dev.memory().write(coeff, c, sizeof c);
  dev.memory().store_scalar(scale, 4, 3);
  dev.memory().store_scalar(counter, 4, 0);
  const uint64_t out = dev.memory().alloc(32 * 4);
  dev.launch(*dev.get_function(mod, "k"), block_of(32), {arg_u64(out)}, dev.symbols(mod));

  const float coeff_sum = 16.0f * 17.0f / 2.0f;   // 1..16
  for (uint32_t t = 0; t < 32; ++t)
    VCHECK_EQ(f32_at(dev, out + 4 * t), coeff_sum * static_cast<float>(t + 1) * 3.0f);
  VCHECK_EQ(u32_at(dev, counter), 32u);
}

// ---------------------------------------------------------------------------
// Mirrors managed_vars.cu: __managed__ variables start from their
// initialisers, and once the runtime moves each onto memory the host shares
// (Device::rebind_global), kernels use the new address -- a host write before
// the launch is what the kernel sees, and the kernel's updates are what the
// host reads after.
VTEST(managed_variables_move_to_shared_memory_and_kernels_follow) {
  runtime::Runtime rt(load_gpu("nvidia/rtx3060"));
  auto& dev = rt.device(0);
  // table = {1, 2, 3, 4} as floats; pair = {int 7, double 0.5}.
  const uint64_t mod = dev.load_module(std::string(kHeader) + R"(
.global .attribute(.managed) .align 4 .u32 counter = 5;
.global .attribute(.managed) .align 4 .b8 table[16] = {0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64};
.global .attribute(.managed) .align 8 .b8 pair[16] = {7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 63};
.global .attribute(.managed) .align 4 .u32 untouched_by_host = 11;
.global .align 4 .u32 plain = 3;
.global .align 8 .u64 counter_ptr = counter;
.visible .entry bump(.param .u32 by)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    .reg .f32 %f<4>;
    .reg .f64 %fd<4>;
    .reg .b64 %rd<6>;
    ld.param.u32 %r1, [by];
    mov.u64 %rd1, counter;
    atom.global.add.u32 %r2, [%rd1], %r1;
    mov.u32 %r3, %tid.x;
    mov.u64 %rd2, table;
    mul.wide.u32 %rd3, %r3, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f1, %f1;
    st.global.f32 [%rd4], %f2;
    setp.ne.u32 %p1, %r3, 0;
    @%p1 bra DONE;
    ld.global.u32 %r4, [pair];
    ld.global.f64 %fd1, [pair+8];
    setp.gt.f64 %p2, %fd1, 0d0000000000000000;
    selp.s32 %r5, 1, -1, %p2;
    add.s32 %r6, %r4, %r5;
    st.global.u32 [pair], %r6;
    mul.f64 %fd2, %fd1, 0d4010000000000000;
    st.global.f64 [pair+8], %fd2;
    ld.global.u32 %r7, [untouched_by_host];
    add.s32 %r7, %r7, 1;
    st.global.u32 [untouched_by_host], %r7;
DONE:
    ret;
}
)");
  const std::vector<std::string> managed = dev.managed_globals(mod);
  VCHECK_EQ(managed.size(), size_t{4});
  VCHECK_EQ(managed[0], std::string("counter"));
  VCHECK_EQ(managed[1], std::string("table"));
  VCHECK_EQ(managed[2], std::string("pair"));
  VCHECK_EQ(managed[3], std::string("untouched_by_host"));

  // Initial values, before any kernel has run.
  const uint64_t old_counter = global_of(dev, mod, "counter");
  VCHECK_EQ(u32_at(dev, old_counter), 5u);
  VCHECK_EQ(f32_at(dev, global_of(dev, mod, "table") + 12), 4.0f);
  VCHECK_EQ(u32_at(dev, global_of(dev, mod, "pair")), 7u);
  VCHECK_EQ(f64_at(dev, global_of(dev, mod, "pair") + 8), 0.5);

  // What the runtime does at registration: a new home for each, holding its
  // current contents.
  for (const std::string& name : managed) {
    uint64_t size = 0;
    const uint64_t from = global_of(dev, mod, name, &size);
    std::vector<uint8_t> bytes(size);
    dev.memory().read(from, bytes.data(), size);
    const uint64_t to = dev.memory().alloc(size);
    dev.memory().write(to, bytes.data(), size);
    dev.rebind_global(mod, name, to);
    VCHECK_EQ(global_of(dev, mod, name), to);
  }
  const uint64_t counter = global_of(dev, mod, "counter");
  VCHECK(counter != old_counter);
  // A global initialised with the variable's address now holds the new one.
  VCHECK_EQ(dev.memory().load_scalar(global_of(dev, mod, "counter_ptr"), 8), counter);

  dev.memory().store_scalar(counter, 4, 100);   // the host writes it
  dev.launch(*dev.get_function(mod, "bump"), block_of(4), {arg_u32(3)}, dev.symbols(mod));
  VCHECK_EQ(u32_at(dev, counter), 112u);        // the host's write plus four adds of 3
  VCHECK_EQ(u32_at(dev, old_counter), 5u);      // nothing used the old address
  const uint64_t table = global_of(dev, mod, "table");
  VCHECK_EQ(f32_at(dev, table), 2.0f);
  VCHECK_EQ(f32_at(dev, table + 12), 8.0f);
  const uint64_t pair = global_of(dev, mod, "pair");
  VCHECK_EQ(u32_at(dev, pair), 8u);
  VCHECK_EQ(f64_at(dev, pair + 8), 2.0);
  VCHECK_EQ(u32_at(dev, global_of(dev, mod, "untouched_by_host")), 12u);
  VCHECK_EQ(u32_at(dev, global_of(dev, mod, "plain")), 3u);
}

// ---------------------------------------------------------------------------
// The device runtime's calls, as libcudadevrt implements them on the driver's
// entry points: cudaPeekAtLastError reads the thread's last error,
// cudaGetLastError reads it and sets it back to 0.
namespace {
const char* kDevrtPtx = R"(
.extern .func (.param .b64 func_retval0) __cudaCDP2GetParameterBufferV2
(.param .b64 f, .param .align 4 .b8 g[12], .param .align 4 .b8 b[12], .param .b32 s);
.extern .func (.param .b32 func_retval0) __cudaCDP2LaunchDeviceV2
(.param .b64 buf, .param .b64 stream);
.extern .func (.param .b32 func_retval0) __cuda_syscall_cnpv2GetLastError
(
)
;
.extern .func (.param .b32 func_retval0) __cuda_syscall_cnpv2SetLastError
(
.param .b32 __cuda_syscall_cnpv2SetLastError_param_0
)
;
.extern .func (.param .b32 func_retval0) __cuda_syscall_cnpv2GetDevice (.param .b64 d);
.extern .func (.param .b32 func_retval0) __cuda_syscall_cnpv2GetDeviceCount (.param .b64 c);

.visible .func (.param .b32 func_retval0) cudaPeekAtLastError()
{
    .reg .b32 %r<3>;
    {
    .param .b32 retval0;
    call.uni (retval0), __cuda_syscall_cnpv2GetLastError, ();
    ld.param.b32 %r1, [retval0];
    }
    st.param.b32 [func_retval0], %r1;
    ret;
}

.visible .func (.param .b32 func_retval0) cudaGetLastError()
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    {
    .param .b32 retval0;
    call.uni (retval0), __cuda_syscall_cnpv2GetLastError, ();
    ld.param.b32 %r2, [retval0];
    }
    setp.eq.s32 %p1, %r2, 0;
    @%p1 bra $L__CLEAN;
    {
    .param .b32 param0;
    st.param.b32 [param0], 0;
    .param .b32 retval0;
    call.uni (retval0), __cuda_syscall_cnpv2SetLastError, (param0);
    ld.param.b32 %r3, [retval0];
    }
$L__CLEAN:
    st.param.b32 [func_retval0], %r2;
    ret;
}
)";

std::string devrt_call(const char* fn, const char* dst_reg) {
  return std::string("    {\n    .param .b32 retval0;\n    call.uni (retval0), ") + fn +
         ", ();\n    ld.param.b32 " + dst_reg + ", [retval0];\n    }\n";
}
}  // namespace

// Mirrors device_last_error.cu: the device-side last error is per thread. Odd
// threads launch a child with a zero-sized block (cudaErrorInvalidConfiguration,
// 9), even threads a valid one; each thread then sees
// {peek before, peek, peek, get, get, peek} = {0, e, e, e, 0, 0}, and the even
// threads' children each add 1 to data[0].
VTEST(device_side_last_error_is_per_thread_and_get_clears_it) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const std::string parent = R"(
.visible .entry child(.param .u64 p)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [p];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd2, %r1, 4;
    add.s64 %rd3, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    add.s32 %r3, %r2, 1;
    st.global.u32 [%rd3], %r3;
    ret;
}
.visible .entry parent(.param .u64 out, .param .u64 data)
{
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [data];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 24;
    add.s64 %rd4, %rd1, %rd3;
)" + devrt_call("cudaPeekAtLastError", "%r10") + R"(
    st.global.u32 [%rd4+0], %r10;
    and.b32 %r2, %r1, 1;
    xor.b32 %r21, %r2, 1;
    mov.u32 %r20, 1;
    mov.u64 %rd19, child;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd19;
    .param .align 4 .b8 param1[12];
    st.param.b32 [param1+0], %r20;
    st.param.b32 [param1+4], 1;
    st.param.b32 [param1+8], 1;
    .param .align 4 .b8 param2[12];
    st.param.b32 [param2+0], %r21;
    st.param.b32 [param2+4], 1;
    st.param.b32 [param2+8], 1;
    .param .b32 param3;
    st.param.b32 [param3+0], 0;
    .param .b64 retval0;
    call.uni (retval0), __cudaCDP2GetParameterBufferV2, (param0, param1, param2, param3);
    ld.param.b64 %rd20, [retval0+0];
    }
    st.u64 [%rd20], %rd2;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd20;
    .param .b64 param1;
    st.param.b64 [param1+0], 0;
    .param .b32 retval0;
    call.uni (retval0), __cudaCDP2LaunchDeviceV2, (param0, param1);
    ld.param.b32 %r22, [retval0+0];
    }
)" + devrt_call("cudaPeekAtLastError", "%r11") + "    st.global.u32 [%rd4+4], %r11;\n" +
                             devrt_call("cudaPeekAtLastError", "%r12") + "    st.global.u32 [%rd4+8], %r12;\n" +
                             devrt_call("cudaGetLastError", "%r13") + "    st.global.u32 [%rd4+12], %r13;\n" +
                             devrt_call("cudaGetLastError", "%r14") + "    st.global.u32 [%rd4+16], %r14;\n" +
                             devrt_call("cudaPeekAtLastError", "%r15") + "    st.global.u32 [%rd4+20], %r15;\n" +
                             "    ret;\n}\n";
  const uint64_t mod = dev.load_module(std::string(kHeader) + kDevrtPtx + parent);
  const uint64_t out = dev.memory().alloc(4 * 6 * 4);
  const uint64_t data = dev.memory().alloc(64 * 4);
  const uint8_t zero = 0;
  dev.memory().fill(out, &zero, 1, 4 * 6 * 4);
  dev.memory().fill(data, &zero, 1, 64 * 4);
  dev.launch(*dev.get_function(mod, "parent"), block_of(4), {arg_u64(out), arg_u64(data)}, dev.symbols(mod));
  for (uint32_t t = 0; t < 4; ++t) {
    const uint32_t e = (t & 1) ? 9 : 0;
    const uint32_t want[6] = {0, e, e, e, 0, 0};
    for (uint32_t j = 0; j < 6; ++j) VCHECK_EQ(u32_at(dev, out + 4 * (t * 6 + j)), want[j]);
  }
  VCHECK_EQ(u32_at(dev, data), 2u);   // the even threads' children ran
}

// Mirrors rdc_device_api.cu: device-side cudaGetDevice and cudaGetDeviceCount
// answer the launching device's ordinal and the machine's count, written
// through a pointer to the thread's own local variable.
VTEST(device_side_get_device_answers_the_launching_device) {
  runtime::Runtime rt(load_gpu("nvidia/a10"), 2);
  const std::string ask = R"(
.visible .entry ask(.param .u64 out)
{
    .local .align 4 .b8 __local_depot0[8];
    .reg .b64 %SP, %SPL;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    mov.u64 %SPL, __local_depot0;
    cvta.local.u64 %SP, %SPL;
    ld.param.u64 %rd1, [out];
    mov.u32 %r5, -9;
    st.local.u32 [%SPL+0], %r5;
    st.local.u32 [%SPL+4], %r5;
    add.u64 %rd2, %SP, 0;
    add.u64 %rd3, %SP, 4;
    {
    .param .b64 param0;
    st.param.b64 [param0], %rd2;
    .param .b32 retval0;
    call.uni (retval0), __cuda_syscall_cnpv2GetDevice, (param0);
    ld.param.b32 %r1, [retval0];
    }
    {
    .param .b64 param0;
    st.param.b64 [param0], %rd3;
    .param .b32 retval0;
    call.uni (retval0), __cuda_syscall_cnpv2GetDeviceCount, (param0);
    ld.param.b32 %r3, [retval0];
    }
    ld.local.u32 %r2, [%SPL+0];
    ld.local.u32 %r4, [%SPL+4];
    st.global.u32 [%rd1+0], %r1;
    st.global.u32 [%rd1+4], %r2;
    st.global.u32 [%rd1+8], %r3;
    st.global.u32 [%rd1+12], %r4;
    ret;
}
)";
  for (int d = 0; d < rt.device_count(); ++d) {
    auto& dev = rt.device(d);
    const uint64_t mod = dev.load_module(std::string(kHeader) + kDevrtPtx + ask);
    const uint64_t out = dev.memory().alloc(16);
    dev.launch(*dev.get_function(mod, "ask"), LaunchConfig{}, {arg_u64(out)}, dev.symbols(mod));
    VCHECK_EQ(u32_at(dev, out), 0u);                              // cudaSuccess
    VCHECK_EQ(u32_at(dev, out + 4), static_cast<uint32_t>(d));    // this device
    VCHECK_EQ(u32_at(dev, out + 8), 0u);
    VCHECK_EQ(u32_at(dev, out + 12), 2u);                         // the host's count
  }
}

// ---------------------------------------------------------------------------
// Mirrors warp_spin_lock.cu: lanes of one warp contending for a spin lock.
// Since Volta the lane holding the lock runs while its warp-mates spin, so
// the classic atomicCAS lock works inside a warp; a lowest-pc-first warp would
// run the spinners forever. Two blocks of two warps, two rounds each.
VTEST(lanes_of_one_warp_can_hand_a_spin_lock_to_each_other) {
  runtime::Runtime rt(load_gpu("nvidia/rtx3060"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + R"(
.global .align 4 .u32 lock_word;
.global .align 4 .u32 counter;
.visible .entry locked_increment(.param .u32 rounds)
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;
    ld.param.u32 %r1, [rounds];
    mov.u32 %r2, 0;
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra DONE;
ROUND:
    atom.global.cas.b32 %r3, [lock_word], 0, 1;
    setp.ne.s32 %p2, %r3, 0;
    @%p2 bra ROUND;
    membar.gl;
    ld.volatile.global.u32 %r4, [counter];
    add.s32 %r5, %r4, 1;
    st.volatile.global.u32 [counter], %r5;
    membar.gl;
    atom.global.exch.b32 %r6, [lock_word], 0;
    add.s32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra ROUND;
DONE:
    ret;
}
)");
  constexpr uint32_t blocks = 2, threads = 64, rounds = 2;
  dev.launch(*dev.get_function(mod, "locked_increment"), block_of(threads, blocks), {arg_u32(rounds)},
             dev.symbols(mod));
  VCHECK_EQ(u32_at(dev, global_of(dev, mod, "counter")), blocks * threads * rounds);
  VCHECK_EQ(u32_at(dev, global_of(dev, mod, "lock_word")), 0u);
}

// ---------------------------------------------------------------------------
// Mirrors graph_conditional.cu (set_to, and "a kernel outside a graph that
// calls cudaGraphSetConditional faults with an illegal address"): a kernel sets
// a conditional handle's value in the graph's table, the value outlives the
// launch, and a kernel with no graph -- or a handle no graph made -- faults.
VTEST(graph_set_conditional_writes_the_graphs_handle) {
  runtime::Runtime rt(load_gpu("nvidia/rtx3060"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(std::string(kHeader) + R"(
.extern .func cudaGraphSetConditional (.param .b64 h, .param .b32 v);
.visible .entry set_to(.param .u64 h, .param .u32 v)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [h];
    ld.param.u32 %r1, [v];
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd1;
    .param .b32 param1;
    st.param.b32 [param1+0], %r1;
    call.uni cudaGraphSetConditional, (param0, param1);
    }
    ret;
}
)");
  const ptx::EntryFn* fn = dev.get_function(mod, "set_to");
  exec::GraphConditionals table;
  table.values[0x100] = 0;
  table.values[0x200] = 7;
  LaunchConfig cfg;
  cfg.conditionals = &table;
  dev.launch(*fn, cfg, {arg_u64(0x100), arg_u32(5)});
  VCHECK_EQ(table.values[0x100], 5u);
  VCHECK_EQ(table.values[0x200], 7u);   // the other handle is untouched
  dev.launch(*fn, cfg, {arg_u64(0x200), arg_u32(0)});
  VCHECK_EQ(table.values[0x100], 5u);   // a value outlives the launch that set it
  VCHECK_EQ(table.values[0x200], 0u);

  auto unknown = VCAPTURE(Error, dev.launch(*fn, cfg, {arg_u64(0x300), arg_u32(1)}));
  VCHECK(unknown.code() == Err::InvalidPointer);
  auto outside = VCAPTURE(Error, dev.launch(*fn, LaunchConfig{}, {arg_u64(0x100), arg_u32(1)}));
  VCHECK(outside.code() == Err::InvalidPointer);
  VCHECK_CONTAINS(outside.message(), "not running in a graph");
}

// ---------------------------------------------------------------------------
// Mirrors runtime_conformance.cu ("reset releases the device's allocations",
// "a pointer from before the reset no longer frees") and device_limits.cu ("a
// reset starts over"): cudaDeviceReset destroys everything the device held --
// allocations, modules and their globals, texture objects -- and the device
// stays usable, with modules loading fresh from their initialisers.
VTEST(a_device_reset_releases_everything_and_starts_over) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const std::string ptx = std::string(kHeader) + R"(
.global .align 4 .u32 g = 7;
.visible .entry read_g(.param .u64 out)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [out];
    ld.global.u32 %r1, [g];
    st.global.u32 [%rd1], %r1;
    ret;
}
)";
  const uint64_t used_empty = dev.memory().used();
  const uint64_t mod = dev.load_module(ptx);
  dev.memory().store_scalar(global_of(dev, mod, "g"), 4, 99);   // changed before the reset
  const uint64_t held = dev.memory().alloc(256ull << 20);
  dev.textures()[0x42] = exec::TextureDesc{};
  VCHECK(dev.memory().used() >= used_empty + (256ull << 20));

  dev.reset();
  VCHECK_EQ(dev.memory().used(), used_empty);
  VCHECK(dev.textures().empty());
  auto freed = VCAPTURE(Error, dev.memory().free(held));
  VCHECK(freed.code() != Err::Internal);
  uint32_t scratch = 0;
  VCAPTURE(Error, dev.memory().read(held, &scratch, 4));
  auto gone = VCAPTURE(Error, dev.get_function(mod, "read_g"));
  VCHECK(gone.code() == Err::NotFound);

  // Usable again: the module loads afresh, its global from the initialiser.
  const uint64_t mod2 = dev.load_module(ptx);
  const uint64_t out = dev.memory().alloc(4);
  dev.launch(*dev.get_function(mod2, "read_g"), LaunchConfig{}, {arg_u64(out)}, dev.symbols(mod2));
  VCHECK_EQ(u32_at(dev, out), 7u);
}

VTEST_MAIN
