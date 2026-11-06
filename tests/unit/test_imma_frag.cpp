// IMMA.16816 / IMMA.16832 with 8-bit operands, from the registers (include/vgpu/exec/imma_frag.hpp), against the definition: D = C + A x B in 32-bit
// wrapping arithmetic, over matrices written into registers by the layout of PTX ISA 9.7.14 spelled out here by element, so that a mistake in the kernel's
// register arithmetic is not shared with the oracle. s8 and u8 operands in all four combinations, random bytes and the extremes (-128, 127, 255, 0),
// accumulators from zero to the largest (the sum wraps), no accumulator (RZ), and the registers at the end of a mapped page (nothing past a register is read).
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "vgpu/exec/imma_frag.hpp"
#include "vtest.hpp"

#ifdef VGPU_IMMA_FRAG
namespace {

struct Edge {   // `words` 32-bit words ending at the last byte of a mapped page, the page after it not mapped
  char* base = nullptr;
  size_t map_len = 0;
  Edge() {
    const size_t pg = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    map_len = pg;
    void* m = mmap(nullptr, map_len + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) throw vtest::Failure("mmap failed");
    base = static_cast<char*>(m);
    if (mprotect(base + map_len, pg, PROT_NONE) != 0) throw vtest::Failure("mprotect failed");
  }
  Edge(const Edge&) = delete;
  Edge& operator=(const Edge&) = delete;
  ~Edge() { munmap(base, map_len + static_cast<size_t>(sysconf(_SC_PAGESIZE))); }
  uint32_t* place(const std::vector<uint32_t>& v) {
    uint32_t* p = reinterpret_cast<uint32_t*>(base + map_len) - v.size();
    std::memcpy(p, v.data(), v.size() * sizeof(uint32_t));
    return p;
  }
};

template <bool SA, bool SB>
long run(std::mt19937_64& rng, unsigned K, int rounds) {
  static Edge ea, eb, ec, eo;
  long calls = 0;
  for (int round = 0; round < rounds; ++round) {
    const int mode = round % 5;
    const auto byte = [&]() -> int {   // an element value: a random byte, or an extreme
      const uint32_t r = static_cast<uint32_t>(rng());
      if (mode == 1) return (r & 3) == 0 ? 0 : (r & 4) ? (SA ? -128 : 255) : (SA ? 127 : 128);
      if (mode == 2) return 0;
      return static_cast<int>(r & 0xff);
    };
    std::vector<int> A(16 * size_t{K}), B(size_t{K} * 8);
    for (int& x : A) x = SA ? static_cast<int8_t>(byte()) : (byte() & 0xff);
    for (int& x : B) x = SB ? static_cast<int8_t>(byte()) : (byte() & 0xff);
    std::vector<uint32_t> C(16 * 8);
    const bool with_c = round % 7 != 0;
    for (uint32_t& x : C) {
      const uint32_t r = static_cast<uint32_t>(rng());
      x = !with_c ? 0u : mode == 3 ? 0x7fffffffu - (r & 0xff) : mode == 4 ? 0x80000000u + (r & 0xff) : r;
    }
    // registers, by the layout
    std::vector<uint32_t> ra(size_t{K / 8} * 32, 0), rb(size_t{K / 16} * 32, 0), rc(4 * 32, 0);
    const auto put_byte = [](std::vector<uint32_t>& regs, unsigned reg, unsigned lane, unsigned which, uint8_t v) {
      uint32_t& w = regs[size_t{reg} * 32 + lane];
      w = (w & ~(0xffu << (8 * which))) | (static_cast<uint32_t>(v) << (8 * which));
    };
    for (unsigned i = 0; i < 16; ++i)   // A[i][k]: register 2 (k / 16) + i / 8, lane 4 (i % 8) + (k % 16) / 4, byte k % 4
      for (unsigned k = 0; k < K; ++k) put_byte(ra, 2 * (k / 16) + i / 8, 4 * (i % 8) + (k % 16) / 4, k % 4, static_cast<uint8_t>(A[i * K + k]));
    for (unsigned k = 0; k < K; ++k)    // B[k][j]: register k / 16, lane 4 j + (k % 16) / 4, byte k % 4
      for (unsigned j = 0; j < 8; ++j) put_byte(rb, k / 16, 4 * j + (k % 16) / 4, k % 4, static_cast<uint8_t>(B[k * 8 + j]));
    for (unsigned i = 0; i < 16; ++i)   // C[i][j]: register 2 (i / 8) + j % 2, lane 4 (i % 8) + j / 2
      for (unsigned j = 0; j < 8; ++j) rc[size_t{2 * (i / 8) + j % 2} * 32 + 4 * (i % 8) + j / 2] = C[i * 8 + j];
    std::vector<uint32_t> want(4 * 32, 0);
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned j = 0; j < 8; ++j) {
        int64_t s = static_cast<int32_t>(C[i * 8 + j]);
        for (unsigned k = 0; k < K; ++k) s += int64_t{A[i * K + k]} * B[k * 8 + j];
        want[size_t{2 * (i / 8) + j % 2} * 32 + 4 * (i % 8) + j / 2] = static_cast<uint32_t>(s);
      }
    const uint32_t* pa = ea.place(ra);
    const uint32_t* pb = eb.place(rb);
    const uint32_t* pc = with_c ? ec.place(rc) : nullptr;
    uint32_t* po = reinterpret_cast<uint32_t*>(eo.base + eo.map_len) - 4 * 32;
    for (unsigned i = 0; i < 4 * 32; ++i) po[i] = 0xdeadbeefu;
    vgpu::exec::imma_frag_m16n8<SA, SB>(K, pa, pb, pc, po);
    ++calls;
    for (unsigned i = 0; i < 4 * 32; ++i)
      if (po[i] != want[i])
        throw vtest::Failure("K " + std::to_string(K) + (SA ? " s8" : " u8") + (SB ? " s8" : " u8") + " mode " + std::to_string(mode) + " round " + std::to_string(round) +
                             ": word " + std::to_string(i) + " is 0x" + [&] { char b[16]; std::snprintf(b, sizeof b, "%08x", po[i]); return std::string(b); }() +
                             ", the definition gives 0x" + [&] { char b[16]; std::snprintf(b, sizeof b, "%08x", want[i]); return std::string(b); }());
  }
  return calls;
}

}  // namespace
#endif

VTEST(imma_from_registers_equals_the_definition) {
#ifdef VGPU_IMMA_FRAG
  __builtin_cpu_init();
  if (!(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl"))) {
    std::printf("  [ SKIP ] this CPU has no AVX-512\n");
    return;
  }
  std::mt19937_64 rng(0x1337a);
  long calls = 0;
  for (unsigned K : {16u, 32u}) {
    calls += run<true, true>(rng, K, 700);
    calls += run<true, false>(rng, K, 700);
    calls += run<false, true>(rng, K, 700);
    calls += run<false, false>(rng, K, 700);
  }
  std::printf("  %ld IMMA calls compared with the definition\n", calls);
  VCHECK(calls == 5600);
#else
  std::printf("  [ SKIP ] not an x86-64 build\n");
#endif
}

VTEST_MAIN
