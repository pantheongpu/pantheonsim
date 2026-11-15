// One approximate-math instruction (MUFU) of a real NVIDIA card over all 2^32 inputs (docs/numerics/math-functions.md).
//
// The kernels are tools/numerics/mufu_kernel.cu, compiled to a cubin. This program loads the cubin, finds every MUFU
// instruction in its code (128-bit instructions whose opcode field, bits 0-8, is 0x108), rewrites the sub-operation field
// (bits 74-77) to the one asked for, and runs it through the driver API. So the instruction that runs is the lone
// MUFU.<op> Rd, Rs with nothing around it; --cubin-out writes the rewritten code for nvdisasm to confirm it.
//
//   nvcc -cubin -arch=sm_<card> mufu_kernel.cu -o mufu_kernel.cubin
//   nvcc -std=c++17 -I ../../include/vgpu/exec mufu_probe.cu -o mufu_probe -lcuda
//
//   mufu_probe <cubin> <op> --hash [--chunk-bits 26]            one line a chunk of 2^chunk-bits inputs: a hash of the results
//   mufu_probe <cubin> <op> --dump <file> [--start S] [--count C]   the results of inputs S .. S+C-1 as raw little-endian words
//   mufu_probe <cubin> <op> --check [--start S] [--count C]     every result against the simulator's model (include/vgpu/exec/mufu_model.hpp):
//                                                               the number that differ, and the first of them
//   mufu_probe <cubin> <op> --vectors <n>                       edge inputs and n sampled inputs with the card's results, as text
//   mufu_probe <cubin> <op> --cubin-out <file>                  only write the rewritten cubin
//   --abs, --neg (with any of them)                             the instruction with the operand modifier: MUFU.<op> Rd, |Rs| or -Rs
//
// <op>: COS SIN EX2 LG2 RCP RSQ RCP64H RSQ64H SQRT TANH (TANH: Ampere and later). The input is the 32-bit pattern itself:
// a float's bits, or for the 64H forms the upper word of a double. Default range: all 2^32.
//
// The hash (the same in --hash, in the card's own kernel, and for the model): the results are taken in blocks of 4096
// consecutive inputs; a block's hash is FNV-1a 64 over its results, one 32-bit result a step (h = (h ^ r) * prime); a
// chunk's hash is FNV-1a 64 over its blocks' hashes the same way (one 64-bit value a step).
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if __has_include("mufu_model.hpp")
#include "mufu_model.hpp"
#define HAVE_MODEL 1
#else
#define HAVE_MODEL 0
#endif

static const char* const kOps[] = {"COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "RCP64H", "RSQ64H", "SQRT", "TANH"};
static const uint64_t kFnvBasis = 0xcbf29ce484222325ull, kFnvPrime = 0x100000001b3ull;
static const unsigned kLog2Block = 12;

static void ok(CUresult r, const char* what) {
  if (r == CUDA_SUCCESS) return;
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  fprintf(stderr, "mufu_probe: %s: %s\n", what, s ? s : "?");
  exit(3);
}

// Rewrites the sub-operation of every MUFU in the ELF's .text sections; returns how many.
static int patch(std::vector<uint8_t>& elf, unsigned op, bool src_abs, bool src_neg) {
  if (elf.size() < 64 || memcmp(elf.data(), "\177ELF", 4) != 0 || elf[4] != 2) { fprintf(stderr, "mufu_probe: not a 64-bit ELF\n"); exit(2); }
  auto rd = [&](size_t off, int bytes) { uint64_t v = 0; memcpy(&v, elf.data() + off, bytes); return v; };
  const uint64_t shoff = rd(0x28, 8);
  const unsigned shentsize = rd(0x3a, 2), shnum = rd(0x3c, 2), shstrndx = rd(0x3e, 2);
  const uint64_t stroff = rd(shoff + static_cast<uint64_t>(shstrndx) * shentsize + 0x18, 8);
  int count = 0;
  for (unsigned i = 0; i < shnum; ++i) {
    const uint64_t sh = shoff + static_cast<uint64_t>(i) * shentsize;
    const char* name = reinterpret_cast<const char*>(elf.data() + stroff + rd(sh, 4));
    if (strncmp(name, ".text.", 6) != 0) continue;
    const uint64_t off = rd(sh + 0x18, 8), size = rd(sh + 0x20, 8);
    for (uint64_t p = off; p + 16 <= off + size; p += 16) {
      uint64_t lo, hi;
      memcpy(&lo, elf.data() + p, 8); memcpy(&hi, elf.data() + p + 8, 8);
      if ((lo & 0x1ff) != 0x108) continue;
      hi = (hi & ~(0xfull << 10)) | (static_cast<uint64_t>(op) << 10);
      memcpy(elf.data() + p + 8, &hi, 8);
      lo = (lo & ~(3ull << 62)) | (static_cast<uint64_t>(src_abs) << 62) | (static_cast<uint64_t>(src_neg) << 63);   // MUFU Rd, |Rs| / -Rs
      memcpy(elf.data() + p, &lo, 8);
      ++count;
    }
  }
  return count;
}

struct Card {
  CUcontext ctx; CUmodule mod; CUfunction dump, hash; CUdeviceptr buf; size_t buf_bytes;
  void run_dump(uint32_t base, uint32_t n, uint32_t* out) {
    void* args[] = {&buf, &base, &n};
    ok(cuLaunchKernel(dump, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr), "launch (dump)");
    ok(cuCtxSynchronize(), "the kernel (dump)");
    ok(cuMemcpyDtoH(out, buf, static_cast<size_t>(n) * 4), "copy");
  }
  // Hashes of the n blocks of 2^kLog2Block inputs from base.
  void run_hash(uint32_t base, uint32_t n, uint64_t* out) {
    uint32_t lb = kLog2Block;
    void* args[] = {&buf, &base, &lb, &n};
    ok(cuLaunchKernel(hash, (n + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr), "launch (hash)");
    ok(cuCtxSynchronize(), "the kernel (hash)");
    ok(cuMemcpyDtoH(out, buf, static_cast<size_t>(n) * 8), "copy");
  }
};

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: mufu_probe <cubin> <op> --hash | --dump <file> | --check | --vectors <n> | --cubin-out <file>  [--start S] [--count C] [--chunk-bits B]\n"); return 2; }
  int op = -1;
  for (int i = 0; i < 10; ++i) if (strcmp(argv[2], kOps[i]) == 0) op = i;
  if (op < 0) { fprintf(stderr, "mufu_probe: unknown op %s\n", argv[2]); return 2; }
  enum { Hash, Dump, Check, Vectors, CubinOut } mode = Hash;
  const char* file = nullptr;
  uint64_t start = 0, count = 1ull << 32;
  unsigned chunk_bits = 26, nvec = 0;
  bool src_abs = false, src_neg = false;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--hash") mode = Hash;
    else if (a == "--dump" && i + 1 < argc) { mode = Dump; file = argv[++i]; }
    else if (a == "--check") mode = Check;
    else if (a == "--vectors" && i + 1 < argc) { mode = Vectors; nvec = strtoul(argv[++i], nullptr, 0); }
    else if (a == "--cubin-out" && i + 1 < argc) { mode = CubinOut; file = argv[++i]; }
    else if (a == "--start" && i + 1 < argc) start = strtoull(argv[++i], nullptr, 0);
    else if (a == "--count" && i + 1 < argc) count = strtoull(argv[++i], nullptr, 0);
    else if (a == "--chunk-bits" && i + 1 < argc) chunk_bits = strtoul(argv[++i], nullptr, 0);
    else if (a == "--abs") src_abs = true;
    else if (a == "--neg") src_neg = true;
    else { fprintf(stderr, "mufu_probe: unknown argument %s\n", argv[i]); return 2; }
  }
  if (start + count > (1ull << 32) || count == 0 || chunk_bits < kLog2Block || chunk_bits > 32) { fprintf(stderr, "mufu_probe: bad range\n"); return 2; }

  std::vector<uint8_t> elf;
  { FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "mufu_probe: cannot read %s\n", argv[1]); return 2; }
    uint8_t tmp[65536]; size_t n;
    while ((n = fread(tmp, 1, sizeof tmp, f)) > 0) elf.insert(elf.end(), tmp, tmp + n);
    fclose(f); }
  const int patched = patch(elf, static_cast<unsigned>(op), src_abs, src_neg);
  if (patched == 0) { fprintf(stderr, "mufu_probe: no MUFU instruction in %s\n", argv[1]); return 2; }
  if (mode == CubinOut) {
    FILE* f = fopen(file, "wb");
    if (!f || fwrite(elf.data(), 1, elf.size(), f) != elf.size()) { fprintf(stderr, "mufu_probe: cannot write %s\n", file); return 2; }
    fclose(f);
    printf("%s: %d MUFU instructions rewritten to MUFU.%s\n", file, patched, kOps[op]);
    return 0;
  }

  Card c{};
  CUdevice dev;
  ok(cuInit(0), "cuInit");
  ok(cuDeviceGet(&dev, 0), "cuDeviceGet");
  ok(cuCtxCreate(&c.ctx, nullptr, 0, dev), "cuCtxCreate");
  ok(cuModuleLoadData(&c.mod, elf.data()), "cuModuleLoadData");
  ok(cuModuleGetFunction(&c.dump, c.mod, "mufu_dump"), "mufu_dump");
  ok(cuModuleGetFunction(&c.hash, c.mod, "mufu_hash"), "mufu_hash");
  const uint32_t step = 1u << 24;   // results a launch (dump, check)
  c.buf_bytes = static_cast<size_t>(step) * 4;
  ok(cuMemAlloc(&c.buf, c.buf_bytes), "cuMemAlloc");

  if (mode == Hash) {
    char name[256] = "?";
    cuDeviceGetName(name, sizeof name, dev);
    printf("# mufu_probe hash op=%s card=%s chunk=2^%u block=2^%u\n", kOps[op], name, chunk_bits, kLog2Block);
    const uint64_t chunk = 1ull << chunk_bits, blocks = chunk >> kLog2Block;
    std::vector<uint64_t> bh(static_cast<size_t>(std::min<uint64_t>(blocks, step / 2)));
    uint64_t total = kFnvBasis;
    for (uint64_t s = 0; s < (1ull << 32); s += chunk) {
      uint64_t h = kFnvBasis;
      for (uint64_t b = 0; b < blocks; b += bh.size()) {
        const uint32_t n = static_cast<uint32_t>(std::min<uint64_t>(bh.size(), blocks - b));
        c.run_hash(static_cast<uint32_t>(s + (b << kLog2Block)), n, bh.data());
        for (uint32_t i = 0; i < n; ++i) { h = (h ^ bh[i]) * kFnvPrime; total = (total ^ bh[i]) * kFnvPrime; }
      }
      printf("%s %08llx %016llx\n", kOps[op], static_cast<unsigned long long>(s), static_cast<unsigned long long>(h));
      fflush(stdout);
    }
    printf("%s all %016llx\n", kOps[op], static_cast<unsigned long long>(total));
    return 0;
  }

  std::vector<uint32_t> r(step);
  if (mode == Vectors) {
    // Edge inputs (each also negated by the sign bit), then nvec inputs from a fixed generator; "input result" in hex.
    std::vector<uint32_t> in = {0x00000000u, 0x00000001u, 0x00000002u, 0x00400000u, 0x007fffffu, 0x00800000u, 0x00800001u, 0x00ffffffu,
                                0x01000000u, 0x3f000000u, 0x3f7fffffu, 0x3f800000u, 0x3f800001u, 0x3f800002u, 0x3fb504f3u, 0x3fb504f4u,
                                0x3fc00000u, 0x3fffffffu, 0x40000000u, 0x40000001u, 0x40490fdbu, 0x3e800000u, 0x3e7fffffu, 0x3e000000u,
                                0x3f400000u, 0x3d800000u, 0x42fe0000u, 0x42ffffffu, 0x43000000u, 0x43000001u, 0xc2fc0000u, 0xc2fc0001u,
                                0xc3150000u, 0xc3160000u, 0x4b000000u, 0x4b800000u, 0x4b800001u, 0x4c000000u, 0x5f000000u, 0x7e800000u,
                                0x7effffffu, 0x7f000000u, 0x7f000001u, 0x7f7fffffu, 0x7f800000u, 0x7f800001u, 0x7fa00000u, 0x7fc00000u,
                                0x7fffffffu, 0x00100000u, 0x000fffffu, 0x7fe00000u, 0x7fefffffu, 0x7ff00000u, 0x7ff00001u, 0x7ff80000u,
                                0x3ff00000u, 0x3fefffffu, 0x3ff00001u, 0x3ff80000u, 0x40000000u, 0x3fe00000u, 0x00200000u, 0x001fffffu,
                                0x41200000u, 0x40400000u, 0x3eaaaaabu, 0x3dcccccdu, 0x3a83126fu, 0x358637bdu, 0x20000000u, 0x60000000u,
                                0x3b7fffffu, 0x3b800000u, 0x3b800040u, 0x40ffffffu, 0x41000000u, 0x3f000001u, 0x3effffffu, 0x3e800001u};
    const size_t edges = in.size();
    for (size_t i = 0; i < edges; ++i) in.push_back(in[i] ^ 0x80000000u);
    uint64_t z = 0x6d756675ull + static_cast<uint64_t>(op) * 0x9E3779B97F4A7C15ull;
    for (unsigned i = 0; i < nvec; ++i) {
      z += 0x9E3779B97F4A7C15ull;
      uint64_t x = z; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; x ^= x >> 31;
      uint32_t u = static_cast<uint32_t>(x >> 32);
      // Three in four with the exponent drawn near 1 (where the functions are not flat), the rest any pattern.
      if (i % 4 != 3) {
        const unsigned e = (op == 6 || op == 7) ? 0x3ff - 24 + (x & 0xff) % 48 : 127 - 24 + (x & 0xff) % 48;
        u = (op == 6 || op == 7) ? (u & 0x800fffffu) | (e << 20) : (u & 0x807fffffu) | (e << 23);
      }
      in.push_back(u);
    }
    for (uint32_t u : in) { uint32_t v; c.run_dump(u, 1, &v); printf("%08x %08x\n", u, v); }
    return 0;
  }

  FILE* f = nullptr;
  if (mode == Dump) { f = fopen(file, "wb"); if (!f) { fprintf(stderr, "mufu_probe: cannot write %s\n", file); return 2; } }
  uint64_t differ = 0, shown = 0;
  uint64_t hist[6] = {0, 0, 0, 0, 0, 0};   // the model's result minus the card's, as integers: -2 or less, -1, +1, +2 or more; then NaN/other
  for (uint64_t s = start; s < start + count; s += step) {
    const uint32_t n = static_cast<uint32_t>(std::min<uint64_t>(step, start + count - s));
    c.run_dump(static_cast<uint32_t>(s), n, r.data());
    if (mode == Dump) {
      if (fwrite(r.data(), 4, n, f) != n) { fprintf(stderr, "mufu_probe: write failed\n"); return 2; }
      continue;
    }
#if HAVE_MODEL
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t x = static_cast<uint32_t>(s) + i;
      // (The operand modifiers: |x| first, then -x, on the sign bit; the 64H forms take them on the upper word's sign bit too.)
      const uint32_t m = vgpu::mufu::eval(static_cast<unsigned>(op), (src_abs ? x & 0x7fffffffu : x) ^ (src_neg ? 0x80000000u : 0u));
      if (m == r[i]) continue;
      ++differ;
      const int64_t d = static_cast<int64_t>(m) - static_cast<int64_t>(r[i]);
      ++hist[d <= -2 ? 0 : d == -1 ? 1 : d == 1 ? 2 : 3];
      if (shown < 12) { printf("  %s(%08x): card %08x, model %08x\n", kOps[op], x, r[i], m); ++shown; }
    }
#else
    fprintf(stderr, "mufu_probe: built without include/vgpu/exec/mufu_model.hpp; --check is not available\n");
    return 2;
#endif
  }
  if (mode == Dump) { fclose(f); return 0; }
  printf("%s check start=%08llx count=%llu: %llu differ (model - card: <=-2: %llu, -1: %llu, +1: %llu, >=+2: %llu)\n", kOps[op],
         static_cast<unsigned long long>(start), static_cast<unsigned long long>(count), static_cast<unsigned long long>(differ),
         static_cast<unsigned long long>(hist[0]), static_cast<unsigned long long>(hist[1]), static_cast<unsigned long long>(hist[2]),
         static_cast<unsigned long long>(hist[3]));
  return differ ? 1 : 0;
}
