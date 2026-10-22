// How fast the CDNA interpreter runs a kernel: loads a code object, runs one
// kernel of it with the arguments given, and reports the wave-instructions it
// retired per second. For measuring the interpreter, not a GPU.
//
//   amd_exec_bench <code object> <kernel name, or part of it> <groups> <group size> [arg ...]
//
// Each argument is `i:<value>` (a 32-bit integer), `l:<value>` (64-bit), or
// `b:<bytes>` (a zeroed device buffer, passed by address), in the kernel's
// own order. VGPU_BENCH_REPEAT=N runs it N times.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/memory.hpp"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <code object> <kernel> <groups> <group size> [i:N | l:N | b:BYTES ...]\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "no code object at %s\n", argv[1]);
    return 1;
  }
  const vgpu::amd::CodeObject o =
      vgpu::amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), argv[1]);
  const vgpu::amd::Kernel* k = nullptr;
  for (const auto& c : o.kernels)
    if (c.name.find(argv[2]) != std::string::npos) {
      k = &c;
      break;
    }
  if (!k) {
    std::fprintf(stderr, "no kernel matching %s; the object has:\n", argv[2]);
    for (const auto& c : o.kernels) std::fprintf(stderr, "  %s\n", c.name.c_str());
    return 1;
  }
  vgpu::MemoryManager mem(8ull << 30);
  std::vector<std::pair<uint64_t, uint64_t>> buffers;   // (address, bytes) of the b: arguments, for VGPU_BENCH_DUMP
  std::vector<uint8_t> kernarg(k->kernarg_size, 0);
  size_t arg = 0;
  for (int i = 5; i < argc; ++i, ++arg) {
    while (arg < k->args.size() && k->args[arg].hidden()) ++arg;
    if (arg >= k->args.size()) {
      std::fprintf(stderr, "more arguments than the kernel takes\n");
      return 1;
    }
    const vgpu::amd::KernelArg& a = k->args[arg];
    const std::string s = argv[i];
    uint64_t v = 0;
    if (s.rfind("b:", 0) == 0) {
      const uint64_t bytes = std::strtoull(s.c_str() + 2, nullptr, 0);
      v = mem.alloc(bytes ? bytes : 1);
      buffers.emplace_back(v, bytes);
      if (const char* dump = std::getenv("VGPU_BENCH_DUMP"); dump && *dump && *dump != '0') {   // data to read, not zeros
        std::vector<uint8_t> fill(bytes);
        uint64_t x = 0x9E3779B97F4A7C15ull ^ bytes;
        for (uint8_t& b : fill) x ^= x << 13, x ^= x >> 7, x ^= x << 17, b = static_cast<uint8_t>(x >> 24);
        if (bytes) mem.write(v, fill.data(), bytes);
      }
    } else if (s.rfind("i:", 0) == 0 || s.rfind("l:", 0) == 0) {
      v = std::strtoull(s.c_str() + 2, nullptr, 0);
    } else {
      std::fprintf(stderr, "argument %s is not i:, l: or b:\n", s.c_str());
      return 1;
    }
    std::memcpy(kernarg.data() + a.offset, &v, a.size < 8 ? a.size : 8);
  }
  vgpu::amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(kernarg.empty() ? 1 : kernarg.size());
  if (!kernarg.empty()) mem.write(d.kernarg, kernarg.data(), kernarg.size());
  d.groups[0] = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 0));
  d.group_size[0] = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 0));
  const char* rep = std::getenv("VGPU_BENCH_REPEAT");
  const int repeat = rep ? std::atoi(rep) : 1;
  uint64_t instructions = 0, waves = 0;
  vgpu::amd::DispatchStats last;
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    const vgpu::amd::DispatchStats s = vgpu::amd::execute(d, mem);
    instructions += s.instructions;
    waves += s.waves;
    last = s;
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: %llu waves, %llu wave-instructions in %.3f s = %.2f M wave-instructions/s\n", k->name.c_str(),
              static_cast<unsigned long long>(waves), static_cast<unsigned long long>(instructions), sec,
              instructions / sec / 1e6);
  // VGPU_BENCH_DUMP=1: everything the dispatch counted and a hash of the buffers it was given, for comparing two builds or the tier
  // on and off (nothing here is timed).
  if (const char* dump = std::getenv("VGPU_BENCH_DUMP"); dump && *dump && *dump != '0') {
    uint64_t h = 1469598103934665603ull;
    for (const auto& [addr, bytes] : buffers) {
      std::vector<uint8_t> data(bytes);
      if (bytes) mem.read(addr, data.data(), bytes);
      for (uint8_t b : data) h = (h ^ b) * 1099511628211ull;
    }
    const vgpu::amd::InstructionCounts& c = last.counts;
    std::printf("dump: buffers=%016llx waves=%llu instructions=%llu barriers=%llu valu=%llu mfma=%llu salu=%llu smem=%llu vmem=%llu flat=%llu lds=%llu "
                "branch=%llu sendmsg=%llu gds=%llu flat_read=%llu flat_write=%llu flat_atomic=%llu vmem_rd=%llu vmem_wr=%llu lt16=%llu lt32=%llu lt48=%llu "
                "lt64=%llu eq64=%llu\n",
                static_cast<unsigned long long>(h), static_cast<unsigned long long>(last.waves), static_cast<unsigned long long>(last.instructions),
                static_cast<unsigned long long>(last.barriers), static_cast<unsigned long long>(c.valu), static_cast<unsigned long long>(c.mfma),
                static_cast<unsigned long long>(c.salu), static_cast<unsigned long long>(c.smem), static_cast<unsigned long long>(c.vmem),
                static_cast<unsigned long long>(c.flat), static_cast<unsigned long long>(c.lds), static_cast<unsigned long long>(c.branch),
                static_cast<unsigned long long>(c.sendmsg), static_cast<unsigned long long>(c.gds), static_cast<unsigned long long>(c.flat_read),
                static_cast<unsigned long long>(c.flat_write), static_cast<unsigned long long>(c.flat_atomic), static_cast<unsigned long long>(c.vmem_rd),
                static_cast<unsigned long long>(c.vmem_wr), static_cast<unsigned long long>(last.waves_lt16), static_cast<unsigned long long>(last.waves_lt32),
                static_cast<unsigned long long>(last.waves_lt48), static_cast<unsigned long long>(last.waves_lt64), static_cast<unsigned long long>(last.waves_eq64));
    std::printf("mix:");
    for (uint64_t v : c.mix) std::printf(" %llu", static_cast<unsigned long long>(v));
    std::printf("\nmops:");
    for (uint64_t v : c.mops) std::printf(" %llu", static_cast<unsigned long long>(v));
    std::printf("\n");
  }
  return 0;
}
