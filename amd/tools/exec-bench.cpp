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
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; ++r) {
    const vgpu::amd::DispatchStats s = vgpu::amd::execute(d, mem);
    instructions += s.instructions;
    waves += s.waves;
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: %llu waves, %llu wave-instructions in %.3f s = %.2f M wave-instructions/s\n", k->name.c_str(),
              static_cast<unsigned long long>(waves), static_cast<unsigned long long>(instructions), sec,
              instructions / sec / 1e6);
  return 0;
}
