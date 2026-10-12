// Throwaway: print each kernel's decoded instructions with the fields the
// executor reads, and a static op histogram. Usage: r7dump <cubin> [kernel]
#include <map>

#include "common.hpp"

using namespace vgpu;
using namespace vgpu::sass;

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  r7::Env env(argv[1]);
  const std::string only = argc > 2 ? argv[2] : "";
  std::map<std::string, size_t> hist;
  for (const CubinKernel& k : env.mod->cubin.kernels) {
    if (!only.empty() && k.name != only) continue;
    const Code& code = env.mod->code[env.mod->code_index.at(k.text_section)];
    std::printf("== %s regs=%u param_base=0x%x param_size=%u frame=%u shared=%llu\n", k.name.c_str(), k.regs, k.param_base,
                k.param_size, k.frame_size, static_cast<unsigned long long>(k.shared_bytes));
    for (const CubinParam& p : k.params) std::printf("   param %u offset %u size %u\n", p.ordinal, p.offset, p.size);
    for (size_t i = 0; i < code.count; ++i) {
      const Instr& ins = code.instr(i);
      std::printf("%04zx  %-52s op=%s f=[%u %u %u %u %u %u %u %u] g=%u%s\n", i * 16, to_text(ins).c_str(), op_name(ins.op),
                  ins.f[0], ins.f[1], ins.f[2], ins.f[3], ins.f[4], ins.f[5], ins.f[6], ins.f[7], ins.guard,
                  ins.guard_not ? "!" : "");
      ++hist[op_name(ins.op)];
    }
  }
  std::printf("-- static op histogram\n");
  for (auto& [n, c] : hist) std::printf("%-10s %zu\n", n.c_str(), c);
  return 0;
}
