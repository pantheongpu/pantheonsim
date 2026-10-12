// Throwaway: static coverage of a set of cubins. For each kernel: instructions, loops, and whether the
// prototype's subset compiles it (and the first reason when not). Aggregates op histograms for the whole
// code and for loop bodies (instructions between a backward branch's target and the branch).
#include <map>

#include "common.hpp"
#include "engine.hpp"

using namespace vgpu;
using namespace vgpu::sass;

int main(int argc, char** argv) {
  std::map<std::string, size_t> all, loops;
  size_t kernels = 0, ok = 0, instrs = 0, loop_instrs = 0, loop_supported = 0;
  std::map<std::string, size_t> why;
  std::map<std::string, size_t> unsup_ops_in_loops, unsup_why_instr;
  size_t instr_ok = 0, loop_instr_ok = 0;
  for (int a = 1; a < argc; ++a) {
    try {
      r7::Env env(argv[a]);
      for (const CubinKernel& k : env.mod->cubin.kernels) {
        const Code& code = env.mod->code[env.mod->code_index.at(k.text_section)];
        ++kernels;
        instrs += code.count;
        std::vector<bool> in_loop(code.count, false);
        for (size_t i = 0; i < code.count; ++i) {
          const Instr& ins = code.instr(i);
          ++all[op_name(ins.op)];
          if (ins.op == Op::BRA && !ins.src.empty() && ins.src[0].kind == Kind::Label) {
            const uint64_t t = static_cast<uint64_t>(ins.src[0].imm);
            if (t <= ins.pc)
              for (uint64_t p = t; p <= ins.pc; p += 16) in_loop[p / 16] = true;
          }
        }
        for (size_t i = 0; i < code.count; ++i) {
          const Instr& ins = code.instr(i);
          if (ins.op == Op::NOP) continue;
          std::string w;
          const bool s = r7::supports(*env.mod, k, ins, &w);
          if (s) {
            ++instr_ok;
            if (in_loop[i]) ++loop_instr_ok;
          } else {
            ++unsup_why_instr[std::string(op_name(ins.op)) + ": " + w];
          }
        }
        for (size_t i = 0; i < code.count; ++i)
          if (in_loop[i]) {
            ++loop_instrs;
            ++loops[op_name(code.instr(i).op)];
          }
        r7::LaunchInfo li;
        li.bank0.assign(0x400, 0);
        try {
          (void)r7::compile(*env.mod, k, li);
          ++ok;
          loop_supported += std::count(in_loop.begin(), in_loop.end(), true);
        } catch (const r7::Unsupported& u) {
          std::string w = u.what();
          if (w.rfind("uniform guard", 0) == 0) w = "uniform guard";
          ++why[w];
        }
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: %s\n", argv[a], e.what());
    }
  }
  std::printf("cubins %d kernels %zu instructions %zu; prototype subset compiles %zu kernels (%.1f%%)\n", argc - 1, kernels, instrs, ok, 100.0 * ok / kernels);
  std::printf("loop-body instructions %zu (%.1f%% of all); in kernels the subset compiles: %zu\n", loop_instrs, 100.0 * loop_instrs / instrs, loop_supported);
  std::printf("instruction-level: %zu of %zu non-NOP instructions (%.1f%%) are inside the subset; in loop bodies %zu of %zu (%.1f%%)\n", instr_ok,
              instrs - all["NOP"], 100.0 * instr_ok / (instrs - all["NOP"]), loop_instr_ok, loop_instrs, 100.0 * loop_instr_ok / loop_instrs);
  {
    std::vector<std::pair<size_t, std::string>> u;
    for (auto& [k, v] : unsup_why_instr) u.push_back({v, k});
    std::sort(u.rbegin(), u.rend());
    std::printf("-- unsupported instructions by reason (top 25)\n");
    for (size_t i = 0; i < u.size() && i < 25; ++i) std::printf("%6zu  %s\n", u[i].first, u[i].second.c_str());
  }
  std::printf("-- first unsupported reason per kernel\n");
  std::vector<std::pair<size_t, std::string>> w;
  for (auto& [k, v] : why) w.push_back({v, k});
  std::sort(w.rbegin(), w.rend());
  for (auto& [n, k] : w) std::printf("%5zu  %s\n", n, k.c_str());
  std::printf("-- op histogram (all code | loop bodies)\n");
  std::vector<std::pair<size_t, std::string>> h;
  for (auto& [k, v] : all) h.push_back({v, k});
  std::sort(h.rbegin(), h.rend());
  size_t cum = 0;
  for (auto& [n, k] : h) {
    cum += n;
    std::printf("%-10s %6zu %5.1f%% cum %5.1f%% | loops %6zu %5.1f%%\n", k.c_str(), n, 100.0 * n / instrs, 100.0 * cum / instrs, loops[k], loop_instrs ? 100.0 * loops[k] / loop_instrs : 0.0);
  }
  return 0;
}
