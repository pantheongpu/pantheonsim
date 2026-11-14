// Host-GPU exact mode, without running anything: which kernels of a CUDA program are in the exact set, and which only by an idiom (Round sp68).
// Reads each program's fatbin, takes its sm_<arch> cubins (default 86), decodes every kernel's code and prints one line per kernel:
//
//   <program> <kernel> plain=<in|out> div=<in|out> sqrt=<in|out> both=<in|out> sites=<n> sfu=<MUFU forms> why=<why, when not in with both>
//
// plain: the exact set alone; div: with VGPU_HOST_GPU_DIV=1; sqrt: with VGPU_HOST_GPU_SQRT=1; both: with both (include/vgpu/host_gpu_idiom.hpp idioms_only, the
// function the runtime calls). The runtime refuses some kernels for other reasons first (module variables, device-side launches, textures...): this tool
// looks at the code only, so a kernel "in" here can still be refused by the runtime, never the other way round.
//
//   build (from the repository root, against a build of the library):
//     g++ -std=c++20 -O2 -Iinclude -Isrc -Invidia/src tools/probes/hostgpu_idiom_scan.cpp nvidia/src/fatbin.cpp build/libvgpu.a -ldl -lpthread -o hostgpu_idiom_scan
//   usage: hostgpu_idiom_scan [--arch 86] <program>...
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "fatbin.hpp"
#include "vgpu/host_gpu_idiom.hpp"

using namespace vgpu;
using namespace vgpu::hostgpu;

static std::string slurp(const char* path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

int main(int argc, char** argv) {
  uint32_t arch = 86;
  int first = 1;
  if (argc > 2 && std::strcmp(argv[1], "--arch") == 0) {
    arch = static_cast<uint32_t>(std::atoi(argv[2]));
    first = 3;
  }
  for (int a = first; a < argc; ++a) {
    const std::string bin = slurp(argv[a]);
    std::vector<std::string> fatbins;
    try {
      fatbins = cuda::host_object_fatbins(bin.data(), bin.size());
    } catch (const std::exception& e) {
      std::printf("%s - error=%s\n", argv[a], e.what());
      continue;
    }
    unsigned kernels = 0;
    for (const std::string& fb : fatbins) {
      std::vector<cuda::FatbinPtx> cubins;
      try {
        cubins = cuda::extract_elf(fb.data(), fb.size());
      } catch (const std::exception& e) {
        std::printf("%s - error=%s\n", argv[a], e.what());
        continue;
      }
      for (const cuda::FatbinPtx& cb : cubins) {
        if (cb.arch != arch) continue;
        for (const auto& [name, bytes] : cuda::elf_sections(cb.text.data(), cb.text.size())) {
          if (name.rfind(".text.", 0) != 0 || bytes.size() < 16) continue;
          std::vector<sass::Instr> code;
          bool decoded = true;
          for (size_t off = 0; off + 16 <= bytes.size(); off += 16) {
            sass::Word w;
            std::memcpy(&w.lo, bytes.data() + off, 8);
            std::memcpy(&w.hi, bytes.data() + off + 8, 8);
            try {
              code.push_back(sass::decode(w, off, static_cast<int>(arch)));
            } catch (const std::exception&) {
              decoded = false;
              break;
            }
          }
          ++kernels;
          if (!decoded) {
            std::printf("%s %s undecoded\n", argv[a], name.c_str() + 6);
            continue;
          }
          std::map<std::string, unsigned> sfu;
          for (const sass::Instr& i : code)
            if (i.mnemonic == "MUFU") {
              std::string k;
              for (const std::string& m : i.mods) k += (k.empty() ? "" : ".") + m;
              ++sfu[k];
            }
          std::string forms;
          for (const auto& [k, n] : sfu) forms += (forms.empty() ? "" : ",") + k + "x" + std::to_string(n);
          const auto in = [](bool b) { return b ? "in" : "out"; };
          const idiom::KernelVerdict plain = idiom::idioms_only(code, false, false), div = idiom::idioms_only(code, true, false),
                                     sq = idiom::idioms_only(code, false, true), both = idiom::idioms_only(code, true, true);
          std::printf("%s %s plain=%s div=%s sqrt=%s both=%s sites=%u sfu=%s%s%s\n", argv[a], name.c_str() + 6, in(plain.ok), in(div.ok), in(sq.ok), in(both.ok),
                      both.sqrt_sites, forms.empty() ? "-" : forms.c_str(), both.ok ? "" : " why=", both.ok ? "" : both.why);
          if (const char* d = std::getenv("SCAN_DUMP"); d && name.find(d) != std::string::npos)
            for (const sass::Instr& i : code) std::printf("    %04llx %s\n", static_cast<unsigned long long>(i.pc), sass::to_text(i).c_str());
          // every MUFU.RSQ of a plain register that a site could start at, and why it is not one (only for kernels the idioms do not take)
          if (!both.ok && std::getenv("SCAN_SITES")) {
            const idiom::Code c(code);
            for (size_t i = 0; i < code.size(); ++i)
              if (code[i].mnemonic == "MUFU" && code[i].mods.size() == 1 && code[i].mods[0] == "RSQ" && !code[i].src.empty() && code[i].src[0].kind == sass::Kind::Reg) {
                const idiom::Site st = idiom::sqrt_site(c, i);
                std::printf("  site %s 0x%llx %s: %s (join 0x%llx, live %d)\n", name.c_str() + 6, static_cast<unsigned long long>(code[i].pc), sass::to_text(code[i]).c_str(),
                            st.ok ? "ok" : st.why, static_cast<unsigned long long>(st.join < code.size() ? code[st.join].pc : 0), st.live);
              }
          }
        }
      }
    }
    if (!kernels) std::printf("%s - no sm_%u kernels\n", argv[a], arch);
  }
  return 0;
}
