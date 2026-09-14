// vgpu — VirtualGPU command-line interface.
//
// Commands:
//   vgpu list-gpus              list built-in virtual GPU models
//   vgpu info --gpu <id>        show one device profile (--json for JSON)
//   vgpu demo vectoradd         (added with the execution engine)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "args.hpp"
#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"

namespace {

constexpr const char* kVersion = "0.1.0";

int usage(FILE* to) {
  std::fprintf(to,
               "VirtualGPU %s — functional GPU emulation on CPUs (no performance modeling)\n"
               "\n"
               "Usage:\n"
               "  vgpu list-gpus                       List built-in virtual GPU models\n"
               "  vgpu info --gpu <vendor/model>       Show a device profile (add --json for JSON)\n"
               "  vgpu demo vectoradd [--gpu <id>] [-n <elems>]\n"
               "                                       Run the built-in vectorAdd kernel end-to-end\n"
               "  vgpu smi [--csv|--rocm|--agents|--lspci|--lspci-dump|--explain]\n"
               "                                       Live view of running virtual GPUs\n"
               "  vgpu serve --gpu <id> [--count N] [--load 0..1] [--alloc-mb N]\n"
               "                                       Present a virtual rack for monitoring tools\n"
               "  vgpu run [opts] <program> [args...]  Run a program against VirtualGPU, with the\n"
               "                                       simulator's CUDA libraries in front of the\n"
               "                                       real ones (vgpu run --help for options)\n"
               "  vgpu counters                        What performance counters this reports, and\n"
               "                                       which it cannot, with the reason\n"
               "  vgpu test --matrix <program> [args]  Run a program on every device profile and\n"
               "                                       compare the results (vgpu test --help)\n"
               "  vgpu shell                           Interactive machine simulator: pick a GPU,\n"
               "                                       CUDA/driver and OS, then get a shell where\n"
               "                                       nvidia-smi/rocm-smi/lspci/dmesg all work\n"
               "  vgpu --version | --help\n",
               kVersion);
  return to == stdout ? 0 : 2;
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

// `vgpu counters` — what this engine can and cannot report, and why.
//
// The question this answers is "which performance counters do you have for my
// GPU", and the honest answer has two halves that are easy to blur. Everything
// here is counted exactly, from what the program actually executed, and is
// therefore identical on every device profile -- these are not hardware
// counters sampled from a particular chip's monitors. Everything a real
// profiler reports that is missing is missing for one reason: it is derived
// from timing, and there is no timing model to derive it from.
//
// Printing both lists together is the point. A gap that is written down is a
// design decision; a gap you discover by its absence is a defect.
int cmd_counters() {
  struct Row { const char* name; const char* what; };
  static const Row kProduced[] = {
      {"blocks, warps", "launched"},
      {"instructions", "warp-level issues, the quantity a profiler calls inst_executed"},
      {"thread_instructions", "summed over active lanes; the ratio to the above is divergence"},
      {"inst_by_class", "nine classes that partition every instruction"},
      {"inst_by_opcode", "per-mnemonic issues, most-used first"},
      {"tensor_instructions", "mma issues, per warp rather than per lane"},
      {"divergent_branches", "branches where the active mask actually split"},
      {"global/shared/local loads, stores, bytes", "per space, per active lane"},
      {"atomics, atomic_bytes", "read-modify-writes and the traffic they moved"},
      {"barriers", "bar.sync executions, per warp"},
      {"global_sectors, local_sectors", "32-byte sectors touched, counted from the addresses"},
      {"global/shared/local_requests", "the denominator for coalescing"},
      {"shared_bank_conflicts", "extra passes serialization forced"},
  };
  static const Row kAbsent[] = {
      {"cycles, IPC, elapsed time", "there is no timing model; %clock64 is a monotonic counter"},
      {"achieved occupancy", "derived from residency over time"},
      {"warp stall reasons", "requires modelling issue and dependency stalls"},
      {"L1/L2/texture hit rates", "there is no cache model"},
      {"DRAM read/write throughput", "requires a memory-hierarchy and timing model"},
      {"issue slot utilisation", "same"},
      {"instruction replay", "a hardware recovery mechanism, not a program property"},
      // Worth naming explicitly, because the local_* counters look like they
      // already answer it and they do not. Register spilling is done by ptxas,
      // *below* the language this engine executes: by the time a kernel reaches
      // us it is still PTX, with virtual registers and no allocation decided,
      // so no spill load or store has been emitted for us to count. The .local
      // traffic that is counted is the depot -- stack frames and arrays nvcc
      // could not keep in virtual registers -- which is a different quantity
      // that happens to live in the same address space. Labelling it "spill"
      // would produce a number that is exact, reproducible, and about something
      // else. What can honestly be said about spilling is said elsewhere:
      // `spilled_regs` reports how many registers exceed the profile's
      // architectural maximum, which is a property of the kernel, not traffic.
      {"register spill/fill traffic", "ptxas allocates registers; we execute PTX, where no spill exists yet"},
  };
  std::printf("Counters VirtualGPU reports. Every one is counted exactly from what the\n"
              "program executed, not sampled -- so they are reproducible run to run, and\n"
              "identical on every device profile. They are not hardware counters.\n\n");
  for (const Row& r : kProduced) std::printf("  %-42s %s\n", r.name, r.what);
  std::printf("\nWhat a hardware profiler reports and this does not, with the reason.\n"
              "These are absent by decision: producing them would mean inventing a\n"
              "timing model, and a plausible wrong number is worse than no number.\n\n");
  for (const Row& r : kAbsent) std::printf("  %-42s %s\n", r.name, r.what);
  std::printf("\nTurn the reported ones on with VGPU_COUNTERS=1.\n"
              "The CUPTI shim advertises zero event domains and zero metrics for the\n"
              "same reason: nvprof gets real activity records and no counters.\n");
  return 0;
}

int cmd_list_gpus() {
  for (const auto& id : vgpu::available_gpus()) std::printf("%s\n", id.c_str());
  return 0;
}

int cmd_info(const std::string& gpu, bool json) {
  vgpu::DeviceProfile p = vgpu::load_gpu(gpu);
  if (json) {
    std::printf("{\n");
    std::printf("  \"id\": \"%s\",\n", json_escape(p.id).c_str());
    std::printf("  \"vendor\": \"%s\",\n", json_escape(p.vendor).c_str());
    std::printf("  \"model\": \"%s\",\n", json_escape(p.model).c_str());
    std::printf("  \"architecture\": \"%s\",\n", json_escape(p.architecture).c_str());
    if (p.vendor == "amd")
      std::printf("  \"gcn_arch\": \"%s\",\n", json_escape(p.gcn_arch).c_str());
    else
      std::printf("  \"compute_capability\": \"%d.%d\",\n", p.cc_major, p.cc_minor);
    std::printf("  \"warp_size\": %u,\n", p.warp_size);
    std::printf("  \"vram_bytes\": %llu,\n", static_cast<unsigned long long>(p.vram_bytes));
    std::printf("  \"verified\": %s,\n", p.verified ? "true" : "false");
    std::printf("  \"limits\": {\n");
    std::printf("    \"max_threads_per_block\": %u,\n", p.limits.max_threads_per_block);
    std::printf("    \"max_block_dim\": [%u, %u, %u],\n", p.limits.max_block_dim[0], p.limits.max_block_dim[1],
                p.limits.max_block_dim[2]);
    std::printf("    \"max_grid_dim\": [%u, %u, %u],\n", p.limits.max_grid_dim[0], p.limits.max_grid_dim[1],
                p.limits.max_grid_dim[2]);
    std::printf("    \"shared_mem_per_block_bytes\": %u,\n", p.limits.shared_mem_per_block);
    std::printf("    \"shared_mem_per_block_optin_bytes\": %u,\n", p.limits.shared_mem_per_block_optin);
    std::printf("    \"registers_per_block\": %u,\n", p.limits.registers_per_block);
    std::printf("    \"multiprocessors\": %u\n", p.limits.multiprocessors);
    std::printf("  },\n");
    std::printf("  \"features\": {");
    bool first = true;
    for (auto& [k, v] : p.features) {
      std::printf("%s\n    \"%s\": %s", first ? "" : ",", json_escape(k).c_str(), v ? "true" : "false");
      first = false;
    }
    std::printf("\n  }\n}\n");
  } else {
    double gib = static_cast<double>(p.vram_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::printf("VirtualGPU device: %s\n", p.id.c_str());
    std::printf("  vendor:              %s\n", p.vendor.c_str());
    std::printf("  model:               %s\n", p.model.c_str());
    std::printf("  architecture:        %s\n", p.architecture.c_str());
    // AMD parts have no compute capability; their gfx target is the equivalent
    // thing, and printing "0.0" for it says nothing true.
    if (p.vendor == "amd")
      std::printf("  gfx target:          %s\n", p.gcn_arch.c_str());
    else
      std::printf("  compute capability:  %d.%d\n", p.cc_major, p.cc_minor);
    std::printf("  warp size:           %u\n", p.warp_size);
    std::printf("  vram:                %.0f GiB (%llu bytes, virtual)\n", gib,
                static_cast<unsigned long long>(p.vram_bytes));
    std::printf("  multiprocessors:     %u\n", p.limits.multiprocessors);
    std::printf("  max threads/block:   %u\n", p.limits.max_threads_per_block);
    std::printf("  max block dim:       [%u, %u, %u]\n", p.limits.max_block_dim[0], p.limits.max_block_dim[1],
                p.limits.max_block_dim[2]);
    std::printf("  max grid dim:        [%u, %u, %u]\n", p.limits.max_grid_dim[0], p.limits.max_grid_dim[1],
                p.limits.max_grid_dim[2]);
    std::printf("  shared mem/block:    %u bytes (opt-in max %u)\n", p.limits.shared_mem_per_block,
                p.limits.shared_mem_per_block_optin);
    std::printf("  registers/block:     %u\n", p.limits.registers_per_block);
    std::printf("  features:            ");
    bool first = true;
    for (auto& [k, v] : p.features) {
      if (v) {
        std::printf("%s%s", first ? "" : ", ", k.c_str());
        first = false;
      }
    }
    std::printf("\n");
    if (!p.verified)
      std::printf(
          "  NOTE: not confirmed by hardware characterization (verified: false).\n"
          "        Where the values came from differs per profile -- some are read from\n"
          "        public documentation, some inherited from a verified profile of the\n"
          "        same die -- so the profile's own header says which.\n");
  }
  return 0;
}

}  // namespace

// Implemented in demo.cpp (stubbed until the execution engine lands).
int demo_vectoradd(const std::string& gpu, long long n);
// Implemented in smi.cpp / serve.cpp.
int cmd_smi(const std::vector<std::string>& args);
int cmd_serve(const std::vector<std::string>& args);
int cmd_shell(const std::vector<std::string>& args);
// Implemented in run.cpp.
int cmd_run(const std::vector<std::string>& args);
// Implemented in test.cpp.
int cmd_test(const std::vector<std::string>& args);

int main(int argc, char** argv) {
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.empty()) return usage(stderr);

  try {
    const std::string& cmd = args[0];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") return usage(stdout);
    if (cmd == "--version") {
      std::printf("vgpu %s\n", kVersion);
      return 0;
    }
    if (cmd == "list-gpus") return cmd_list_gpus();
    if (cmd == "counters") return cmd_counters();
    if (cmd == "smi") return cmd_smi({args.begin() + 1, args.end()});
    if (cmd == "serve") return cmd_serve({args.begin() + 1, args.end()});
    if (cmd == "shell") return cmd_shell({args.begin() + 1, args.end()});
    if (cmd == "run") return cmd_run({args.begin() + 1, args.end()});
    if (cmd == "test") return cmd_test({args.begin() + 1, args.end()});
    if (cmd == "info") {
      std::string gpu;
      bool json = false;
      for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--gpu") {
          if (i + 1 >= args.size()) {
            std::fprintf(stderr, "vgpu info: --gpu needs a value\n");
            return 2;
          }
          gpu = args[++i];
        } else if (args[i] == "--json")
          json = true;
        else {
          std::fprintf(stderr, "vgpu info: unknown argument '%s'\n", args[i].c_str());
          return 2;
        }
      }
      if (gpu.empty()) {
        std::fprintf(stderr, "vgpu info: missing --gpu <vendor/model>\n");
        return 2;
      }
      return cmd_info(gpu, json);
    }
    if (cmd == "demo") {
      std::string kernel = args.size() > 1 ? args[1] : "";
      if (kernel != "vectoradd") {
        std::fprintf(stderr, "vgpu demo: expected 'vectoradd', got '%s'\n", kernel.c_str());
        return 2;
      }
      std::string gpu = "nvidia/h100";
      long long n = 65536;
      for (size_t i = 2; i < args.size(); ++i) {
        if ((args[i] == "--gpu" || args[i] == "-n") && i + 1 >= args.size()) {
          std::fprintf(stderr, "vgpu demo: %s needs a value\n", args[i].c_str());
          return 2;
        }
        if (args[i] == "--gpu") {
          gpu = args[++i];
        } else if (args[i] == "-n") {
          // std::stoll let "abc" escape as an exception whose only text was
          // "stoll", and took "12abc" as 12.
          const std::string& v = args[++i];
          if (!vgpu::cli::parse_int(v, 1, 1ll << 31, &n)) {
            std::fprintf(stderr, "vgpu demo: -n needs a whole number in [1, 2^31], got '%s'\n",
                         v.c_str());
            return 2;
          }
        } else {
          std::fprintf(stderr, "vgpu demo: unknown argument '%s'\n", args[i].c_str());
          return 2;
        }
      }
      return demo_vectoradd(gpu, n);
    }
    std::fprintf(stderr, "vgpu: unknown command '%s'\n\n", cmd.c_str());
    return usage(stderr);
  } catch (const vgpu::Error& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vgpu: %s\n", e.what());
    return 1;
  }
}
