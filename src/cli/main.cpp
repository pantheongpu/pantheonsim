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
          "  NOTE: profile values are placeholders from public documentation;\n"
          "        not yet confirmed by hardware characterization (verified: false)\n");
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
    if (cmd == "smi") return cmd_smi({args.begin() + 1, args.end()});
    if (cmd == "serve") return cmd_serve({args.begin() + 1, args.end()});
    if (cmd == "shell") return cmd_shell({args.begin() + 1, args.end()});
    if (cmd == "run") return cmd_run({args.begin() + 1, args.end()});
    if (cmd == "info") {
      std::string gpu;
      bool json = false;
      for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--gpu" && i + 1 < args.size())
          gpu = args[++i];
        else if (args[i] == "--json")
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
        if (args[i] == "--gpu" && i + 1 < args.size())
          gpu = args[++i];
        else if (args[i] == "-n" && i + 1 < args.size())
          n = std::stoll(args[++i]);
        else {
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
