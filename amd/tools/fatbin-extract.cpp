// Writes out the code objects a library or program carries for one target:
// every offload bundle in the file, compressed or not, in the order they are
// laid out, each target's code object as <outdir>/<n>.co. What ROCm's
// roc-obj-extract does, with VirtualGPU's own bundle reader, for building
// instruction corpora (amd/tools/isa-corpus.py) from real code.
//
//   amd_fatbin_extract <file> <target, e.g. gfx950:sramecc+:xnack-> <outdir>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vgpu/amd_bundle.hpp"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <file> <target> <outdir>\n", argv[0]);
    return 2;
  }
  std::ifstream in(argv[1], std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(in)), {});
  const std::string target = argv[2], outdir = argv[3];
  const auto* b = reinterpret_cast<const uint8_t*>(bytes.data());
  int written = 0, bundles = 0;
  // Bundles start 8-byte aligned; a compressed one says how long it is, so
  // what is inside it is never mistaken for another.
  for (size_t at = 0; at + 32 <= bytes.size(); at += 8) {
    const bool compressed = std::memcmp(b + at, "CCOB", 4) == 0;
    if (!compressed && std::memcmp(b + at, "__CLANG_OFFLOAD_BUNDLE__", 24) != 0) continue;
    try {
      const std::unique_ptr<vgpu::amd::Bundle> bundle = vgpu::amd::read_bundle(b + at, target);
      if (!bundle) continue;
      ++bundles;
      if (const std::string_view* code = vgpu::amd::code_for(*bundle, target)) {
        std::ofstream(outdir + "/" + std::to_string(written++) + ".co", std::ios::binary)
            .write(code->data(), static_cast<std::streamsize>(code->size()));
      }
      if (compressed) {
        uint64_t total = 0;
        const uint16_t version = static_cast<uint16_t>(b[at + 4] | b[at + 5] << 8);
        if (version == 2) {
          uint32_t t;
          std::memcpy(&t, b + at + 8, 4);
          total = t;
        } else {
          std::memcpy(&total, b + at + 8, 8);
        }
        if (total > 8) at += (total - 8) / 8 * 8;
      }
    } catch (const std::exception&) {
      // Not a bundle after all.
    }
  }
  std::printf("%d bundles, %d code objects for %s\n", bundles, written, target.c_str());
  return 0;
}
