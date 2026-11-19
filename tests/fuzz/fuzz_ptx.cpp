// Fuzz target: the PTX parser and its register analysis. Untrusted PTX must come back as a vgpu::Error (PtxParse,
// UnsupportedPtx, PtxVersion), never a crash, a sanitizer report, a hang, or another kind of exception.
#include <cstddef>
#include <cstdint>
#include <string>

#include "vgpu/error.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/ptx/regalloc.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > (1u << 20)) return 0;
  const std::string src(reinterpret_cast<const char*>(data), size);
  try {
    const vgpu::ptx::Module m = vgpu::ptx::parse(src);
    for (const auto& e : m.entries) (void)vgpu::ptx::analyze_registers(e);
  } catch (const vgpu::Error&) {
    // The documented answer to input that is not PTX the simulator runs.
  }
  return 0;
}
