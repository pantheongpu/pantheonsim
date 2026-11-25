// Fuzz target: the SASS decoder and the cubin reader. Input byte 0 picks the architecture and the mode.
//   mode 0: the rest is 16-byte instruction words; each is decoded and printed.
//   mode 1: the rest is a cubin image: parse_cubin, then every kernel's text section is decoded.
// Anything that is not an instruction or an image the simulator takes must come back as a vgpu::Error.
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/sass/cubin.hpp"
#include "vgpu/sass/sass.hpp"

namespace {
constexpr int kArchs[] = {75, 80, 86, 87, 89, 90, 100, 103, 120, 121};

void decode_words(const uint8_t* p, size_t n, int sm) {
  for (size_t at = 0; at + 16 <= n; at += 16) {
    vgpu::sass::Word w;
    std::memcpy(&w.lo, p + at, 8);
    std::memcpy(&w.hi, p + at + 8, 8);
    try {
      const vgpu::sass::Instr ins = vgpu::sass::decode(w, at, sm);
      (void)vgpu::sass::to_text(ins);
    } catch (const vgpu::Error&) {
    }
  }
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 1 || size > (1u << 20)) return 0;
  const int sm = kArchs[(data[0] >> 1) % (sizeof kArchs / sizeof *kArchs)];
  const bool image = data[0] & 1;
  ++data;
  --size;
  if (!image) {
    decode_words(data, size, sm);
    return 0;
  }
  try {
    (void)vgpu::sass::cubin_arch_specific(data, size);
    (void)vgpu::sass::is_cubin(data, size);
    const vgpu::sass::Cubin c = vgpu::sass::parse_cubin(data, size);
    for (const auto& k : c.kernels)
      if (const auto* s = c.section(k.text_section)) decode_words(s->bytes.data(), s->bytes.size(), c.sm ? c.sm : sm);
  } catch (const vgpu::Error&) {
  }
  return 0;
}
