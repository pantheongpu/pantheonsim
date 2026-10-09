// libvgpunvcomp_cpu -- nvCOMP's CPU library, presented as libnvcomp_cpu.so.5.
//
// NVIDIA ships the host-side halves of nvCOMP in a library of their own: the
// GDeflate compressor and decompressor that run on the CPU
// (nvcomp/native/gdeflate_cpu.h; they write and read the same chunks
// nvcompBatchedGdeflate* do on the GPU) and an LZ4 manager. The GDeflate
// functions are here, over the simulator's own GDeflate codec
// (nvcomp_gdeflate.cpp, which decodes every stream NVIDIA's library wrote and
// writes streams it decodes). The compressed bytes are this encoder's, not
// NVIDIA's; the sizes, the limits and the exceptions are NVIDIA's, as measured
// on libnvcomp_cpu.so.5 5.3 (it needs no GPU).
//
// Not here: nvcomp::LZ4CPUManager, the high-level manager over host buffers.
// Applications that construct one fail to link, naming it.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/vgpu_nvcomp.hpp"
#include "nvcomp_gdeflate.hpp"

namespace {

constexpr size_t kMaxChunk = 1u << 16;

[[noreturn]] void raise(const std::string& msg, nvcompStatus_t code = static_cast<nvcompStatus_t>(10)) {
  throw nvcomp::NVCompException(code, msg);
}

// GDeflate's effort scale (0 entropy coding only, 1..5 searches harder) for
// the library's 0..12 compression levels.
int effort_of(int level) { return level == 0 ? 0 : (level <= 2 ? 1 : (level <= 5 ? 2 : (level <= 8 ? 3 : (level <= 11 ? 4 : 5)))); }

// The bound compressCPUGetMaxOutputChunkSize reports (65,678 bytes for any chunk, plus
// five for each started ten thousand bytes past the first).
size_t bound_of(size_t n) { return n == 0 ? 0 : 65678 + 5 * ((n - 1) / 10000); }

}  // namespace

namespace gdeflate {

// The bound is 65,678 bytes for any chunk, plus five for each started ten
// thousand bytes past the first (measured over every size to 64 KiB).
__attribute__((visibility("default"))) void compressCPUGetMaxOutputChunkSize(size_t max_uncompressed_chunk_bytes,
                                                                             size_t* max_compressed_chunk_bytes) {
  if (!max_compressed_chunk_bytes) raise("max_compressed_chunk_bytes must not be null", static_cast<nvcompStatus_t>(1));
  if (max_uncompressed_chunk_bytes > kMaxChunk) raise("Maximum allowed chunk size for Gdeflate CPU is 64kB");
  *max_compressed_chunk_bytes = bound_of(max_uncompressed_chunk_bytes);
}

__attribute__((visibility("default"))) void compressCPU(const void* const* in_ptr, const size_t* in_bytes,
                                                         const size_t max_uncompressed_chunk_bytes, size_t batch_size,
                                                         void* const* out_ptr, size_t* out_bytes, int level) {
  if (level < 0 || level > 12) raise("Compression level must be between 0 and 12, both inclusive");
  if (max_uncompressed_chunk_bytes > kMaxChunk) raise("Maximum allowed chunk size for Gdeflate CPU is 64kB");
  if (batch_size && (!in_ptr || !in_bytes || !out_ptr || !out_bytes)) raise("null batch arrays", static_cast<nvcompStatus_t>(1));
  for (size_t i = 0; i < batch_size; ++i)
    if (in_bytes[i] > max_uncompressed_chunk_bytes) raise("max_uncompressed_chunk_bytes cannot be lower than any single chunk size");
  for (size_t i = 0; i < batch_size; ++i) {
    if (in_bytes[i] == 0) {
      out_bytes[i] = 0;
      continue;
    }
    if (!in_ptr[i] || !out_ptr[i]) raise("null chunk pointer", static_cast<nvcompStatus_t>(1));
    vgpu::codec::Bytes c =
        vgpu::codec::gdeflate_compress(static_cast<const uint8_t*>(in_ptr[i]), in_bytes[i], effort_of(level));
    // The caller sized the output by the bound above. Data that does not compress comes
    // out of the entropy coder a little larger than that, so it is stored instead (blocks
    // of at most 64 KiB), which stays within it.
    if (c.size() > bound_of(max_uncompressed_chunk_bytes)) {
      const std::vector<int> stored((in_bytes[i] + 32767) / 32768, 0);
      c = vgpu::codec::gdeflate_compress_blocks(static_cast<const uint8_t*>(in_ptr[i]), in_bytes[i], effort_of(level), stored);
    }
    if (c.size() > bound_of(max_uncompressed_chunk_bytes)) raise("Failed to compress chunk", static_cast<nvcompStatus_t>(12));
    std::memcpy(out_ptr[i], c.data(), c.size());
    out_bytes[i] = c.size();
  }
}

// A chunk that does not decode is "Failed to decompress chunk" (code 12). The
// card's library also decodes a stream whose declared length is shorter than
// the bytes it needs; this one reads only the bytes it is given.
__attribute__((visibility("default"))) void decompressCPU(const void* const* in_ptr, const size_t* in_bytes,
                                                           size_t batch_size, void* const* out_ptr,
                                                           size_t* out_buffer_bytes, size_t* out_bytes) {
  if (batch_size && (!in_ptr || !in_bytes || !out_ptr || !out_buffer_bytes || !out_bytes))
    raise("null batch arrays", static_cast<nvcompStatus_t>(1));
  for (size_t i = 0; i < batch_size; ++i) {
    size_t produced = 0;
    const vgpu::codec::Result r =
        in_bytes[i] == 0 ? vgpu::codec::Result::Corrupt
                         : vgpu::codec::gdeflate_decompress(static_cast<const uint8_t*>(in_ptr[i]), in_bytes[i],
                                                            static_cast<uint8_t*>(out_ptr[i]), out_buffer_bytes[i], &produced);
    if (r != vgpu::codec::Result::Ok) {
      out_bytes[i] = 0;
      raise("Failed to decompress chunk", static_cast<nvcompStatus_t>(12));
    }
    out_bytes[i] = produced;
  }
}

}  // namespace gdeflate
