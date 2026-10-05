// What libvgpunvcomp's low-level API (nvcomp_api.cpp) shares with its
// high-level manager API (nvcomp_manager.cpp): the formats, their bounds and
// codecs, and moving bytes to and from simulated device memory.
#ifndef VGPU_NVCOMP_INTERNAL_HPP
#define VGPU_NVCOMP_INTERNAL_HPP

#include <cstddef>
#include <cstdint>
#include <functional>

#include "../include/vgpu_nvcomp.h"
#include "nvcomp_codecs.hpp"

namespace vgpu::nvcomp_impl {

enum class Fmt { LZ4, Snappy, Deflate, Gdeflate, Gzip, Zstd };

bool quiet();
void say_once(const char* key, const char* msg);
// n bytes at p are readable and writable memory: one device allocation, or
// host memory a kernel could reach.
bool accessible(const void* p, size_t n);
bool get(void* host, const void* src, size_t n);
bool put(void* dst, const void* host, size_t n);
// Runs `work` in stream order (as a host node while the stream is captured).
nvcompStatus_t in_stream_order(cudaStream_t stream, std::function<void()> work);
// nvcompBatched*CompressGetMaxOutputChunkSize's answer for a chunk of n bytes.
size_t max_output(Fmt f, size_t n);
codec::Bytes compress(Fmt f, const uint8_t* in, size_t n, int level);
codec::Result decompress(Fmt f, const uint8_t* in, size_t n, uint8_t* out, size_t cap, size_t* produced);

}  // namespace vgpu::nvcomp_impl

#endif  // VGPU_NVCOMP_INTERNAL_HPP
