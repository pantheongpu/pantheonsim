// Decoding a whole Annex B HEVC stream in memory: the video parser (hevc_parser.hpp) and the decoder
// (hevc_decode.hpp) joined without the cuvid API. NVDEC's cuvid shim does not use this; tests do, and so
// can anything that wants the decoded frames of a file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu_hevc {

struct OutFrame {
  int width = 0, height = 0;          // the visible size (the conformance window of the sequence)
  int bit_depth_luma = 8, bit_depth_chroma = 8;
  std::vector<uint16_t> y, u, v;      // width*height luma, ((width+1)/2)*((height+1)/2) for each chroma plane
  int64_t timestamp = 0;
};

// Decodes `n` bytes of Annex B stream; the frames come out in display order. Returns false (and says why in
// `error`) if a picture could not be decoded; the frames decoded so far are in `out` either way.
bool decode_stream(const uint8_t* data, size_t n, std::vector<OutFrame>* out, std::string* error);

}  // namespace vgpu_hevc
