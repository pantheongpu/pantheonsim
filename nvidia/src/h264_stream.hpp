// Decoding a whole Annex B H.264 stream in memory: the video parser (h264_parser.hpp) and the
// decoder (h264_decode.hpp) joined without the cuvid API. NVDEC's cuvid shim does not use this;
// tests do, and so can anything that wants the decoded frames of a file (the NVENC tests decode
// the encoder's output with it).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu_h264 {

struct OutFrame {
  int width = 0, height = 0;          // the visible size (the cropping rectangle of the sequence)
  std::vector<uint8_t> y, uv;         // width*height luma; interleaved Cb Cr, ((width+1)/2)*((height+1)/2) pairs
  int64_t timestamp = 0;
};

// Decodes `n` bytes of Annex B stream; the frames come out in display order. Returns false (and says why in
// `error`) if a picture could not be decoded; the frames decoded so far are in `out` either way.
bool decode_stream(const uint8_t* data, size_t n, std::vector<OutFrame>* out, std::string* error);

}  // namespace vgpu_h264
