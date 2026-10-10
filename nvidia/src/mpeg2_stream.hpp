// Decoding a whole MPEG-2 video elementary stream in memory: the video parser (mpeg2_parser.hpp) and the decoder
// (mpeg2_decode.hpp) joined without the cuvid API. NVDEC's cuvid shim does not use this; tests do, and so can anything that wants
// the decoded frames of a file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vgpu_mpeg2 {

struct OutFrame {
  int width = 0, height = 0;          // the visible size (the display rectangle of the sequence)
  std::vector<uint8_t> y, u, v;       // width*height luma, ((width+1)/2)*((height+1)/2) for each chroma plane
  int64_t timestamp = 0;
};

// Decodes `n` bytes of elementary stream with a display delay of `delay` pictures; the frames come out in display order. Returns false
// (and says why in `error`) if a picture could not be decoded; the frames decoded so far are in `out` either way.
bool decode_stream(const uint8_t* data, size_t n, std::vector<OutFrame>* out, std::string* error, unsigned delay = 0);

// The frame as the bytes of an NV12 surface cropped to the visible size: the luma rows, then the interleaved chroma rows of
// ((width+1) & ~1) bytes.
std::vector<uint8_t> to_nv12(const OutFrame& f);

}  // namespace vgpu_mpeg2
