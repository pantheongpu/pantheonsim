// The JPEG decoder inside libvgpunvjpeg, for the other libraries that decode
// JPEG on the host (NVDEC's Motion JPEG, libnvcuvid): the planes as the inverse
// DCT leaves them, on the whole MCU grid -- the padding past the picture's edge
// included, which NVDEC's surfaces carry and nvJPEG's cropped output does not.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vgpu_jpeg {

struct Plane {
  int h_samp = 1, v_samp = 1;       // sampling factors
  int width = 0, height = 0;        // samples, padded to the MCU grid
  std::vector<uint8_t> data;        // width * height
};

struct Decoded {
  int width = 0, height = 0;        // the picture
  int hmax = 1, vmax = 1;
  std::vector<Plane> planes;        // one (grey) or three (YCbCr)
};

// True for the 8-bit Huffman pictures nvJPEG decodes with one or three components.
bool decode_planes(const uint8_t* data, size_t length, Decoded* out);

}  // namespace vgpu_jpeg
