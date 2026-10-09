// Writes the field-picture (PAFF) fixtures: the frames of an interlaced MBAFF fixture, decoded, are coded again as pairs of field
// pictures by the NVENC H.264 encoder's field mode (nvenc_h264_enc.cpp). No other encoder at hand writes field pictures: x264 writes
// MBAFF, and the card's NVENC refuses field encoding. The streams are checked against ffmpeg and against the card's NVDEC (the
// transcript in nvidia/tests/e2e/nvcuvid_h264.rtx3060.txt), not against the encoder.
//
//   g++-12 -std=c++20 -I nvidia/src nvidia/tests/data/h264/make_paff.cpp nvidia/src/{nvenc_h264_enc,h264_decode,h264_parser,h264_stream}.cpp
//   -o /tmp/make_paff && cd nvidia/tests/data/h264 && /tmp/make_paff
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "h264_stream.hpp"
#include "nvenc_h264_enc.hpp"

using namespace vgpu_nvenc;

static void make(const char* source, const char* name, int qp, bool tff, int profile, bool deblock, bool all_idr, int frames) {
  std::ifstream f(source, std::ios::binary);
  const std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<vgpu_h264::OutFrame> in;
  std::string err;
  if (!vgpu_h264::decode_stream(data.data(), data.size(), &in, &err) || in.empty()) {
    std::fprintf(stderr, "%s: %s\n", source, err.c_str());
    return;
  }
  const int w = in[0].width, h = in[0].height;
  H264Encoder enc(w, h, 30, 1, profile, deblock, true);
  std::vector<uint8_t> out = enc.parameter_sets();
  for (size_t i = 0; i < in.size() && static_cast<int>(i) < frames; ++i) {
    EncPicture p;
    p.w = w;
    p.h = h;
    p.y = in[i].y;
    const int cw = (w + 1) / 2, ch = (h + 1) / 2;
    p.u.resize(static_cast<size_t>(cw) * ch);
    p.v.resize(p.u.size());
    for (size_t k = 0; k < p.u.size(); ++k) {
      p.u[k] = in[i].uv[2 * k];
      p.v[k] = in[i].uv[2 * k + 1];
    }
    const auto nal = enc.encode_field_pair(p, tff, (i == 0 || all_idr) ? PicType::kIdr : PicType::kInter, qp);
    out.insert(out.end(), nal.begin(), nal.end());
  }
  std::ofstream(std::string(name) + ".h264", std::ios::binary).write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
  std::printf("%s: %zu bytes\n", name, out.size());
}

int main() {
  make("mbaff_field.h264", "paff", 28, true, 100, true, false, 6);
  make("mbaff_field.h264", "paff_bff", 28, false, 100, true, false, 6);
  make("mbaff_field.h264", "paff_nodeblock", 24, true, 77, false, false, 5);
  make("mbaff_field_intra.h264", "paff_intra", 26, true, 100, true, true, 3);
  make("mbaff_field_crop.h264", "paff_crop", 30, true, 100, true, false, 5);
  return 0;
}
