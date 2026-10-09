// The compressing H.264 encoder behind NVENC (nvenc_h264_enc.cpp), read back by the decoder of this
// tree (h264_stream.cpp: the video parser and the H.264 decoder NVDEC uses, bit-exact against the
// card and ffmpeg on every fixture in nvidia/tests/data/h264). No ffmpeg needed:
//
//  * the encoder's own reconstruction (made with the loop filter off) equals what the decoder
//    returns for the bytes it wrote -- intra pictures and P pictures, every size from one
//    macroblock up -- so what the encoder believes it coded is what the stream says;
//  * pictures are close to the input and far smaller than raw ones, smaller as QP rises;
//  * an IDR picture of one frame at one QP is the same bytes every time and a changed pixel
//    changes them (the contract of the pantheon encoder SDC workload);
//  * P pictures find motion, skip static content, and the first P picture without a reference
//    becomes an IDR picture.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../../src/h264_stream.hpp"
#include "../../src/nvenc_h264_enc.hpp"
#include "vtest.hpp"

namespace {

using vgpu_nvenc::EncPicture;
using vgpu_nvenc::EncStats;
using vgpu_nvenc::H264Encoder;
using vgpu_nvenc::PicType;

int tri(int v) {
  v &= 255;
  return v < 128 ? v : 255 - v;
}
int noise(int x, int y, int t, int amp) {
  uint32_t v = static_cast<uint32_t>(x) * 73856093u ^ static_cast<uint32_t>(y) * 19349663u ^ static_cast<uint32_t>(t) * 83492791u;
  v ^= v >> 13;
  v *= 0x5bd1e995u;
  v ^= v >> 15;
  return static_cast<int>(v % static_cast<uint32_t>(2 * amp + 1)) - amp;
}

// Smooth moving shapes with texture. `dx`, `dy` move the whole scene (in samples); noise is fixed to the scene.
EncPicture scene(int w, int h, int dx, int dy, int t = 0) {
  EncPicture p;
  p.w = w;
  p.h = h;
  p.y.resize(static_cast<size_t>(w) * h);
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  p.u.resize(static_cast<size_t>(cw) * ch);
  p.v.resize(p.u.size());
  auto luma = [&](int x, int y) {
    x -= dx;
    y -= dy;
    int v = 60 + tri(x * 3 + y) / 2 + ((x / 12 + y / 12) & 1) * 40 + noise(x, y, 0, 5);
    if ((x - 40) * (x - 40) + (y - 30) * (y - 30) < 400) v += 70;   // a disc
    return static_cast<uint8_t>(std::min(255, std::max(0, v + t)));
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) p.y[static_cast<size_t>(y) * w + x] = luma(x, y);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      p.u[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(110 + tri((x - dx / 2) * 2 + (y - dy / 2)) / 3 + noise(x, y, 1, 2));
      p.v[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(130 - tri((x - dx / 2) + (y - dy / 2) * 2) / 3 + noise(x, y, 2, 2));
    }
  return p;
}

double psnr_y(const EncPicture& in, const vgpu_h264::OutFrame& out) {
  double sq = 0;
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      const double d = static_cast<double>(in.y[static_cast<size_t>(y) * in.w + x]) - out.y[static_cast<size_t>(y) * out.width + x];
      sq += d * d;
    }
  return sq == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * in.w * in.h / sq);
}

// The NAL unit types in an Annex B stream, in order.
std::vector<int> nal_types(const std::vector<uint8_t>& s) {
  std::vector<int> types;
  for (size_t i = 0; i + 3 < s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) types.push_back(s[i + 3] & 31);
  return types;
}

// The decoder's frames of a stream; the stream must decode.
std::vector<vgpu_h264::OutFrame> decode(const std::vector<uint8_t>& stream) {
  std::vector<vgpu_h264::OutFrame> frames;
  std::string err;
  const bool ok = vgpu_h264::decode_stream(stream.data(), stream.size(), &frames, &err);
  if (!ok) std::fprintf(stderr, "decode error: %s\n", err.c_str());
  VCHECK(ok);
  return frames;
}

// True if the decoded frame is exactly the encoder's (pre-loop-filter) reconstruction.
bool equals_recon(const H264Encoder& e, const vgpu_h264::OutFrame& f) {
  const int cw = (f.width + 1) / 2, ch = (f.height + 1) / 2;
  const int stride = e.coded_width(), cstride = e.coded_width() / 2;
  for (int y = 0; y < f.height; ++y)
    for (int x = 0; x < f.width; ++x)
      if (e.recon_y()[static_cast<size_t>(y) * stride + x] != f.y[static_cast<size_t>(y) * f.width + x]) return false;
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      const size_t i = static_cast<size_t>(y) * cw + x;
      if (e.recon_u()[static_cast<size_t>(y) * cstride + x] != f.uv[2 * i] || e.recon_v()[static_cast<size_t>(y) * cstride + x] != f.uv[2 * i + 1]) return false;
    }
  return true;
}

}  // namespace

VTEST(an_intra_picture_decodes_to_the_encoders_reconstruction) {
  const int sizes[][2] = {{16, 16}, {17, 17}, {33, 20}, {64, 48}, {70, 38}, {161, 51}};
  for (const auto& s : sizes)
    for (const int qp : {0, 12, 28, 40, 51}) {
      H264Encoder enc(s[0], s[1], 30, 1, 66, false);
      const EncPicture in = scene(s[0], s[1], 0, 0);
      EncStats st;
      std::vector<uint8_t> stream = enc.parameter_sets();
      const auto nal = enc.encode(in, PicType::kIdr, qp, &st);
      stream.insert(stream.end(), nal.begin(), nal.end());
      const auto frames = decode(stream);
      VCHECK_EQ(frames.size(), size_t{1});
      if (frames.size() != 1) continue;
      VCHECK(equals_recon(enc, frames[0]));
      if (qp <= 28) VCHECK(psnr_y(in, frames[0]) > 34.0);
      VCHECK(frames[0].width >= s[0] && frames[0].width <= s[0] + 1 && frames[0].height >= s[1] && frames[0].height <= s[1] + 1);
    }
}

VTEST(p_pictures_decode_to_the_encoders_reconstruction) {
  const int sizes[][2] = {{16, 16}, {33, 20}, {96, 64}, {161, 51}};
  for (const auto& s : sizes)
    for (const int qp : {10, 26, 45}) {
      H264Encoder enc(s[0], s[1], 30, 1, 100, false);
      std::vector<uint8_t> stream = enc.parameter_sets();
      for (int t = 0; t < 6; ++t) {
        const EncPicture in = scene(s[0], s[1], 2 * t, t);   // the scene moves 2 right and 1 down per picture
        const auto nal = enc.encode(in, t == 0 ? PicType::kIdr : PicType::kInter, qp);
        stream.insert(stream.end(), nal.begin(), nal.end());
      }
      // the whole stream decodes, and the last picture is the encoder's reconstruction of it
      const auto frames = decode(stream);
      VCHECK_EQ(frames.size(), size_t{6});
      if (frames.size() == 6) VCHECK(equals_recon(enc, frames[5]));
    }
}

VTEST(pictures_are_compressed_more_as_qp_rises_and_stay_close_to_the_input) {
  const EncPicture in = scene(160, 96, 0, 0);
  size_t prev_bytes = 1u << 30;
  double prev_psnr = 100;
  for (const int qp : {10, 18, 26, 34, 42}) {
    H264Encoder enc(160, 96, 30, 1, 100, true);
    EncStats st;
    std::vector<uint8_t> stream = enc.parameter_sets();
    const auto nal = enc.encode(in, PicType::kIdr, qp, &st);
    stream.insert(stream.end(), nal.begin(), nal.end());
    const auto frames = decode(stream);
    VCHECK_EQ(frames.size(), size_t{1});
    if (frames.empty()) continue;
    const double psnr = psnr_y(in, frames[0]);
    VCHECK(nal.size() < prev_bytes);
    VCHECK(psnr < prev_psnr + 0.01);
    VCHECK(std::fabs(psnr - st.psnr_y) < 1e-9);   // the encoder's figure is the decoded picture's
    VCHECK(nal.size() < 160u * 96 * 3 / 2 / 2);
    if (qp <= 26) VCHECK(psnr > 30.0);
    prev_bytes = nal.size();
    prev_psnr = psnr;
  }
}

VTEST(one_frame_at_one_qp_is_one_bitstream_and_a_changed_pixel_changes_it) {
  EncPicture in = scene(96, 64, 0, 0);
  H264Encoder a(96, 64, 30, 1, 100, true), b(96, 64, 30, 1, 100, true);
  const auto first = a.encode(in, PicType::kIdr, 28);
  VCHECK(first == b.encode(in, PicType::kIdr, 28));
  // encoding it again, as an IDR picture, gives the same bytes whatever came before
  VCHECK(first == a.encode(in, PicType::kIdr, 28));
  in.y[40 * 96 + 50] = static_cast<uint8_t>(in.y[40 * 96 + 50] ^ 0x5a);
  VCHECK(first != a.encode(in, PicType::kIdr, 28));
  in.y[40 * 96 + 50] = static_cast<uint8_t>(in.y[40 * 96 + 50] ^ 0x5a);
  VCHECK(first == a.encode(in, PicType::kIdr, 28));
  // a chroma change too
  in.v[10 * 48 + 20] = static_cast<uint8_t>(in.v[10 * 48 + 20] ^ 0x5a);
  VCHECK(first != a.encode(in, PicType::kIdr, 28));
  in.v[10 * 48 + 20] = static_cast<uint8_t>(in.v[10 * 48 + 20] ^ 0x5a);
  // a one-level change that quantisation erases from the coded picture still changes the bytes (the digest)
  H264Encoder coarse(96, 64, 30, 1, 100, true);
  const auto base = coarse.encode(in, PicType::kIdr, 51);
  in.y[10 * 96 + 10] = static_cast<uint8_t>(in.y[10 * 96 + 10] + 1);
  const auto bumped = coarse.encode(in, PicType::kIdr, 51);
  VCHECK(base != bumped);
  VCHECK_EQ(base.size(), bumped.size());   // only the digest differs
}

VTEST(p_pictures_follow_motion_and_skip_what_is_still) {
  const int w = 128, h = 96;
  H264Encoder enc(w, h, 30, 1, 100, true);
  EncStats st0, st1, st2, st3;
  const EncPicture f0 = scene(w, h, 0, 0);
  const auto idr = enc.encode(f0, PicType::kIdr, 28, &st0);
  // the same frame again as a P picture: nearly everything is skipped
  const auto still = enc.encode(f0, PicType::kInter, 28, &st1);
  VCHECK(still.size() * 20 < idr.size());
  VCHECK(st1.skipped_mbs * 10 >= (w / 16) * (h / 16) * 9);
  // the scene moved by (5, -3) samples: found by the search, so much cheaper than intra
  const EncPicture moved = scene(w, h, 5, -3);
  const auto mv = enc.encode(moved, PicType::kInter, 28, &st2);
  H264Encoder intra_only(w, h, 30, 1, 100, true);
  const auto intra = intra_only.encode(moved, PicType::kIdr, 28);
  VCHECK(mv.size() * 2 < intra.size());
  VCHECK(st2.inter_mbs + st2.skipped_mbs > st2.intra_mbs * 3);
  // a quarter-sample move (the average of the frame and its neighbour)
  EncPicture half = f0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) half.y[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>((f0.y[static_cast<size_t>(y) * w + x] + f0.y[static_cast<size_t>(y) * w + std::min(x + 1, w - 1)] + 1) / 2);
  enc.encode(f0, PicType::kIdr, 28);
  const auto frac = enc.encode(half, PicType::kInter, 28, &st3);
  H264Encoder intra2(w, h, 30, 1, 100, true);
  VCHECK(frac.size() * 2 < intra2.encode(half, PicType::kIdr, 28).size());
}

VTEST(a_p_picture_without_a_reference_is_an_idr_picture_and_headers_name_the_profile) {
  for (const int profile : {66, 77, 100}) {
    H264Encoder enc(64, 48, 30, 1, profile, true);
    const auto ps = enc.parameter_sets();
    VCHECK(ps.size() > 8);
    VCHECK_EQ(static_cast<int>(ps[3]), 1);                 // start code 00 00 00 01
    VCHECK_EQ(static_cast<int>(ps[4] & 31), 7);            // an SPS
    VCHECK_EQ(static_cast<int>(ps[5]), profile);           // profile_idc
    // an IDR picture is a digest SEI message and an IDR slice; a P picture is one non-IDR slice
    const auto nal = enc.encode(scene(64, 48, 0, 0), PicType::kInter, 30);
    VCHECK(nal_types(nal) == (std::vector<int>{6, 5}));
    const auto next = enc.encode(scene(64, 48, 1, 0), PicType::kInter, 30);
    VCHECK(nal_types(next) == (std::vector<int>{1}));
    enc.reset();
    const auto again = enc.encode(scene(64, 48, 2, 0), PicType::kInter, 30);
    VCHECK(nal_types(again) == (std::vector<int>{6, 5}));
  }
}

VTEST(field_pictures_decode_to_the_encoders_reconstruction_and_predict_across_parities) {
  // Interlaced frames coded as pairs of field pictures (the field coding mode that makes the PAFF fixtures): the second field of
  // every frame is the encoder's last reconstruction, and the decoder must return exactly it, whichever parity comes first.
  for (const bool tff : {true, false})
    for (const int profile : {66, 100}) {
      const int w = 64, h = 64;
      H264Encoder enc(w, h, 30, 1, profile, false, true);
      std::vector<uint8_t> stream = enc.parameter_sets();
      std::vector<EncPicture> input;
      std::vector<std::vector<uint8_t>> second_field;   // the encoder's reconstruction of each frame's second field
      for (int t = 0; t < 4; ++t) {
        EncPicture in = scene(w, h, 3 * t, t);
        EncStats st;
        const auto nal = enc.encode_field_pair(in, tff, t == 0 ? PicType::kIdr : PicType::kInter, 26, &st);
        stream.insert(stream.end(), nal.begin(), nal.end());
        second_field.push_back(enc.recon_y());
        VCHECK(st.psnr_y > 30.0);
        input.push_back(std::move(in));
      }
      const auto frames = decode(stream);
      VCHECK_EQ(frames.size(), size_t{4});
      for (size_t i = 0; i < frames.size() && i < 4; ++i) {
        const int par = tff ? 1 : 0;   // the parity of the second field
        bool same = frames[i].width == w && frames[i].height == h;
        for (int y = 0; y < h / 2 && same; ++y)
          for (int x = 0; x < w; ++x)
            if (second_field[i][static_cast<size_t>(y) * enc.coded_width() + x] != frames[i].y[static_cast<size_t>(2 * y + par) * w + x]) {
              same = false;
              break;
            }
        VCHECK(same);
        VCHECK(psnr_y(input[i], frames[i]) > 30.0);
      }
    }
}

VTEST(the_initial_qp_falls_as_the_bit_rate_rises) {
  int prev = 100;
  for (const long rate : {50000L, 200000L, 800000L, 3000000L, 12000000L}) {
    const int qp = vgpu_nvenc::initial_qp_for(640, 360, 30, 1, rate);
    VCHECK(qp <= prev);
    VCHECK(qp >= 10 && qp <= 48);
    prev = qp;
  }
}

VTEST_MAIN
