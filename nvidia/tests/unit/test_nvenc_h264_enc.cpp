// The compressing H.264 encoder behind NVENC (nvenc_h264_enc.cpp), read back by the decoder of this
// tree (h264_stream.cpp: the video parser and the H.264 decoder NVDEC uses, bit-exact against the
// card and ffmpeg on every fixture in nvidia/tests/data/h264). No ffmpeg needed (where it is installed, the
// last test also has it decode a stream and compares with the decoder of this tree):
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
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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


// ---- the tools beyond the first encoder: CABAC, the 8x8 transform, slices, references, B pictures -----------------------------------------------

namespace {

// Codes `n` pictures of the scene in the order an application with `max_b` B pictures between P pictures would; returns the stream, and the encoder's
// reconstruction of every picture in display order (the loop filter is off, so these are what a decoder returns).
struct CodedSeq {
  std::vector<uint8_t> stream;
  std::vector<std::vector<uint8_t>> recon_y;   // by display order, coded_width() x coded_height()
  std::vector<int> types;                      // coding order
  int stride = 0;
};
CodedSeq code_seq(int w, int h, int n, int qp, vgpu_nvenc::H264Options opt, int profile, int gop = 0) {
  CodedSeq c;
  opt.deblock = false;
  H264Encoder enc(w, h, 30, 1, profile, opt);
  c.stream = enc.parameter_sets();
  c.recon_y.resize(static_cast<size_t>(n));
  c.stride = enc.coded_width();
  auto code = [&](int d, PicType t, int q) {
    const auto nal = enc.encode_at(scene(w, h, 2 * d, d), t, q, d);
    c.stream.insert(c.stream.end(), nal.begin(), nal.end());
    c.types.push_back(static_cast<int>(t));
    c.recon_y[static_cast<size_t>(d)] = enc.recon_y();
  };
  for (int d = 0; d < n;) {
    if (d == 0 || (gop && d % gop == 0)) {
      code(d, PicType::kIdr, qp);
      ++d;
      continue;
    }
    const int remaining = gop ? gop - d % gop : n - d;
    const int span = std::min(std::min(opt.max_b + 1, n - d), remaining);
    const int anchor = d + span - 1;
    code(anchor, PicType::kInter, qp);
    for (int b = d; b < anchor; ++b) code(b, PicType::kBi, qp + 2);
    d = anchor + 1;
  }
  return c;
}

// Every decoded frame is the encoder's reconstruction of that picture (luma; the chroma is checked on the last picture by equals_recon in the tests above).
bool decodes_to_recon(const CodedSeq& c, int w, int h) {
  const auto frames = decode(c.stream);
  if (frames.size() != c.recon_y.size()) return false;
  for (size_t i = 0; i < frames.size(); ++i) {
    if (frames[i].width < w || frames[i].height < h) return false;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        if (c.recon_y[i][static_cast<size_t>(y) * c.stride + x] != frames[i].y[static_cast<size_t>(y) * frames[i].width + x]) return false;
  }
  return true;
}

}  // namespace

VTEST(cabac_and_the_8x8_transform_decode_to_the_encoders_reconstruction) {
  const int sizes[][2] = {{16, 16}, {33, 20}, {96, 64}};
  for (const auto& s : sizes)
    for (const int qp : {0, 28, 51}) {
      for (int variant = 0; variant < 4; ++variant) {
        vgpu_nvenc::H264Options opt;
        opt.cabac = variant & 1;
        opt.transform8x8 = variant & 2;
        opt.num_ref = 2;
        const CodedSeq c = code_seq(s[0], s[1], 6, qp, opt, opt.transform8x8 ? 100 : 77);
        VCHECK(decodes_to_recon(c, s[0], s[1]));
      }
    }
}

VTEST(several_slices_decode_to_the_encoders_reconstruction_and_are_counted) {
  const int w = 96, h = 80;   // 6 x 5 macroblocks
  const struct {
    int mode, data, slices;
  } layouts[] = {{0, 7, 5}, {2, 2, 3}, {3, 4, 4}, {3, 1, 1}, {0, 0, 1}};
  for (const auto& l : layouts)
    for (const bool cabac : {false, true}) {
      vgpu_nvenc::H264Options opt;
      opt.cabac = cabac;
      opt.slice_mode = l.mode;
      opt.slice_data = l.data;
      H264Encoder enc(w, h, 30, 1, 77, opt);
      EncStats st;
      const auto nal = enc.encode(scene(w, h, 0, 0), PicType::kIdr, 28, &st);
      VCHECK_EQ(st.slices, l.slices);
      VCHECK_EQ(enc.last_slices(), l.slices);
      int slice_nals = 0;
      for (int t : nal_types(nal)) slice_nals += t == 5;
      VCHECK_EQ(slice_nals, l.slices);
      opt.deblock = false;
      const CodedSeq c = code_seq(w, h, 4, 24, opt, 77);
      VCHECK(decodes_to_recon(c, w, h));
    }
}

VTEST(b_pictures_decode_to_the_encoders_reconstruction_in_display_order) {
  const int sizes[][2] = {{32, 32}, {97, 53}};
  for (const auto& s : sizes)
    for (const int qp : {14, 34})
      for (const int bf : {1, 4})
        for (const bool cabac : {false, true}) {
          vgpu_nvenc::H264Options opt;
          opt.cabac = cabac;
          opt.transform8x8 = cabac;
          opt.max_b = bf;
          opt.num_ref = 2;
          const CodedSeq c = code_seq(s[0], s[1], 10, qp, opt, 100);
          VCHECK(decodes_to_recon(c, s[0], s[1]));
          VCHECK(c.types.size() == 10u);
          VCHECK(c.types[1] == static_cast<int>(PicType::kInter));
          VCHECK(c.types[2] == static_cast<int>(PicType::kBi));
        }
  // IDR pictures in the middle, a group cut short by the end of the GOP
  vgpu_nvenc::H264Options opt;
  opt.cabac = true;
  opt.max_b = 3;
  const CodedSeq c = code_seq(64, 48, 13, 26, opt, 77, 6);
  VCHECK(decodes_to_recon(c, 64, 48));
}

VTEST(profiles_without_the_tools_do_not_use_them_and_b_pictures_cost_less_than_p_pictures) {
  vgpu_nvenc::H264Options opt;
  opt.cabac = true;
  opt.transform8x8 = true;
  opt.max_b = 3;
  H264Encoder base(64, 48, 30, 1, 66, opt);   // Baseline: CAVLC, the 4x4 transform, no B pictures
  VCHECK(!base.supports_b());
  // the PPS of the Baseline stream says CAVLC (entropy_coding_mode_flag is the third field)
  const auto ps = base.parameter_sets();
  size_t pps = 0;
  for (size_t i = 0; i + 5 < ps.size(); ++i)
    if (ps[i] == 0 && ps[i + 1] == 0 && ps[i + 2] == 0 && ps[i + 3] == 1 && (ps[i + 4] & 31) == 8) pps = i + 5;
  VCHECK(pps != 0);
  VCHECK_EQ(static_cast<int>(ps[pps] >> 5 & 1), 0);   // after the two one-bit ue(0) fields: pic_parameter_set_id 1, seq_parameter_set_id 1, then entropy_coding_mode_flag
  // with B pictures: the B pictures of a group are cheaper than its P picture
  opt.num_ref = 2;
  H264Encoder enc(160, 96, 30, 1, 100, opt);
  VCHECK(enc.supports_b());
  size_t p_bytes = 0, b_bytes = 0;
  enc.encode_at(scene(160, 96, 0, 0), PicType::kIdr, 28, 0);
  for (int g = 0; g < 3; ++g) {
    p_bytes += enc.encode_at(scene(160, 96, 6 * (g + 1), 3 * (g + 1)), PicType::kInter, 28, 3 * (g + 1)).size();
    for (int b = 2; b >= 1; --b) b_bytes += enc.encode_at(scene(160, 96, 2 * (3 * g + b), 3 * g + b), PicType::kBi, 30, 3 * g + b).size();
  }
  VCHECK(b_bytes < 2 * p_bytes);
}

VTEST(rollback_returns_the_encoder_to_the_state_before_a_picture) {
  vgpu_nvenc::H264Options opt;
  opt.cabac = true;
  opt.max_b = 1;
  opt.num_ref = 2;
  H264Encoder a(96, 64, 30, 1, 100, opt), b(96, 64, 30, 1, 100, opt);
  a.encode_at(scene(96, 64, 0, 0), PicType::kIdr, 28, 0);
  b.encode_at(scene(96, 64, 0, 0), PicType::kIdr, 28, 0);
  const auto first_try = a.encode_at(scene(96, 64, 4, 2), PicType::kInter, 18, 2);
  a.rollback();
  const auto p = a.encode_at(scene(96, 64, 4, 2), PicType::kInter, 34, 2);
  VCHECK(p != first_try);
  VCHECK(p == b.encode_at(scene(96, 64, 4, 2), PicType::kInter, 34, 2));
  // a B picture after it, and the next group, are as if the first try had not happened
  VCHECK(a.encode_at(scene(96, 64, 2, 1), PicType::kBi, 34, 1) == b.encode_at(scene(96, 64, 2, 1), PicType::kBi, 34, 1));
  VCHECK(a.encode_at(scene(96, 64, 8, 4), PicType::kInter, 30, 4) == b.encode_at(scene(96, 64, 8, 4), PicType::kInter, 30, 4));
}

VTEST(ffmpeg_decodes_what_the_encoder_wrote_where_it_is_installed) {
  if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
    std::printf("note: ffmpeg not found, so no stream is decoded by it here (the decoder of this tree has read all of them above)\n");
    return;
  }
  vgpu_nvenc::H264Options opt;
  opt.cabac = true;
  opt.transform8x8 = true;
  opt.max_b = 2;
  opt.num_ref = 2;
  opt.slice_mode = 3;
  opt.slice_data = 3;
  const CodedSeq c = code_seq(96, 64, 8, 26, opt, 100);
  const std::string file = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/vgpu_h264_enc_" + std::to_string(getpid()) + ".h264";
  FILE* f = std::fopen(file.c_str(), "wb");
  VCHECK(f != nullptr);
  if (!f) return;
  std::fwrite(c.stream.data(), 1, c.stream.size(), f);
  std::fclose(f);
  FILE* p = popen(("ffmpeg -v error -f h264 -i " + file + " -f rawvideo -pix_fmt yuv420p - 2>/dev/null").c_str(), "r");
  std::vector<uint8_t> dec;
  if (p) {
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) dec.insert(dec.end(), buf, buf + n);
    VCHECK(pclose(p) == 0);
  }
  std::remove(file.c_str());
  const size_t frame_bytes = 96 * 64 * 3 / 2;
  VCHECK_EQ(dec.size(), frame_bytes * 8);
  if (dec.size() != frame_bytes * 8) return;
  for (int i = 0; i < 8; ++i)
    for (int y = 0; y < 64; ++y)
      VCHECK(std::memcmp(&dec[frame_bytes * i + static_cast<size_t>(y) * 96], &c.recon_y[static_cast<size_t>(i)][static_cast<size_t>(y) * c.stride], 96) == 0);
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
