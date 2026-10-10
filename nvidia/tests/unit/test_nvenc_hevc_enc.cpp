// The compressing HEVC encoder behind NVENC (nvenc_hevc_enc.cpp).
//
// There is no HEVC decoder in this tree to read its streams back (the writer of the lossless tuning is checked by
// test_nvenc_hevc with a decoder of its own for the three syntax elements it uses; a decoder for everything this encoder
// writes would be a project of its own). So the checks are of two kinds:
//
//  * with ffmpeg installed, every stream is decoded by ffmpeg's HEVC decoder and each picture must equal the encoder's own
//    reconstruction sample for sample -- intra pictures at every size and QP, P pictures with several references, B pictures
//    (the encoder codes the loop filters off, so what a decoder returns is exactly the reconstruction). That is the proof that
//    the syntax, the arithmetic coder's contexts and the prediction process are those of ITU-T H.265. Without ffmpeg the test
//    prints a note and skips only that comparison;
//  * with or without it: the parameter sets name the picture size and the tools, the coded size follows the QP, the
//    reconstruction is close to the input and P and B pictures cost less than intra pictures, the same picture at the same QP
//    is the same bytes (and a changed sample is not), and rollback() returns the encoder to its state before a picture.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/nvenc_hevc_enc.hpp"
#include "vtest.hpp"

namespace {

using vgpu_nvenc::EncPicture;
using vgpu_nvenc::EncStats;
using vgpu_nvenc::HevcEncoder;
using vgpu_nvenc::HevcOptions;
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

// Moving shapes with texture; `t` moves the scene.
EncPicture scene(int w, int h, int t) {
  EncPicture p;
  p.w = w;
  p.h = h;
  p.y.resize(static_cast<size_t>(w) * h);
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  p.u.resize(static_cast<size_t>(cw) * ch);
  p.v.resize(p.u.size());
  const int dx = 2 * t, dy = t;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int X = x - dx, Y = y - dy;
      int v = 60 + tri(X * 3 + Y) / 2 + ((X / 12 + Y / 12) & 1) * 40 + noise(X, Y, 0, 5);
      if ((X - 40) * (X - 40) + (Y - 30) * (Y - 30) < 400) v += 70;
      p.y[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::min(255, std::max(0, v)));
    }
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      p.u[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(110 + tri((x - t) * 2 + (y - t / 2)) / 3 + noise(x, y, 1, 2));
      p.v[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(130 - tri((x - t) + (y - t / 2) * 2) / 3 + noise(x, y, 2, 2));
    }
  return p;
}

double psnr_y(const EncPicture& in, const HevcEncoder& e) {
  double sq = 0;
  for (int y = 0; y < in.h; ++y)
    for (int x = 0; x < in.w; ++x) {
      const double d = static_cast<double>(in.y[static_cast<size_t>(y) * in.w + x]) - e.recon_y()[static_cast<size_t>(y) * e.coded_width() + x];
      sq += d * d;
    }
  return sq == 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 * in.w * in.h / sq);
}

// ffmpeg, as the reference decoder: the test passes without it and says so.
bool have_ffmpeg() {
  static const bool ok = std::system("command -v ffmpeg >/dev/null 2>&1") == 0;
  return ok;
}
bool note_no_ffmpeg() {
  static bool noted = false;
  if (!have_ffmpeg() && !noted) {
    std::printf("note: ffmpeg not found, so streams are not decoded and compared with the reconstruction here\n");
    noted = true;
  }
  return !have_ffmpeg();
}

// The pictures of `stream` as ffmpeg returns them (yuv420p, display order), (w+1)&~1 by (h+1)&~1.
bool ffmpeg_decode(const std::vector<uint8_t>& stream, std::vector<uint8_t>* out) {
  const std::string file = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/vgpu_hevc_enc_" + std::to_string(getpid()) + ".hevc";
  FILE* f = std::fopen(file.c_str(), "wb");
  if (!f) return false;
  std::fwrite(stream.data(), 1, stream.size(), f);
  std::fclose(f);
  const std::string cmd = "ffmpeg -v error -f hevc -i " + file + " -f rawvideo -pix_fmt yuv420p - 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r");
  bool ok = p != nullptr;
  if (p) {
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, p)) > 0) out->insert(out->end(), buf, buf + n);
    ok = pclose(p) == 0;
  }
  std::remove(file.c_str());
  return ok;
}

// A coded sequence: the stream and the encoder's reconstruction of each picture in display order.
struct Coded {
  std::vector<uint8_t> stream;
  std::vector<std::vector<uint8_t>> recon;   // yuv420p, (w+1)&~1 by (h+1)&~1
  std::vector<size_t> bytes;                 // by display order
  std::vector<int> types;                    // PicType as int, in coding order
  int n = 0;
};

void append_recon(const HevcEncoder& e, int w, int h, std::vector<uint8_t>* r) {
  const int ew = (w + 1) & ~1, eh = (h + 1) & ~1, W = e.coded_width();
  for (int y = 0; y < eh; ++y) r->insert(r->end(), e.recon_y().begin() + static_cast<std::ptrdiff_t>(y) * W, e.recon_y().begin() + static_cast<std::ptrdiff_t>(y) * W + ew);
  for (int y = 0; y < eh / 2; ++y) r->insert(r->end(), e.recon_u().begin() + static_cast<std::ptrdiff_t>(y) * (W / 2), e.recon_u().begin() + static_cast<std::ptrdiff_t>(y) * (W / 2) + ew / 2);
  for (int y = 0; y < eh / 2; ++y) r->insert(r->end(), e.recon_v().begin() + static_cast<std::ptrdiff_t>(y) * (W / 2), e.recon_v().begin() + static_cast<std::ptrdiff_t>(y) * (W / 2) + ew / 2);
}

// Codes `n` pictures of the scene in the order an application with `max_b` B pictures between P pictures would: the P picture of a group first.
Coded code_sequence(int w, int h, int n, int qp, const HevcOptions& opt, int gop = 0) {
  Coded c;
  c.n = n;
  HevcEncoder enc(w, h, 30, 1, opt);
  c.stream = enc.parameter_sets();
  c.recon.resize(static_cast<size_t>(n));
  c.bytes.resize(static_cast<size_t>(n));
  auto code = [&](int d, PicType t, int q) {
    const EncPicture in = scene(w, h, d);
    const auto nal = enc.encode_at(in, t, q, d);
    c.stream.insert(c.stream.end(), nal.begin(), nal.end());
    c.bytes[static_cast<size_t>(d)] = nal.size();
    c.types.push_back(static_cast<int>(t));
    append_recon(enc, w, h, &c.recon[static_cast<size_t>(d)]);
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

// Each picture ffmpeg returns equals the encoder's reconstruction; true where ffmpeg is absent.
bool decodes_to_the_reconstruction(const Coded& c, const char* what) {
  if (note_no_ffmpeg()) return true;
  std::vector<uint8_t> dec;
  if (!ffmpeg_decode(c.stream, &dec)) {
    std::fprintf(stderr, "%s: ffmpeg could not decode the stream\n", what);
    return false;
  }
  size_t total = 0;
  for (const auto& r : c.recon) total += r.size();
  if (dec.size() != total) {
    std::fprintf(stderr, "%s: ffmpeg returned %zu bytes, wanted %zu\n", what, dec.size(), total);
    return false;
  }
  size_t at = 0;
  for (size_t i = 0; i < c.recon.size(); ++i) {
    if (std::memcmp(&dec[at], c.recon[i].data(), c.recon[i].size()) != 0) {
      std::fprintf(stderr, "%s: picture %zu differs from the encoder's reconstruction\n", what, i);
      return false;
    }
    at += c.recon[i].size();
  }
  return true;
}

// The NAL unit types of an Annex B stream, in order.
std::vector<int> nal_types(const std::vector<uint8_t>& s) {
  std::vector<int> types;
  for (size_t i = 0; i + 4 < s.size(); ++i)
    if (s[i] == 0 && s[i + 1] == 0 && s[i + 2] == 1) types.push_back((s[i + 3] >> 1) & 63);
  return types;
}

}  // namespace

VTEST(parameter_sets_name_the_size_and_the_slices_the_picture_type) {
  HevcEncoder enc(161, 51, 30, 1);
  const auto ps = enc.parameter_sets();
  VCHECK((nal_types(ps) == std::vector<int>{32, 33, 34}));
  EncPicture in = scene(161, 51, 0);
  EncStats st;
  const auto idr = enc.encode(in, PicType::kIdr, 28, &st);
  VCHECK((nal_types(idr) == std::vector<int>{39, 19}));   // the digest SEI, an IDR_W_RADL slice
  VCHECK_EQ(st.slices, 1);
  const auto p = enc.encode(scene(161, 51, 1), PicType::kInter, 28);
  VCHECK((nal_types(p) == std::vector<int>{1}));           // TRAIL_R
  enc.reset();
  const auto again = enc.encode(scene(161, 51, 2), PicType::kInter, 28);
  VCHECK((nal_types(again) == std::vector<int>{39, 19}));  // a P picture with no reference becomes an IDR picture
  // The coded size is a multiple of eight; the stream crops to the even size.
  VCHECK_EQ(enc.coded_width(), 168);
  VCHECK_EQ(enc.coded_height(), 56);
}

VTEST(intra_pictures_decode_to_the_reconstruction_at_every_size_and_qp) {
  const int sizes[][2] = {{8, 8}, {17, 17}, {33, 20}, {64, 48}, {70, 38}, {161, 51}};
  for (const auto& s : sizes)
    for (const int qp : {0, 12, 28, 40, 51}) {
      HevcOptions opt;
      const Coded c = code_sequence(s[0], s[1], 1, qp, opt);
      VCHECK(decodes_to_the_reconstruction(c, "intra picture"));
    }
}

VTEST(p_pictures_with_several_references_decode_to_the_reconstruction) {
  const int sizes[][2] = {{16, 16}, {33, 20}, {96, 64}, {161, 51}};
  for (const auto& s : sizes)
    for (const int qp : {10, 26, 45})
      for (const int refs : {1, 2, 3}) {
        HevcOptions opt;
        opt.num_ref = refs;
        const Coded c = code_sequence(s[0], s[1], 6, qp, opt);
        VCHECK(decodes_to_the_reconstruction(c, "P pictures"));
      }
}

VTEST(b_pictures_decode_to_the_reconstruction_in_display_order) {
  const int sizes[][2] = {{32, 32}, {96, 64}, {161, 51}};
  for (const auto& s : sizes)
    for (const int qp : {14, 30})
      for (const int bf : {1, 2, 3}) {
        HevcOptions opt;
        opt.num_ref = 2;
        opt.max_b = bf;
        const Coded c = code_sequence(s[0], s[1], 9, qp, opt);
        VCHECK(decodes_to_the_reconstruction(c, "B pictures"));
        // coding order: the first group's P picture comes before the B pictures that precede it in display order
        VCHECK(c.types.size() == 9u);
        VCHECK(c.types[1] == static_cast<int>(PicType::kInter));
        VCHECK(c.types[2] == static_cast<int>(PicType::kBi));
      }
  // a GOP that ends in the middle of a group, with an IDR picture after it
  HevcOptions opt;
  opt.max_b = 3;
  const Coded c = code_sequence(64, 48, 12, 26, opt, 6);
  VCHECK(decodes_to_the_reconstruction(c, "B pictures with IDR pictures between"));
}

VTEST(pictures_are_compressed_more_as_qp_rises_and_stay_close_to_the_input) {
  const EncPicture in = scene(160, 96, 0);
  size_t prev_bytes = 1u << 30;
  double prev_psnr = 100;
  for (const int qp : {10, 18, 26, 34, 42}) {
    HevcEncoder enc(160, 96, 30, 1);
    EncStats st;
    const auto nal = enc.encode(in, PicType::kIdr, qp, &st);
    const double psnr = psnr_y(in, enc);
    VCHECK(nal.size() < prev_bytes);
    VCHECK(psnr < prev_psnr + 0.01);
    VCHECK(std::fabs(psnr - st.psnr_y) < 1e-9);
    VCHECK(nal.size() < 160u * 96 * 3 / 2 / 2);
    if (qp <= 26) VCHECK(psnr > 30.0);
    prev_bytes = nal.size();
    prev_psnr = psnr;
  }
}

VTEST(p_and_b_pictures_cost_less_than_intra_pictures) {
  HevcOptions opt;
  opt.num_ref = 2;
  opt.max_b = 2;
  const Coded c = code_sequence(160, 96, 7, 28, opt);
  for (int d = 1; d < 7; ++d) VCHECK(c.bytes[static_cast<size_t>(d)] * 2 < c.bytes[0]);
  // a B picture (display order 1, 2 and 4, 5) costs less than the P picture that follows it
  VCHECK(c.bytes[1] < c.bytes[3]);
  VCHECK(c.bytes[4] < c.bytes[6]);
}

VTEST(one_frame_at_one_qp_is_one_bitstream_and_a_changed_sample_changes_it) {
  EncPicture in = scene(96, 64, 0);
  HevcEncoder a(96, 64, 30, 1), b(96, 64, 30, 1);
  const auto first = a.encode(in, PicType::kIdr, 28);
  VCHECK(first == b.encode(in, PicType::kIdr, 28));
  VCHECK(first == a.encode(in, PicType::kIdr, 28));   // whatever came before
  in.y[40 * 96 + 50] = static_cast<uint8_t>(in.y[40 * 96 + 50] ^ 0x5a);
  VCHECK(first != a.encode(in, PicType::kIdr, 28));
  in.y[40 * 96 + 50] = static_cast<uint8_t>(in.y[40 * 96 + 50] ^ 0x5a);
  in.v[10 * 48 + 20] = static_cast<uint8_t>(in.v[10 * 48 + 20] ^ 0x5a);
  VCHECK(first != a.encode(in, PicType::kIdr, 28));
  in.v[10 * 48 + 20] = static_cast<uint8_t>(in.v[10 * 48 + 20] ^ 0x5a);
  // a one-level change that quantisation erases from the coded picture still changes the bytes (the digest)
  HevcEncoder coarse(96, 64, 30, 1);
  const auto base = coarse.encode(in, PicType::kIdr, 51);
  in.y[10 * 96 + 10] = static_cast<uint8_t>(in.y[10 * 96 + 10] + 1);
  const auto bumped = coarse.encode(in, PicType::kIdr, 51);
  VCHECK(base != bumped);
  VCHECK_EQ(base.size(), bumped.size());
}

VTEST(rollback_returns_the_encoder_to_the_state_before_a_picture) {
  HevcOptions opt;
  opt.num_ref = 2;
  const EncPicture f0 = scene(96, 64, 0), f1 = scene(96, 64, 1), f2 = scene(96, 64, 2);
  HevcEncoder a(96, 64, 30, 1, opt), b(96, 64, 30, 1, opt);
  a.encode_at(f0, PicType::kIdr, 28, 0);
  b.encode_at(f0, PicType::kIdr, 28, 0);
  const auto p1_first_try = a.encode_at(f1, PicType::kInter, 20, 1);   // too many bits: try again at another QP
  a.rollback();
  const auto p1 = a.encode_at(f1, PicType::kInter, 34, 1);
  VCHECK(p1 != p1_first_try);
  const auto p1_direct = b.encode_at(f1, PicType::kInter, 34, 1);
  VCHECK(p1 == p1_direct);
  // what follows is as if the first try had not happened
  VCHECK(a.encode_at(f2, PicType::kInter, 30, 2) == b.encode_at(f2, PicType::kInter, 30, 2));
}

VTEST_MAIN
