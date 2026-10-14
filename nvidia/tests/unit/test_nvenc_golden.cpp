// The bytes the compressing encoders write, as a digest per configuration: the encoders are pure functions of the
// pictures, the types, the QPs and their options ("an IDR picture of the same frame at the same QP is the same bytes every
// time"), so a change that makes them faster must not change one byte of any stream. The digests below were taken from
// the encoders before their inner loops were reworked for speed (the pictures are test_nvenc_h264_enc.cpp's synthetic
// scene); `test_nvenc_golden --print` prints the current ones.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/nvenc_h264_enc.hpp"
#include "../../src/nvenc_hevc_enc.hpp"
#include "vtest.hpp"

namespace {

using vgpu_nvenc::EncPicture;
using vgpu_nvenc::H264Encoder;
using vgpu_nvenc::H264Options;
using vgpu_nvenc::HevcEncoder;
using vgpu_nvenc::HevcOptions;
using vgpu_nvenc::PicType;

bool g_print = false;

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

// Smooth moving shapes with texture; (dx, dy) move the whole scene.
EncPicture scene(int w, int h, int dx, int dy) {
  EncPicture p;
  p.w = w;
  p.h = h;
  p.y.resize(static_cast<size_t>(w) * h);
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  p.u.resize(static_cast<size_t>(cw) * ch);
  p.v.resize(p.u.size());
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int X = x - dx, Y = y - dy;
      // (more texture than test_nvenc_h264_enc.cpp's scene -- fine noise, a second moving pattern, edges -- so residuals, many
      // modes and sub-sample motion are all in play)
      int v = 60 + tri(X * 3 + Y) / 2 + ((X / 12 + Y / 12) & 1) * 40 + noise(X, Y, 0, 14);
      if ((X - 40) * (X - 40) + (Y - 30) * (Y - 30) < 400) v += 70;
      if ((X * 7 + Y * 3) % 29 < 4) v -= 45;
      v += tri((x + 2 * dx) * 5 - (y - dy) * 2) / 5;
      p.y[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::min(255, std::max(0, v)));
    }
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x) {
      p.u[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(110 + tri((x - dx / 2) * 2 + (y - dy / 2)) / 3 + noise(x, y, 1, 6));
      p.v[static_cast<size_t>(y) * cw + x] = static_cast<uint8_t>(130 - tri((x - dx / 2) + (y - dy / 2) * 2) / 3 + noise(x, y, 2, 6));
    }
  return p;
}

uint64_t fnv(const std::vector<uint8_t>& v) {
  uint64_t h = 1469598103934665603ull;
  for (uint8_t b : v) {
    h ^= b;
    h *= 1099511628211ull;
  }
  return h;
}

// n pictures in the order an application with `max_b` B pictures between P pictures would code them (the P picture of a
// group first, then its B pictures); the stream of parameter sets and every picture.
template <class Enc>
std::vector<uint8_t> code_seq(Enc& enc, int w, int h, int n, int qp, int max_b) {
  std::vector<uint8_t> stream = enc.parameter_sets();
  auto code = [&](int d, PicType t, int q) {
    const auto nal = enc.encode_at(scene(w, h, 2 * d, d), t, q, d);
    stream.insert(stream.end(), nal.begin(), nal.end());
  };
  code(0, PicType::kIdr, qp);
  for (int d = 1; d < n;) {
    const int span = std::min(max_b + 1, n - d);
    const int anchor = d + span - 1;
    code(anchor, PicType::kInter, qp);
    for (int b = d; b < anchor; ++b) code(b, PicType::kBi, qp + 2);
    d = anchor + 1;
  }
  return stream;
}

struct H264Case {
  const char* name;
  int w, h, profile;
  H264Options opt;
  int qp;
};
struct HevcCase {
  const char* name;
  int w, h;
  HevcOptions opt;
  int qp;
};

H264Options h264(bool cabac, bool t8, int num_ref, int max_b, int slice_mode = 0, int slice_data = 0, bool deblock = true) {
  H264Options o;
  o.cabac = cabac;
  o.transform8x8 = t8;
  o.num_ref = num_ref;
  o.max_b = max_b;
  o.slice_mode = slice_mode;
  o.slice_data = slice_data;
  o.deblock = deblock;
  return o;
}
HevcOptions hevc(int num_ref, int max_b, bool deblock = true) {
  HevcOptions o;
  o.num_ref = num_ref;
  o.max_b = max_b;
  o.deblock = deblock;
  return o;
}

const H264Case kH264[] = {
    {"h264 baseline cavlc", 96, 80, 66, h264(false, false, 1, 0), 24},
    {"h264 baseline cavlc odd size", 130, 70, 66, h264(false, false, 1, 0), 20},
    {"h264 main cabac 2 refs", 96, 80, 77, h264(true, false, 2, 0), 24},
    {"h264 main cabac b pictures", 112, 64, 77, h264(true, false, 2, 2), 26},
    {"h264 high cabac 8x8 3 refs slices", 128, 96, 100, h264(true, true, 3, 0, 2, 2), 22},
    {"h264 high cavlc 8x8 b", 96, 80, 100, h264(false, true, 2, 1), 28},
    {"h264 high cabac no deblock", 96, 80, 100, h264(true, true, 2, 1, 0, 0, false), 18},
};
const HevcCase kHevc[] = {
    {"hevc default", 96, 80, hevc(1, 0), 24},
    {"hevc odd size", 100, 70, hevc(1, 0), 22},
    {"hevc 2 refs", 128, 96, hevc(2, 0), 26},
    {"hevc b pictures", 112, 64, hevc(2, 2), 24},
    {"hevc no deblock", 96, 80, hevc(1, 1, false), 20},
};

// The digests, in the order of the cases above.
const uint64_t kH264Digest[] = {
    0x4dac403396fc564eull,
    0x8dbd3a3fdf1883cbull,
    0x7c8dd6e9af9f18f6ull,
    0x32600cd2dbc57f64ull,
    0x93b6d401e2949890ull,
    0x7918aa96cde38e83ull,
    0xb797bf1f72beb151ull,
};
const uint64_t kHevcDigest[] = {
    0xe3ba4eb36b9251a2ull,
    0x46c24923720273ffull,
    0xb2c6725229d34364ull,
    0x5c936a1913e1b8ebull,
    0x43b59dad9efe8528ull,
};

}  // namespace

VTEST(h264_encoder_bytes_are_unchanged) {
  size_t i = 0;
  int bad = 0;
  for (const H264Case& c : kH264) {
    H264Encoder enc(c.w, c.h, 30, 1, c.profile, c.opt);
    const auto stream = code_seq(enc, c.w, c.h, 6, c.qp, c.opt.max_b);
    const uint64_t d = fnv(stream);
    if (g_print) std::printf("  %s: 0x%016llxull,  // %zu bytes\n", c.name, static_cast<unsigned long long>(d), stream.size());
    else if (i >= sizeof kH264Digest / sizeof kH264Digest[0] || d != kH264Digest[i]) {
      std::printf("  %s: got %016llx\n", c.name, static_cast<unsigned long long>(d));
      ++bad;
    }
    ++i;
  }
  VCHECK_EQ(bad, 0);
}

VTEST(hevc_encoder_bytes_are_unchanged) {
  size_t i = 0;
  int bad = 0;
  for (const HevcCase& c : kHevc) {
    HevcEncoder enc(c.w, c.h, 30, 1, c.opt);
    const auto stream = code_seq(enc, c.w, c.h, 5, c.qp, c.opt.max_b);
    const uint64_t d = fnv(stream);
    if (g_print) std::printf("  %s: 0x%016llxull,  // %zu bytes\n", c.name, static_cast<unsigned long long>(d), stream.size());
    else if (i >= sizeof kHevcDigest / sizeof kHevcDigest[0] || d != kHevcDigest[i]) {
      std::printf("  %s: got %016llx\n", c.name, static_cast<unsigned long long>(d));
      ++bad;
    }
    ++i;
  }
  VCHECK_EQ(bad, 0);
}

int main(int argc, char** argv) {
  g_print = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  return ::vtest::run_all(g_print ? 1 : argc, argv);
}
