// The HEVC writer behind VirtualGPU's NVENC (nvenc_hevc.hpp), read back by a decoder
// of its own: the parameter sets are parsed as ITU-T H.265 7.3.2 lays them out, and
// the slice data by the CABAC decoding process of 9.3.4.3 for exactly the syntax the
// writer emits (split_cu_flag, pcm_flag and end_of_slice_segment_flag, then the PCM
// samples). Every sample written must come back, for sizes that are and are not
// multiples of 16 and for odd ones (coded as the next even size and cropped), and
// through frames long enough to walk the split_cu_flag context through all its
// states. The table rangeTabLPS this decoder shares with the writer was checked
// against ffmpeg's decoder (nvidia/tests/e2e/nvenc_h264.cpp does it again where
// ffmpeg is installed). Needs no decoder installed.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../src/nvenc_hevc.hpp"
#include "vtest.hpp"

namespace {

struct Reader {
  const std::vector<uint8_t>& d;
  size_t pos = 0;   // bit position
  explicit Reader(const std::vector<uint8_t>& v) : d(v) {}
  uint32_t bit() {
    const uint32_t b = pos / 8 < d.size() ? (d[pos / 8] >> (7 - pos % 8)) & 1 : 0;
    ++pos;
    return b;
  }
  uint32_t u(int n) {
    uint32_t v = 0;
    while (n--) v = v << 1 | bit();
    return v;
  }
  uint32_t ue() {
    int zeros = 0;
    while (!bit() && zeros < 32) ++zeros;
    return ((1u << zeros) - 1) + u(zeros);
  }
  int32_t se() {
    const uint32_t k = ue();
    return k & 1 ? static_cast<int32_t>((k + 1) / 2) : -static_cast<int32_t>(k / 2);
  }
  void align() { pos = (pos + 7) & ~size_t{7}; }
};

struct Nal {
  int type;
  std::vector<uint8_t> rbsp;   // emulation prevention removed
};

std::vector<Nal> split(const std::vector<uint8_t>& s) {
  std::vector<Nal> out;
  size_t i = 0;
  auto start4 = [&](size_t k) { return k + 4 <= s.size() && s[k] == 0 && s[k + 1] == 0 && s[k + 2] == 0 && s[k + 3] == 1; };
  while (i < s.size() && !start4(i)) ++i;
  while (i < s.size()) {
    i += 4;
    size_t j = i;
    while (j < s.size() && !start4(j)) ++j;
    Nal n;
    n.type = s[i] >> 1 & 63;
    int zeros = 0;
    for (size_t k = i + 2; k < j; ++k) {   // the header is two bytes
      if (zeros >= 2 && s[k] == 3) {
        zeros = 0;
        continue;
      }
      n.rbsp.push_back(s[k]);
      zeros = s[k] == 0 ? zeros + 1 : 0;
    }
    out.push_back(std::move(n));
    i = j;
  }
  return out;
}

// The decoding engine of 9.3.4.3 over the bytes of the slice data.
struct Engine {
  const std::vector<uint8_t>& d;
  Reader r;
  uint32_t range = 510, offset = 0;
  int state = 0;   // pStateIdx of the split_cu_flag context
  explicit Engine(const std::vector<uint8_t>& v, size_t pos) : d(v), r(v) {
    r.pos = pos;
    init();
  }
  void init() {
    range = 510;
    offset = r.u(9);
  }
  int decision() {
    static const uint8_t lps_q3[63] = {240, 227, 216, 205, 195, 185, 175, 166, 158, 150, 142, 135, 128, 122, 116, 110, 104, 99, 94,
                                       89,  85,  80,  76,  72,  69,  65,  62,  59,  56,  53,  50,  48,  45,  43,  41,  39,  37, 35,
                                       33,  31,  30,  28,  27,  25,  24,  23,  22,  21,  20,  19,  18,  17,  16,  15,  14,  14, 13,
                                       12,  12,  11,  11,  10,  9};
    // Only column 3 is needed: the engine starts each coding tree unit at 508 or 510.
    if (((range >> 6) & 3) != 3) return -1;
    const uint32_t lps = lps_q3[state];
    range -= lps;
    int bin = 0;   // the most probable symbol is 0
    if (offset >= range) {
      bin = 1;
      offset -= range;
      range = lps;
    } else if (state < 62) {
      ++state;
    }
    renorm();
    return bin;
  }
  int terminate() {
    range -= 2;
    if (offset >= range) return 1;
    renorm();
    return 0;
  }
  void renorm() {
    while (range < 256) {
      range <<= 1;
      offset = offset << 1 | r.bit();
    }
  }
};

struct Decoded {
  int w = 0, h = 0;
  std::vector<uint8_t> y, u, v;
  bool ok = false;
};

Decoded decode(const std::vector<uint8_t>& stream) {
  Decoded r;
  int ctus_x = 0, ctus_y = 0, crop_r = 0, crop_b = 0;
  bool have_sps = false;
  for (const Nal& n : split(stream)) {
    Reader b(n.rbsp);
    if (n.type == 33) {
      b.u(4);   // vps id
      b.u(3);   // max sub layers
      b.u(1);
      b.u(2 + 1 + 5);   // profile space, tier, profile idc
      b.u(32);          // compatibility flags
      b.u(4);           // progressive, interlaced, non-packed, frame-only
      b.u(32);
      b.u(12);
      b.u(8);           // level
      b.ue();           // sps id
      if (b.ue() != 1) return r;   // chroma_format_idc
      ctus_x = static_cast<int>(b.ue()) / 16;
      ctus_y = static_cast<int>(b.ue()) / 16;
      if (b.bit()) {
        b.ue();
        crop_r = static_cast<int>(b.ue()) * 2;
        b.ue();
        crop_b = static_cast<int>(b.ue()) * 2;
      }
      have_sps = true;
    } else if (n.type == 19 && have_sps) {
      b.bit();   // first_slice_segment_in_pic_flag
      b.bit();   // no_output_of_prior_pics_flag
      b.ue();    // pps id
      if (b.ue() != 2) return r;   // slice_type I
      b.se();                      // slice_qp_delta
      if (!b.bit()) return r;      // byte_alignment(): a one
      b.align();
      Engine e(n.rbsp, b.pos);
      const int W = ctus_x * 16, H = ctus_y * 16;
      std::vector<uint8_t> y(static_cast<size_t>(W) * H), u(static_cast<size_t>(W) * H / 4), v(u.size());
      for (int cu = 0; cu < ctus_x * ctus_y; ++cu) {
        const int mx = cu % ctus_x, my = cu / ctus_x;
        if (e.decision() != 0) return r;       // split_cu_flag
        if (e.terminate() != 1) return r;      // pcm_flag
        // The decoder has read exactly as many bits as the encoder's flush wrote;
        // the zero bits to the byte boundary, then the samples.
        e.r.align();
        for (int j = 0; j < 16; ++j)
          for (int i = 0; i < 16; ++i) y[static_cast<size_t>(my * 16 + j) * W + mx * 16 + i] = static_cast<uint8_t>(e.r.u(8));
        for (std::vector<uint8_t>* p : {&u, &v})
          for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i) (*p)[static_cast<size_t>(my * 8 + j) * (W / 2) + mx * 8 + i] = static_cast<uint8_t>(e.r.u(8));
        e.init();
        const int last = e.terminate();        // end_of_slice_segment_flag
        if (last != (cu + 1 == ctus_x * ctus_y)) return r;
      }
      r.w = W - crop_r;
      r.h = H - crop_b;
      for (int j = 0; j < r.h; ++j) r.y.insert(r.y.end(), y.begin() + static_cast<long>(j) * W, y.begin() + static_cast<long>(j) * W + r.w);
      for (int j = 0; j < r.h / 2; ++j) {
        r.u.insert(r.u.end(), u.begin() + static_cast<long>(j) * (W / 2), u.begin() + static_cast<long>(j) * (W / 2) + r.w / 2);
        r.v.insert(r.v.end(), v.begin() + static_cast<long>(j) * (W / 2), v.begin() + static_cast<long>(j) * (W / 2) + r.w / 2);
      }
      r.ok = true;
    }
  }
  return r;
}

uint8_t luma_at(int x, int y) { return static_cast<uint8_t>(x * 7 + y * 13 + (x * y) % 17); }
uint8_t chroma_at(int plane, int x, int y) { return static_cast<uint8_t>(plane * 91 + x * 5 + y * 11); }

}  // namespace

VTEST(pcm_pictures_round_trip_at_every_shape) {
  // 320x180 is 220 coding tree units: the context walks through all 63 states.
  const int sizes[][2] = {{16, 16}, {32, 48}, {145, 49}, {146, 50}, {192, 128}, {257, 65}, {1, 1}, {17, 33}, {320, 180}, {1280, 64}};
  for (const auto& sz : sizes) {
    vgpu_nvenc::HevcStream st;
    st.width = sz[0];
    st.height = sz[1];
    std::vector<uint8_t> bits = st.parameter_sets();
    const auto idr = st.idr(luma_at, chroma_at);
    bits.insert(bits.end(), idr.begin(), idr.end());
    const Decoded d = decode(bits);
    VCHECK(d.ok);
    // An odd size is coded as the next even one, repeating the last row and column.
    VCHECK_EQ(d.w, st.coded_width());
    VCHECK_EQ(d.h, st.coded_height());
    bool same = d.ok;
    for (int y = 0; same && y < d.h; ++y)
      for (int x = 0; x < d.w; ++x)
        if (d.y[static_cast<size_t>(y) * d.w + x] != luma_at(std::min(x, st.width - 1), std::min(y, st.height - 1))) {
          same = false;
          break;
        }
    const int cw = (st.width + 1) / 2, ch = (st.height + 1) / 2;
    for (int y = 0; same && y < d.h / 2; ++y)
      for (int x = 0; x < d.w / 2; ++x) {
        const int sx = std::min(x, cw - 1), sy = std::min(y, ch - 1);
        if (d.u[static_cast<size_t>(y) * (d.w / 2) + x] != chroma_at(0, sx, sy) || d.v[static_cast<size_t>(y) * (d.w / 2) + x] != chroma_at(1, sx, sy)) {
          same = false;
          break;
        }
      }
    VCHECK(same);
  }
}

VTEST(emulation_prevention_keeps_start_codes_out_of_the_payload) {
  // All-zero samples put long runs of zero bytes in the payload; none may form a start code.
  vgpu_nvenc::HevcStream st;
  st.width = 64;
  st.height = 32;
  const auto idr = st.idr([](int, int) { return uint8_t{0}; }, [](int, int, int) { return uint8_t{0}; });
  for (size_t i = 6; i + 3 <= idr.size(); ++i) VCHECK(!(idr[i] == 0 && idr[i + 1] == 0 && idr[i + 2] <= 2));
  auto all = st.parameter_sets();
  all.insert(all.end(), idr.begin(), idr.end());
  const Decoded d = decode(all);
  VCHECK(d.ok);
  bool zero = d.ok;
  for (uint8_t b : d.y) zero = zero && b == 0;
  VCHECK(zero);
}

VTEST(level_follows_the_picture_size_and_rate) {
  // general_level_idc is the byte after the 12 bytes of profile and constraint flags.
  auto level = [](int w, int h, int fps) {
    vgpu_nvenc::HevcStream st;
    st.width = w;
    st.height = h;
    st.fps_num = fps;
    const std::vector<uint8_t> sps = st.sps();
    // sps_video_parameter_set_id, max sub layers and nesting: 1 byte; profile_tier_level: 12 bytes.
    return static_cast<int>(sps[1 + 11]);
  };
  VCHECK_EQ(level(1920, 1080, 30), 120);
  VCHECK_EQ(level(3840, 2160, 60), 153);
  VCHECK_EQ(level(64, 64, 15), 30);
}

VTEST_MAIN
