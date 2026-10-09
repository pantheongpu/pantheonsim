// The H.264 writer behind VirtualGPU's NVENC (nvenc_h264.hpp), read back by a
// decoder of its own: a bit reader that follows ITU-T H.264 clauses 7.3.2.1.1
// (sequence parameter set), 7.3.2.2 (picture parameter set) and 7.3.3 / 7.3.5
// (slice header, macroblock layer) for exactly the syntax the writer emits, and
// returns the I_PCM samples. Every sample written must come back, for sizes that
// are and are not multiples of 16 and for odd ones (coded as the next even size
// and cropped). Needs no decoder installed; nvidia/tests/e2e/nvenc_h264.cpp does
// the same through ffmpeg when it is there.
#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../src/nvenc_h264.hpp"
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
  int ref_idc, type;
  std::vector<uint8_t> rbsp;   // emulation prevention removed
};

// Annex B: split on 00 00 01 / 00 00 00 01 start codes and remove 00 00 03.
std::vector<Nal> split(const std::vector<uint8_t>& s) {
  std::vector<Nal> out;
  size_t i = 0;
  auto start_at = [&](size_t k) { return k + 3 <= s.size() && s[k] == 0 && s[k + 1] == 0 && s[k + 2] == 1; };
  while (i < s.size() && !start_at(i)) ++i;
  while (i < s.size()) {
    i += 3;
    size_t j = i;
    while (j < s.size() && !start_at(j) && !(j + 4 <= s.size() && s[j] == 0 && s[j + 1] == 0 && s[j + 2] == 0 && s[j + 3] == 1)) ++j;
    Nal n;
    n.ref_idc = s[i] >> 5 & 3;
    n.type = s[i] & 31;
    int zeros = 0;
    for (size_t k = i + 1; k < j; ++k) {
      if (zeros >= 2 && s[k] == 3) {
        zeros = 0;
        continue;
      }
      n.rbsp.push_back(s[k]);
      zeros = s[k] == 0 ? zeros + 1 : 0;
    }
    out.push_back(std::move(n));
    i = j;
    if (i < s.size() && !start_at(i)) ++i;   // the zero of a four-byte start code
  }
  return out;
}

struct Decoded {
  int w = 0, h = 0;                 // cropped picture size
  std::vector<uint8_t> y, u, v;
  bool ok = false;
};

Decoded decode(const std::vector<uint8_t>& stream) {
  Decoded r;
  const std::vector<Nal> nals = split(stream);
  int mbs_x = 0, mbs_y = 0, crop_r = 0, crop_b = 0;
  bool have_sps = false;
  for (const Nal& n : nals) {
    Reader b(n.rbsp);
    if (n.type == 7) {
      if (b.u(8) != 66) return r;   // Baseline
      b.u(8);
      b.u(8);
      b.ue();                       // sps id
      b.ue();                       // log2_max_frame_num_minus4
      if (b.ue() != 2) return r;    // pic_order_cnt_type
      b.ue();                       // max_num_ref_frames
      b.bit();
      mbs_x = static_cast<int>(b.ue()) + 1;
      mbs_y = static_cast<int>(b.ue()) + 1;
      if (!b.bit()) return r;       // frame_mbs_only_flag
      b.bit();
      if (b.bit()) {                // frame_cropping_flag
        b.ue();
        crop_r = static_cast<int>(b.ue()) * 2;
        b.ue();
        crop_b = static_cast<int>(b.ue()) * 2;
      }
      have_sps = true;
    } else if (n.type == 5 && have_sps) {
      b.ue();                       // first_mb_in_slice
      b.ue();                       // slice_type
      b.ue();                       // pps id
      b.u(4);                       // frame_num
      b.ue();                       // idr_pic_id
      b.u(2);                       // dec_ref_pic_marking
      b.se();                       // slice_qp_delta
      if (b.ue() != 1) return r;    // disable_deblocking_filter_idc
      const int W = mbs_x * 16, H = mbs_y * 16;
      std::vector<uint8_t> y(static_cast<size_t>(W) * H), u(static_cast<size_t>(W) * H / 4), v(u.size());
      for (int my = 0; my < mbs_y; ++my)
        for (int mx = 0; mx < mbs_x; ++mx) {
          if (b.ue() != 25) return r;   // I_PCM
          b.align();
          for (int j = 0; j < 16; ++j)
            for (int i = 0; i < 16; ++i) y[static_cast<size_t>(my * 16 + j) * W + mx * 16 + i] = static_cast<uint8_t>(b.u(8));
          for (std::vector<uint8_t>* p : {&u, &v})
            for (int j = 0; j < 8; ++j)
              for (int i = 0; i < 8; ++i) (*p)[static_cast<size_t>(my * 8 + j) * (W / 2) + mx * 8 + i] = static_cast<uint8_t>(b.u(8));
        }
      // The trailing bits end the NAL: a one, then zeros.
      if (!b.bit()) return r;
      r.w = W - crop_r;
      r.h = H - crop_b;
      const int cw = r.w / 2, ch = r.h / 2;
      for (int j = 0; j < r.h; ++j)
        r.y.insert(r.y.end(), y.begin() + static_cast<long>(j) * W, y.begin() + static_cast<long>(j) * W + r.w);
      for (int j = 0; j < ch; ++j) {
        r.u.insert(r.u.end(), u.begin() + static_cast<long>(j) * (W / 2), u.begin() + static_cast<long>(j) * (W / 2) + cw);
        r.v.insert(r.v.end(), v.begin() + static_cast<long>(j) * (W / 2), v.begin() + static_cast<long>(j) * (W / 2) + cw);
      }
      r.ok = true;
    }
  }
  return r;
}

uint8_t luma_at(int x, int y, int t) { return static_cast<uint8_t>(x * 7 + y * 13 + t * 29 + (x * y) % 17); }
uint8_t chroma_at(int plane, int x, int y, int t) { return static_cast<uint8_t>(plane * 91 + x * 5 + y * 11 + t * 3); }

}  // namespace

VTEST(i_pcm_pictures_round_trip_at_every_shape) {
  const int sizes[][2] = {{16, 16}, {32, 48}, {145, 49}, {146, 50}, {192, 128}, {257, 65}, {1, 1}, {17, 33}, {320, 180}};
  for (const auto& sz : sizes) {
    vgpu_nvenc::H264Stream st;
    st.width = sz[0];
    st.height = sz[1];
    for (int t = 0; t < 3; ++t) {
      std::vector<uint8_t> bits = st.parameter_sets();
      const auto idr = st.idr([&](int x, int y) { return luma_at(x, y, t); },
                              [&](int plane, int x, int y) { return chroma_at(plane, x, y, t); }, t);
      bits.insert(bits.end(), idr.begin(), idr.end());
      const Decoded d = decode(bits);
      VCHECK(d.ok);
      // An odd size is coded as the next even one, repeating the last row and column.
      VCHECK_EQ(d.w, st.coded_width());
      VCHECK_EQ(d.h, st.coded_height());
      bool same = d.ok;
      for (int y = 0; same && y < d.h; ++y)
        for (int x = 0; x < d.w; ++x)
          if (d.y[static_cast<size_t>(y) * d.w + x] != luma_at(std::min(x, st.width - 1), std::min(y, st.height - 1), t)) {
            same = false;
            break;
          }
      const int cw = (st.width + 1) / 2, ch = (st.height + 1) / 2;
      for (int y = 0; same && y < d.h / 2; ++y)
        for (int x = 0; x < d.w / 2; ++x) {
          const int sx = std::min(x, cw - 1), sy = std::min(y, ch - 1);
          if (d.u[static_cast<size_t>(y) * (d.w / 2) + x] != chroma_at(0, sx, sy, t) ||
              d.v[static_cast<size_t>(y) * (d.w / 2) + x] != chroma_at(1, sx, sy, t)) {
            same = false;
            break;
          }
        }
      VCHECK(same);
    }
  }
}

VTEST(emulation_prevention_keeps_start_codes_out_of_the_payload) {
  // All-zero samples would put 00 00 00 inside the PCM payload; the writer must
  // escape them and the reader above must get them back.
  vgpu_nvenc::H264Stream st;
  st.width = 64;
  st.height = 32;
  const auto idr = st.idr([](int, int) { return uint8_t{0}; }, [](int, int, int) { return uint8_t{0}; }, 0);
  for (size_t i = 5; i + 3 <= idr.size(); ++i)
    VCHECK(!(idr[i] == 0 && idr[i + 1] == 0 && idr[i + 2] <= 2));
  auto all = st.parameter_sets();
  all.insert(all.end(), idr.begin(), idr.end());
  const Decoded d = decode(all);
  VCHECK(d.ok);
  bool zero = d.ok;
  for (uint8_t b : d.y) zero = zero && b == 0;
  VCHECK(zero);
}

VTEST(level_follows_the_frame_size_and_rate) {
  vgpu_nvenc::H264Stream st;
  st.width = 1920;
  st.height = 1080;
  st.fps_num = 30;
  VCHECK_EQ(st.level_idc(), 40);
  st.fps_num = 60;
  VCHECK_EQ(st.level_idc(), 42);
  st.width = 64;
  st.height = 64;
  st.fps_num = 15;
  VCHECK_EQ(st.level_idc(), 10);
}

VTEST_MAIN
