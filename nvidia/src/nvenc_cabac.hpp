// The CABAC encoder of VirtualGPU's NVENC, for both codecs. H.264 (ITU-T H.264 clause 9.3.4) and HEVC (ITU-T H.265 clause 9.3.4.3)
// define the same arithmetic coding engine -- the same range and state tables, the same renormalisation, bypass and terminate
// bins, the same flush -- and differ in the syntax elements, their binarisations and their context numbers, which the two
// encoders (nvenc_h264_enc.cpp, nvenc_hevc_enc.cpp) supply.
//
// Two coders with one interface, so that a macroblock or coding unit is written by one function for both purposes:
//   * CabacWriter  produces the bytes;
//   * CabacCost    only adds up what the bins would cost (in 1/256 bit, from each context's probability state), updating its
//                  own copy of the contexts as it goes. Rate-distortion decisions copy the writer's contexts into a CabacCost,
//                  run the syntax writer over a candidate, and read the cost.
// Contexts are numbered by the caller; a coder holds N of them.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgpu_nvenc {

// Generated from the decoder's tables (h264_tables.inc); the costs are -log2 of pLPS(s) = 0.5 * alpha^s (alpha = (0.01875 / 0.5)^(1/63)) and of 1 - pLPS(s), in 1/256 bit.
inline constexpr uint8_t kCabacRangeLps[64][4] = {
    {128, 176, 208, 240},
    {128, 167, 197, 227},
    {128, 158, 187, 216},
    {123, 150, 178, 205},
    {116, 142, 169, 195},
    {111, 135, 160, 185},
    {105, 128, 152, 175},
    {100, 122, 144, 166},
    {95, 116, 137, 158},
    {90, 110, 130, 150},
    {85, 104, 123, 142},
    {81, 99, 117, 135},
    {77, 94, 111, 128},
    {73, 89, 105, 122},
    {69, 85, 100, 116},
    {66, 80, 95, 110},
    {62, 76, 90, 104},
    {59, 72, 86, 99},
    {56, 69, 81, 94},
    {53, 65, 77, 89},
    {51, 62, 73, 85},
    {48, 59, 69, 80},
    {46, 56, 66, 76},
    {43, 53, 63, 72},
    {41, 50, 59, 69},
    {39, 48, 56, 65},
    {37, 45, 54, 62},
    {35, 43, 51, 59},
    {33, 41, 48, 56},
    {32, 39, 46, 53},
    {30, 37, 43, 50},
    {29, 35, 41, 48},
    {27, 33, 39, 45},
    {26, 31, 37, 43},
    {24, 30, 35, 41},
    {23, 28, 33, 39},
    {22, 27, 32, 37},
    {21, 26, 30, 35},
    {20, 24, 29, 33},
    {19, 23, 27, 31},
    {18, 22, 26, 30},
    {17, 21, 25, 28},
    {16, 20, 23, 27},
    {15, 19, 22, 25},
    {14, 18, 21, 24},
    {14, 17, 20, 23},
    {13, 16, 19, 22},
    {12, 15, 18, 21},
    {12, 14, 17, 20},
    {11, 14, 16, 19},
    {11, 13, 15, 18},
    {10, 12, 15, 17},
    {10, 12, 14, 16},
    {9, 11, 13, 15},
    {9, 11, 12, 14},
    {8, 10, 12, 14},
    {8, 9, 11, 13},
    {7, 9, 11, 12},
    {7, 9, 10, 12},
    {7, 8, 10, 11},
    {6, 8, 9, 11},
    {6, 7, 9, 10},
    {6, 7, 8, 9},
    {2, 2, 2, 2}};

inline constexpr uint8_t kCabacTransLps[64] = {0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9,  11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63};

inline constexpr uint16_t kCabacCostMps[64] = {256, 238, 221, 206, 192, 180, 168, 157, 148, 139, 130, 122, 115, 108, 102, 96, 90, 85, 80, 76, 72, 68, 64, 60, 57, 54, 51, 48, 46, 43, 41, 39, 37, 35, 33, 31, 29, 28, 26, 25, 24, 22, 21, 20, 19, 18, 17, 16, 15, 15, 14, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7};

inline constexpr uint16_t kCabacCostLps[64] = {256, 275, 294, 314, 333, 352, 371, 391, 410, 429, 448, 468, 487, 506, 525, 545, 564, 583, 602, 622, 641, 660, 679, 699, 718, 737, 756, 776, 795, 814, 833, 853, 872, 891, 910, 930, 949, 968, 987, 1007, 1026, 1045, 1064, 1084, 1103, 1122, 1141, 1161, 1180, 1199, 1218, 1238, 1257, 1276, 1295, 1315, 1334, 1353, 1372, 1392, 1411, 1430, 1449, 1469};

// A context model: pStateIdx and valMps.
struct CabacCtx {
  uint8_t state = 0, mps = 0;
};

// Initialises a context from the (m, n) pair of its table (H.264 Table 9-12 ff.) or the initValue of HEVC's tables
// (slopeIdx = initValue >> 4, offsetIdx = initValue & 15; m = slopeIdx * 5 - 45, n = (offsetIdx << 3) - 16), and the slice QP.
inline CabacCtx cabac_init_ctx(int m, int n, int slice_qp) {
  const int qp = slice_qp < 0 ? 0 : (slice_qp > 51 ? 51 : slice_qp);
  int pre = ((m * qp) >> 4) + n;
  pre = pre < 1 ? 1 : (pre > 126 ? 126 : pre);
  CabacCtx c;
  if (pre <= 63) {
    c.mps = 0;
    c.state = static_cast<uint8_t>(63 - pre);
  } else {
    c.mps = 1;
    c.state = static_cast<uint8_t>(pre - 64);
  }
  return c;
}
inline CabacCtx cabac_init_hevc(int init_value, int slice_qp) {
  return cabac_init_ctx((init_value >> 4) * 5 - 45, ((init_value & 15) << 3) - 16, slice_qp);
}

inline void cabac_update(CabacCtx& c, int bin) {
  if (bin == c.mps) {
    if (c.state < 62) ++c.state;
  } else {
    if (c.state == 0) c.mps = static_cast<uint8_t>(1 - c.mps);
    c.state = kCabacTransLps[c.state];
  }
}

// Writes the bins of a slice's data into bytes (9.3.4.2). `out` receives whole bytes; call finish() after the last terminate(1).
template <int N>
class CabacWriter {
 public:
  explicit CabacWriter(std::vector<uint8_t>& out) : out_(out) { start(); }
  CabacCtx ctx[N];
  void start() {
    low_ = 0;
    range_ = 510;
    first_ = true;
    outstanding_ = 0;
  }
  void decision(int idx, int bin) {
    CabacCtx& c = ctx[idx];
    const uint32_t lps = kCabacRangeLps[c.state][(range_ >> 6) & 3];
    range_ -= lps;
    if (bin != c.mps) {
      low_ += range_;
      range_ = lps;
    }
    cabac_update(c, bin);
    renorm();
  }
  void bypass(int bin) {
    low_ <<= 1;
    if (bin) low_ += range_;
    if (low_ >= 1024) {
      put_bit(1);
      low_ -= 1024;
    } else if (low_ < 512) {
      put_bit(0);
    } else {
      low_ -= 512;
      ++outstanding_;
    }
  }
  void bypass_bits(uint32_t v, int n) {
    for (int i = n - 1; i >= 0; --i) bypass(static_cast<int>((v >> i) & 1));
  }
  // The terminate bin; a 1 ends the arithmetic code word (EncodeFlush): the last bit written is a one, which is the
  // rbsp_stop_one_bit when the bin is end_of_slice(_segment)_flag.
  void terminate(int bin) {
    range_ -= 2;
    if (bin) {
      low_ += range_;
      range_ = 2;
      renorm();
      put_bit(static_cast<int>((low_ >> 9) & 1));
      write_bit(static_cast<int>((low_ >> 8) & 1));
      write_bit(1);
    } else {
      renorm();
    }
  }
  // After a terminate(1): pad to a byte boundary with zero bits.
  void finish() {
    while (nbits_) write_bit(0);
  }
  // The engine's state after pcm samples (H.264 9.3.1.2, HEVC 9.3.2.5): initialised again, contexts kept.
  void restart_engine() { start(); }
  // Raw bits between a terminate(1) and restart_engine() (PCM samples): the stream is at a byte boundary after finish().
  void raw_byte(uint8_t b) { out_.push_back(b); }
  size_t bytes_written() const { return out_.size(); }

 private:
  void renorm() {
    while (range_ < 256) {
      if (low_ < 256) {
        put_bit(0);
      } else if (low_ >= 512) {
        low_ -= 512;
        put_bit(1);
      } else {
        low_ -= 256;
        ++outstanding_;
      }
      range_ <<= 1;
      low_ <<= 1;
    }
  }
  void put_bit(int b) {
    if (first_) {
      first_ = false;
    } else {
      write_bit(b);
    }
    for (; outstanding_ > 0; --outstanding_) write_bit(1 - b);
  }
  void write_bit(int b) {
    cur_ = static_cast<uint8_t>(cur_ << 1 | (b & 1));
    if (++nbits_ == 8) {
      out_.push_back(cur_);
      cur_ = 0;
      nbits_ = 0;
    }
  }
  std::vector<uint8_t>& out_;
  uint32_t low_ = 0, range_ = 510;
  bool first_ = true;
  int outstanding_ = 0;
  uint8_t cur_ = 0;
  int nbits_ = 0;
};

// The same interface, adding up costs instead of bytes.
template <int N>
class CabacCost {
 public:
  CabacCtx ctx[N];
  uint32_t cost = 0;   // 1/256 bit
  CabacCost() = default;
  template <class Other>
  explicit CabacCost(const Other& from) {
    std::memcpy(ctx, from.ctx, sizeof ctx);
  }
  void decision(int idx, int bin) {
    CabacCtx& c = ctx[idx];
    cost += bin == c.mps ? kCabacCostMps[c.state] : kCabacCostLps[c.state];
    cabac_update(c, bin);
  }
  void bypass(int) { cost += 256; }
  void bypass_bits(uint32_t, int n) { cost += 256u * static_cast<uint32_t>(n); }
  void terminate(int bin) { cost += bin ? 2 * 256 : 2; }   // a 1 flushes about two bits; a 0 costs next to nothing
};

}  // namespace vgpu_nvenc
