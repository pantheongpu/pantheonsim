// Internal types of the H.264 decoder (h264_decode.cpp and its parts): the
// per-macroblock record, the CABAC engine, and the decoder object.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "h264_decode.hpp"

namespace vgpu_h264 {

inline int clip3(int lo, int hi, int v) { return v < lo ? lo : (v > hi ? hi : v); }
inline uint8_t clip1(int v) { return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v)); }
inline int median3(int a, int b, int c) { return std::max(std::min(a, b), std::min(std::max(a, b), c)); }

// Raster position of the 4x4 block luma4x4BlkIdx inside a macroblock, in 4x4 units.
inline int blk_x(int idx) { return ((idx >> 2) & 1) * 2 + (idx & 1); }
inline int blk_y(int idx) { return ((idx >> 3) & 1) * 2 + ((idx >> 1) & 1); }
inline int blk_raster(int idx) { return blk_y(idx) * 4 + blk_x(idx); }

enum MbKind : uint8_t { kI4x4 = 0, kI8x8, kI16x16, kIPCM, kInter };

// What a macroblock leaves behind: for the neighbours that entropy decoding and
// prediction look at, for the deblocking filter, and (in a reference picture) for
// direct prediction of later B pictures.
struct MbInfo {
  int16_t slice = -1;             // slice number within the picture; -1 = not decoded
  uint8_t kind = kInter;
  bool intra = false, skip = false, field = false, t8x8 = false, direct16 = false;
  uint8_t cbp = 0;                // luma bits 0..3, chroma in bits 4..5
  uint8_t chroma_mode = 0, i16_mode = 0;
  int8_t qp = 0;                  // QPY
  int8_t qpc[2] = {0, 0};         // QPC of Cb and Cr (for deblocking)
  bool qp_delta_nonzero = false;  // for the CABAC context of the next mb_qp_delta
  int8_t ipred[16] = {};          // Intra4x4PredMode (raster 4x4 order), or Intra8x8PredMode replicated
  uint8_t nz[3][16] = {};         // coefficients per block: luma raster 4x4; Cb and Cr raster 2x2 (4 used)
  uint8_t cbf_dc = 0;             // CABAC coded_block_flag of the DC blocks: bit 0 luma (Intra16x16), 1 Cb, 2 Cr
  uint8_t pred_flags[4] = {};     // per 8x8: bit 0 list 0, bit 1 list 1 used for prediction (any mode)
  uint8_t coded_flags[4] = {};    // per 8x8: lists whose ref_idx / mvd were coded (not direct, not skipped)
  uint8_t direct8 = 0;            // bit per 8x8: predicted by direct mode
  int8_t ref_idx[2][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};
  int32_t ref_id[2][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}};   // identity of the referenced picture
  int16_t mv[2][16][2] = {};      // raster 4x4
  uint8_t mvd[2][16][2] = {};     // |mvd| clipped to 255, raster 4x4
};

// Slice parameters the deblocking filter needs.
struct SliceParams {
  int disable_idc = 0, alpha_off = 0, beta_off = 0;
  bool is_b = false;
};

struct PicData {
  int mbs_w = 0, mbs_h = 0;       // of this picture (a field has half the frame's height)
  bool field_pic = false, bottom = false, mbaff = false;
  int poc[2] = {0, 0};
  std::vector<MbInfo> mb;
  std::vector<SliceParams> slices;
};

// A reference picture as one slice sees it.
struct RefPic {
  Frame* frame = nullptr;
  int structure = 0;              // 0 frame, 1 top field, 2 bottom field
  int poc = 0;
  bool long_term = false;
  int id = -1;                    // frame uid * 4 + structure
  int fn_wrap = 0;                // PicNum or LongTermPicNum
};

// ---- CABAC ------------------------------------------------------------------------------
class Cabac {
 public:
  void init_contexts(int slice_qp, int idc, bool intra_slice) {
    const int set = intra_slice ? 0 : idc + 1;
    const int qp = clip3(0, 51, slice_qp);
    for (int i = 0; i < 1024; ++i) {
      const int m = kCabacInit[i][set][0], n = kCabacInit[i][set][1];
      const int pre = clip3(1, 126, ((m * qp) >> 4) + n);
      if (pre <= 63) {
        state_[i] = static_cast<uint8_t>(63 - pre);
        mps_[i] = 0;
      } else {
        state_[i] = static_cast<uint8_t>(pre - 64);
        mps_[i] = 1;
      }
    }
  }
  void start(BitReader* br) {
    br_ = br;
    range_ = 510;
    offset_ = br->u(9);
  }
  int decision(int ctx) {
    uint8_t& st = state_[ctx];
    const uint32_t rlps = kRangeTabLps[st][(range_ >> 6) & 3];
    range_ -= rlps;
    int bin;
    if (offset_ >= range_) {
      bin = !mps_[ctx];
      offset_ -= range_;
      range_ = rlps;
      if (st == 0) mps_[ctx] = static_cast<uint8_t>(1 - mps_[ctx]);
      st = kTransIdxLps[st];
    } else {
      bin = mps_[ctx];
      if (st < 62) ++st;
    }
    while (range_ < 256) {
      range_ <<= 1;
      offset_ = offset_ << 1 | br_->bit();
    }
    return bin;
  }
  int bypass() {
    offset_ = offset_ << 1 | br_->bit();
    if (offset_ >= range_) {
      offset_ -= range_;
      return 1;
    }
    return 0;
  }
  int terminate() {
    range_ -= 2;
    if (offset_ >= range_) return 1;
    while (range_ < 256) {
      range_ <<= 1;
      offset_ = offset_ << 1 | br_->bit();
    }
    return 0;
  }
  BitReader* reader() { return br_; }

 private:
  static constexpr uint8_t kTransIdxLps[64] = {0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9,  11, 11, 12, 13, 13, 15, 15, 16, 16,
                                               18, 18, 19, 19, 21, 21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30,
                                               31, 32, 32, 33, 33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63};
  uint8_t state_[1024];
  uint8_t mps_[1024];
  uint32_t range_ = 510, offset_ = 0;
  BitReader* br_ = nullptr;
};

}  // namespace vgpu_h264
