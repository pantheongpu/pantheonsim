// A software H.264 decoder: Baseline, Main and High profile pictures at 4:2:0 and
// 8 bits -- CAVLC and CABAC, I, P and B slices, 4x4 and 8x8 transforms, scaling
// matrices, explicit and implicit weighted prediction, spatial and temporal direct
// prediction, multiple slices, the deblocking filter, and interlaced coding (field
// pictures and macroblock-adaptive frame/field frames). Written from ITU-T H.264
// (03/2009) clauses 6 to 9 and nothing else. It is bit-exact by construction: H.264
// defines the output sample for sample, and the card's NVDEC and FFmpeg's decoder
// agree with it on every stream in nvidia/tests/data/h264 (frame pictures, progressive
// and macroblock-adaptive; no stream with field pictures has been available to check
// that path).
//
// VirtualGPU's NVDEC (nvcuvid_api.cpp) calls decode_picture() once per
// cuvidDecodePicture with the parameters the application's picture parameter
// structure carries and the slice NAL units; the DPB, picture order counts and
// reference marking are the caller's (the video parser's, h264_parser.hpp).
//
// Not decoded (decode_picture returns false and says why): flexible macroblock
// ordering / slice groups, redundant pictures, SP and SI slices, data partitioning,
// chroma formats other than 4:2:0, bit depths other than 8, lossless
// (qpprime_y_zero_transform_bypass) coding. The card's NVDEC refuses the same
// profiles in cuvidGetDecoderCaps.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "h264_syntax.hpp"

namespace vgpu_h264 {

struct PicData;   // per-macroblock mode and motion data of one coded picture

// A frame store: the samples of a frame (or of two fields, interleaved) and the
// motion data of the pictures decoded into it.
struct Frame {
  int uid = 0;                 // identifies the frame to the motion data that refers to it
  int mbs_w = 0, mbs_h = 0;    // frame size in macroblocks
  int stride_y = 0, stride_c = 0;
  std::vector<uint8_t> y, u, v;
  std::shared_ptr<PicData> pic[3];   // [0] frame, [1] top field, [2] bottom field (as coded)
  bool coded_as_frame = true;        // false once a field picture has been decoded into it
  int poc[2] = {0, 0};
  bool long_term = false;
  void alloc(int w_mbs, int h_mbs);
};

struct DpbEntry {
  Frame* frame = nullptr;
  int frame_num = 0;           // FrameNum of a short-term picture, LongTermFrameIdx of a long-term one
  bool long_term = false;
  bool not_existing = false;
  int used = 3;                // 1 top field, 2 bottom field, 3 both (a frame): which fields are used for reference
  int poc[2] = {0, 0};         // top, bottom
};

struct PicParams {
  // sequence
  int log2_max_frame_num = 4, poc_type = 0, log2_max_poc_lsb = 4, delta_pic_order_always_zero = 0;
  int frame_mbs_only = 1, direct_8x8_inference = 0, num_ref_frames = 1;
  int chroma_format_idc = 1, bit_depth_luma = 8, bit_depth_chroma = 8, transform_bypass = 0;
  // picture parameter set
  int cabac = 0, bottom_field_pic_order_in_frame_present = 0, redundant_pic_cnt_present = 0;
  int num_ref_idx_default[2] = {1, 1};
  int weighted_pred = 0, weighted_bipred_idc = 0;
  int pic_init_qp = 26, deblocking_control_present = 0, constrained_intra_pred = 0, transform_8x8_mode = 0;
  int chroma_qp_offset[2] = {0, 0};
  int num_slice_groups = 1, slice_group_map_type = 0;
  uint8_t ws4[6][16];          // weightScale4x4, raster order: Intra Y, Cb, Cr, Inter Y, Cb, Cr
  uint8_t ws8[2][64];          // weightScale8x8, raster order: Intra Y, Inter Y
  // picture
  int mbs_w = 0, mbs_h = 0;    // frame size in macroblocks
  int mbaff = 0, field_pic = 0, bottom_field = 0, second_field = 0;
  int frame_num = 0;
  int poc[2] = {0, 0};         // CurrFieldOrderCnt top, bottom
  int ref_pic = 0;             // nal_ref_idc != 0
  int idr = 0;
  std::vector<DpbEntry> dpb;   // reference pictures (frames, or frames with a field marked)
  PicParams() {
    std::memset(ws4, 16, sizeof ws4);
    std::memset(ws8, 16, sizeof ws8);
  }
};

// Decodes the slices of one picture into `cur`. `slices` are the RBSP-unescaped slice
// NAL units (header byte included, as they appear after the start code), in bitstream order.
// Returns false (and an explanation) if the picture cannot be decoded; the samples of
// macroblocks that were decoded stay in `cur`.
struct SliceData {
  const uint8_t* nal = nullptr;   // NAL unit bytes after the start code, starting with the NAL header byte
  size_t size = 0;
};
bool decode_picture(Frame* cur, const PicParams& pp, const std::vector<SliceData>& slices, std::string* error);

// The pictures in decoding order for the parameters of a parameter set pair: the HeaderCtx
// that slice-header parsing needs, from the decoder's parameters.
HeaderCtx header_ctx(const PicParams& pp);

}  // namespace vgpu_h264
