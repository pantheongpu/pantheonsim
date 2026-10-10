// A software HEVC (ITU-T H.265) decoder: Main and Main 10 profile pictures (and the 4:2:0 range
// extension tools at 12 bits) -- CABAC, I, P and B slices, coding trees of 8x8 to 64x64 samples, intra
// prediction with 35 modes, merge and AMVP motion vector prediction with temporal candidates, the
// transforms and transform skip, scaling lists, PCM, lossless coding, tiles, wavefront entropy sync,
// dependent slice segments, the deblocking filter and sample adaptive offset. Written from Rec. ITU-T
// H.265 v4 (12/2016) clauses 6 to 9 and nothing else. It is bit-exact by construction: HEVC defines
// the output sample for sample, and the card's NVDEC and FFmpeg's decoder agree with it on every stream
// in nvidia/tests/data/hevc.
//
// VirtualGPU's NVDEC (nvcuvid_api.cpp) calls decode_picture() once per cuvidDecodePicture with the
// parameters the application's CUVIDHEVCPICPARAMS carries and the slice segment NAL units; the DPB,
// picture order counts and the reference picture set are the caller's (the video parser's,
// hevc_parser.hpp), exactly as for the hardware: the slice segment headers are parsed here, from the
// bits NVDEC is given.
//
// Not decoded (decode_picture returns false and says why): chroma formats other than 4:2:0,
// extended_precision_processing, cabac_bypass_alignment, and the screen content, 3D and multilayer
// extensions. The card's NVDEC reports the same combinations in cuvidGetDecoderCaps.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vgpu_hevc {

// The motion of a 16x16 block of a decoded picture, kept for temporal motion vector prediction: the vectors, and the
// order counts of the pictures they refer to with whether each was a long-term reference (8.5.3.2.9).
struct ColMv {
  int16_t mv[2][2] = {};
  int32_t ref_poc[2] = {0, 0};
  uint8_t flags = 0;   // bit 0: list 0 used, bit 1: list 1 used, bit 2: list 0 reference is long-term, bit 3: list 1 likewise
};

// A frame store: the samples of a picture at 4:2:0 and the motion kept for the pictures that use it as the collocated picture.
struct Frame {
  int uid = 0;
  int width = 0, height = 0;          // luma samples
  int cwidth = 0, cheight = 0;
  int stride_y = 0, stride_c = 0;
  std::vector<uint16_t> y, u, v;
  int poc = 0;
  std::vector<ColMv> col;
  int col_w = 0, col_h = 0;           // in 16x16 blocks
  void alloc(int w, int h);
};

struct RefPic {
  Frame* frame = nullptr;
  int poc = 0;
  bool long_term = false;
};

// Everything a picture's decoding depends on that is not in the slice segment NAL units: the sequence and picture parameter set
// fields of CUVIDHEVCPICPARAMS, with the reference picture set the video parser derived.
struct PicParams {
  // ---- sequence parameter set
  int width = 0, height = 0;
  int log2_min_cb = 3, log2_diff_cb = 0, log2_min_tb = 2, log2_diff_tb = 0;
  int max_th_depth_intra = 0, max_th_depth_inter = 0;
  bool amp = false, sao = false, scaling_list_enabled = false, strong_intra_smoothing = false, sps_temporal_mvp = false;
  bool pcm = false, pcm_loop_filter_disabled = false;
  int pcm_bit_depth_luma = 8, pcm_bit_depth_chroma = 8, log2_min_pcm = 3, log2_diff_pcm = 0;
  int bit_depth_luma = 8, bit_depth_chroma = 8;
  int chroma_format_idc = 1;
  bool separate_colour_plane = false;
  int log2_max_poc_lsb = 4;
  int num_short_term_ref_pic_sets = 0;
  bool long_term_ref_pics_present = false;
  int num_long_term_ref_pics_sps = 0;
  bool irap = false, idr = false;
  // range extension
  bool sps_range_extension = false, pps_range_extension = false;
  bool transform_skip_rotation = false, transform_skip_context = false, implicit_rdpcm = false, explicit_rdpcm = false;
  bool extended_precision = false, intra_smoothing_disabled = false, high_precision_offsets = false;
  bool persistent_rice_adaptation = false, cabac_bypass_alignment = false;
  int log2_max_transform_skip = 2;
  int log2_sao_offset_scale_luma = 0, log2_sao_offset_scale_chroma = 0;
  bool cross_component_prediction = false, chroma_qp_offset_list_enabled = false;
  int diff_cu_chroma_qp_offset_depth = 0, chroma_qp_offset_list_len = 0;
  int cb_qp_offset_list[6] = {}, cr_qp_offset_list[6] = {};
  // ---- picture parameter set
  bool dependent_slice_segments_enabled = false, slice_segment_header_extension_present = false;
  bool sign_data_hiding = false, cu_qp_delta_enabled = false;
  int diff_cu_qp_delta_depth = 0;
  int init_qp = 26;
  int cb_qp_offset = 0, cr_qp_offset = 0;
  bool constrained_intra_pred = false, weighted_pred = false, weighted_bipred = false, transform_skip_enabled = false;
  bool transquant_bypass_enabled = false, entropy_coding_sync = false;
  int log2_parallel_merge_level = 2;
  int num_extra_slice_header_bits = 0;
  bool loop_filter_across_tiles = true, loop_filter_across_slices = false, output_flag_present = false;
  int num_ref_idx_default[2] = {1, 1};
  bool lists_modification_present = false, cabac_init_present = false, slice_chroma_qp_offsets_present = false;
  bool deblocking_override_enabled = false, pps_deblocking_disabled = false;
  int pps_beta_offset_div2 = 0, pps_tc_offset_div2 = 0;
  bool tiles_enabled = false, uniform_spacing = true;
  int num_tile_cols = 1, num_tile_rows = 1;
  int column_width_minus1[21] = {}, row_height_minus1[21] = {};
  // ---- reference picture set
  int num_bits_st_rps = 0;
  int num_poc_total_curr = 0, num_st_before = 0, num_st_after = 0, num_lt_curr = 0;
  int curr_poc = 0;
  RefPic refs[16];                    // by slot; frame == nullptr: no picture
  uint8_t st_before[8] = {}, st_after[8] = {}, lt_curr[8] = {};   // slots of refs[]
  // ---- scaling lists (diagonal scan order), as resolved by the caller
  uint8_t sl4[6][16] = {};
  uint8_t sl8[6][64] = {}, sl16[6][64] = {};
  uint8_t sl32[2][64] = {};
  uint8_t dc16[6] = {}, dc32[2] = {};
};

// One slice segment NAL unit: the bytes after the start code, starting with the two-byte NAL unit header.
struct SliceData {
  const uint8_t* nal = nullptr;
  size_t size = 0;
};

// Decodes the slice segments of one picture into `cur` (allocated by the caller to the picture size). Returns false (and an
// explanation) if the picture could not be decoded completely; the samples that were decoded stay in `cur`.
bool decode_picture(Frame* cur, const PicParams& pp, const std::vector<SliceData>& slices, std::string* error);

}  // namespace vgpu_hevc
