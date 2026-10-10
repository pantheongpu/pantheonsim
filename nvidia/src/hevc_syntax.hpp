// HEVC (ITU-T H.265) bitstream syntax: NAL unit headers, the RBSP bit reader, sequence and picture
// parameter sets (with the VUI fields a decoder front end reports), short-term reference picture sets,
// scaling lists and slice segment headers. Written from clauses 7.3 and 7.4 of Rec. ITU-T H.265 v4
// (12/2016) and Annex E; shared by VirtualGPU's NVDEC (nvcuvid_api.cpp), the software decoder behind
// it (hevc_decode.cpp), the video parser (hevc_parser.cpp) and the tests.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgpu_hevc {

// ---- bit reader ---------------------------------------------------------------------------
class BitReader {
 public:
  BitReader() = default;
  BitReader(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  uint32_t bit() {
    if (pos_ >= n_ * 8) {
      ++pos_;
      overrun_ = true;
      return 0;
    }
    const uint32_t b = (d_[pos_ >> 3] >> (7 - (pos_ & 7))) & 1;
    ++pos_;
    return b;
  }
  uint32_t u(int bits) {
    uint32_t v = 0;
    for (int i = 0; i < bits; ++i) v = v << 1 | bit();
    return v;
  }
  void skip(size_t bits) { pos_ += bits; if (pos_ > n_ * 8) overrun_ = true; }
  uint32_t ue() {
    int zeros = 0;
    while (!bit()) {
      if (++zeros > 32 || overrun_) {
        overrun_ = true;
        return 0;
      }
    }
    if (zeros == 32) return 0xFFFFFFFFu;
    return ((1u << zeros) - 1) + u(zeros);
  }
  int32_t se() {
    const uint32_t k = ue();
    return (k & 1) ? static_cast<int32_t>((k + 1) >> 1) : -static_cast<int32_t>(k >> 1);
  }
  size_t pos() const { return pos_; }
  void set_pos(size_t p) { pos_ = p; }
  size_t size_bits() const { return n_ * 8; }
  bool overrun() const { return overrun_; }
  bool byte_aligned() const { return (pos_ & 7) == 0; }
  const uint8_t* data() const { return d_; }
  size_t bytes() const { return n_; }
  // more_rbsp_data(): is there anything before the rbsp_trailing_bits (the last 1 bit)?
  bool more_rbsp_data() const {
    size_t last = n_;
    while (last > 0 && d_[last - 1] == 0) --last;
    if (last == 0) return false;
    const uint8_t b = d_[last - 1];
    int tz = 0;
    while (!((b >> tz) & 1)) ++tz;
    const size_t stop_bit = (last - 1) * 8 + static_cast<size_t>(7 - tz);
    return pos_ < stop_bit;
  }

 private:
  const uint8_t* d_ = nullptr;
  size_t n_ = 0;
  size_t pos_ = 0;
  bool overrun_ = false;
};

// ---- NAL units ----------------------------------------------------------------------------
enum NalType {
  kTrailN = 0, kTrailR = 1, kTsaN = 2, kTsaR = 3, kStsaN = 4, kStsaR = 5, kRadlN = 6, kRadlR = 7, kRaslN = 8, kRaslR = 9,
  kBlaWLp = 16, kBlaWRadl = 17, kBlaNLp = 18, kIdrWRadl = 19, kIdrNLp = 20, kCra = 21, kRsvIrap22 = 22, kRsvIrap23 = 23,
  kVps = 32, kSps = 33, kPps = 34, kAud = 35, kEos = 36, kEob = 37, kFd = 38, kPrefixSei = 39, kSuffixSei = 40
};

inline bool is_irap(int t) { return t >= 16 && t <= 23; }
inline bool is_idr(int t) { return t == kIdrWRadl || t == kIdrNLp; }
inline bool is_bla(int t) { return t >= kBlaWLp && t <= kBlaNLp; }
inline bool is_rasl(int t) { return t == kRaslN || t == kRaslR; }
inline bool is_radl(int t) { return t == kRadlN || t == kRadlR; }
// sub-layer non-reference pictures: even types below 16 (TRAIL_N, TSA_N, STSA_N, RADL_N, RASL_N, RSV_VCL_N10/12/14)
inline bool is_sub_layer_non_ref(int t) { return t < 16 && (t & 1) == 0; }

// The RBSP of a NAL unit: emulation prevention bytes (00 00 03) removed; `skip` bytes of header are dropped.
// `removed` (when given) receives, for each dropped byte, its offset within the input after `skip`.
inline std::vector<uint8_t> unescape(const uint8_t* d, size_t n, size_t skip, std::vector<uint32_t>* removed = nullptr) {
  std::vector<uint8_t> out;
  out.reserve(n);
  int zeros = 0;
  for (size_t i = skip; i < n; ++i) {
    if (zeros >= 2 && d[i] == 3) {
      zeros = 0;
      if (removed) removed->push_back(static_cast<uint32_t>(i - skip));
      continue;
    }
    out.push_back(d[i]);
    zeros = d[i] == 0 ? zeros + 1 : 0;
  }
  return out;
}

// ---- parameter sets -----------------------------------------------------------------------
// A short-term reference picture set (7.4.8), in the form of equations 7-61 to 7-71.
struct StRps {
  int num_neg = 0, num_pos = 0;
  int delta_s0[16] = {}, delta_s1[16] = {};
  bool used_s0[16] = {}, used_s1[16] = {};
  int num_delta() const { return num_neg + num_pos; }
};

// Scaling lists (7.3.4, 7.4.5): ScalingList[sizeId][matrixId][i] in up-right diagonal scan order.
// For sizeId 3 only matrixId 0 and 3 are coded; they are kept in slots 0 and 1.
struct ScalingLists {
  uint8_t list[4][6][64];
  int dc[2][6];          // scaling_list_dc_coef_minus8 + 8 for sizeId 2 and 3
  ScalingLists() { set_default(); }
  void set_default();
  // Reads scaling_list_data(); false if the data is malformed.
  bool parse(BitReader& b);
};

struct Sps {
  bool valid = false;
  bool unsupported_extension = false;   // a multilayer, 3D or screen content extension this decoder does not implement
  int vps_id = 0, max_sub_layers = 1, id = 0;
  int profile_idc = 0, level_idc = 0, tier = 0;
  uint32_t profile_compat = 0;
  int chroma_format_idc = 1, separate_colour_plane = 0;
  int width = 0, height = 0;                 // pic_width_in_luma_samples, pic_height_in_luma_samples
  bool conformance_window = false;
  int conf_win[4] = {0, 0, 0, 0};            // left, right, top, bottom offsets in chroma units
  int bit_depth_luma = 8, bit_depth_chroma = 8;
  int log2_max_poc_lsb = 4;
  bool sub_layer_ordering_info_present = false;
  int max_dec_pic_buffering[8] = {}, max_num_reorder[8] = {}, max_latency_increase_plus1[8] = {};   // the _minus1 for the first
  int log2_min_cb = 3, log2_ctb = 4;         // MinCbLog2SizeY, CtbLog2SizeY
  int log2_min_tb = 2, log2_max_tb = 5;      // MinTbLog2SizeY, MaxTbLog2SizeY
  int max_th_depth_inter = 0, max_th_depth_intra = 0;
  bool scaling_list_enabled = false, scaling_list_data_present = false;
  ScalingLists scaling;
  bool amp = false, sao = false, pcm = false;
  int pcm_bit_depth_luma = 8, pcm_bit_depth_chroma = 8, log2_min_pcm_cb = 3, log2_max_pcm_cb = 3;
  bool pcm_loop_filter_disabled = false;
  std::vector<StRps> st_rps;
  bool long_term_present = false;
  int num_lt_sps = 0;
  int lt_poc_lsb_sps[33] = {};
  bool lt_used_sps[33] = {};
  bool temporal_mvp = false, strong_intra_smoothing = false;
  // range extension
  bool range_extension = false;              // sps_range_extension_flag
  bool transform_skip_rotation = false, transform_skip_context = false, implicit_rdpcm = false, explicit_rdpcm = false;
  bool extended_precision = false, intra_smoothing_disabled = false, high_precision_offsets = false;
  bool persistent_rice_adaptation = false, cabac_bypass_alignment = false;
  // VUI
  bool vui_present = false;
  bool aspect_present = false;
  int aspect_idc = 0, sar_w = 0, sar_h = 0;
  bool video_signal_present = false;
  int video_format = 5, video_full_range = 0;
  bool colour_description_present = false;
  int colour_primaries = 2, transfer = 2, matrix = 2;
  bool field_seq = false, frame_field_info_present = false;
  bool default_display_window = false;
  int def_disp_win[4] = {0, 0, 0, 0};
  bool timing_present = false;
  uint32_t num_units_in_tick = 0, time_scale = 0;
  bool hrd_present = false;
  uint32_t hrd_bit_rate = 0;                 // bit rate of the first CPB of the highest sub-layer, in bits per second
  bool restriction_present = false;
  // derived
  int sub_width_c() const { return chroma_format_idc == 1 || chroma_format_idc == 2 ? 2 : 1; }
  int sub_height_c() const { return chroma_format_idc == 1 ? 2 : 1; }
  int chroma_array_type() const { return separate_colour_plane ? 0 : chroma_format_idc; }
  int ctb_size() const { return 1 << log2_ctb; }
  int width_ctbs() const { return (width + ctb_size() - 1) >> log2_ctb; }
  int height_ctbs() const { return (height + ctb_size() - 1) >> log2_ctb; }
  int highest_tid() const { return max_sub_layers - 1; }
};

struct Pps {
  bool valid = false;
  bool unsupported_extension = false;
  int id = 0, sps_id = 0;
  bool dependent_slice_segments_enabled = false, output_flag_present = false;
  int num_extra_slice_header_bits = 0;
  bool sign_data_hiding = false, cabac_init_present = false;
  int num_ref_idx_default[2] = {1, 1};
  int init_qp = 26;                          // 26 + init_qp_minus26
  bool constrained_intra_pred = false, transform_skip_enabled = false, cu_qp_delta_enabled = false;
  int diff_cu_qp_delta_depth = 0;
  int cb_qp_offset = 0, cr_qp_offset = 0;
  bool slice_chroma_qp_offsets_present = false, weighted_pred = false, weighted_bipred = false, transquant_bypass_enabled = false;
  bool tiles_enabled = false, entropy_coding_sync = false;
  int num_tile_cols = 1, num_tile_rows = 1;
  bool uniform_spacing = true;
  int column_width_minus1[20] = {}, row_height_minus1[20] = {};
  bool loop_filter_across_tiles = true, loop_filter_across_slices = false;
  bool deblocking_control_present = false, deblocking_override_enabled = false, deblocking_disabled = false;
  int beta_offset_div2 = 0, tc_offset_div2 = 0;
  bool scaling_list_data_present = false;
  ScalingLists scaling;
  bool lists_modification_present = false;
  int log2_parallel_merge_level = 2;
  bool slice_header_extension_present = false;
  // range extension
  bool range_extension = false;              // pps_range_extension_flag
  int log2_max_transform_skip_size = 2;
  bool cross_component_prediction = false, chroma_qp_offset_list_enabled = false;
  int diff_cu_chroma_qp_offset_depth = 0, chroma_qp_offset_list_len = 0;
  int cb_qp_offset_list[6] = {}, cr_qp_offset_list[6] = {};
  int log2_sao_offset_scale_luma = 0, log2_sao_offset_scale_chroma = 0;
};

// Parses an SPS / PPS RBSP (the bytes after the 2-byte NAL header, emulation prevention removed). The PPS needs the SPS table
// for its scaling-list handling only through `sps_by_id` (may be null).
bool parse_sps(const uint8_t* rbsp, size_t n, Sps* out);
bool parse_pps(const uint8_t* rbsp, size_t n, Pps* out);

// Tile layout (6.5.1): column boundaries and row boundaries in CTBs, the conversion tables between raster and tile scan, TileId.
struct TileLayout {
  int cols = 1, rows = 1;
  std::vector<int> col_bd, row_bd;           // cols+1 / rows+1 boundaries
  std::vector<int> rs_to_ts, ts_to_rs, tile_id;   // indexed by CtbAddrInRs / CtbAddrInTs
  std::vector<int> col_of_x, row_of_y;       // tile column of CTB column, tile row of CTB row
  bool build(const Sps& s, const Pps& p);    // false if the tile sizes do not fit the picture
};

// ---- slice segment headers ------------------------------------------------------------------
struct PredWeights {
  int luma_log2_denom = 0, chroma_log2_denom = 0;
  // [list][refIdx]: LumaWeightLX, luma_offset_lX; ChromaWeightLX[j], ChromaOffsetLX[j]
  int luma_weight[2][16], luma_offset[2][16], chroma_weight[2][16][2], chroma_offset[2][16][2];
  bool luma_flag[2][16], chroma_flag[2][16];
  PredWeights() { std::memset(this, 0, sizeof *this); }
};

enum SliceType { kSliceB = 0, kSliceP = 1, kSliceI = 2 };

struct SliceHeader {
  bool first_slice_segment_in_pic = false, no_output_of_prior_pics = false, dependent = false;
  int pps_id = 0;
  uint32_t segment_address = 0;
  int slice_type = kSliceI;
  bool pic_output_flag = true;
  int colour_plane_id = 0;
  int poc_lsb = 0;
  bool st_rps_sps_flag = false;
  int st_rps_idx = 0;
  StRps st_rps;                              // the RPS in use (from the SPS or coded in the header); only when parsed with the SPS table
  int st_rps_bits = 0;                       // bits the inline st_ref_pic_set() took in the header
  int st_rps_ref_num_delta = 0;              // NumDeltaPocs[ RefRpsIdx ] when the inline set is predicted from another, else 0
  int num_lt = 0;                            // num_long_term_sps + num_long_term_pics
  int lt_poc_lsb[33] = {};
  bool lt_used[33] = {};
  bool lt_msb_present[33] = {};
  int lt_delta_msb_cycle[33] = {};           // DeltaPocMsbCycleLt (cumulative)
  bool temporal_mvp = false;
  bool sao_luma = false, sao_chroma = false;
  int num_ref_idx[2] = {0, 0};               // num_ref_idx_lX_active_minus1 + 1
  bool list_mod[2] = {false, false};
  int list_entry[2][16] = {};
  bool mvd_l1_zero = false, cabac_init_flag = false, collocated_from_l0 = true;
  int collocated_ref_idx = 0;
  PredWeights pw;
  int max_num_merge_cand = 5;
  int slice_qp_delta = 0, cb_qp_offset = 0, cr_qp_offset = 0;
  bool cu_chroma_qp_offset_enabled = false;
  bool deblocking_override = false, deblocking_disabled = false;
  int beta_offset_div2 = 0, tc_offset_div2 = 0;
  bool loop_filter_across_slices = false;
  int num_entry_points = 0;
  std::vector<uint32_t> entry_offset;        // entry_point_offset_minus1[i] + 1
  int num_pic_total_curr = 0;
  size_t data_bit_offset = 0;                // RBSP bit offset of the first slice segment data bit (byte aligned)
  bool is_intra() const { return slice_type == kSliceI; }
};

// What slice header parsing depends on in the SPS and PPS (7.3.6.1). Built from parsed parameter sets (the video parser), or
// from the picture parameters an application hands cuvidDecodePicture (the decoder, which gets the derived values directly).
struct SliceCtx {
  int log2_ctb = 4, width_ctbs = 1, height_ctbs = 1;
  int log2_max_poc_lsb = 4;
  int num_st_rps = 0;
  const std::vector<StRps>* sps_rps = nullptr;      // when set, the RPS of the header is derived (parser); else skipped (decoder)
  int skip_st_rps_bits = 0;                         // decoder: NumBitsForShortTermRPSInSlice
  int fixed_num_pic_total_curr = -1;                // decoder: NumPicTotalCurr from the application
  bool long_term_present = false;
  int num_lt_sps = 0;
  const int* lt_poc_lsb_sps = nullptr;
  const bool* lt_used_sps = nullptr;
  bool temporal_mvp = false, sao = false;
  int chroma_array_type = 1;
  bool separate_colour_plane = false;
  bool high_precision_offsets = false;
  int bit_depth_chroma = 8;
  bool dependent_slice_segments_enabled = false, output_flag_present = false;
  int num_extra_slice_header_bits = 0;
  bool lists_modification_present = false, cabac_init_present = false;
  int num_ref_idx_default[2] = {1, 1};
  bool weighted_pred = false, weighted_bipred = false;
  bool slice_chroma_qp_offsets_present = false, chroma_qp_offset_list_enabled = false;
  bool deblocking_override_enabled = false, pps_deblocking_disabled = false;
  int pps_beta_offset_div2 = 0, pps_tc_offset_div2 = 0;
  bool pps_loop_filter_across_slices = false;
  bool tiles_enabled = false, entropy_coding_sync = false;
  bool slice_header_extension_present = false;
  int init_qp = 26;
};

// The id of the PPS a slice segment refers to (first fields of the header); false on a truncated header.
bool peek_slice_pps_id(const uint8_t* rbsp, size_t n, int nal_type, int* pps_id, bool* first_slice);

// Parses slice_segment_header() from `b` (positioned at the start of the RBSP after the NAL header). `prev` is the
// preceding independent slice segment header of the picture, required for a dependent slice segment. Returns false if the header
// is malformed; on success b is at the first bit of the slice segment data.
bool parse_slice_header(BitReader& b, const SliceCtx& ctx, int nal_type, const SliceHeader* prev, SliceHeader* out);

// The derived lists of a short-term RPS in the SPS table, for the video parser.
int ceil_log2(uint32_t v);

}  // namespace vgpu_hevc
