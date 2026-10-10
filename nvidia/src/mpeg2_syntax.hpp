// The MPEG-2 video headers (H.262 6.2.2, 6.2.3): sequence, group of pictures and picture level syntax and the extensions the video parser
// (mpeg2_parser.hpp) uses.
#pragma once

#include <cstddef>
#include <cstdint>

namespace vgpu_mpeg2 {

struct SeqHeader {
  int horizontal_size = 0, vertical_size = 0;    // with the extension bits
  int aspect_ratio_information = 0, frame_rate_code = 0;
  unsigned bit_rate = 0;                         // in units of 400 bit/s, with the extension bits
  int vbv_buffer_size = 0;
  bool constrained_parameters = false;
  uint8_t intra_matrix[64] = {}, inter_matrix[64] = {};   // raster order
  bool loaded_intra = false, loaded_inter = false;
  // sequence_extension
  bool has_extension = false;
  int profile_and_level = 0, progressive_sequence = 1, chroma_format = 1, low_delay = 0, frame_rate_ext_n = 0, frame_rate_ext_d = 0;
  // sequence_display_extension
  bool has_display = false;
  int video_format = 5, colour_description = 0, colour_primaries = 2, transfer_characteristics = 2, matrix_coefficients = 2;
  int display_horizontal_size = 0, display_vertical_size = 0;
};

struct GopHeader {
  uint32_t time_code = 0;
  bool closed_gop = false, broken_link = false;
};

struct PicHeader {
  int temporal_reference = 0, picture_coding_type = 0, vbv_delay = 0;
  bool full_pel_forward = false, full_pel_backward = false;
  int forward_f_code = 0, backward_f_code = 0;
  // picture_coding_extension
  bool has_extension = false;
  int f_code[2][2] = {{15, 15}, {15, 15}};
  int intra_dc_precision = 0, picture_structure = 3;
  bool top_field_first = false, frame_pred_frame_dct = false, concealment_motion_vectors = false, q_scale_type = false, intra_vlc_format = false;
  bool alternate_scan = false, repeat_first_field = false, chroma_420_type = false, progressive_frame = true, composite_display = false;
};

// The bytes after the start code. Return false if the header is cut short.
bool parse_sequence_header(const uint8_t* d, size_t n, SeqHeader* h);
bool parse_sequence_extension(const uint8_t* d, size_t n, SeqHeader* h);       // after the extension_start_code
bool parse_sequence_display_extension(const uint8_t* d, size_t n, SeqHeader* h);
// A quant_matrix_extension: updates the matrices in `h` (raster order), setting the loaded flags.
bool parse_quant_matrix_extension(const uint8_t* d, size_t n, uint8_t* intra, uint8_t* inter, bool* loaded_intra, bool* loaded_inter);
bool parse_gop_header(const uint8_t* d, size_t n, GopHeader* h);
bool parse_picture_header(const uint8_t* d, size_t n, PicHeader* h);
bool parse_picture_coding_extension(const uint8_t* d, size_t n, PicHeader* h);   // after the extension_start_code

// Table 7-6 etc. helpers.
void default_matrices(uint8_t* intra, uint8_t* inter);

}  // namespace vgpu_mpeg2
