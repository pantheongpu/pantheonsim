// See mpeg2_syntax.hpp.
#include "mpeg2_syntax.hpp"

#include <cstring>

namespace vgpu_mpeg2 {
namespace {

class Rd {
 public:
  Rd(const uint8_t* d, size_t n) : d_(d), n_(n) {}
  uint32_t get(int bits) {
    uint32_t v = 0;
    for (int i = 0; i < bits; ++i) {
      const size_t byte = pos_ >> 3;
      const uint32_t bit = byte < n_ ? (d_[byte] >> (7 - (pos_ & 7))) & 1u : (over_ = true, 0u);
      v = (v << 1) | bit;
      ++pos_;
    }
    return v;
  }
  bool over() const { return over_; }

 private:
  const uint8_t* d_;
  size_t n_;
  size_t pos_ = 0;
  bool over_ = false;
};

// zigzag scan of Figure 7-2, n -> v * 8 + u
const uint8_t kZigzag[64] = {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                             41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                             30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
const uint8_t kDefaultIntra[64] = {8,  16, 19, 22, 26, 27, 29, 34, 16, 16, 22, 24, 27, 29, 34, 37, 19, 22, 26, 27, 29, 34,
                                   34, 38, 22, 22, 26, 27, 29, 34, 37, 40, 22, 26, 27, 29, 32, 35, 40, 48, 26, 27, 29, 32,
                                   35, 40, 48, 58, 26, 27, 29, 34, 38, 46, 56, 69, 27, 29, 35, 38, 46, 56, 69, 83};

void read_matrix(Rd& r, uint8_t* m) {
  for (int i = 0; i < 64; ++i) m[kZigzag[i]] = static_cast<uint8_t>(r.get(8));
}

}  // namespace

void default_matrices(uint8_t* intra, uint8_t* inter) {
  std::memcpy(intra, kDefaultIntra, 64);
  std::memset(inter, 16, 64);
}

bool parse_sequence_header(const uint8_t* d, size_t n, SeqHeader* h) {
  Rd r(d, n);
  h->horizontal_size = static_cast<int>(r.get(12));
  h->vertical_size = static_cast<int>(r.get(12));
  h->aspect_ratio_information = static_cast<int>(r.get(4));
  h->frame_rate_code = static_cast<int>(r.get(4));
  h->bit_rate = r.get(18);
  r.get(1);
  h->vbv_buffer_size = static_cast<int>(r.get(10));
  h->constrained_parameters = r.get(1);
  default_matrices(h->intra_matrix, h->inter_matrix);
  h->loaded_intra = h->loaded_inter = false;
  if (r.get(1)) {
    read_matrix(r, h->intra_matrix);
    h->loaded_intra = true;
  }
  if (r.get(1)) {
    read_matrix(r, h->inter_matrix);
    h->loaded_inter = true;
  }
  h->has_extension = h->has_display = false;
  h->progressive_sequence = 0;
  h->chroma_format = 1;
  h->low_delay = 0;
  h->frame_rate_ext_n = h->frame_rate_ext_d = 0;
  return !r.over();
}

bool parse_sequence_extension(const uint8_t* d, size_t n, SeqHeader* h) {
  Rd r(d, n);
  if (r.get(4) != 1) return false;
  h->profile_and_level = static_cast<int>(r.get(8));
  h->progressive_sequence = static_cast<int>(r.get(1));
  h->chroma_format = static_cast<int>(r.get(2));
  h->horizontal_size |= static_cast<int>(r.get(2)) << 12;
  h->vertical_size |= static_cast<int>(r.get(2)) << 12;
  h->bit_rate |= r.get(12) << 18;
  r.get(1);
  h->vbv_buffer_size |= static_cast<int>(r.get(8)) << 10;
  h->low_delay = static_cast<int>(r.get(1));
  h->frame_rate_ext_n = static_cast<int>(r.get(2));
  h->frame_rate_ext_d = static_cast<int>(r.get(5));
  h->has_extension = true;
  return !r.over();
}

bool parse_sequence_display_extension(const uint8_t* d, size_t n, SeqHeader* h) {
  Rd r(d, n);
  if (r.get(4) != 2) return false;
  h->video_format = static_cast<int>(r.get(3));
  h->colour_description = static_cast<int>(r.get(1));
  if (h->colour_description) {
    h->colour_primaries = static_cast<int>(r.get(8));
    h->transfer_characteristics = static_cast<int>(r.get(8));
    h->matrix_coefficients = static_cast<int>(r.get(8));
  } else {
    h->colour_primaries = h->transfer_characteristics = h->matrix_coefficients = 2;
  }
  h->display_horizontal_size = static_cast<int>(r.get(14));
  r.get(1);
  h->display_vertical_size = static_cast<int>(r.get(14));
  h->has_display = true;
  return !r.over();
}

bool parse_quant_matrix_extension(const uint8_t* d, size_t n, uint8_t* intra, uint8_t* inter, bool* loaded_intra, bool* loaded_inter) {
  Rd r(d, n);
  if (r.get(4) != 3) return false;
  if (r.get(1)) {
    read_matrix(r, intra);
    *loaded_intra = true;
  }
  if (r.get(1)) {
    read_matrix(r, inter);
    *loaded_inter = true;
  }
  // the chroma matrices only apply to 4:2:2 and 4:4:4
  return !r.over();
}

bool parse_gop_header(const uint8_t* d, size_t n, GopHeader* h) {
  Rd r(d, n);
  h->time_code = r.get(25);
  h->closed_gop = r.get(1);
  h->broken_link = r.get(1);
  return !r.over();
}

bool parse_picture_header(const uint8_t* d, size_t n, PicHeader* h) {
  Rd r(d, n);
  *h = PicHeader();
  h->temporal_reference = static_cast<int>(r.get(10));
  h->picture_coding_type = static_cast<int>(r.get(3));
  h->vbv_delay = static_cast<int>(r.get(16));
  if (h->picture_coding_type == 2 || h->picture_coding_type == 3) {
    h->full_pel_forward = r.get(1);
    h->forward_f_code = static_cast<int>(r.get(3));
  }
  if (h->picture_coding_type == 3) {
    h->full_pel_backward = r.get(1);
    h->backward_f_code = static_cast<int>(r.get(3));
  }
  return !r.over();
}

bool parse_picture_coding_extension(const uint8_t* d, size_t n, PicHeader* h) {
  Rd r(d, n);
  if (r.get(4) != 8) return false;
  for (int s = 0; s < 2; ++s)
    for (int t = 0; t < 2; ++t) h->f_code[s][t] = static_cast<int>(r.get(4));
  h->intra_dc_precision = static_cast<int>(r.get(2));
  h->picture_structure = static_cast<int>(r.get(2));
  h->top_field_first = r.get(1);
  h->frame_pred_frame_dct = r.get(1);
  h->concealment_motion_vectors = r.get(1);
  h->q_scale_type = r.get(1);
  h->intra_vlc_format = r.get(1);
  h->alternate_scan = r.get(1);
  h->repeat_first_field = r.get(1);
  h->chroma_420_type = r.get(1);
  h->progressive_frame = r.get(1);
  h->composite_display = r.get(1);
  h->has_extension = true;
  return !r.over();
}

}  // namespace vgpu_mpeg2
