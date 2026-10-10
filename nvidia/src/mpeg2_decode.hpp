// A software MPEG-2 video (ITU-T H.262 | ISO/IEC 13818-2) decoder for Main profile pictures: 4:2:0, frame pictures and field
// pictures, I, P and B pictures, frame and field motion compensation, 16x8 motion compensation, dual prime, concealment motion
// vectors, field and frame DCTs, both scans, both coefficient tables, both quantiser scale tables, all four DC precisions and
// downloaded quantiser matrices. Written from Rec. ITU-T H.262 (02/2000) clauses 6 and 7 and Annex B, and the card's output.
//
// VirtualGPU's NVDEC (nvcuvid_api.cpp) calls decode_picture() once per cuvidDecodePicture with the fields CUVIDMPEG2PICPARAMS
// carries and the slice data; the parser (mpeg2_parser.hpp) does the headers, reference selection and display order, as the
// hardware's parser does.
//
// The inverse DCT is not specified bit for bit (Annex A only bounds its error, IEEE 1180), so decoders differ in the last bit of some
// samples. mpeg2_idct.inc is the card's: see there.
//
// Not decoded (decode_picture returns false and says why): chroma formats other than 4:2:0, scalable extensions and MPEG-1 streams.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vgpu_mpeg2 {

// A frame store: both fields of a frame, 4:2:0, 8 bits per sample. Dimensions are those of the coded macroblock array.
struct Frame {
  int width = 0, height = 0;          // luma samples (multiples of 16)
  int stride_y = 0, stride_c = 0;
  std::vector<uint8_t> y, u, v;
  void alloc(int w, int h);
};

// What CUVIDMPEG2PICPARAMS and CUVIDPICPARAMS hand the hardware decoder.
struct PicParams {
  int mb_width = 0, mb_height = 0;    // of the frame (PicWidthInMbs, FrameHeightInMbs)
  bool field_pic = false, bottom_field = false, second_field = false;
  int picture_coding_type = 1;        // 1 I, 2 P, 3 B
  bool full_pel_forward = false, full_pel_backward = false;   // MPEG-1 only
  int f_code[2][2] = {{15, 15}, {15, 15}};
  int intra_dc_precision = 0;
  bool frame_pred_frame_dct = false, concealment_motion_vectors = false, q_scale_type = false, intra_vlc_format = false;
  bool alternate_scan = false, top_field_first = false;
  uint8_t intra_matrix[64] = {}, inter_matrix[64] = {};   // raster order: [v * 8 + u]
};

// One slice: the bytes from its slice_start_code (00 00 01 xx) to the next start code.
struct SliceData {
  const uint8_t* data = nullptr;
  size_t size = 0;
};

// Decodes the slices of one picture into `cur` (allocated by the caller to the frame size). `fwd` and `bwd` are the reference frames of
// P and B pictures; for the second field of a P frame the first field of `cur` is a reference too. Returns false (and an explanation) if
// the picture could not be decoded completely; the samples that were decoded stay in `cur`.
bool decode_picture(Frame* cur, const Frame* fwd, const Frame* bwd, const PicParams& pp, const std::vector<SliceData>& slices, std::string* error);

}  // namespace vgpu_mpeg2
